// SarvamVoiceClient — replaces the OpenAI realtime WebRTC plumbing.
// Owns two WebSockets to the backend (TTS + STT proxies) and the WebAudio
// graph: mic capture worklet -> upstream WS, downstream WAV chunks -> playback
// worklet -> speakers.

export type SarvamSessionConfig = {
  session_id: string
  voice: string
  language_id: string
  language_code: string
  tts_language_code?: string
  stt_language_code?: string
  tts_ws_path: string
  stt_ws_path: string
  tts_sample_rate: number
  stt_sample_rate: number
  tts_model: string
  stt_model: string
  stt_mode?: string
}

type Listeners = {
  onReady?: () => void
  onPartialTranscript?: (text: string, languageCode: string) => void
  onFinalTranscript?: (text: string, languageCode: string) => void
  onPlaybackStart?: (utteranceId: string) => void
  onPlaybackEnd?: (utteranceId: string, chars: number) => void
  onPlaybackInterrupted?: (utteranceId: string) => void
  onError?: (message: string) => void
  onDisconnect?: () => void
}

function wsUrlFor(path: string): string {
  const base = (import.meta as unknown as { env: { VITE_API_BASE_URL?: string } }).env
    .VITE_API_BASE_URL
  if (base) {
    const u = new URL(base)
    u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:'
    u.pathname = (u.pathname.replace(/\/$/, '') + path) || path
    return u.toString()
  }
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}${path}`
}

export class SarvamVoiceClient {
  private config: SarvamSessionConfig | null = null
  private ttsWs: WebSocket | null = null
  private sttWs: WebSocket | null = null
  private audioCtx: AudioContext | null = null
  private playbackNode: AudioWorkletNode | null = null
  private micNode: AudioWorkletNode | null = null
  private micSource: MediaStreamAudioSourceNode | null = null
  private mediaStream: MediaStream | null = null
  private pendingAudioMeta: { utteranceId: string; sampleRate: number; format: string } | null = null
  private playbackState: { utteranceId: string; chars: number; serverEnded: boolean; receivedAudio: boolean } | null =
    null
  private currentUtteranceId: string | null = null
  private listeners: Listeners = {}
  private muted = false
  // Treat agent playback as active until the server has finished streaming the
  // utterance and the local playback queue has actually drained.
  private agentSpeaking = false
  private armMicTimer: number | null = null

  setListeners(listeners: Listeners): void {
    this.listeners = { ...this.listeners, ...listeners }
  }

  async connect(config: SarvamSessionConfig): Promise<void> {
    this.config = config
    await this._openTts()
    await this._openStt()
  }

  async startMic(stream: MediaStream): Promise<void> {
    if (!this.config) throw new Error('SarvamVoiceClient not connected')
    this.mediaStream = stream

    if (!this.audioCtx) {
      const Ctor =
        (window as unknown as { AudioContext: typeof AudioContext; webkitAudioContext: typeof AudioContext })
          .AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      this.audioCtx = new Ctor()
    }
    await this.audioCtx.audioWorklet.addModule('/mic-capture-worklet.js')
    await this.audioCtx.audioWorklet.addModule('/tts-playback-worklet.js')

    if (!this.playbackNode) {
      this.playbackNode = new AudioWorkletNode(this.audioCtx, 'tts-playback-processor')
      this.playbackNode.connect(this.audioCtx.destination)
      this.playbackNode.port.onmessage = (event) => {
        const data = event.data as { type?: string } | undefined
        if (data?.type === 'idle' && this.agentSpeaking) {
          const playback = this.playbackState
          if (!playback || !playback.serverEnded || !playback.receivedAudio) return
          this.playbackState = null
          this.currentUtteranceId = null
          this.agentSpeaking = false
          if (this.armMicTimer !== null) {
            window.clearTimeout(this.armMicTimer)
            this.armMicTimer = null
          }
          this.listeners.onPlaybackEnd?.(playback.utteranceId, playback.chars)
        }
      }
    }

    this.micSource = this.audioCtx.createMediaStreamSource(stream)
    this.micNode = new AudioWorkletNode(this.audioCtx, 'mic-capture-processor', {
      processorOptions: { targetSampleRate: this.config.stt_sample_rate },
    })
    this.micNode.port.onmessage = (event) => {
      if (this.muted) return
      const data = event.data
      // VAD control frame from the worklet.
      if (data && typeof data === 'object' && (data as { type?: string }).type === 'vad') {
        const state = (data as { state?: string }).state
        if (state === 'speech_end') {
          this.flushStt()
        } else if (state === 'drop') {
          this.discardSttBuffer()
        } else if (state === 'speech_start') {
          // Barge-in: cancel agent playback the moment a sustained mic-energy
          // window fires. Browser AEC (echoCancellation:true on getUserMedia)
          // suppresses our own TTS output, so this should rarely false-fire.
          // The worklet now requires 250ms of high-energy speech (threshold
          // 0.05) before posting speech_start, which filters out coughs and
          // breath.
          if (this.agentSpeaking) {
            this.cancelSpeech('barge_in')
          }
        }
        return
      }
      const buf = data as ArrayBuffer
      if (this.sttWs && this.sttWs.readyState === WebSocket.OPEN) {
        try {
          this.sttWs.send(buf)
        } catch {
          // socket closing
        }
      }
    }
    this.micSource.connect(this.micNode)
    // Don't connect mic to destination — we don't want self-echo.
  }

  speak(text: string, languageCode?: string, utteranceId?: string): string {
    if (!this.ttsWs || this.ttsWs.readyState !== WebSocket.OPEN) {
      this.listeners.onError?.('TTS socket not open')
      return ''
    }
    const id = utteranceId || `utt_${Math.random().toString(36).slice(2, 10)}`
    this.ttsWs.send(
      JSON.stringify({
        type: 'speak',
        utterance_id: id,
        text,
        language_code: languageCode || this.config?.tts_language_code || this.config?.language_code || 'hi-IN',
      }),
    )
    return id
  }

  cancelSpeech(reason: 'app' | 'barge_in' = 'app'): void {
    const interruptedUtteranceId = this.playbackState?.utteranceId ?? this.currentUtteranceId ?? null
    // 1. Flush the playback worklet IMMEDIATELY so already-buffered audio
    //    stops within one render quantum (~3 ms).
    this.playbackNode?.port.postMessage({ type: 'flush' })
    this.playbackState = null
    this.currentUtteranceId = reason === 'barge_in' ? interruptedUtteranceId : null
    this.agentSpeaking = false
    if (this.armMicTimer !== null) {
      window.clearTimeout(this.armMicTimer)
      this.armMicTimer = null
    }
    // 2. Tell backend to close upstream Sarvam TTS WS so no more chunks arrive.
    if (this.ttsWs && this.ttsWs.readyState === WebSocket.OPEN) {
      try {
        this.ttsWs.send(JSON.stringify({ type: 'cancel' }))
      } catch {
        // ignore
      }
    }
    if (reason === 'barge_in' && interruptedUtteranceId) {
      this.listeners.onPlaybackInterrupted?.(interruptedUtteranceId)
    }
  }

  flushStt(): void {
    if (this.sttWs && this.sttWs.readyState === WebSocket.OPEN) {
      try {
        this.sttWs.send(JSON.stringify({ type: 'flush' }))
      } catch {
        // ignore
      }
    }
  }

  discardSttBuffer(): void {
    if (this.sttWs && this.sttWs.readyState === WebSocket.OPEN) {
      try {
        this.sttWs.send(JSON.stringify({ type: 'discard' }))
      } catch {
        // ignore
      }
    }
  }

  setMicMuted(muted: boolean): void {
    this.muted = muted
  }

  disconnect(): void {
    if (this.armMicTimer !== null) {
      window.clearTimeout(this.armMicTimer)
      this.armMicTimer = null
    }
    this.agentSpeaking = false
    this.currentUtteranceId = null
    try {
      this.sttWs?.send(JSON.stringify({ type: 'stop' }))
    } catch {
      // ignore
    }
    this.ttsWs?.close()
    this.sttWs?.close()
    this.ttsWs = null
    this.sttWs = null

    this.micNode?.disconnect()
    this.micSource?.disconnect()
    this.playbackNode?.disconnect()
    this.micNode = null
    this.micSource = null
    this.playbackNode = null

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop())
      this.mediaStream = null
    }
    if (this.audioCtx) {
      void this.audioCtx.close()
      this.audioCtx = null
    }
    this.listeners.onDisconnect?.()
  }

  private async _openTts(): Promise<void> {
    if (!this.config) return
    const url = wsUrlFor(this.config.tts_ws_path)
    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'
    this.ttsWs = ws

    await new Promise<void>((resolve, reject) => {
      ws.addEventListener('open', () => {
        ws.send(
          JSON.stringify({
            type: 'hello',
            session_id: this.config?.session_id,
            voice: this.config?.voice,
            language_code: this.config?.tts_language_code || this.config?.language_code,
          }),
        )
        resolve()
      })
      ws.addEventListener('error', () => reject(new Error('TTS WebSocket failed to open')))
    })

    ws.addEventListener('message', (event) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data)
          this._handleTtsControl(msg)
        } catch {
          // ignore malformed
        }
        return
      }
      void this._handleTtsAudio(event.data as ArrayBuffer)
    })
    ws.addEventListener('close', () => {
      if (this.ttsWs === ws) this.ttsWs = null
    })
  }

  private async _openStt(): Promise<void> {
    if (!this.config) return
    const url = wsUrlFor(this.config.stt_ws_path)
    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'
    this.sttWs = ws

    await new Promise<void>((resolve, reject) => {
      ws.addEventListener('open', () => {
        ws.send(
          JSON.stringify({
            type: 'hello',
            session_id: this.config?.session_id,
            language_code: this.config?.stt_language_code || this.config?.language_code,
            sample_rate: this.config?.stt_sample_rate,
          }),
        )
        resolve()
      })
      ws.addEventListener('error', () => reject(new Error('STT WebSocket failed to open')))
    })

    ws.addEventListener('message', (event) => {
      if (typeof event.data !== 'string') return
      try {
        const msg = JSON.parse(event.data)
        this._handleSttControl(msg)
      } catch {
        // ignore
      }
    })
    ws.addEventListener('close', () => {
      if (this.sttWs === ws) this.sttWs = null
    })
  }

  private _handleTtsControl(msg: Record<string, unknown>): void {
    const type = msg.type as string | undefined
    if (type === 'ready') {
      this.listeners.onReady?.()
      return
    }
    if (type === 'audio_start') {
      const utteranceId = String(msg.utterance_id || '')
      this.currentUtteranceId = utteranceId
      const sampleRate = Number(msg.sample_rate || 22050)
      this.pendingAudioMeta = {
        utteranceId,
        sampleRate,
        format: String(msg.format || 'wav'),
      }
      this.playbackState = {
        utteranceId,
        chars: 0,
        serverEnded: false,
        receivedAudio: false,
      }
      this.agentSpeaking = true
      if (this.armMicTimer !== null) {
        window.clearTimeout(this.armMicTimer)
        this.armMicTimer = null
      }
      this.listeners.onPlaybackStart?.(utteranceId)
      return
    }
    if (type === 'audio_end') {
      const utteranceId = String(msg.utterance_id || '')
      this.currentUtteranceId = utteranceId
      const chars = Number(msg.chars || 0)
      if (!this.playbackState || this.playbackState.utteranceId !== utteranceId) {
        this.playbackState = {
          utteranceId,
          chars,
          serverEnded: true,
          receivedAudio: false,
        }
      } else {
        this.playbackState.serverEnded = true
        this.playbackState.chars = chars
      }
      return
    }
    if (type === 'error') {
      this.listeners.onError?.(String(msg.message || 'unknown TTS error'))
      return
    }
    if (type === 'cancelled') {
      const utteranceId = this.currentUtteranceId
      this.playbackState = null
      this.agentSpeaking = false
      if (this.armMicTimer !== null) {
        window.clearTimeout(this.armMicTimer)
        this.armMicTimer = null
      }
      this.playbackNode?.port.postMessage({ type: 'flush' })
      if (utteranceId) {
        this.listeners.onPlaybackInterrupted?.(utteranceId)
      }
      return
    }
  }

  private async _handleTtsAudio(buf: ArrayBuffer): Promise<void> {
    const meta = this.pendingAudioMeta
    // Streaming TTS keeps the meta sticky across chunks of one utterance until
    // the next audio_start arrives — don't null it.
    if (!this.audioCtx || !this.playbackNode) return
    if (this.playbackState) this.playbackState.receivedAudio = true
    const sampleRate = meta?.sampleRate ?? 22050
    const format = meta?.format ?? 'pcm_s16le'

    try {
      let float: Float32Array
      if (format === 'pcm_s16le') {
        // Raw int16 little-endian PCM chunk.
        const view = new DataView(buf)
        const samples = view.byteLength / 2
        float = new Float32Array(samples)
        for (let i = 0; i < samples; i++) {
          const s = view.getInt16(i * 2, true)
          float[i] = s < 0 ? s / 0x8000 : s / 0x7fff
        }
      } else {
        // WAV/MP3/etc — let WebAudio decode the full buffer.
        const decoded = await this.audioCtx.decodeAudioData(buf.slice(0))
        const channel = decoded.numberOfChannels > 0 ? decoded.getChannelData(0) : new Float32Array(0)
        float = new Float32Array(channel)
        if (decoded.sampleRate !== this.audioCtx.sampleRate) {
          float = resampleLinear(float, decoded.sampleRate, this.audioCtx.sampleRate)
        }
        this.playbackNode.port.postMessage({ type: 'push', buffer: float.buffer }, [float.buffer])
        return
      }

      // Resample PCM to AudioContext rate.
      const ctxRate = this.audioCtx.sampleRate
      const finalBuf = sampleRate === ctxRate ? float : resampleLinear(float, sampleRate, ctxRate)
      this.playbackNode.port.postMessage({ type: 'push', buffer: finalBuf.buffer }, [finalBuf.buffer])
    } catch (err) {
      this.listeners.onError?.(`audio decode failed: ${(err as Error).message}`)
    }
  }

  private _handleSttControl(msg: Record<string, unknown>): void {
    const type = msg.type as string | undefined
    if (type === 'partial') {
      this.listeners.onPartialTranscript?.(String(msg.text || ''), String(msg.language_code || 'hi-IN'))
      return
    }
    if (type === 'final') {
      this.listeners.onFinalTranscript?.(String(msg.text || ''), String(msg.language_code || 'hi-IN'))
      return
    }
    if (type === 'error') {
      this.listeners.onError?.(String(msg.message || 'unknown STT error'))
      return
    }
  }
}

function resampleLinear(input: Float32Array, inRate: number, outRate: number): Float32Array {
  if (inRate === outRate) return new Float32Array(input)
  const ratio = inRate / outRate
  const outLen = Math.floor(input.length / ratio)
  const out = new Float32Array(outLen)
  for (let i = 0; i < outLen; i++) {
    const src = i * ratio
    const i0 = Math.floor(src)
    const i1 = Math.min(i0 + 1, input.length - 1)
    const t = src - i0
    out[i] = input[i0] * (1 - t) + input[i1] * t
  }
  return out
}
