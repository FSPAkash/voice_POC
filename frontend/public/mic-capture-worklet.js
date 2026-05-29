// Downsample mic input (usually 48 kHz mono Float32) to 16 kHz int16 PCM and
// post Int16Array chunks to the main thread. The main thread forwards them
// over a WebSocket to the backend STT proxy.

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    const opts = (options && options.processorOptions) || {}
    this._targetSampleRate = opts.targetSampleRate || 16000
    this._inputSampleRate = sampleRate
    this._ratio = this._inputSampleRate / this._targetSampleRate
    this._buffer = []
    this._bufferLength = 0
    this._chunkSize = Math.round(this._targetSampleRate * 0.05) // ~50 ms per post
    this._enabled = true
    // VAD state — RMS energy threshold over short frames.
    // Browser AEC (echoCancellation:true on getUserMedia) suppresses our own
    // TTS output before it reaches the mic, so we can run a strict threshold
    // for true barge-in without false-firing on agent self-echo.
    this._vadThreshold = opts.vadThreshold || 0.05
    this._vadSpeechFrames = 0
    this._vadSilenceFrames = 0
    this._speechFrameCount = 0  // frames classified as speech in the current utterance
    // ~120 ms of sustained energy before we declare "speaking" — fast enough
    // for natural barge-in, still long enough to ignore clicks/breath.
    this._vadSpeechStartFrames = Math.round((this._inputSampleRate * 0.12) / 128)
    // ~280 ms silence to declare "done speaking" so the agent does not leave
    // a dead gap after the customer stops.
    this._vadSilenceEndFrames = Math.round((this._inputSampleRate * 0.28) / 128)
    // Drop utterances shorter than ~180 ms total speech — they're almost
    // always coughs/clicks that STT will hallucinate around.
    this._vadMinSpeechFrames = Math.round((this._inputSampleRate * 0.18) / 128)
    this._inSpeech = false
    this.port.onmessage = (event) => {
      if (event.data && event.data.type === 'enable') this._enabled = !!event.data.value
    }
  }

  process(inputs) {
    if (!this._enabled) return true
    const input = inputs[0]
    if (!input || input.length === 0) return true
    const channel = input[0]
    if (!channel || channel.length === 0) return true

    // VAD energy on the input frame (pre-downsample).
    let sumSq = 0
    for (let i = 0; i < channel.length; i++) sumSq += channel[i] * channel[i]
    const rms = Math.sqrt(sumSq / channel.length)
    if (rms >= this._vadThreshold) {
      this._vadSpeechFrames++
      this._vadSilenceFrames = 0
      if (this._inSpeech) this._speechFrameCount++
      if (!this._inSpeech && this._vadSpeechFrames >= this._vadSpeechStartFrames) {
        this._inSpeech = true
        this._speechFrameCount = this._vadSpeechStartFrames
        this.port.postMessage({ type: 'vad', state: 'speech_start' })
      }
    } else {
      this._vadSpeechFrames = 0
      if (this._inSpeech) {
        this._vadSilenceFrames++
        if (this._vadSilenceFrames >= this._vadSilenceEndFrames) {
          const totalSpeech = this._speechFrameCount
          this._inSpeech = false
          this._vadSilenceFrames = 0
          this._speechFrameCount = 0
          if (totalSpeech >= this._vadMinSpeechFrames) {
            this.port.postMessage({ type: 'vad', state: 'speech_end' })
          } else {
            // Too short to be real speech — drop the buffer instead of
            // flushing it to STT where it would hallucinate.
            this.port.postMessage({ type: 'vad', state: 'drop' })
          }
        }
      }
    }

    // Linear-interpolation downsample.
    const outLen = Math.floor(channel.length / this._ratio)
    if (outLen <= 0) return true
    const out = new Int16Array(outLen)
    for (let i = 0; i < outLen; i++) {
      const srcIndex = i * this._ratio
      const i0 = Math.floor(srcIndex)
      const i1 = Math.min(i0 + 1, channel.length - 1)
      const t = srcIndex - i0
      const sample = channel[i0] * (1 - t) + channel[i1] * t
      let s = Math.max(-1, Math.min(1, sample))
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff
    }

    this._buffer.push(out)
    this._bufferLength += out.length

    while (this._bufferLength >= this._chunkSize) {
      const chunk = new Int16Array(this._chunkSize)
      let offset = 0
      while (offset < this._chunkSize && this._buffer.length > 0) {
        const head = this._buffer[0]
        const take = Math.min(head.length, this._chunkSize - offset)
        chunk.set(head.subarray(0, take), offset)
        offset += take
        if (take >= head.length) {
          this._buffer.shift()
        } else {
          this._buffer[0] = head.subarray(take)
        }
      }
      this._bufferLength -= this._chunkSize
      this.port.postMessage(chunk.buffer, [chunk.buffer])
    }
    return true
  }
}

registerProcessor('mic-capture-processor', MicCaptureProcessor)
