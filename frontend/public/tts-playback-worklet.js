// Plays back PCM Float32 audio frames pushed from the main thread.
// Main thread decodes WAV chunks (from Sarvam) into Float32, posts them here,
// and this processor outputs them to the destination one render quantum at a time.

class TtsPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this._queue = []          // Array<Float32Array>
    this._cursor = 0          // index within the head buffer
    this.port.onmessage = (event) => {
      const data = event.data
      if (!data) return
      if (data.type === 'push' && data.buffer) {
        this._queue.push(new Float32Array(data.buffer))
        return
      }
      if (data.type === 'flush') {
        this._queue = []
        this._cursor = 0
        return
      }
    }
  }

  process(_inputs, outputs) {
    const output = outputs[0]
    if (!output || output.length === 0) return true
    const channel = output[0]
    const frames = channel.length
    let written = 0

    while (written < frames && this._queue.length > 0) {
      const head = this._queue[0]
      const available = head.length - this._cursor
      const need = frames - written
      const take = Math.min(available, need)
      channel.set(head.subarray(this._cursor, this._cursor + take), written)
      written += take
      this._cursor += take
      if (this._cursor >= head.length) {
        this._queue.shift()
        this._cursor = 0
        this.port.postMessage({ type: 'consumed' })
      }
    }

    if (written < frames) {
      channel.fill(0, written)
      if (this._queue.length === 0) this.port.postMessage({ type: 'idle' })
    }
    return true
  }
}

registerProcessor('tts-playback-processor', TtsPlaybackProcessor)
