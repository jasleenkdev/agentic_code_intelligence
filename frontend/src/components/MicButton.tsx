import clsx from 'clsx'
import { Mic, Square } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { Spinner } from './ui'

type State = 'idle' | 'recording' | 'transcribing'

/** Records from the browser microphone and transcribes locally via /api/transcribe. */
export function MicButton({ onText, onError, disabled }: {
  onText: (text: string) => void; onError: (msg: string) => void; disabled?: boolean
}) {
  const [state, setState] = useState<State>('idle')
  const [seconds, setSeconds] = useState(0)
  const recorder = useRef<MediaRecorder | null>(null)
  const chunks = useRef<Blob[]>([])

  useEffect(() => {
    if (state !== 'recording') return
    setSeconds(0)
    const t = setInterval(() => setSeconds((s) => s + 1), 1000)
    return () => clearInterval(t)
  }, [state])

  useEffect(() => () => recorder.current?.stream.getTracks().forEach((t) => t.stop()), [])

  async function start() {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      onError('This browser cannot record audio. Type the question instead.')
      return
    }
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      onError('Microphone access was denied. Allow it in the browser, or type the question.')
      return
    }
    chunks.current = []
    const rec = new MediaRecorder(stream)
    rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data)
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop())
      const blob = new Blob(chunks.current, { type: rec.mimeType || 'audio/webm' })
      if (!blob.size) { setState('idle'); return }
      setState('transcribing')
      try {
        const { text } = await api.transcribe(blob)
        if (text.trim()) onText(text.trim())
        else onError('No speech was recognised. Try again a little closer to the microphone.')
      } catch (e) {
        onError((e as Error).message)
      } finally {
        setState('idle')
      }
    }
    recorder.current = rec
    rec.start()
    setState('recording')
  }

  function stop() {
    if (recorder.current?.state === 'recording') recorder.current.stop()
  }

  const label = state === 'recording' ? 'Stop recording' : 'Ask by voice'
  return (
    <button
      type="button"
      onClick={state === 'recording' ? stop : start}
      disabled={disabled || state === 'transcribing'}
      title={label}
      aria-label={label}
      className={clsx(
        'flex h-11 items-center gap-2 rounded-lg border px-3 text-sm transition disabled:opacity-50',
        state === 'recording'
          ? 'border-rose-500/50 bg-rose-500/10 text-rose-600 dark:text-rose-300'
          : 'border-app hover:bg-zinc-500/10',
      )}
    >
      {state === 'transcribing' ? <Spinner /> : state === 'recording'
        ? <Square className="h-4 w-4 fill-current" /> : <Mic className="h-4 w-4" />}
      {state === 'recording' && <span className="font-mono text-xs">{seconds}s</span>}
      {state === 'transcribing' && <span className="text-xs">transcribing…</span>}
    </button>
  )
}
