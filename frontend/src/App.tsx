import {
  startTransition,
  useDeferredValue,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from 'react'
import { StatusPill } from './components/StatusPill'
import { SupervisorBoard } from './components/SupervisorBoard'
import {
  chatTurn,
  createRealtimeSession,
  customerTurn,
  detectCustomerLanguage,
  evaluateSupervisor,
  fetchBootstrap,
  fetchCallHistory,
  fetchCosts,
  fetchSupervisorBoard,
  invokeTool,
  logCall,
  moveSupervisorIssue,
  recordCostEvent,
  resetCostLedger,
  resetDemo,
  summarizeCall,
  type CallSummary,
} from './lib/api'
import { formatCurrency, formatDate, formatNumber, formatTime, formatUsd } from './lib/format'
import {
  buildScriptedResponse,
  buildOpeningText,
  buildSessionUpdate,
  detectLanguageComplianceIssue,
  detectSystemPromptLeak,
  extractRealtimeFunctionCalls,
  extractRealtimeText,
} from './lib/realtime'
import { SarvamVoiceClient, type SarvamSessionConfig } from './lib/sarvam'

// Extract "LINE TO SPEAK (verbatim): xxx" payload from buildScriptedResponse output.
function extractSpokenLine(event: Record<string, unknown>): string {
  const response = (event.response as Record<string, unknown>) || {}
  const instructions = String(response.instructions || '')
  const match = instructions.match(/LINE TO SPEAK \(verbatim\):\s*([\s\S]+?)$/)
  return (match ? match[1] : '').trim()
}
import type {
  BootstrapResponse,
  CallHistoryRecord,
  CallDisposition,
  CostState,
  LanguageAdvice,
  LanguageOption,
  RealtimeModelOption,
  SupervisorBoardState,
  ToolCallEntry,
  TranscriptEntry,
} from './types'

type CallState = 'loading' | 'ready' | 'starting' | 'connected' | 'ending' | 'error'
type TabId = 'call' | 'wrap' | 'supervisor'
type InteractionMode = 'voice' | 'chat'
type PricingCard = {
  id: string
  title: string
  model: string
  lines: string[]
  active?: boolean
}

type CallRecord = CallHistoryRecord

function formatInrRate(amount: number): string {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount)
}

function formatUsdRate(amount: number): string {
  return `$${amount.toFixed(2)}`
}

type CostEventPayload = {
  event_id: string
  session_id: string
  source: 'agent' | 'supervisor' | 'language_coach' | 'sarvam'
  usage_type: 'response' | 'transcription' | 'tts' | 'stt'
  model: string
  usage: Record<string, unknown>
  chars?: number
  seconds?: number
}

type AgentActivityEntry = {
  id: string
  agent: 'caller' | 'language_coach' | 'supervisor' | 'summarizer' | 'tool' | 'chat_agent'
  status: 'started' | 'completed' | 'error'
  summary: string
  timestamp: string
}

const AGENT_NODES: { id: AgentActivityEntry['agent']; label: string; model: string; idle: string }[] = [
  { id: 'caller', label: 'Live Caller', model: 'bulbul:v3 + saaras:v3', idle: 'Idle — voice mode only' },
  { id: 'chat_agent', label: 'Chat Agent', model: 'gpt-4.1 + deterministic fallback', idle: 'Idle — text mode only' },
  { id: 'language_coach', label: 'Language Coach', model: 'rule-based', idle: 'Awaiting customer turn' },
  { id: 'supervisor', label: 'Supervisor', model: 'rule-based', idle: 'Awaiting agent turn' },
  { id: 'tool', label: 'Tool Layer', model: 'mock SAP', idle: 'No tool calls yet' },
  { id: 'summarizer', label: 'Summariser', model: 'gpt-4.1-mini', idle: 'Runs on call wrap-up' },
]

const STT_HALLUCINATION_MARKERS = [
  'transcribe faithfully',
  'do not hallucinate',
  'if audio is unclear',
  '[unclear]',
  'collections call. the agent',
  'primary mode starts in hinglish',
  'indian regional languages at any time',
  'prefer english text for english',
]

const STT_PROMPT_VOCAB_TOKENS = new Set([
  'dhl',
  'express',
  'india',
  'mybill',
  'virtual',
  'account',
  'number',
  'invoice',
  'invoices',
  'overdue',
  'promise',
  'to',
  'pay',
  'credit',
  'note',
  'waybill',
  'accounts',
  'payable',
  'hinglish',
])

function isLikelySttHallucination(text: string): boolean {
  if (!text) return false
  const lowered = text.toLowerCase().trim()
  if (!lowered) return true
  const markerHit = STT_HALLUCINATION_MARKERS.find((marker) => lowered.includes(marker))
  if (markerHit) {
    const remainder = lowered.replace(markerHit, '').replace(/[^a-z0-9]+/g, ' ').trim()
    if (remainder.length < 8) return true
  }
  // Whisper sometimes regurgitates the transcription `prompt` vocab list on
  // silence. Only treat as hallucination when the turn looks like a verbatim
  // recital of that list — i.e. short-ish, no verbs/connectives, and almost
  // every token comes from the vocab. Real customer speech that happens to
  // mention "invoice" / "DHL" must pass through.
  const tokens = lowered.split(/[^a-z0-9]+/).filter(Boolean)
  if (tokens.length >= 6 && tokens.length <= 30) {
    const vocabHits = tokens.filter((t) => STT_PROMPT_VOCAB_TOKENS.has(t)).length
    const hasConnective = tokens.some((t) =>
      ['i', 'you', 'we', 'is', 'are', 'was', 'be', 'have', 'do', 'can', 'will', 'the', 'a', 'an', 'and', 'but', 'so', 'because', 'please', 'yes', 'no', 'not'].includes(t),
    )
    if (!hasConnective && vocabHits / tokens.length >= 0.85) return true
  }
  return false
}

function agentLabel(id: AgentActivityEntry['agent']): string {
  return AGENT_NODES.find((node) => node.id === id)?.label ?? id
}

function emptyCosts(): CostState {
  return {
    agent: {
      model: 'bulbul:v3',
      events: 0,
      response_usage: {
        text_input_tokens: 0,
        text_cached_input_tokens: 0,
        text_output_tokens: 0,
        audio_input_tokens: 0,
        audio_cached_input_tokens: 0,
        audio_output_tokens: 0,
        estimated_cost_usd: 0,
      },
      transcription_usage: {
        model: 'saaras:v3',
        audio_input_tokens: 0,
        text_input_tokens: 0,
        text_output_tokens: 0,
        estimated_cost_usd: 0,
      },
      total_tokens: 0,
      estimated_cost_usd: 0,
    },
    supervisor: {
      model: 'deterministic-call-engine',
      events: 0,
      text_input_tokens: 0,
      text_cached_input_tokens: 0,
      text_output_tokens: 0,
      total_tokens: 0,
      estimated_cost_usd: 0,
    },
    language_coach: {
      model: 'gpt-4.1',
      events: 0,
      text_input_tokens: 0,
      text_cached_input_tokens: 0,
      text_output_tokens: 0,
      total_tokens: 0,
      estimated_cost_usd: 0,
    },
    chat_agent: {
      model: 'gpt-4.1-mini',
      events: 0,
      text_input_tokens: 0,
      text_cached_input_tokens: 0,
      text_output_tokens: 0,
      total_tokens: 0,
      estimated_cost_usd: 0,
    },
    combined: {
      total_tokens: 0,
      estimated_cost_usd: 0,
    },
    updated_at: '',
    session_id: '',
    price_table_version: '',
    price_table: {},
  }
}

function defaultLanguageAdvice(languageId = 'hinglish'): LanguageAdvice {
  return {
    detected_language_id: languageId,
    suggested_language_id: languageId,
    transcription_language_id: languageId,
    transcript_quality: 'good',
    confidence: 'high',
    should_switch: false,
    nudge: 'Open in Hinglish and switch only when the customer clearly prefers another language.',
    rationale: 'Default call opening behavior.',
  }
}

function languageLabel(languages: LanguageOption[], languageId: string) {
  return languages.find((language) => language.id === languageId)?.label ?? languageId
}

function realtimeModelLabel(models: RealtimeModelOption[], modelId: string) {
  return models.find((model) => model.id === modelId)?.label ?? modelId
}

function emptyBoard(): SupervisorBoardState {
  return {
    columns: [
      { id: 'new', title: 'New', issues: [] },
      { id: 'reviewing', title: 'Reviewing', issues: [] },
      { id: 'accepted', title: 'Accepted', issues: [] },
      { id: 'dismissed', title: 'Dismissed', issues: [] },
    ],
    updated_at: '',
  }
}

function determineDisposition(toolName: string): CallDisposition | null {
  switch (toolName) {
    case 'log_promise_to_pay':
      return 'Promise to pay logged'
    case 'log_already_paid':
      return 'Already paid claimed'
    case 'resend_invoice':
      return 'Invoice resend requested'
    case 'log_dispute':
      return 'Dispute raised'
    case 'update_contact':
      return 'Alternate contact captured'
    case 'transfer_to_human':
      return 'Transferred to human'
    default:
      return null
  }
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

const PENDING_COST_EVENTS_STORAGE_KEY = 'dhl_pending_cost_events_v1'

function costEventKey(payload: Pick<CostEventPayload, 'event_id' | 'session_id'>): string {
  return `${payload.session_id}:${payload.event_id}`
}

function loadPendingCostEvents(): CostEventPayload[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(PENDING_COST_EVENTS_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter((item): item is CostEventPayload => {
      if (!item || typeof item !== 'object') return false
      return (
        typeof item.event_id === 'string' &&
        item.event_id.trim().length > 0 &&
        typeof item.session_id === 'string' &&
        item.session_id.trim().length > 0 &&
        typeof item.source === 'string' &&
        typeof item.usage_type === 'string' &&
        typeof item.model === 'string' &&
        item.usage !== null &&
        typeof item.usage === 'object'
      )
    })
  } catch {
    return []
  }
}

function persistPendingCostEvents(events: Iterable<CostEventPayload>) {
  if (typeof window === 'undefined') return
  try {
    const next = Array.from(events)
    if (next.length === 0) {
      window.localStorage.removeItem(PENDING_COST_EVENTS_STORAGE_KEY)
      return
    }
    window.localStorage.setItem(PENDING_COST_EVENTS_STORAGE_KEY, JSON.stringify(next))
  } catch {
    // storage unavailable - ignore
  }
}

function buildFallbackCostEventId(prefix: string): string {
  return `${prefix}:${Date.now()}:${Math.random().toString(36).slice(2, 10)}`
}

type AppProps = {
  username?: string
  onLogout?: () => void
}

export default function App({ username, onLogout }: AppProps = {}) {
  const [bootstrap, setBootstrap] = useState<BootstrapResponse | null>(null)
  const [costs, setCosts] = useState<CostState>(emptyCosts)
  const [board, setBoard] = useState<SupervisorBoardState>(emptyBoard)
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([])
  const [toolCalls, setToolCalls] = useState<ToolCallEntry[]>([])
  const [callState, setCallState] = useState<CallState>('loading')
  const [activeTab, setActiveTab] = useState<TabId>('call')
  const [errorMessage, setErrorMessage] = useState('')
  const [customerSpeaking, setCustomerSpeaking] = useState(false)
  const [thinking, setThinking] = useState(false)
  const [disposition, setDisposition] = useState<CallDisposition>('Awaiting call')
  const [elapsed, setElapsed] = useState(0)
  const [micLevel, setMicLevel] = useState(0)
  const [selectedLanguageId, setSelectedLanguageId] = useState('hinglish')
  const [activeLanguageId, setActiveLanguageId] = useState('hinglish')
  const [selectedRealtimeModel, setSelectedRealtimeModel] = useState('bulbul:v3')
  const [languageAdvice, setLanguageAdvice] = useState<LanguageAdvice>(defaultLanguageAdvice)
  const [callSummary, setCallSummary] = useState<CallSummary | null>(null)
  const [summarizing, setSummarizing] = useState(false)
  const [callHistory, setCallHistory] = useState<CallRecord[]>(() => {
    if (typeof window === 'undefined') return []
    try {
      const raw = window.localStorage.getItem('dhl_call_history_v1')
      if (!raw) return []
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) ? (parsed as CallRecord[]) : []
    } catch {
      return []
    }
  })

  useEffect(() => {
    try {
      window.localStorage.setItem('dhl_call_history_v1', JSON.stringify(callHistory))
    } catch {
      // storage unavailable — ignore
    }
  }, [callHistory])
  const [mode, setMode] = useState<InteractionMode>('voice')
  const [headless, setHeadless] = useState<boolean>(false)
  const [muted, setMuted] = useState(false)
  const [ambienceGain, setAmbienceGain] = useState<number>(() => {
    if (typeof window === 'undefined') return 0.4
    try {
      const raw = window.localStorage.getItem('dhl_ambience_gain_v1')
      const parsed = raw ? Number(raw) : NaN
      return Number.isFinite(parsed) && parsed >= 0 && parsed <= 1 ? parsed : 0.4
    } catch {
      return 0.4
    }
  })
  const [headlessNotices, setHeadlessNotices] = useState<{ id: string; text: string }[]>([])
  const [chatInput, setChatInput] = useState('')
  const [chatBusy, setChatBusy] = useState(false)
  const [agentActivity, setAgentActivity] = useState<AgentActivityEntry[]>([])
  const [openInfo, setOpenInfo] = useState<'account' | 'language' | 'invoices' | 'tools' | null>(
    null,
  )

  useEffect(() => {
    if (headlessNotices.length === 0) return
    const id = window.setTimeout(() => {
      setHeadlessNotices((previous) => previous.slice(1))
    }, 4000)
    return () => window.clearTimeout(id)
  }, [headlessNotices])

  const deferredTranscript = useDeferredValue(transcript)

  const remoteAudioRef = useRef<HTMLAudioElement | null>(null)
  const transcriptScrollRef = useRef<HTMLDivElement | null>(null)
  const sarvamClientRef = useRef<SarvamVoiceClient | null>(null)
  const sarvamSessionRef = useRef<SarvamSessionConfig | null>(null)
  // Turn-commit debounce: buffer rapid fragments from STT into one logical
  // customer turn before firing the policy engine. Keep this short so the call
  // never feels dropped between the customer's turn and the agent response.
  const turnCommitTimerRef = useRef<number | null>(null)
  const turnCommitBufferRef = useRef<string[]>([])
  const turnCommitDelayMs = 250
  const pendingBargeInRef = useRef<{ responseId: string; at: number } | null>(null)
  const policyReplyRequestSeqRef = useRef(0)
  // Wall-clock timestamp of the most recent agent audio_start. Used to detect
  // self-echo hallucinations: very short transcripts arriving immediately
  // after playback begins are usually not real customer speech.
  const lastAgentSpeakStartRef = useRef<number | null>(null)
  const localStreamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const micRafRef = useRef<number | null>(null)
  const processedCallIdsRef = useRef<Set<string>>(new Set())
  const assistantMessageIdRef = useRef<string | null>(null)
  const assistantBufferRef = useRef('')
  const languageRepairAttemptedRef = useRef(false)
  const activeResponseIdRef = useRef<string | null>(null)
  const pendingResponseQueueRef = useRef<Array<Record<string, unknown>>>([])
  const pendingSessionUpdateRef = useRef<Record<string, unknown> | null>(null)
  const streamingViolationCancelledRef = useRef(false)
  const discardCurrentResponseRef = useRef(false)
  const lastApprovedScriptRef = useRef<string>('')
  const turnNumberRef = useRef(0)
  const bootstrapRequestIdRef = useRef(0)
  const callStartRef = useRef<number | null>(null)
  const sessionStartRef = useRef<number | null>(null)
  const costRetryTimeoutRef = useRef<number | null>(null)

  const bootstrapRef = useRef<BootstrapResponse | null>(null)
  const costsRef = useRef<CostState>(emptyCosts())
  const transcriptRef = useRef<TranscriptEntry[]>([])
  const toolCallsRef = useRef<ToolCallEntry[]>([])
  const dispositionRef = useRef<CallDisposition>('Awaiting call')
  const callStateRef = useRef<CallState>('loading')
  const selectedLanguageRef = useRef('hinglish')
  const activeLanguageRef = useRef('hinglish')
  const selectedRealtimeModelRef = useRef('bulbul:v3')
  const languageAdviceRef = useRef<LanguageAdvice>(defaultLanguageAdvice())
  const coachingHintsRef = useRef<string[]>([])
  const pendingCostEventsRef = useRef<Map<string, CostEventPayload>>(
    new Map(loadPendingCostEvents().map((event) => [costEventKey(event), event])),
  )

  useEffect(() => {
    bootstrapRef.current = bootstrap
  }, [bootstrap])

  useEffect(() => {
    costsRef.current = costs
  }, [costs])

  useEffect(() => {
    transcriptRef.current = transcript
  }, [transcript])

  useEffect(() => {
    toolCallsRef.current = toolCalls
  }, [toolCalls])

  useEffect(() => {
    dispositionRef.current = disposition
  }, [disposition])

  useEffect(() => {
    callStateRef.current = callState
  }, [callState])

  useEffect(() => {
    selectedLanguageRef.current = selectedLanguageId
  }, [selectedLanguageId])

  useEffect(() => {
    activeLanguageRef.current = activeLanguageId
  }, [activeLanguageId])

  useEffect(() => {
    selectedRealtimeModelRef.current = selectedRealtimeModel
  }, [selectedRealtimeModel])

  useEffect(() => {
    languageAdviceRef.current = languageAdvice
  }, [languageAdvice])

  useEffect(() => {
    return () => {
      if (costRetryTimeoutRef.current !== null) {
        window.clearTimeout(costRetryTimeoutRef.current)
      }
    }
  }, [])

  const syncPendingCostEvents = () => {
    persistPendingCostEvents(pendingCostEventsRef.current.values())
  }

  const flushPendingCostEvents = useEffectEvent(async () => {
    const pending = Array.from(pendingCostEventsRef.current.values())
    if (pending.length === 0) return

    let latestCosts: CostState | null = null
    for (const event of pending) {
      try {
        latestCosts = await recordCostEvent(event)
        pendingCostEventsRef.current.delete(costEventKey(event))
        syncPendingCostEvents()
      } catch {
        if (costRetryTimeoutRef.current === null) {
          costRetryTimeoutRef.current = window.setTimeout(() => {
            costRetryTimeoutRef.current = null
            void flushPendingCostEvents()
          }, 3000)
        }
        return
      }
    }

    if (latestCosts) {
      costsRef.current = latestCosts
      startTransition(() => setCosts(latestCosts))
    }
  })

  // Auto-scroll transcript on new entries.
  useEffect(() => {
    const node = transcriptScrollRef.current
    if (!node) return
    node.scrollTop = node.scrollHeight
  }, [deferredTranscript.length])

  // Session elapsed timer.
  useEffect(() => {
    if (callState !== 'connected') {
      if (callState !== 'ending') {
        callStartRef.current = null
        setElapsed(0)
      }
      return undefined
    }
    if (callStartRef.current === null) {
      callStartRef.current = Date.now()
    }
    const id = window.setInterval(() => {
      const start = callStartRef.current ?? Date.now()
      setElapsed(Math.floor((Date.now() - start) / 1000))
    }, 500)
    return () => window.clearInterval(id)
  }, [callState])

  useEffect(() => {
    const stream = localStreamRef.current
    if (!stream) return
    stream.getAudioTracks().forEach((track) => {
      track.enabled = !muted
    })
  }, [muted])

  const ambienceRef = useRef<HTMLAudioElement | null>(null)
  const ambienceFadeRef = useRef<number | null>(null)

  useEffect(() => {
    const audio = ambienceRef.current
    if (!audio) return
    const isOnCall = callState === 'connected' || callState === 'starting'
    const baseGain = Math.max(0, Math.min(1, ambienceGain))
    const targetVolume = !isOnCall || muted ? 0 : thinking || customerSpeaking ? baseGain : baseGain * 0.5

    if (isOnCall && audio.paused) {
      audio.play().catch((err) => console.warn('[ambience] resume failed', err))
    }

    if (ambienceFadeRef.current !== null) {
      window.clearInterval(ambienceFadeRef.current)
    }
    const step = 0.025
    ambienceFadeRef.current = window.setInterval(() => {
      const current = audio.volume
      if (Math.abs(current - targetVolume) < step) {
        audio.volume = targetVolume
        if (ambienceFadeRef.current !== null) {
          window.clearInterval(ambienceFadeRef.current)
          ambienceFadeRef.current = null
        }
        if (!isOnCall && targetVolume === 0 && !audio.paused) {
          audio.pause()
        }
        return
      }
      audio.volume = current < targetVolume
        ? Math.min(targetVolume, current + step)
        : Math.max(targetVolume, current - step)
    }, 50)
  }, [callState, thinking, customerSpeaking, muted, ambienceGain])

  useEffect(() => {
    try {
      window.localStorage.setItem('dhl_ambience_gain_v1', String(ambienceGain))
    } catch {
      // storage unavailable — ignore
    }
  }, [ambienceGain])


  const getCallStateLabel = () => {
    switch (callState) {
      case 'loading':
        return 'Loading'
      case 'ready':
        return 'Ready'
      case 'starting':
        return 'Connecting'
      case 'connected':
        return 'Live'
      case 'ending':
        return 'Ending'
      case 'error':
        return 'Error'
      default:
        return callState
    }
  }

  const getStageLine = () => {
    if (callState === 'loading') return 'Loading local demo configuration'
    if (callState === 'error') return errorMessage || 'Call needs attention'
    if (callState === 'starting') return 'Connecting voice session'
    if (callState === 'ending') return 'Closing the session'
    if (callState !== 'connected') return 'Click Start Call to dial Mind Your Business Inc.'
    if (customerSpeaking) return 'Customer speaking'
    if (thinking) return 'Agent reasoning'
    return 'Listening'
  }

  const syncBootstrap = (payload: BootstrapResponse) => {
    setBootstrap(payload)
    setBoard(payload.board)
    costsRef.current = payload.costs
    setCosts(payload.costs)
    setCallHistory(Array.isArray(payload.call_history) ? payload.call_history : [])
    const defaultLanguageId = payload.config.default_language_id ?? 'hinglish'
    const defaultRealtimeModel = payload.config.realtime_model ?? 'bulbul:v3'
    setSelectedLanguageId(defaultLanguageId)
    setActiveLanguageId(defaultLanguageId)
    setSelectedRealtimeModel(defaultRealtimeModel)
    setLanguageAdvice(defaultLanguageAdvice(defaultLanguageId))
    coachingHintsRef.current = []
  }

  const loadBootstrapData = useEffectEvent(async () => {
    const requestId = bootstrapRequestIdRef.current + 1
    bootstrapRequestIdRef.current = requestId
    setCallState('loading')
    setErrorMessage('')

    try {
      const payload = await fetchBootstrap()
      if (bootstrapRequestIdRef.current !== requestId) return
      startTransition(() => {
        syncBootstrap(payload)
        setCallState('ready')
        setDisposition('Awaiting call')
      })
    } catch (error) {
      if (bootstrapRequestIdRef.current !== requestId) return
      const message = error instanceof Error ? error.message : 'Failed to load the demo.'
      startTransition(() => {
        setCallState('error')
        setErrorMessage(message)
      })
    }
  })

  useEffect(() => {
    void loadBootstrapData()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (callState === 'loading' || callState === 'error') return
    void flushPendingCostEvents()
  }, [callState])

  const refreshRuntimeState = useEffectEvent(async () => {
    try {
      const [nextCosts, nextBoard, nextHistory] = await Promise.all([
        fetchCosts(),
        fetchSupervisorBoard(),
        fetchCallHistory(),
      ])
      costsRef.current = nextCosts
      startTransition(() => {
        setCosts(nextCosts)
        setBoard(nextBoard)
        setCallHistory(nextHistory.history)
      })
    } catch {
      // Background refresh is best-effort.
    }
  })

  useEffect(() => {
    if (callState === 'loading' || callState === 'error') return undefined
    const intervalId = window.setInterval(() => {
      void refreshRuntimeState()
    }, 5000)
    return () => window.clearInterval(intervalId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [callState])

  useEffect(() => {
    const snapshot = bootstrapRef.current
    if (!snapshot || callState !== 'connected') return
    sendRealtimeEvent(
      buildSessionUpdate(
        snapshot.agent_prompt,
        snapshot.realtime_tools,
        snapshot.config.realtime_voice,
        selectedRealtimeModelRef.current,
        snapshot.config.transcription_model,
        snapshot.config.supported_languages,
        activeLanguageId,
      ),
    )
  }, [activeLanguageId, callState])

  const resetAssistantDraft = () => {
    assistantMessageIdRef.current = null
    assistantBufferRef.current = ''
  }

  const dropAssistantDraft = () => {
    const currentId = assistantMessageIdRef.current
    if (!currentId) {
      resetAssistantDraft()
      return
    }
    startTransition(() => {
      setTranscript((previous) => previous.filter((entry) => entry.id !== currentId))
    })
    resetAssistantDraft()
  }

  const upsertAssistantDraft = (nextChunk: string) => {
    if (!nextChunk.trim()) return
    const currentId = assistantMessageIdRef.current ?? `assistant_${Date.now()}`
    const nextText = `${assistantBufferRef.current}${nextChunk}`
    assistantMessageIdRef.current = currentId
    assistantBufferRef.current = nextText

    startTransition(() => {
      setTranscript((previous) => {
        const exists = previous.some((entry) => entry.id === currentId)
        if (!exists) {
          return [
            ...previous,
            {
              id: currentId,
              role: 'assistant',
              text: nextText.trim(),
              timestamp: new Date().toISOString(),
              status: 'streaming',
            },
          ]
        }
        return previous.map((entry) =>
          entry.id === currentId
            ? { ...entry, text: nextText.trim(), status: 'streaming' as const }
            : entry,
        )
      })
    })
  }

  const finalizeAssistantDraft = (finalText: string) => {
    const normalized = finalText.trim() || assistantBufferRef.current.trim()
    if (!normalized) {
      resetAssistantDraft()
      return
    }
    const currentId = assistantMessageIdRef.current ?? `assistant_${Date.now()}`
    assistantMessageIdRef.current = currentId
    assistantBufferRef.current = normalized

    startTransition(() => {
      setTranscript((previous) => {
        const exists = previous.some((entry) => entry.id === currentId)
        if (!exists) {
          return [
            ...previous,
            {
              id: currentId,
              role: 'assistant',
              text: normalized,
              timestamp: new Date().toISOString(),
              status: 'final',
            },
          ]
        }
        return previous.map((entry) =>
          entry.id === currentId ? { ...entry, text: normalized, status: 'final' as const } : entry,
        )
      })
    })

    resetAssistantDraft()
  }

  const sealAssistantDraft = () => {
    const current = assistantBufferRef.current.trim()
    if (!current) {
      resetAssistantDraft()
      return
    }
    finalizeAssistantDraft(current)
  }

  const appendCustomerTranscript = (text: string, id?: string) => {
    const normalized = text.trim()
    if (!normalized) return

    startTransition(() => {
      setTranscript((previous) => {
        if (id && previous.some((entry) => entry.id === id)) return previous
        return [
          ...previous,
          {
            id: id ?? `customer_${Date.now()}`,
            role: 'customer',
            text: normalized,
            timestamp: new Date().toISOString(),
            status: 'final',
          },
        ]
      })
    })
  }

  const appendSystemTranscript = (text: string) => {
    const normalized = text.trim()
    if (!normalized) return

    if (headless) {
      setHeadlessNotices((previous) => [
        ...previous.slice(-2),
        { id: `notice_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`, text: normalized },
      ])
    }

    startTransition(() => {
      setTranscript((previous) => [
        ...previous,
        {
          id: `system_${Date.now()}`,
          role: 'system',
          text: normalized,
          timestamp: new Date().toISOString(),
          status: 'final',
        },
      ])
    })
  }

  const pushAgentActivity = (entry: Omit<AgentActivityEntry, 'id' | 'timestamp'>) => {
    const next: AgentActivityEntry = {
      ...entry,
      id: `act_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      timestamp: new Date().toISOString(),
    }
    startTransition(() => {
      setAgentActivity((previous) => [next, ...previous].slice(0, 40))
    })
  }

  // Shim: original code posted realtime events over an OpenAI WebRTC data
  // channel. With Sarvam we only need `response.create` (-> speak text) and
  // `response.cancel` (-> stop playback). `session.update` and other event
  // types are no-ops; the Sarvam config is set once at connect.
  const sendRealtimeEvent = (event: Record<string, unknown>) => {
    const client = sarvamClientRef.current
    if (!client) return
    const type = String(event.type ?? '')
    if (type === 'response.create') {
      const line = extractSpokenLine(event)
      if (!line) return
      const langCode = sarvamLangCodeFor(activeLanguageRef.current)
      const utteranceId = client.speak(line, langCode)
      // Synth the realtime-style created event so the legacy state machine
      // bookkeeps activeResponseIdRef correctly.
      void handleRealtimeEvent({
        type: 'response.created',
        response: { id: utteranceId, _spoken_text: line },
      })
      return
    }
    if (type === 'response.cancel') {
      client.cancelSpeech('app')
      void handleRealtimeEvent({ type: 'response.cancelled' })
      return
    }
    // session.update etc. — Sarvam handles language per-utterance; ignore.
  }

  const sarvamLangCodeFor = (languageId: string): string => {
    const map = bootstrapRef.current?.config.sarvam_language_codes
    if (map && map[languageId]) return map[languageId]
    if (languageId === 'english') return 'en-IN'
    if (languageId === 'bengali') return 'bn-IN'
    if (languageId === 'marathi') return 'mr-IN'
    if (languageId === 'tamil') return 'ta-IN'
    return 'hi-IN'
  }

  const languageIdForSarvamCode = (languageCode: string): string | null => {
    const normalized = languageCode.trim()
    if (!normalized) return null
    const map = bootstrapRef.current?.config.sarvam_language_codes
    if (map) {
      const entry = Object.entries(map).find(([, value]) => value === normalized)
      if (entry && entry[0] !== 'hinglish' && entry[0] !== 'hindi') {
        return entry[0]
      }
    }
    if (normalized === 'bn-IN') return 'bengali'
    if (normalized === 'mr-IN') return 'marathi'
    if (normalized === 'ta-IN') return 'tamil'
    return null
  }

  const flushPendingRealtime = () => {
    if (activeResponseIdRef.current) return
    if (pendingSessionUpdateRef.current) {
      const update = pendingSessionUpdateRef.current
      pendingSessionUpdateRef.current = null
      sendRealtimeEvent(update)
    }
    while (!activeResponseIdRef.current && pendingResponseQueueRef.current.length > 0) {
      const next = pendingResponseQueueRef.current.shift()
      if (next) sendRealtimeEvent(next)
    }
  }

  const queueSessionUpdate = (event: Record<string, unknown>) => {
    if (activeResponseIdRef.current) {
      pendingSessionUpdateRef.current = event
      return
    }
    sendRealtimeEvent(event)
  }

  const queueResponseCreate = (event: Record<string, unknown>) => {
    if (activeResponseIdRef.current) {
      pendingResponseQueueRef.current.push(event)
      sendRealtimeEvent({ type: 'response.cancel' })
      return
    }
    sendRealtimeEvent(event)
  }

  const applyCostUpdate = useEffectEvent(async (body: CostEventPayload) => {
    pendingCostEventsRef.current.set(costEventKey(body), body)
    syncPendingCostEvents()

    try {
      const nextCosts = await recordCostEvent(body)
      pendingCostEventsRef.current.delete(costEventKey(body))
      syncPendingCostEvents()
      costsRef.current = nextCosts
      startTransition(() => setCosts(nextCosts))
      if (pendingCostEventsRef.current.size > 0) {
        void flushPendingCostEvents()
      }
    } catch {
      if (costRetryTimeoutRef.current === null) {
        costRetryTimeoutRef.current = window.setTimeout(() => {
          costRetryTimeoutRef.current = null
          void flushPendingCostEvents()
        }, 3000)
      }
    }
  })

  const applyBackendToolCalls = useEffectEvent((calls: ToolCallEntry[]) => {
    if (!calls || calls.length === 0) return
    startTransition(() => setToolCalls((previous) => [...calls.slice().reverse(), ...previous]))
    for (const call of calls) {
      pushAgentActivity({
        agent: 'tool',
        status: call.status === 'error' ? 'error' : 'completed',
        summary: `${call.name}(${Object.keys(call.args || {}).join(', ')})`,
      })
      const nextDisp = determineDisposition(call.name)
      if (nextDisp) startTransition(() => setDisposition(nextDisp))
      if (call.name === 'get_invoices' && Array.isArray(call.result?.invoices)) {
        startTransition(() => {
          setBootstrap((previous) =>
            previous ? { ...previous, invoices: call.result.invoices as BootstrapResponse['invoices'] } : previous,
          )
        })
      }
      if (call.name === 'get_customer' && call.result?.customer) {
        startTransition(() => {
          setBootstrap((previous) =>
            previous ? { ...previous, customer: call.result.customer as BootstrapResponse['customer'] } : previous,
          )
        })
      }
    }
  })

  const sendGuidedResponse = useEffectEvent(async (
    context: 'customer_turn' | 'tool_followup',
    languageAdviceOverride?: LanguageAdvice,
  ) => {
    const snapshot = bootstrapRef.current
    if (!snapshot) return
    const requestSeq = ++policyReplyRequestSeqRef.current
    setThinking(true)
    pushAgentActivity({
      agent: 'caller',
      status: 'started',
      summary:
        context === 'tool_followup'
          ? 'Preparing approved follow-up after backend action'
          : 'Preparing approved reply from backend policy engine',
    })

    try {
      const result = await chatTurn({
        account_number: snapshot.account_number,
        voice: snapshot.config.realtime_voice,
        messages: transcriptRef.current
          .filter((entry) => entry.role === 'assistant' || entry.role === 'customer')
          .map((entry) => ({ role: entry.role as 'assistant' | 'customer', text: entry.text })),
        coaching_hints: coachingHintsRef.current.slice(0, 5),
        language_advice: languageAdviceOverride ?? languageAdviceRef.current,
      })
      if (requestSeq !== policyReplyRequestSeqRef.current) return
      costsRef.current = result.costs
      startTransition(() => setCosts(result.costs))
      applyBackendToolCalls(result.tool_calls)

      if (result.assistant_text && result.assistant_text.trim()) {
        lastApprovedScriptRef.current = result.assistant_text
        queueResponseCreate(buildScriptedResponse(result.assistant_text))
        pushAgentActivity({
          agent: 'caller',
          status: 'completed',
          summary: `Approved reply ready (${result.model})`,
        })
        return
      }

      setThinking(false)
      pushAgentActivity({
        agent: 'caller',
        status: 'completed',
        summary: 'No spoken follow-up was needed for this turn',
      })
    } catch (error) {
      if (requestSeq !== policyReplyRequestSeqRef.current) return
      const message = error instanceof Error ? error.message : 'Backend reply generation failed.'
      appendSystemTranscript(`Call policy error: ${message}`)
      setThinking(false)
      pushAgentActivity({ agent: 'caller', status: 'error', summary: message })
    }
  })

  const runLanguageCoach = useEffectEvent(async (transcriptText: string) => {
    const snapshot = bootstrapRef.current
    if (!snapshot) return
    languageRepairAttemptedRef.current = false

    pushAgentActivity({
      agent: 'language_coach',
      status: 'started',
      summary: 'Detecting customer language and transcript quality',
    })

    try {
      const result = await detectCustomerLanguage({
        transcript: transcriptText,
        current_language_id: activeLanguageRef.current,
        preferred_language_id: selectedLanguageRef.current,
        recent_transcript: transcriptRef.current.slice(-6),
      })

      const nextAdvice = result.advice
      const nextLanguageId = nextAdvice.suggested_language_id || activeLanguageRef.current
      languageAdviceRef.current = nextAdvice
      activeLanguageRef.current = nextLanguageId
      costsRef.current = result.costs
      startTransition(() => {
        setCosts(result.costs)
        setLanguageAdvice(nextAdvice)
        setActiveLanguageId(nextLanguageId)
      })

      if (nextAdvice.should_switch || nextAdvice.transcript_quality !== 'good') {
        appendSystemTranscript(`Language coach: ${nextAdvice.nudge}`)
      }
      pushAgentActivity({
        agent: 'language_coach',
        status: 'completed',
        summary: `Detected ${nextAdvice.detected_language_id}, suggested ${nextAdvice.suggested_language_id} (${nextAdvice.transcript_quality})`,
      })

      queueSessionUpdate(
        buildSessionUpdate(
          snapshot.agent_prompt,
          snapshot.realtime_tools,
          snapshot.config.realtime_voice,
          selectedRealtimeModelRef.current,
          snapshot.config.transcription_model,
          snapshot.config.supported_languages,
          nextAdvice.transcription_language_id || nextLanguageId,
        ),
      )
    } catch {
      const fallbackAdvice = defaultLanguageAdvice(activeLanguageRef.current)
      fallbackAdvice.nudge = 'Keep the reply compact, and confirm the customer language if the transcript sounded off.'
      fallbackAdvice.confidence = 'low'
      languageAdviceRef.current = fallbackAdvice
      startTransition(() => setLanguageAdvice(fallbackAdvice))
    }
  })

  const runUnifiedCustomerTurn = useEffectEvent(async (transcriptText: string) => {
    const snapshot = bootstrapRef.current
    if (!snapshot) return
    languageRepairAttemptedRef.current = false
    const requestSeq = ++policyReplyRequestSeqRef.current
    setThinking(true)

    pushAgentActivity({
      agent: 'language_coach',
      status: 'started',
      summary: 'Detecting customer language and transcript quality',
    })
    pushAgentActivity({
      agent: 'caller',
      status: 'started',
      summary: 'Preparing approved reply from backend policy engine',
    })

    try {
      const result = await customerTurn({
        transcript: transcriptText,
        current_language_id: activeLanguageRef.current,
        preferred_language_id: selectedLanguageRef.current,
        recent_transcript: transcriptRef.current.slice(-6),
        account_number: snapshot.account_number,
        voice: snapshot.config.realtime_voice,
        messages: transcriptRef.current
          .filter((entry) => entry.role === 'assistant' || entry.role === 'customer')
          .map((entry) => ({ role: entry.role as 'assistant' | 'customer', text: entry.text })),
        coaching_hints: coachingHintsRef.current.slice(0, 5),
      })
      if (requestSeq !== policyReplyRequestSeqRef.current) return

      const nextAdvice = result.advice
      const nextLanguageId = nextAdvice.suggested_language_id || activeLanguageRef.current
      languageAdviceRef.current = nextAdvice
      activeLanguageRef.current = nextLanguageId
      costsRef.current = result.costs
      startTransition(() => {
        setCosts(result.costs)
        setLanguageAdvice(nextAdvice)
        setActiveLanguageId(nextLanguageId)
      })

      if (nextAdvice.should_switch || nextAdvice.transcript_quality !== 'good') {
        appendSystemTranscript(`Language coach: ${nextAdvice.nudge}`)
      }
      pushAgentActivity({
        agent: 'language_coach',
        status: 'completed',
        summary: `Detected ${nextAdvice.detected_language_id}, suggested ${nextAdvice.suggested_language_id} (${nextAdvice.transcript_quality})`,
      })

      queueSessionUpdate(
        buildSessionUpdate(
          snapshot.agent_prompt,
          snapshot.realtime_tools,
          snapshot.config.realtime_voice,
          selectedRealtimeModelRef.current,
          snapshot.config.transcription_model,
          snapshot.config.supported_languages,
          nextAdvice.transcription_language_id || nextLanguageId,
        ),
      )

      applyBackendToolCalls(result.tool_calls)

      if (result.assistant_text && result.assistant_text.trim()) {
        lastApprovedScriptRef.current = result.assistant_text
        queueResponseCreate(buildScriptedResponse(result.assistant_text))
        pushAgentActivity({
          agent: 'caller',
          status: 'completed',
          summary: `Approved reply ready (${result.model})`,
        })
        return
      }

      setThinking(false)
      pushAgentActivity({
        agent: 'caller',
        status: 'completed',
        summary: 'No spoken follow-up was needed for this turn',
      })
    } catch (error) {
      if (requestSeq !== policyReplyRequestSeqRef.current) return
      const message = error instanceof Error ? error.message : 'Backend turn failed.'
      appendSystemTranscript(`Call policy error: ${message}`)
      setThinking(false)
      pushAgentActivity({ agent: 'caller', status: 'error', summary: message })
    }
  })

  const runSupervisorReview = useEffectEvent(async () => {
    const snapshot = bootstrapRef.current
    if (!snapshot) return
    pushAgentActivity({
      agent: 'supervisor',
      status: 'started',
      summary: 'Reviewing latest agent turn for grounding and policy',
    })
    try {
      const result = await evaluateSupervisor({
        customer: snapshot.customer,
        invoices: snapshot.invoices,
        transcript: transcriptRef.current,
        tool_calls: toolCallsRef.current,
        disposition: dispositionRef.current,
        turn_number: turnNumberRef.current,
      })
      costsRef.current = result.costs
      startTransition(() => {
        setBoard(result.board)
        setCosts(result.costs)
      })
      coachingHintsRef.current = result.issues.map((issue) => issue.suggested_fix).filter(Boolean)
      if (result.issues.length > 0) {
        appendSystemTranscript(`Supervisor coach: ${result.issues[0].suggested_fix}`)
      }
      pushAgentActivity({
        agent: 'supervisor',
        status: 'completed',
        summary:
          result.issues.length === 0
            ? 'Reviewed turn — no issues raised'
            : `Raised ${result.issues.length} flag(s): ${result.issues[0].title}`,
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Supervisor review failed'
      pushAgentActivity({
        agent: 'supervisor',
        status: 'error',
        summary: `Supervisor error: ${message.slice(0, 160)}`,
      })
    }
  })

  const sendChatMessage = useEffectEvent(async () => {
    const snapshot = bootstrapRef.current
    if (!snapshot) return
    const text = chatInput.trim()
    if (!text || chatBusy) return

    setChatInput('')
    setChatBusy(true)
    appendCustomerTranscript(text)
    pushAgentActivity({
      agent: 'chat_agent',
      status: 'started',
      summary: 'Generating reply (text mode, no voice billing)',
    })

    const history = [...transcriptRef.current, {
      id: `local_${Date.now()}`,
      role: 'customer' as const,
      text,
      timestamp: new Date().toISOString(),
      status: 'final' as const,
    }]

    try {
      await runLanguageCoach(text)
      const hintsForTurn = coachingHintsRef.current.slice(0, 5)
      const result = await chatTurn({
        account_number: snapshot.account_number,
        voice: snapshot.config.realtime_voice,
        messages: history
          .filter((entry) => entry.role === 'assistant' || entry.role === 'customer')
          .map((entry) => ({ role: entry.role as 'assistant' | 'customer', text: entry.text })),
        coaching_hints: hintsForTurn,
        language_advice: languageAdviceRef.current,
      })

      if (hintsForTurn.length > 0) {
        pushAgentActivity({
          agent: 'chat_agent',
          status: 'completed',
          summary: `Coached by supervisor (${hintsForTurn.length} hint${hintsForTurn.length === 1 ? '' : 's'} applied)`,
        })
      }

      costsRef.current = result.costs
      startTransition(() => setCosts(result.costs))

      applyBackendToolCalls(result.tool_calls)

      if (result.assistant_text) {
        finalizeAssistantDraft(result.assistant_text)
        turnNumberRef.current += 1
        pushAgentActivity({
          agent: 'chat_agent',
          status: 'completed',
          summary: `Replied (${result.model})`,
        })
        void runSupervisorReview()
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Chat turn failed.'
      appendSystemTranscript(`Chat error: ${message}`)
      pushAgentActivity({ agent: 'chat_agent', status: 'error', summary: message })
    } finally {
      setChatBusy(false)
    }
  })

  const executeFunctionCalls = useEffectEvent(
    async (functionCalls: ReturnType<typeof extractRealtimeFunctionCalls>) => {
      if (functionCalls.length === 0) return

      setThinking(true)

      for (const functionCall of functionCalls) {
        if (processedCallIdsRef.current.has(functionCall.callId)) continue
        processedCallIdsRef.current.add(functionCall.callId)

        try {
          const result = await invokeTool(functionCall.name, functionCall.args)
          const toolEntry: ToolCallEntry = {
            id: functionCall.callId,
            name: functionCall.name,
            args: functionCall.args,
            result,
            timestamp: new Date().toISOString(),
            status: 'completed',
          }

          startTransition(() => setToolCalls((previous) => [toolEntry, ...previous]))
          pushAgentActivity({
            agent: 'tool',
            status: 'completed',
            summary: `${functionCall.name}(${Object.keys(functionCall.args || {}).join(', ')})`,
          })

          const nextDisposition = determineDisposition(functionCall.name)
          if (nextDisposition) {
            startTransition(() => setDisposition(nextDisposition))
          }

          if (functionCall.name === 'get_invoices' && Array.isArray(result.invoices)) {
            startTransition(() => {
              setBootstrap((previous) =>
                previous ? { ...previous, invoices: result.invoices as BootstrapResponse['invoices'] } : previous,
              )
            })
          }

          if (functionCall.name === 'get_customer' && result.customer) {
            startTransition(() => {
              setBootstrap((previous) =>
                previous
                  ? { ...previous, customer: result.customer as BootstrapResponse['customer'] }
                  : previous,
              )
            })
          }

          sendRealtimeEvent({
            type: 'conversation.item.create',
            item: {
              type: 'function_call_output',
              call_id: functionCall.callId,
              status: 'completed',
              output: JSON.stringify(result),
            },
          })
        } catch (error) {
          const result = {
            ok: false,
            error: error instanceof Error ? error.message : 'Tool execution failed.',
          }
          const toolEntry: ToolCallEntry = {
            id: functionCall.callId,
            name: functionCall.name,
            args: functionCall.args,
            result,
            timestamp: new Date().toISOString(),
            status: 'error',
          }
          startTransition(() => setToolCalls((previous) => [toolEntry, ...previous]))
          sendRealtimeEvent({
            type: 'conversation.item.create',
            item: {
              type: 'function_call_output',
              call_id: functionCall.callId,
              status: 'completed',
              output: JSON.stringify(result),
            },
          })
        }
      }

      sendGuidedResponse('tool_followup')
    },
  )

  const handleRealtimeEvent = useEffectEvent(async (event: Record<string, unknown>) => {
    const eventType = String(event.type ?? '')

    if (eventType === 'input_audio_buffer.speech_started') {
      setCustomerSpeaking(true)
      setThinking(false)
      // Do NOT cancel an active agent response on speech_started — VAD fires
      // on coughs, breaths, and self-echo and was truncating legitimate agent
      // turns mid-sentence. Real barge-in is confirmed downstream when the
      // transcription completes with substantive content (>=3 tokens), which
      // is where we cancel ([response.create] path in transcription.completed).
      return
    }
    if (eventType === 'response.created') {
      const resp =
        typeof event.response === 'object' && event.response ? (event.response as Record<string, unknown>) : {}
      const id = typeof resp.id === 'string' ? resp.id : null
      if (id) activeResponseIdRef.current = id
      const spokenText = typeof resp._spoken_text === 'string' ? resp._spoken_text.trim() : ''
      if (spokenText && !assistantMessageIdRef.current) {
        upsertAssistantDraft(spokenText)
      }
      streamingViolationCancelledRef.current = false
      return
    }
    if (eventType === 'response.cancelled' || eventType === 'response.canceled') {
      if (assistantMessageIdRef.current && assistantBufferRef.current.trim()) {
        sealAssistantDraft()
      }
      pendingBargeInRef.current = null
      activeResponseIdRef.current = null
      lastAgentSpeakStartRef.current = null
      flushPendingRealtime()
      return
    }
    if (eventType === 'input_audio_buffer.speech_stopped') {
      setCustomerSpeaking(false)
      setThinking(true)
      return
    }
    if (eventType === 'conversation.item.input_audio_transcription.completed') {
      const transcriptText = String(event.transcript ?? '').trim()
      if (activeResponseIdRef.current) {
        const tokens = transcriptText.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean)
        const looksSubstantive = tokens.length >= 3
        if (!looksSubstantive) {
          appendSystemTranscript('Dropped short transcription captured during agent speech (likely echo).')
          // Drop path skips runUnifiedCustomerTurn, which means nothing else
          // will clear the spinner that speech_stopped turned on. Clear it
          // here so the UI doesn't get stuck in "Agent reasoning…".
          setThinking(false)
          return
        }
        // Real barge-in. Cancel agent and process the customer turn.
        sendRealtimeEvent({ type: 'response.cancel' })
        activeResponseIdRef.current = null
      }
      if (isLikelySttHallucination(transcriptText)) {
        appendSystemTranscript('Dropped transcription hallucination (STT echoed prompt on silence).')
        setThinking(false)
        return
      }
      appendCustomerTranscript(transcriptText, String(event.item_id ?? `customer_${Date.now()}`))
      const usage = event.usage
      if (usage && typeof usage === 'object') {
        await applyCostUpdate({
          event_id:
            typeof event.item_id === 'string'
              ? `agent-transcription:${event.item_id}`
              : buildFallbackCostEventId('agent-transcription'),
          session_id: costsRef.current.session_id || bootstrapRef.current?.costs.session_id || '',
          source: 'agent',
          usage_type: 'transcription',
          model: bootstrapRef.current?.config.transcription_model ?? 'saaras:v3',
          usage: usage as Record<string, unknown>,
        })
      }
      // Single backend roundtrip: deterministic language coach runs inline
      // server-side, then chat/turn uses fresh advice. No client-side
      // serial waterfall, no language-drift repair turn.
      void runUnifiedCustomerTurn(transcriptText)
      return
    }
    if (eventType === 'conversation.item.input_audio_transcription.failed') {
      languageRepairAttemptedRef.current = false
      const fallbackAdvice = defaultLanguageAdvice(activeLanguageRef.current)
      fallbackAdvice.transcript_quality = 'suspect'
      fallbackAdvice.confidence = 'low'
      fallbackAdvice.nudge =
        'I may have misheard you. Please repeat that once more and tell me your preferred language.'
      languageAdviceRef.current = fallbackAdvice
      startTransition(() => setLanguageAdvice(fallbackAdvice))
      appendSystemTranscript(`Language coach: ${fallbackAdvice.nudge}`)
      sendGuidedResponse('customer_turn', fallbackAdvice)
      return
    }
    if (eventType === 'response.output_audio_transcript.delta' || eventType === 'response.output_text.delta') {
      const delta = String(event.delta ?? '')
      if (delta) upsertAssistantDraft(delta)
      // Mid-stream language compliance is now enforced by the backend
      // (reply_violates_english_lock + retry). Front-end no longer cancels
      // mid-utterance — that caused truncated speech and infinite re-render
      // loops when the same approved text was re-issued.
      return
    }
    if (eventType === 'response.done') {
      pendingBargeInRef.current = null
      activeResponseIdRef.current = null
      lastAgentSpeakStartRef.current = null
      const responsePayload =
        typeof event.response === 'object' && event.response ? (event.response as Record<string, unknown>) : {}
      const usage = responsePayload.usage
      if (usage && typeof usage === 'object') {
        await applyCostUpdate({
          event_id:
            typeof responsePayload.id === 'string'
              ? `agent-response:${responsePayload.id}`
              : buildFallbackCostEventId('agent-response'),
          session_id: costsRef.current.session_id || bootstrapRef.current?.costs.session_id || '',
          source: 'agent',
          usage_type: 'response',
          model: selectedRealtimeModelRef.current || bootstrapRef.current?.config.realtime_model || 'bulbul:v3',
          usage: usage as Record<string, unknown>,
        })
      }

      const output = responsePayload.output
      const finalText = extractRealtimeText(output)
      const functionCalls = extractRealtimeFunctionCalls(output)
      const completedText = finalText || assistantBufferRef.current
      const languageIssue =
        completedText && functionCalls.length === 0
          ? detectLanguageComplianceIssue(completedText, languageAdviceRef.current)
          : null
      const promptLeak =
        completedText && functionCalls.length === 0 && detectSystemPromptLeak(completedText)

      if (discardCurrentResponseRef.current) {
        discardCurrentResponseRef.current = false
        resetAssistantDraft()
        flushPendingRealtime()
        return
      }

      if (promptLeak) {
        const snapshot = bootstrapRef.current
        appendSystemTranscript('Suppressed system-prompt leak from voice agent. Re-requesting approved reply.')
        pushAgentActivity({
          agent: 'caller',
          status: 'error',
          summary: 'Voice agent leaked internal instructions — discarded turn and re-fetching approved reply',
        })
        dropAssistantDraft()
        if (snapshot) {
          if (lastApprovedScriptRef.current) {
            queueResponseCreate(buildScriptedResponse(lastApprovedScriptRef.current))
          } else {
            void sendGuidedResponse('customer_turn', languageAdviceRef.current)
          }
          setThinking(true)
          return
        }
        setThinking(false)
        flushPendingRealtime()
        return
      }

      if (finalText) {
        turnNumberRef.current += 1
        finalizeAssistantDraft(finalText)
        setThinking(false)
        void runSupervisorReview()
      } else if (assistantBufferRef.current) {
        turnNumberRef.current += 1
        finalizeAssistantDraft(assistantBufferRef.current)
        setThinking(false)
        void runSupervisorReview()
      }

      if (languageIssue && !languageRepairAttemptedRef.current) {
        const snapshot = bootstrapRef.current
        languageRepairAttemptedRef.current = true
        appendSystemTranscript(`Language coach: ${languageIssue}`)
        pushAgentActivity({
          agent: 'language_coach',
          status: 'started',
          summary: 'Detected a language mismatch in the live reply and forcing an immediate repair',
        })
        if (snapshot) {
          discardCurrentResponseRef.current = true
          dropAssistantDraft()
          void sendGuidedResponse('customer_turn', languageAdviceRef.current)
          setThinking(true)
          return
        }
      }

      if (functionCalls.length > 0) {
        await executeFunctionCalls(functionCalls)
        return
      }
      setThinking(false)
      flushPendingRealtime()
      return
    }
    if (eventType === 'error') {
      const errorObject =
        typeof event.error === 'object' && event.error ? (event.error as Record<string, unknown>) : {}
      const message = String(errorObject.message ?? 'Realtime connection error')
      const code = typeof errorObject.code === 'string' ? errorObject.code : ''
      console.error('[realtime] server error', errorObject, event)
      if (code === 'conversation_already_has_active_response' || /active response/i.test(message)) {
        pendingResponseQueueRef.current = []
        return
      }
      setErrorMessage(message)
      setCallState('error')
    }
  })

  const stopMicMeter = () => {
    if (micRafRef.current !== null) {
      cancelAnimationFrame(micRafRef.current)
      micRafRef.current = null
    }
    analyserRef.current = null
    if (audioContextRef.current) {
      void audioContextRef.current.close().catch(() => undefined)
      audioContextRef.current = null
    }
    setMicLevel(0)
  }

  const startMicMeter = (stream: MediaStream) => {
    try {
      const ctx = new AudioContext()
      audioContextRef.current = ctx
      const source = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser
      const buf = new Uint8Array(analyser.frequencyBinCount)
      const tick = () => {
        if (!analyserRef.current) return
        analyserRef.current.getByteTimeDomainData(buf)
        let sum = 0
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128
          sum += v * v
        }
        const rms = Math.sqrt(sum / buf.length)
        setMicLevel(Math.min(1, rms * 4))
        micRafRef.current = requestAnimationFrame(tick)
      }
      micRafRef.current = requestAnimationFrame(tick)
    } catch {
      // Mic meter is cosmetic.
    }
  }

  const closeMediaResources = () => {
    stopMicMeter()
    if (turnCommitTimerRef.current !== null) {
      window.clearTimeout(turnCommitTimerRef.current)
      turnCommitTimerRef.current = null
    }
    turnCommitBufferRef.current = []
    pendingBargeInRef.current = null
    policyReplyRequestSeqRef.current += 1
    sarvamClientRef.current?.disconnect()
    sarvamClientRef.current = null
    sarvamSessionRef.current = null
    localStreamRef.current?.getTracks().forEach((track) => track.stop())
    localStreamRef.current = null
    if (remoteAudioRef.current) {
      remoteAudioRef.current.srcObject = null
    }
    processedCallIdsRef.current = new Set()
    activeResponseIdRef.current = null
    pendingResponseQueueRef.current = []
    pendingSessionUpdateRef.current = null
    resetAssistantDraft()
    setCustomerSpeaking(false)
    setThinking(false)
    setMuted(false)
    setHeadlessNotices([])
    coachingHintsRef.current = []
  }

  const startCall = async () => {
    const snapshot = bootstrapRef.current
    if (!snapshot || callStateRef.current !== 'ready') return

    // Prime ambience inside the user-gesture tick so autoplay is allowed.
    const ambience = ambienceRef.current
    if (ambience) {
      ambience.volume = 0
      ambience.muted = false
      ambience.play().catch(() => {
        // autoplay blocked — fade effect will retry on next state change
      })
    }

    setCallState('starting')
    setErrorMessage('')
    setCustomerSpeaking(false)
    setThinking(false)
    setDisposition('Call in progress')
    sessionStartRef.current = Date.now()
    setTranscript([])
    setToolCalls([])
    transcriptRef.current = []
    toolCallsRef.current = []
    turnNumberRef.current = 0
    processedCallIdsRef.current = new Set()
    resetAssistantDraft()
    coachingHintsRef.current = []
    setActiveLanguageId(selectedLanguageRef.current)
    setLanguageAdvice(defaultLanguageAdvice(selectedLanguageRef.current))
    setCallSummary(null)

    try {
      const resetCosts = await resetCostLedger({
        model: selectedRealtimeModelRef.current,
        transcription_model: snapshot.config.transcription_model,
      })
      costsRef.current = resetCosts
      startTransition(() => setCosts(resetCosts))

      const session = await createRealtimeSession({
        session_id: resetCosts.session_id,
        voice: snapshot.config.realtime_voice,
        language_id: selectedLanguageRef.current,
      })

      if (!session.session_id) {
        throw new Error('Sarvam session did not return a session_id.')
      }
      sarvamSessionRef.current = session

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      localStreamRef.current = stream
      startMicMeter(stream)

      const client = new SarvamVoiceClient()
      sarvamClientRef.current = client

      const utteranceTextById = new Map<string, string>()
      const dispatchSpoken = (utteranceId: string, chars: number) => {
        // Synthesize a realtime-style `response.done` so the legacy state
        // machine finalises the assistant draft and triggers supervisor review.
        const spoken = utteranceTextById.get(utteranceId) ?? ''
        utteranceTextById.delete(utteranceId)
        lastAgentSpeakStartRef.current = null
        void handleRealtimeEvent({
          type: 'response.done',
          response: {
            id: utteranceId,
            output: [
              {
                type: 'message',
                content: [
                  { type: 'output_audio_transcript', transcript: spoken },
                ],
              },
            ],
          },
        })
        // Record TTS cost out-of-band.
        const sessionId = costsRef.current.session_id || bootstrapRef.current?.costs.session_id || ''
        void applyCostUpdate({
          event_id: `tts_${utteranceId}`,
          session_id: sessionId,
          source: 'sarvam',
          usage_type: 'tts',
          model: session.tts_model,
          usage: {},
          chars,
        } as unknown as CostEventPayload)
      }

      client.setListeners({
        onReady: () => undefined,
        onPartialTranscript: (text) => {
          const trimmed = text.trim()
          const interruptedResponseId = activeResponseIdRef.current
          if (!trimmed || !interruptedResponseId) return
          if (isLikelySttHallucination(trimmed)) return
          pendingBargeInRef.current = { responseId: interruptedResponseId, at: Date.now() }
          client.cancelSpeech('barge_in')
          void handleRealtimeEvent({
            type: 'response.cancelled',
            response: { id: interruptedResponseId },
          })
        },
        onPlaybackStart: (utteranceId) => {
          lastAgentSpeakStartRef.current = Date.now()
          void utteranceId
        },
        onPlaybackEnd: (utteranceId, chars) => dispatchSpoken(utteranceId, chars),
        onPlaybackInterrupted: (utteranceId) => {
          void handleRealtimeEvent({
            type: 'response.cancelled',
            response: { id: utteranceId },
          })
        },
        onFinalTranscript: (text, languageCode) => {
          const trimmed = text.trim()
          if (!trimmed) return
          const pendingBargeIn = pendingBargeInRef.current
          const recentBargeIn = !!pendingBargeIn && Date.now() - pendingBargeIn.at < 2000
          // Drop short interjections that arrive while the agent's TTS audio
          // was still playing (within ~0.9s of last audio_start). Almost always
          // a Saarika hallucination from self-echo, not a real barge-in.
          const lastSpeakAt = lastAgentSpeakStartRef.current
          if (
            !recentBargeIn &&
            lastSpeakAt !== null &&
            activeResponseIdRef.current &&
            Date.now() - lastSpeakAt < 900 &&
            trimmed.split(/\s+/).length <= 2
          ) {
            return
          }
          pendingBargeInRef.current = null

          // Language switch detection still runs per-fragment so the next
          // agent turn uses the right TTS voice.
          const mapped = languageIdForSarvamCode(languageCode)
          if (mapped && mapped !== activeLanguageRef.current) {
            activeLanguageRef.current = mapped
            const nextAdvice = defaultLanguageAdvice(mapped)
            nextAdvice.detected_language_id = mapped
            nextAdvice.suggested_language_id = mapped
            nextAdvice.should_switch = true
            nextAdvice.confidence = 'high'
            nextAdvice.nudge = `Customer switched to ${mapped}. Reply in ${mapped}.`
            languageAdviceRef.current = nextAdvice
            startTransition(() => {
              setActiveLanguageId(mapped)
              setLanguageAdvice(nextAdvice)
            })
          }

          // Show the fragment in the transcript right away so the user sees
          // their words land. But do NOT call the LLM yet — buffer fragments
          // into one logical turn.
          appendCustomerTranscript(
            trimmed,
            `customer_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
          )
          turnCommitBufferRef.current.push(trimmed)

          if (turnCommitTimerRef.current !== null) {
            window.clearTimeout(turnCommitTimerRef.current)
          }
          turnCommitTimerRef.current = window.setTimeout(() => {
            turnCommitTimerRef.current = null
            const merged = turnCommitBufferRef.current.join(' ').trim()
            turnCommitBufferRef.current = []
            if (!merged) return
            // Drop suspected STT hallucinations BEFORE dispatching to the LLM.
            if (isLikelySttHallucination(merged)) {
              appendSystemTranscript('Dropped transcript hallucination (STT echoed prompt on silence).')
              setThinking(false)
              return
            }
            // Hand the merged turn to the existing pipeline. We synthesize
            // the realtime-style event with the merged text but skip the
            // appendCustomerTranscript inside handleRealtimeEvent's branch
            // (it would duplicate what we already showed). The handler
            // pattern is: kick supervisor + chat agent. We reproduce just
            // that here, without the duplicate UI append.
            void runUnifiedCustomerTurn(merged)
          }, turnCommitDelayMs)
        },
        onError: (message) => {
          appendSystemTranscript(`Sarvam error: ${message}`)
        },
        onDisconnect: () => {
          if (callStateRef.current !== 'ending') {
            startTransition(() => setCallState('ready'))
          }
        },
      })

      // Wrap speak so dispatchSpoken knows the text to feed downstream.
      const originalSpeak = client.speak.bind(client)
      client.speak = (text: string, langCode?: string, utteranceId?: string) => {
        const id = originalSpeak(text, langCode, utteranceId)
        if (id) utteranceTextById.set(id, text)
        return id
      }

      await client.connect(session)
      await client.startMic(stream)

      startTransition(() => setCallState('connected'))
      activeResponseIdRef.current = null
      pendingResponseQueueRef.current = []
      pendingSessionUpdateRef.current = null

      const openingLang =
        snapshot.config.supported_languages.find(
          (l) => l.id === selectedLanguageRef.current,
        )?.agent_label ?? 'Hinglish'
      const openingText = buildOpeningText(snapshot.customer, snapshot.agent_persona, openingLang)
      lastApprovedScriptRef.current = openingText
      sendRealtimeEvent(buildScriptedResponse(openingText))
    } catch (error) {
      closeMediaResources()
      const message = error instanceof Error ? error.message : 'Failed to start the browser call.'
      startTransition(() => {
        setCallState('error')
        setErrorMessage(message)
      })
    }
  }

  const endCall = async () => {
    if (!bootstrapRef.current) return

    setCallState('ending')
    closeMediaResources()

    await flushPendingCostEvents()
    try {
      const syncedCosts = await fetchCosts()
      costsRef.current = syncedCosts
      startTransition(() => setCosts(syncedCosts))
    } catch {
      // Best effort. We'll fall back to the latest local snapshot below.
    }

    const startedAt = sessionStartRef.current ?? Date.now()
    const endedAt = Date.now()
    let finalCosts: CostState | null = null
    let summary: CallSummary | null = null
    if (transcriptRef.current.length > 0 || toolCallsRef.current.length > 0) {
      setSummarizing(true)
      pushAgentActivity({
        agent: 'summarizer',
        status: 'started',
        summary: 'Building structured call summary',
      })
      try {
        const result = await summarizeCall({
          customer: bootstrapRef.current.customer,
          invoices: bootstrapRef.current.invoices,
          transcript: transcriptRef.current,
          tool_calls: toolCallsRef.current,
          disposition: dispositionRef.current,
        })
        summary = result.summary
        finalCosts = result.costs
        costsRef.current = result.costs
        startTransition(() => {
          setCallSummary(result.summary)
          setCosts(result.costs)
        })
        if (result.summary?.headline) {
          appendSystemTranscript(`Call summary: ${result.summary.headline}`)
        }
        pushAgentActivity({
          agent: 'summarizer',
          status: 'completed',
          summary: result.summary?.headline || 'Summary generated',
        })
      } catch {
        pushAgentActivity({
          agent: 'summarizer',
          status: 'error',
          summary: 'Summary failed; raw transcript still logged',
        })
      } finally {
        setSummarizing(false)
      }

      try {
        const durationSec = Math.max(0, Math.round((endedAt - startedAt) / 1000))
        const loggedCosts = finalCosts ?? costsRef.current
        const modeCostUsd =
          mode === 'voice'
            ? loggedCosts.agent.estimated_cost_usd
            : loggedCosts.chat_agent?.estimated_cost_usd ?? 0
        const modeTokens =
          mode === 'voice'
            ? loggedCosts.agent.total_tokens
            : loggedCosts.chat_agent?.total_tokens ?? 0
        await logCall({
          account_number: bootstrapRef.current.account_number,
          mode,
          disposition: dispositionRef.current,
          transcript: transcriptRef.current,
          tool_calls: toolCallsRef.current,
          duration_sec: durationSec,
          cost_usd: loggedCosts.combined.estimated_cost_usd,
          total_units: loggedCosts.combined.total_tokens,
          mode_cost_usd: modeCostUsd,
          mode_tokens: modeTokens,
          costs: loggedCosts,
          summary: summary ?? undefined,
        })
      } catch {
        // Best effort.
      }

      try {
        const syncedCosts = await fetchCosts()
        finalCosts = syncedCosts
        costsRef.current = syncedCosts
        startTransition(() => setCosts(syncedCosts))
      } catch {
        // Best effort.
      }

      const snap = finalCosts ?? costsRef.current
      const modeCostUsd =
        mode === 'voice'
          ? snap.agent.estimated_cost_usd
          : snap.chat_agent?.estimated_cost_usd ?? 0
      const modeTokens =
        mode === 'voice'
          ? snap.agent.total_tokens
          : snap.chat_agent?.total_tokens ?? 0
      const record: CallRecord = {
        id: `call_${endedAt}`,
        startedAt,
        endedAt,
        durationSec: Math.max(0, Math.round((endedAt - startedAt) / 1000)),
        mode,
        disposition: dispositionRef.current,
        costUsd: snap.combined.estimated_cost_usd,
        totalTokens: snap.combined.total_tokens,
        modeCostUsd,
        modeTokens,
        summary,
      }
      setCallHistory((prev) => [record, ...prev])
    }

    sessionStartRef.current = null
    startTransition(() => {
      setCallState('ready')
      setThinking(false)
      setCustomerSpeaking(false)
      setDisposition('Call ended')
      setActiveLanguageId(selectedLanguageRef.current)
      setLanguageAdvice(defaultLanguageAdvice(selectedLanguageRef.current))
    })
    void refreshRuntimeState()
  }

  const resetDemoState = async () => {
    closeMediaResources()
    setCallState('loading')
    setTranscript([])
    setToolCalls([])
    setAgentActivity([])
    setDisposition('Awaiting call')
    setErrorMessage('')
    setCallSummary(null)
    setCallHistory([])
    try {
      await resetDemo()
    } finally {
      void loadBootstrapData()
    }
  }

  const beginChatSession = useEffectEvent(async () => {
    const snapshot = bootstrapRef.current
    if (!snapshot || chatBusy) return
    setTranscript([])
    setToolCalls([])
    setAgentActivity([])
    setDisposition('Call in progress')
    setCallSummary(null)
    transcriptRef.current = []
    toolCallsRef.current = []
    turnNumberRef.current = 0
    coachingHintsRef.current = []
    sessionStartRef.current = Date.now()

    setChatBusy(true)
    pushAgentActivity({
      agent: 'chat_agent',
      status: 'started',
      summary: 'Opening chat session — drafting greeting',
    })

    try {
      const resetCosts = await resetCostLedger({
        model: selectedRealtimeModelRef.current,
        transcription_model: snapshot.config.transcription_model,
      })
      costsRef.current = resetCosts
      startTransition(() => setCosts(resetCosts))

      const result = await chatTurn({
        account_number: snapshot.account_number,
        voice: snapshot.config.realtime_voice,
        messages: [
          {
            role: 'customer',
            text: '(The customer has just picked up. Begin the call now per your instructions.)',
          },
        ],
      })
      costsRef.current = result.costs
      startTransition(() => setCosts(result.costs))
      if (result.assistant_text) {
        finalizeAssistantDraft(result.assistant_text)
        turnNumberRef.current += 1
        pushAgentActivity({
          agent: 'chat_agent',
          status: 'completed',
          summary: `Greeted (${result.model})`,
        })
      }
      if (result.tool_calls && result.tool_calls.length > 0) {
        startTransition(() =>
          setToolCalls((previous) => [...result.tool_calls.slice().reverse(), ...previous]),
        )
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to start chat.'
      appendSystemTranscript(`Chat error: ${message}`)
      pushAgentActivity({ agent: 'chat_agent', status: 'error', summary: message })
    } finally {
      setChatBusy(false)
    }
  })

  const handleMoveIssue = async (
    issueId: string,
    status: 'new' | 'reviewing' | 'accepted' | 'dismissed',
  ) => {
    const result = await moveSupervisorIssue(issueId, status)
    startTransition(() => setBoard(result.board))
  }

  const customer = bootstrap?.customer
  const invoices = bootstrap?.invoices ?? []
  const supportedLanguages = bootstrap?.config.supported_languages ?? []
  const totalOutstanding = bootstrap?.total_outstanding ?? 0
  const totalFlags = board.columns.reduce((sum, column) => sum + column.issues.length, 0)
  const canStart = callState === 'ready' && Boolean(bootstrap)
  const canEnd = callState === 'connected' || callState === 'starting'
  const isLive = callState === 'connected'
  const supportedRealtimeModels =
    bootstrap?.config.supported_realtime_models ?? [
      { id: 'bulbul:v3', label: 'Sarvam Bulbul v3' },
    ]
  const selectedRealtimeModelLabel = realtimeModelLabel(
    supportedRealtimeModels,
    selectedRealtimeModel,
  )
  const selectedLanguageLabel = languageLabel(supportedLanguages, selectedLanguageId)
  const activeLanguageLabel = languageLabel(supportedLanguages, activeLanguageId)
  const detectedLanguageLabel = languageLabel(
    supportedLanguages,
    languageAdvice.detected_language_id || activeLanguageId,
  )
  const sarvamPricingReference = bootstrap?.config.pricing_reference?.sarvam
  const sarvamPricingCards: PricingCard[] = []
  const ttsModelId = bootstrap?.config.tts_model ?? selectedRealtimeModel
  if (sarvamPricingReference) {
    const ttsInr = sarvamPricingReference.tts_inr_per_10k_chars[ttsModelId]
    if (typeof ttsInr === 'number') {
      sarvamPricingCards.push({
        id: 'sarvam-tts',
        title: 'Voice synthesis',
        model: ttsModelId,
        active: ttsModelId === selectedRealtimeModel,
        lines: [`Text out ${formatInrRate(ttsInr)}/10k chars`],
      })
    }

    const sttModelId =
      bootstrap?.config.stt_model ??
      bootstrap?.config.transcription_model ??
      costs.agent.transcription_usage.model ??
      'saaras:v3'
    const sttInr = sarvamPricingReference.stt_inr_per_hour[sttModelId]
    if (typeof sttInr === 'number') {
      sarvamPricingCards.push({
        id: 'sarvam-stt',
        title: 'Transcription',
        model: sttModelId,
        lines: [`Audio in ${formatInrRate(sttInr)}/hour`],
      })
    }
  }

  const buildUsdModelCard = (id: string, title: string, modelId: string): PricingCard | null => {
    const pricing = costs.price_table[modelId] ?? {}
    const lines: string[] = []
    if ((pricing.audio_input_per_million ?? 0) > 0) {
      lines.push(`Audio in ${formatUsdRate(pricing.audio_input_per_million ?? 0)}/1M`)
    }
    if ((pricing.audio_cached_input_per_million ?? 0) > 0) {
      lines.push(`Cached audio in ${formatUsdRate(pricing.audio_cached_input_per_million ?? 0)}/1M`)
    }
    if ((pricing.audio_output_per_million ?? 0) > 0) {
      lines.push(`Audio out ${formatUsdRate(pricing.audio_output_per_million ?? 0)}/1M`)
    }
    if ((pricing.text_input_per_million ?? 0) > 0) {
      lines.push(`Text in ${formatUsdRate(pricing.text_input_per_million ?? 0)}/1M`)
    }
    if ((pricing.text_cached_input_per_million ?? 0) > 0) {
      lines.push(`Cached text in ${formatUsdRate(pricing.text_cached_input_per_million ?? 0)}/1M`)
    }
    if ((pricing.text_output_per_million ?? 0) > 0) {
      lines.push(`Text out ${formatUsdRate(pricing.text_output_per_million ?? 0)}/1M`)
    }
    if (lines.length === 0) return null
    return {
      id,
      title,
      model: modelId,
      lines,
    }
  }

  const usdPricingCards = [
    buildUsdModelCard('chat-agent-pricing', 'Chat agent', bootstrap?.config.chat_model ?? costs.chat_agent?.model ?? 'gpt-4.1'),
    buildUsdModelCard('supervisor-pricing', 'Supervisor', bootstrap?.config.supervisor_model ?? costs.supervisor.model),
    buildUsdModelCard('language-coach-pricing', 'Language coach', bootstrap?.config.language_coach_model ?? costs.language_coach.model),
  ].filter((card): card is PricingCard => Boolean(card))
  const sarvamPricingNote = sarvamPricingReference
    ? `Sarvam speech pricing is shown in INR. Ledger totals are still normalized to USD using ₹${sarvamPricingReference.inr_per_usd.toFixed(2)} per USD.`
    : null
  const policyStackCostUsd =
    (costs.chat_agent?.estimated_cost_usd ?? 0) +
    costs.supervisor.estimated_cost_usd +
    costs.language_coach.estimated_cost_usd
  const policyStackTokens =
    (costs.chat_agent?.total_tokens ?? 0) +
    costs.supervisor.total_tokens +
    costs.language_coach.total_tokens
  const voiceTtsChars = costs.agent.response_usage.text_output_tokens
  const voiceSttSeconds = costs.agent.transcription_usage.audio_input_tokens
  const chatModelLabel = bootstrap?.config.chat_model ?? costs.chat_agent?.model ?? 'gpt-4.1'
  const supervisorModelLabel = bootstrap?.config.supervisor_model ?? costs.supervisor.model
  const languageCoachModelLabel = bootstrap?.config.language_coach_model ?? costs.language_coach.model

  const stageTone =
    callState === 'error'
      ? 'danger'
      : callState === 'connected'
        ? customerSpeaking
          ? 'warn'
          : thinking
            ? 'neutral'
            : 'good'
        : 'neutral'

  const meterBars = 28
  const activeBars = isLive ? Math.round(micLevel * meterBars) : 0

  return (
    <div className="app">
      <audio autoPlay className="remote-audio" ref={remoteAudioRef} />
      <audio loop preload="auto" ref={ambienceRef} src="/sound/call_center_background.wav" />

      <header className="topbar">
        <div className="topbar__brand">
          <div className="brand-lockup" aria-label="DHL | Findability Sciences">
            <img className="brand-lockup__logo brand-lockup__logo--dhl" src="/logos/DHL.png" alt="DHL" />
            <span className="brand-lockup__x" aria-hidden>|</span>
            <img className="brand-lockup__logo brand-lockup__logo--fs" src="/logos/FSSML.png" alt="Findability Sciences" />
          </div>
          <div>
            <div className="topbar__title">Collections Voice POC</div>
          </div>
        </div>

        <nav className="topbar__tabs">
          <button
            className={activeTab === 'call' ? 'tab tab--active' : 'tab'}
            onClick={() => setActiveTab('call')}
            type="button"
          >
            Live Call
          </button>
          <button
            className={activeTab === 'wrap' ? 'tab tab--active' : 'tab'}
            onClick={() => setActiveTab('wrap')}
            type="button"
          >
            Wrap-up
            {callHistory.length > 0 ? <span className="tab__badge">{callHistory.length}</span> : null}
          </button>
          <button
            className={activeTab === 'supervisor' ? 'tab tab--active' : 'tab'}
            onClick={() => setActiveTab('supervisor')}
            type="button"
          >
            Supervisor
            {totalFlags > 0 ? <span className="tab__badge">{totalFlags}</span> : null}
          </button>
        </nav>

        <div className="topbar__actions">
          <StatusPill tone={stageTone}>{getCallStateLabel()}</StatusPill>
          {isLive ? <span className="topbar__timer">{formatElapsed(elapsed)}</span> : null}
          <button className="btn btn--ghost" onClick={() => void resetDemoState()} type="button">
            Reset
          </button>
          {onLogout ? (
            <button
              className="btn btn--ghost"
              onClick={onLogout}
              type="button"
              title={username ? `Signed in as ${username}` : 'Log out'}
            >
              Log out{username ? ` (${username})` : ''}
            </button>
          ) : null}
        </div>
      </header>

      {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}

      {activeTab === 'call' ? (
        <main className="layout">
          <section className="stage">
            <div className="stage__toolbar">
              <div className="mode-switch" role="tablist" aria-label="Interaction mode">
                <button
                  type="button"
                  role="tab"
                  aria-selected={mode === 'voice'}
                  className={mode === 'voice' ? 'mode-switch__btn mode-switch__btn--on' : 'mode-switch__btn'}
                  onClick={() => setMode('voice')}
                  disabled={isLive || chatBusy}
                >
                  Voice
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={mode === 'chat'}
                  className={mode === 'chat' ? 'mode-switch__btn mode-switch__btn--on' : 'mode-switch__btn'}
                  onClick={() => setMode('chat')}
                  disabled={isLive || chatBusy}
                >
                  Chat
                </button>
              </div>
              {mode === 'voice' ? (
                <label className="stage__model">
                  <span>Model</span>
                  <select
                    value={selectedRealtimeModel}
                    onChange={(event) => setSelectedRealtimeModel(event.target.value)}
                    disabled={isLive || callState === 'starting' || chatBusy}
                  >
                    {supportedRealtimeModels.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
              {mode === 'voice' ? (
                <label className="headless-toggle" title="Hide live transcript and show a phone-call view. Quality coaching still runs in background; full transcript appears in Wrap-Up.">
                  <input
                    type="checkbox"
                    checked={headless}
                    onChange={(event) => setHeadless(event.target.checked)}
                  />
                  <span>Headless</span>
                </label>
              ) : null}
              {mode === 'voice' ? (
                <label className="ambience-slider" title="Background call-center ambience volume (does not affect agent voice).">
                  <span>Ambience</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={ambienceGain}
                    onChange={(event) => setAmbienceGain(Number(event.target.value))}
                  />
                  <span className="ambience-slider__value">{Math.round(ambienceGain * 100)}%</span>
                </label>
              ) : null}
              <div className="stage__toolbar-spacer" />
              {mode === 'voice' ? (
                <>
                  <button
                    className="btn btn--primary"
                    disabled={!canStart}
                    onClick={() => void startCall()}
                    type="button"
                  >
                    Start Call
                  </button>
                  <button
                    className="btn btn--secondary"
                    disabled={!canEnd}
                    onClick={() => void endCall()}
                    type="button"
                  >
                    End
                  </button>
                </>
              ) : (
                <>
                  <button
                    className="btn btn--primary"
                    disabled={chatBusy || callState !== 'ready'}
                    onClick={() => void beginChatSession()}
                    type="button"
                  >
                    {transcript.length === 0 ? 'Begin Chat' : 'Restart Chat'}
                  </button>
                  <button
                    className="btn btn--secondary"
                    disabled={chatBusy || transcript.length === 0}
                    onClick={() => void endCall()}
                    type="button"
                  >
                    Wrap Up
                  </button>
                </>
              )}
            </div>
            {mode === 'voice' && headless ? (
              <div className={`phone phone--popup phone--${stageTone}`} role="dialog" aria-label="Voice call">
                <div
                  className="phone__backdrop"
                  aria-hidden
                  onClick={() => setHeadless(false)}
                />
                <div className="phone__frame">
                  <div className="phone__notch" />
                  <button
                    type="button"
                    className="phone__close"
                    onClick={() => setHeadless(false)}
                    aria-label="Exit headless view (call continues)"
                    title="Exit headless view (call continues)"
                  >
                    ×
                  </button>
                  <div className="phone__screen">
                    <div className="phone__status-row">
                      <span>{isLive ? formatElapsed(elapsed) : 'INCOMING'}</span>
                      <span className="phone__status-dot" />
                      <span>{isLive ? disposition : 'DHL EXPRESS INDIA'}</span>
                    </div>
                    <div className={`phone__avatar phone__avatar--brand${customerSpeaking ? ' phone__avatar--talking' : ''}${thinking ? ' phone__avatar--thinking' : ''}`}>
                      <div className="phone__avatar-ring" />
                      <div className="phone__avatar-ring phone__avatar-ring--delay" />
                      <div className="phone__avatar-core">
                        <img src="/logos/DHL.png" alt="DHL" className="phone__avatar-img" />
                      </div>
                    </div>
                    <div className="phone__caller">
                      <div className="phone__name">DHL Express India</div>
                      <div className="phone__sub">
                        {callState === 'starting'
                          ? 'Calling…'
                          : callState === 'connected'
                            ? customerSpeaking
                              ? `${customer?.contact_name ?? 'You'} speaking…`
                              : thinking
                                ? 'Agent thinking…'
                                : `On call · ${customer?.contact_name ?? 'Customer'}`
                            : callState === 'ending'
                              ? 'Wrapping up…'
                              : `Incoming call for ${customer?.contact_name ?? 'customer'}`}
                      </div>
                    </div>
                    <div className="phone__meter" aria-hidden>
                      {Array.from({ length: meterBars }).map((_, i) => (
                        <span
                          key={i}
                          className={`phone__bar${i < activeBars ? ' phone__bar--on' : ''}${
                            thinking ? ' phone__bar--pulse' : ''
                          }`}
                        />
                      ))}
                    </div>
                    <div className="phone__notices" aria-live="polite">
                      {headlessNotices.map((notice) => (
                        <div className="phone__notice" key={notice.id}>{notice.text}</div>
                      ))}
                    </div>
                    <div className="phone__controls">
                      {isLive || callState === 'starting' ? (
                        <>
                          <button
                            type="button"
                            className={`phone__btn phone__btn--mute${muted ? ' phone__btn--on' : ''}`}
                            disabled={!isLive}
                            onClick={() => setMuted((value) => !value)}
                            aria-label={muted ? 'Unmute microphone' : 'Mute microphone'}
                          >
                            <span>{muted ? 'Unmute' : 'Mute'}</span>
                          </button>
                          <button
                            type="button"
                            className="phone__btn phone__btn--end"
                            disabled={!canEnd}
                            onClick={() => void endCall()}
                            aria-label="End call"
                          >
                            <span>End</span>
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            type="button"
                            className="phone__btn phone__btn--decline"
                            disabled
                            aria-label="Decline (preview only)"
                            title="Preview UI — use Accept to start the simulated call"
                          >
                            <span>Decline</span>
                          </button>
                          <button
                            type="button"
                            className="phone__btn phone__btn--accept"
                            disabled={!canStart}
                            onClick={() => void startCall()}
                            aria-label="Accept call"
                          >
                            <span>Accept</span>
                          </button>
                        </>
                      )}
                    </div>
                    <div className="phone__footer">{selectedRealtimeModelLabel}</div>
                  </div>
                </div>
              </div>
            ) : mode === 'voice' ? (
              <div className={`stage__hero stage__hero--compact stage__hero--${stageTone}`}>
                <div className="stage__line">{getStageLine()}</div>
                <div className="stage__meter stage__meter--slim" aria-hidden>
                  {Array.from({ length: meterBars }).map((_, i) => (
                    <span
                      key={i}
                      className={`stage__bar${i < activeBars ? ' stage__bar--on' : ''}${
                        thinking ? ' stage__bar--pulse' : ''
                      }`}
                    />
                  ))}
                </div>
                <div className="stage__chips">
                  <StatusPill tone={customerSpeaking ? 'warn' : 'neutral'}>
                    {customerSpeaking ? 'Customer speaking' : 'Customer idle'}
                  </StatusPill>
                  <StatusPill tone={thinking ? 'neutral' : 'good'}>
                    {thinking ? 'Agent reasoning' : 'Agent ready'}
                  </StatusPill>
                  <StatusPill tone="neutral">{selectedRealtimeModelLabel}</StatusPill>
                  <StatusPill tone="neutral">{disposition}</StatusPill>
                </div>
              </div>
            ) : (
              <div className="stage__hero stage__hero--compact stage__hero--neutral stage__hero--chat">
                <div className="stage__line">
                  {chatBusy
                    ? 'Chat agent thinking…'
                    : transcript.length === 0
                      ? 'Click Begin Chat to open a text session — voice billing is paused in this mode.'
                      : 'Type as the customer. The agentic stack will reply.'}
                </div>
                <div className="stage__chips">
                  <StatusPill tone={chatBusy ? 'neutral' : 'good'}>
                    {chatBusy ? 'Agent reasoning' : 'Agent ready'}
                  </StatusPill>
                  <StatusPill tone="neutral">Mode: chat (text-only)</StatusPill>
                  <StatusPill tone="neutral">{disposition}</StatusPill>
                </div>
              </div>
            )}

            {mode === 'voice' && headless ? null : (
            <div className="transcript" ref={transcriptScrollRef}>
              {deferredTranscript.length === 0 ? (
                <div className="transcript__empty">
                  Transcript appears here in real time. Click <strong>Start Call</strong> and speak as the customer.
                </div>
              ) : null}
              {deferredTranscript.map((entry) => (
                <div className={`bubble bubble--${entry.role}`} key={entry.id}>
                  <div className="bubble__meta">
                    <span>{entry.role === 'assistant' ? 'Agent' : entry.role === 'customer' ? 'You' : 'System'}</span>
                    <span>{formatTime(entry.timestamp)}</span>
                  </div>
                  <p>
                    {entry.text}
                    {entry.status === 'streaming' ? <span className="bubble__caret" /> : null}
                  </p>
                </div>
              ))}
              {customerSpeaking ? (
                <div className="bubble bubble--customer bubble--ghost">
                  <div className="bubble__meta">
                    <span>You</span>
                    <span>transcribing…</span>
                  </div>
                  <p>
                    <span className="dot-flash" />
                    <span className="dot-flash" />
                    <span className="dot-flash" />
                  </p>
                </div>
              ) : null}
              {mode === 'chat' && chatBusy ? (
                <div className="bubble bubble--assistant bubble--ghost">
                  <div className="bubble__meta">
                    <span>Agent</span>
                    <span>thinking…</span>
                  </div>
                  <p>
                    <span className="dot-flash" />
                    <span className="dot-flash" />
                    <span className="dot-flash" />
                  </p>
                </div>
              ) : null}
            </div>
            )}

            {mode === 'chat' ? (
              <form
                className="composer"
                onSubmit={(event) => {
                  event.preventDefault()
                  void sendChatMessage()
                }}
              >
                <textarea
                  className="composer__input"
                  rows={2}
                  value={chatInput}
                  placeholder={
                    transcript.length === 0
                      ? 'Click Begin Chat to start, or just type to greet the agent…'
                      : 'Type as the customer (Shift+Enter for new line)'
                  }
                  onChange={(event) => setChatInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault()
                      void sendChatMessage()
                    }
                  }}
                  disabled={chatBusy}
                />
                <button
                  className="btn btn--primary"
                  type="submit"
                  disabled={chatBusy || !chatInput.trim()}
                >
                  Send
                </button>
              </form>
            ) : null}
          </section>

          <aside className="side">
            <div className="cost-strip cost-strip--side">
              {mode === 'voice' ? (
                <>
                <div className="cost-block">
                  <div className="cost-block__head">
                    <span>Sarvam Speech</span>
                    <small>{`${costs.agent.model} + ${bootstrap?.config.transcription_model ?? costs.agent.transcription_usage.model}`}</small>
                  </div>
                  <div className="cost-block__value">{formatUsd(costs.agent.estimated_cost_usd)}</div>
                  <div className="cost-block__meta">
                    <span>tts {formatUsd(costs.agent.response_usage.estimated_cost_usd)}</span>
                    <span>stt {formatUsd(costs.agent.transcription_usage.estimated_cost_usd)}</span>
                    <span>chars {formatNumber(voiceTtsChars)}</span>
                    <span>secs {formatNumber(voiceSttSeconds)}</span>
                  </div>
                </div>
                <div className="cost-block">
                  <div className="cost-block__head">
                    <span>GPT Policy</span>
                    <small>{chatModelLabel}</small>
                  </div>
                  <div className="cost-block__value">{formatUsd(policyStackCostUsd)}</div>
                  <div className="cost-block__meta">
                    <span>{formatNumber(policyStackTokens)} tok</span>
                    <span>chat {formatUsd(costs.chat_agent?.estimated_cost_usd ?? 0)}</span>
                    <span>coach {formatUsd(costs.language_coach.estimated_cost_usd)}</span>
                    <span>supervisor {formatUsd(costs.supervisor.estimated_cost_usd)}</span>
                  </div>
                </div>
                <div className="cost-block cost-block--combined">
                  <div className="cost-block__head">
                    <span>Total Call</span>
                    <small>{`${chatModelLabel} + speech`}</small>
                  </div>
                  <div className="cost-block__value">{formatUsd(costs.combined.estimated_cost_usd)}</div>
                  <div className="cost-block__meta">
                    <span>{formatNumber(costs.combined.total_tokens)} units</span>
                    <span>{costs.agent.model}</span>
                    <span>{supervisorModelLabel}</span>
                    <span>{languageCoachModelLabel}</span>
                  </div>
                </div>
                </>
              ) : (
                <div className="cost-block">
                  <div className="cost-block__head">
                    <span>Chat Agent (text)</span>
                    <small>{costs.chat_agent?.model ?? bootstrap?.config.chat_model ?? 'gpt-4.1'}</small>
                  </div>
                  <div className="cost-block__value">
                    {formatUsd(costs.chat_agent?.estimated_cost_usd ?? 0)}
                  </div>
                  <div className="cost-block__meta">
                    <span>{formatNumber(costs.chat_agent?.total_tokens ?? 0)} tok</span>
                    <span>cached {formatNumber(costs.chat_agent?.text_cached_input_tokens ?? 0)}</span>
                    <span>· in {formatNumber(costs.chat_agent?.text_input_tokens ?? 0)}</span>
                    <span>· out {formatNumber(costs.chat_agent?.text_output_tokens ?? 0)}</span>
                  </div>
                </div>
              )}
            </div>
            <div className="card agents-card">
              <div className="card__head">
                <span className="card__eyebrow">Agent Stack</span>
                <span className="card__pill">{agentActivity.length} events</span>
              </div>
              <div className="agents-grid">
                {AGENT_NODES.map((node) => {
                  const last = agentActivity.find((entry) => entry.agent === node.id)
                  const live = last?.status === 'started'
                  return (
                    <div
                      key={node.id}
                      className={`agent-node${live ? ' agent-node--live' : ''}${
                        last?.status === 'error' ? ' agent-node--error' : ''
                      }`}
                    >
                      <div className="agent-node__row">
                        <span className="agent-node__name">{node.label}</span>
                        <span className="agent-node__model">
                          {node.id === 'caller' ? selectedRealtimeModel : node.model}
                        </span>
                      </div>
                      <div className="agent-node__sub">{last?.summary ?? node.idle}</div>
                    </div>
                  )
                })}
              </div>
              <div className="agent-feed">
                {agentActivity.length === 0 ? (
                  <div className="tool-feed__empty">
                    Agent activity will stream here once a session starts.
                  </div>
                ) : null}
                {agentActivity.slice(0, 12).map((entry) => (
                  <div className={`agent-feed__row agent-feed__row--${entry.status}`} key={entry.id}>
                    <div className="agent-feed__head">
                      <strong>{agentLabel(entry.agent)}</strong>
                      <span>{formatTime(entry.timestamp)}</span>
                    </div>
                    <p>{entry.summary}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="info-strip">
              {(
                [
                  { id: 'account' as const, label: 'Account', count: customer?.account_number ?? '--' },
                  { id: 'language' as const, label: 'Language', count: selectedLanguageLabel },
                  { id: 'invoices' as const, label: 'Invoices', count: invoices.length },
                  { id: 'tools' as const, label: 'Tools', count: toolCalls.length },
                ]
              ).map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  className={`info-strip__btn${openInfo === tab.id ? ' info-strip__btn--on' : ''}`}
                  onClick={() => setOpenInfo(openInfo === tab.id ? null : tab.id)}
                >
                  <span>{tab.label}</span>
                  <small>{tab.count}</small>
                </button>
              ))}
            </div>

            {openInfo === 'account' ? (
              <div className="card">
                <div className="card__head">
                  <span className="card__eyebrow">Account</span>
                  <span className="card__pill">{customer?.account_number ?? '--'}</span>
                </div>
                <h3 className="card__title">{customer?.company_name ?? 'Loading'}</h3>
                <div className="card__row">
                  <div>
                    <span>Outstanding</span>
                    <strong>{formatCurrency(totalOutstanding)}</strong>
                  </div>
                  <div>
                    <span>Invoices</span>
                    <strong>{invoices.length}</strong>
                  </div>
                  <div>
                    <span>Flags</span>
                    <strong>{totalFlags}</strong>
                  </div>
                </div>
                <div className="card__contact">
                  <span>{customer?.contact_name ?? '--'}</span>
                  <span>{customer?.registered_email ?? ''}</span>
                </div>
              </div>
            ) : null}

            {openInfo === 'language' ? (
              <div className="card">
                <div className="card__head">
                  <span className="card__eyebrow">Language Control</span>
                  <span className="card__pill">{selectedLanguageLabel}</span>
                </div>
                <div className="language-stack">
                  <label className="field">
                    <span>Start language</span>
                    <select
                      value={selectedLanguageId}
                      onChange={(event) => {
                        const nextLanguageId = event.target.value
                        setSelectedLanguageId(nextLanguageId)
                        setActiveLanguageId(nextLanguageId)
                        setLanguageAdvice(defaultLanguageAdvice(nextLanguageId))
                      }}
                    >
                      {supportedLanguages.map((language) => (
                        <option key={language.id} value={language.id}>
                          {language.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>Voice model</span>
                    <select
                      value={selectedRealtimeModel}
                      onChange={(event) => setSelectedRealtimeModel(event.target.value)}
                      disabled={isLive || callState === 'starting' || chatBusy}
                    >
                      {supportedRealtimeModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="coach-grid">
                    <div>
                      <span>Live reply</span>
                      <strong>{activeLanguageLabel}</strong>
                    </div>
                    <div>
                      <span>Detected</span>
                      <strong>{detectedLanguageLabel}</strong>
                    </div>
                    <div>
                      <span>Quality</span>
                      <strong>{languageAdvice.transcript_quality}</strong>
                    </div>
                    <div>
                      <span>Confidence</span>
                      <strong>{languageAdvice.confidence}</strong>
                    </div>
                  </div>
                  <div className="pricing-stack">
                    {sarvamPricingCards.length > 0 ? (
                      <div className="pricing-section">
                        <div className="pricing-section__head">
                          <span className="card__eyebrow">Sarvam Speech Pricing</span>
                          <span>INR reference</span>
                        </div>
                        <div className="realtime-rate-grid">
                          {sarvamPricingCards.map((card) => (
                            <div
                              className={`realtime-rate-card${card.active ? ' realtime-rate-card--active' : ''}`}
                              key={card.id}
                            >
                              <div className="realtime-rate-card__head">
                                <strong>{card.title}</strong>
                                <span>{card.model}</span>
                              </div>
                              <div className="realtime-rate-card__meta">
                                {card.lines.map((line) => (
                                  <span key={line}>{line}</span>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                        {sarvamPricingNote ? <div className="pricing-note">{sarvamPricingNote}</div> : null}
                      </div>
                    ) : null}
                    {usdPricingCards.length > 0 ? (
                      <div className="pricing-section">
                        <div className="pricing-section__head">
                          <span className="card__eyebrow">Policy Stack Pricing</span>
                          <span>USD per 1M tokens</span>
                        </div>
                        <div className="realtime-rate-grid">
                          {usdPricingCards.map((card) => (
                            <div className="realtime-rate-card" key={card.id}>
                              <div className="realtime-rate-card__head">
                                <strong>{card.title}</strong>
                                <span>{card.model}</span>
                              </div>
                              <div className="realtime-rate-card__meta">
                                {card.lines.map((line) => (
                                  <span key={line}>{line}</span>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                  <div className="coach-note">{languageAdvice.nudge}</div>
                </div>
              </div>
            ) : null}

            {openInfo === 'invoices' ? (
              <div className="card">
                <div className="card__head">
                  <span className="card__eyebrow">Invoices</span>
                  <span className="card__pill">{invoices.length}</span>
                </div>
                <ul className="invoice-list">
                  {invoices.map((invoice) => (
                    <li className="invoice-row" key={invoice.invoice_no}>
                      <div>
                        <strong>{invoice.invoice_no}</strong>
                        <span>Due {formatDate(invoice.due_date)} · {invoice.overdue_days}d overdue</span>
                      </div>
                      <span className="invoice-row__amt">{formatCurrency(invoice.amount, invoice.currency)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            {callSummary || summarizing ? (
              <div className="card">
                <div className="card__head">
                  <span className="card__eyebrow">Call Summary</span>
                  <span className="card__pill">
                    {summarizing ? 'generating' : callSummary?.customer_mood ?? 'unknown'}
                  </span>
                </div>
                {summarizing && !callSummary ? (
                  <div className="tool-feed__empty">Summarising the call…</div>
                ) : callSummary ? (
                  <div className="summary-stack">
                    {callSummary.headline ? <p className="summary-headline">{callSummary.headline}</p> : null}
                    <div className="summary-block summary-block--metrics">
                      <span>This call</span>
                      <p>
                        {formatUsd(costs.combined.estimated_cost_usd)} ·{' '}
                        {formatElapsed(elapsed)} · {formatNumber(costs.combined.total_tokens)} units
                      </p>
                    </div>
                    {callSummary.agent_tone_assessment ? (
                      <div className="summary-block">
                        <span>Agent tone</span>
                        <p>{callSummary.agent_tone_assessment}</p>
                      </div>
                    ) : null}
                    {callSummary.agreements && callSummary.agreements.length > 0 ? (
                      <div className="summary-block">
                        <span>Customer agreed to</span>
                        <ul>
                          {callSummary.agreements.map((line, idx) => (
                            <li key={`a${idx}`}>{line}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {callSummary.customer_requests && callSummary.customer_requests.length > 0 ? (
                      <div className="summary-block">
                        <span>Customer asked us to</span>
                        <ul>
                          {callSummary.customer_requests.map((line, idx) => (
                            <li key={`r${idx}`}>{line}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {callSummary.agent_commitments && callSummary.agent_commitments.length > 0 ? (
                      <div className="summary-block">
                        <span>Agent committed to</span>
                        <ul>
                          {callSummary.agent_commitments.map((line, idx) => (
                            <li key={`c${idx}`}>{line}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {callSummary.follow_ups && callSummary.follow_ups.length > 0 ? (
                      <div className="summary-block">
                        <span>Follow-ups</span>
                        <ul>
                          {callSummary.follow_ups.map((line, idx) => (
                            <li key={`f${idx}`}>{line}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {callSummary.risk_flags && callSummary.risk_flags.length > 0 ? (
                      <div className="summary-block summary-block--risk">
                        <span>Risk flags</span>
                        <ul>
                          {callSummary.risk_flags.map((line, idx) => (
                            <li key={`x${idx}`}>{line}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {callSummary.next_action ? (
                      <div className="summary-block">
                        <span>Next action</span>
                        <p>{callSummary.next_action}</p>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : null}

            {openInfo === 'tools' ? (
              <div className="card">
                <div className="card__head">
                  <span className="card__eyebrow">Tool calls</span>
                  <span className="card__pill">{toolCalls.length}</span>
                </div>
                <div className="tool-feed">
                  {toolCalls.length === 0 ? (
                    <div className="tool-feed__empty">No tools fired yet.</div>
                  ) : null}
                  {toolCalls.slice(0, 12).map((toolCall) => (
                    <div className={`tool-row tool-row--${toolCall.status}`} key={toolCall.id}>
                      <div className="tool-row__head">
                        <strong>{toolCall.name}</strong>
                        <span>{formatTime(toolCall.timestamp)}</span>
                      </div>
                      <code>{JSON.stringify(toolCall.args)}</code>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </aside>

        </main>
      ) : activeTab === 'wrap' ? (
        <WrapUpView history={callHistory} />
      ) : (
        <SupervisorBoard board={board} costs={costs} onMoveIssue={handleMoveIssue} />
      )}
    </div>
  )
}

function WrapUpView({ history }: { history: CallRecord[] }) {
  const totalCalls = history.length
  const totalSec = history.reduce((sum, r) => sum + r.durationSec, 0)
  const totalCost = history.reduce((sum, r) => sum + r.costUsd, 0)
  const totalTokens = history.reduce((sum, r) => sum + r.totalTokens, 0)

  return (
    <main className="wrap-view">
      <header className="wrap-view__stats">
        <div className="wrap-stat">
          <span>Calls</span>
          <strong>{totalCalls}</strong>
        </div>
        <div className="wrap-stat">
          <span>Total time</span>
          <strong>{formatElapsed(totalSec)}</strong>
        </div>
        <div className="wrap-stat">
          <span>Total cost</span>
          <strong>{formatUsd(totalCost)}</strong>
        </div>
        <div className="wrap-stat">
          <span>Total units</span>
          <strong>{formatNumber(totalTokens)}</strong>
        </div>
      </header>

      {totalCalls === 0 ? (
        <div className="wrap-view__empty">
          No completed calls yet. End a call to log it here.
        </div>
      ) : (
        <ul className="wrap-list">
          {history.map((rec) => (
            <li className="wrap-card" key={rec.id}>
              <div className="wrap-card__head">
                <div className="wrap-card__meta">
                  <strong>
                    <span className={`wrap-card__mode wrap-card__mode--${rec.mode}`}>
                      {rec.mode === 'voice' ? 'Voice' : 'Chat'}
                    </span>
                    {rec.summary?.headline ?? '(no summary)'}
                  </strong>
                  <span>
                    {new Date(rec.startedAt).toLocaleTimeString()} · {rec.disposition}
                  </span>
                </div>
                <div className="wrap-card__metrics">
                  <span>{formatElapsed(rec.durationSec)}</span>
                  <strong>{formatUsd(rec.costUsd)}</strong>
                  <span>{formatNumber(rec.totalTokens)} units</span>
                </div>
              </div>
              {rec.summary ? (
                <div className="summary-stack">
                  {rec.summary.agent_tone_assessment ? (
                    <div className="summary-block">
                      <span>Agent tone</span>
                      <p>{rec.summary.agent_tone_assessment}</p>
                    </div>
                  ) : null}
                  {rec.summary.agreements && rec.summary.agreements.length > 0 ? (
                    <div className="summary-block">
                      <span>Customer agreed to</span>
                      <ul>
                        {rec.summary.agreements.map((line, idx) => (
                          <li key={`a${idx}`}>{line}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                  {rec.summary.customer_requests && rec.summary.customer_requests.length > 0 ? (
                    <div className="summary-block">
                      <span>Customer asked us to</span>
                      <ul>
                        {rec.summary.customer_requests.map((line, idx) => (
                          <li key={`r${idx}`}>{line}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                  {rec.summary.agent_commitments && rec.summary.agent_commitments.length > 0 ? (
                    <div className="summary-block">
                      <span>Agent committed to</span>
                      <ul>
                        {rec.summary.agent_commitments.map((line, idx) => (
                          <li key={`c${idx}`}>{line}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                  {rec.summary.follow_ups && rec.summary.follow_ups.length > 0 ? (
                    <div className="summary-block">
                      <span>Follow-ups</span>
                      <ul>
                        {rec.summary.follow_ups.map((line, idx) => (
                          <li key={`f${idx}`}>{line}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                  {rec.summary.risk_flags && rec.summary.risk_flags.length > 0 ? (
                    <div className="summary-block summary-block--risk">
                      <span>Risk flags</span>
                      <ul>
                        {rec.summary.risk_flags.map((line, idx) => (
                          <li key={`x${idx}`}>{line}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                  {rec.summary.next_action ? (
                    <div className="summary-block">
                      <span>Next action</span>
                      <p>{rec.summary.next_action}</p>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </main>
  )
}
