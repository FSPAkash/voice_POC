import type {
  BootstrapResponse,
  CallHistoryRecord,
  CostState,
  LanguageCoachResponse,
  SessionResponse,
  SupervisorBoardState,
  SupervisorEvaluationResponse,
  TranscriptEntry,
  ToolCallEntry,
} from '../types'

const API_ROOT = import.meta.env.VITE_API_BASE_URL ?? ''

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed with ${response.status}`)
  }

  return (await response.json()) as T
}

export function fetchBootstrap() {
  return requestJson<BootstrapResponse>('/api/bootstrap')
}

export function startExotelCall(body: {
  to_number: string
  language_id?: string
  voice?: string
}) {
  return requestJson<{ session?: { session_id?: string } }>('/api/exotel/calls/start', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export type ExotelCallSnapshot = {
  session_id: string
  status: string
  active: boolean
  target_number?: string
  started_at?: string | null
  ended_at?: string | null
  disposition?: string
}

export function fetchExotelActiveCall() {
  return requestJson<{ active_call: ExotelCallSnapshot | null; last_call: ExotelCallSnapshot | null }>(
    '/api/exotel/calls/active',
  )
}

export function fetchCallHistory() {
  return requestJson<{ history: CallHistoryRecord[] }>('/api/call/history')
}

export function createRealtimeSession(body: {
  session_id?: string
  voice?: string
  language_id?: string
}) {
  return requestJson<SessionResponse>('/api/session', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function invokeTool(name: string, payload: Record<string, unknown>) {
  return requestJson<Record<string, unknown>>(`/api/tool/${name}`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function evaluateSupervisor(payload: {
  customer: unknown
  invoices: unknown
  transcript: TranscriptEntry[]
  tool_calls: ToolCallEntry[]
  disposition: string
  turn_number: number
}) {
  return requestJson<SupervisorEvaluationResponse>('/api/supervisor/evaluate', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function detectCustomerLanguage(payload: {
  transcript: string
  current_language_id: string
  preferred_language_id: string
  recent_transcript: TranscriptEntry[]
}) {
  return requestJson<LanguageCoachResponse>('/api/language/detect', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function fetchSupervisorBoard() {
  return requestJson<SupervisorBoardState>('/api/supervisor/issues')
}

export function moveSupervisorIssue(issueId: string, status: string) {
  return requestJson<{ board: SupervisorBoardState }>(`/api/supervisor/issues/${issueId}`, {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  })
}

export function fetchCosts() {
  return requestJson<CostState>('/api/metrics/costs')
}

export function resetCostLedger(body?: { model?: string; transcription_model?: string }) {
  return requestJson<CostState>('/api/metrics/costs/reset', {
    method: 'POST',
    body: JSON.stringify(body ?? {}),
  })
}

export function recordCostEvent(body: {
  event_id: string
  session_id: string
  source: 'agent' | 'supervisor' | 'language_coach' | 'voice'
  usage_type: 'response' | 'transcription' | 'tts' | 'stt'
  model: string
  usage: Record<string, unknown>
  chars?: number
  seconds?: number
}) {
  return requestJson<CostState>('/api/metrics/costs/event', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export type CallSummary = {
  headline?: string
  customer_mood?: string
  customer_sentiment_score?: number
  agent_tone_assessment?: string
  rapport_built?: boolean
  agreements?: string[]
  customer_requests?: string[]
  agent_commitments?: string[]
  follow_ups?: string[]
  next_action?: string
  key_decisions?: string[]
  disposition?: string
  risk_flags?: string[]
}

export function summarizeCall(body: {
  customer: unknown
  invoices: unknown
  transcript: TranscriptEntry[]
  tool_calls: ToolCallEntry[]
  disposition: string
}) {
  return requestJson<{ summary: CallSummary; costs: CostState }>('/api/call/summarize', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function chatTurn(body: {
  messages: { role: 'customer' | 'assistant'; text: string }[]
  account_number: string
  voice?: string
  coaching_hints?: string[]
  language_advice?: Record<string, unknown>
}) {
  return requestJson<{
    assistant_text: string
    tool_calls: ToolCallEntry[]
    costs: CostState
    model: string
    tone?: string
    end_call?: boolean
    end_reason?: string
    parting_message?: string
  }>('/api/chat/turn', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function customerTurn(body: {
  transcript: string
  current_language_id: string
  preferred_language_id: string
  recent_transcript: TranscriptEntry[]
  messages: { role: 'customer' | 'assistant'; text: string }[]
  account_number: string
  voice?: string
  coaching_hints?: string[]
}) {
  return requestJson<{
    advice: import('../types').LanguageAdvice
    assistant_text: string
    tool_calls: ToolCallEntry[]
    costs: CostState
    model: string
    tone?: string
    end_call?: boolean
    end_reason?: string
    parting_message?: string
  }>('/api/turn/customer', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function logCall(body: {
  account_number: string
  mode?: 'voice' | 'chat'
  disposition: string
  transcript: TranscriptEntry[]
  tool_calls: ToolCallEntry[]
  duration_sec?: number
  cost_usd?: number
  total_units?: number
  mode_cost_usd?: number
  mode_tokens?: number
  costs?: CostState
  summary?: CallSummary
  notes?: string
}) {
  return requestJson<{ ok: boolean; entry_id: string }>('/api/call/log', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function resetDemo() {
  return requestJson<{ board: SupervisorBoardState; costs: CostState }>('/api/demo/reset', {
    method: 'POST',
  })
}
