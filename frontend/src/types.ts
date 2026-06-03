export type TranscriptRole = 'assistant' | 'customer' | 'system'

export type TranscriptEntry = {
  id: string
  role: TranscriptRole
  text: string
  timestamp: string
  status?: 'streaming' | 'final'
}

export type ToolCallEntry = {
  id: string
  name: string
  args: Record<string, unknown>
  result: Record<string, unknown>
  timestamp: string
  status: 'completed' | 'error'
}

export type Customer = {
  account_number: string
  company_name: string
  contact_name: string
  alternate_contact_name?: string
  registered_email: string
  phone: string
  billing_city?: string
  language_preferences?: string[]
  payment_terms?: string
  collection_notes?: string[]
}

export type LanguageOption = {
  id: string
  label: string
  agent_label: string
  transcription_language?: string | null
}

export type LanguageAdvice = {
  detected_language_id: string
  suggested_language_id: string
  transcription_language_id: string
  transcript_quality: 'good' | 'unclear' | 'suspect'
  confidence: 'low' | 'medium' | 'high'
  should_switch: boolean
  nudge: string
  rationale: string
}

export type Invoice = {
  invoice_no: string
  invoice_type: string
  amount: number
  currency: string
  invoice_date: string
  due_date: string
  overdue_days: number
  history: string[]
}

export type SupervisorIssue = {
  id: string
  title: string
  category: string
  severity: 'low' | 'medium' | 'high'
  evidence: string
  suggested_fix: string
  turn_number: number
  status: 'new' | 'reviewing' | 'accepted' | 'dismissed'
  created_at: string
  updated_at?: string
}

export type SupervisorColumn = {
  id: 'new' | 'reviewing' | 'accepted' | 'dismissed'
  title: string
  issues: SupervisorIssue[]
}

export type SupervisorBoardState = {
  columns: SupervisorColumn[]
  updated_at: string
}

export type AgentCostState = {
  model: string
  events: number
  response_usage: {
    text_input_tokens: number
    text_cached_input_tokens: number
    text_output_tokens: number
    audio_input_tokens: number
    audio_cached_input_tokens: number
    audio_output_tokens: number
    estimated_cost_usd: number
  }
  transcription_usage: {
    model: string
    audio_input_tokens: number
    text_input_tokens: number
    text_output_tokens: number
    estimated_cost_usd: number
  }
  total_tokens: number
  estimated_cost_usd: number
}

export type SupervisorCostState = {
  model: string
  events: number
  text_input_tokens: number
  text_cached_input_tokens: number
  text_output_tokens: number
  total_tokens: number
  estimated_cost_usd: number
}

export type LanguageCoachCostState = {
  model: string
  events: number
  text_input_tokens: number
  text_cached_input_tokens: number
  text_output_tokens: number
  total_tokens: number
  estimated_cost_usd: number
}

export type ChatAgentCostState = {
  model: string
  events: number
  text_input_tokens: number
  text_cached_input_tokens: number
  text_output_tokens: number
  total_tokens: number
  estimated_cost_usd: number
}

export type CostState = {
  agent: AgentCostState
  supervisor: SupervisorCostState
  language_coach: LanguageCoachCostState
  chat_agent?: ChatAgentCostState
  combined: {
    total_tokens: number
    estimated_cost_usd: number
  }
  updated_at: string
  session_id: string
  price_table_version: string
  price_table: Record<string, Record<string, number>>
}

export type RealtimeTool = {
  type: 'function'
  name: string
  description: string
  parameters: Record<string, unknown>
}

export type RealtimeModelOption = {
  id: string
  label: string
}

export type SarvamVoiceOption = {
  id: string
  label: string
  gender: 'female' | 'male' | 'neutral'
}

export type PricingReference = {
  openai_currency?: 'USD'
  sarvam?: {
    currency?: 'INR'
    inr_per_usd: number
    tts_inr_per_10k_chars: Record<string, number>
    stt_inr_per_hour: Record<string, number>
  }
}

export type CallHistoryRecord = {
  id: string
  startedAt: number
  endedAt: number
  durationSec: number
  mode: 'voice' | 'chat'
  disposition: string
  costUsd: number
  totalTokens: number
  modeCostUsd?: number
  modeTokens?: number
  summary: {
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
  } | null
}

export type BootstrapResponse = {
  account_number: string
  customer: Customer
  invoices: Invoice[]
  total_outstanding: number
  human_agent: {
    name: string
    phone: string
    team: string
  }
  agent_prompt: string
  agent_persona?: {
    name: string
    gender: string
    pronouns: string
  }
  realtime_tools: RealtimeTool[]
  board: SupervisorBoardState
  costs: CostState
  call_history?: CallHistoryRecord[]
  config: {
    realtime_model: string
    supported_realtime_models: RealtimeModelOption[]
    realtime_voice: string
    transcription_model: string
    supervisor_model: string
    language_coach_model: string
    default_language_id: string
    supported_languages: LanguageOption[]
    chat_model?: string
    tts_provider?: 'sarvam' | 'openai'
    stt_provider?: 'sarvam' | 'openai'
    tts_model?: string
    stt_model?: string
    sarvam_voices?: SarvamVoiceOption[]
    sarvam_language_codes?: Record<string, string>
    pricing_reference?: PricingReference
    tts_sample_rate?: number
    stt_sample_rate?: number
  }
}

export type SessionResponse = {
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

export type SupervisorEvaluationResponse = {
  issues: SupervisorIssue[]
  board: SupervisorBoardState
  costs: CostState
  turn_number: number
}

export type LanguageCoachResponse = {
  advice: LanguageAdvice
  costs: CostState
}

export type CallDisposition =
  | 'Awaiting call'
  | 'Call in progress'
  | 'Promise to pay logged'
  | 'Already paid claimed'
  | 'Invoice resend requested'
  | 'Dispute raised'
  | 'Alternate contact captured'
  | 'Transferred to human'
  | 'Call ended'
