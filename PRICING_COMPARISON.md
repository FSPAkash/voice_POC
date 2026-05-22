# Voice Stack Pricing Comparison - DHL Collections POC

**Date:** 2026-05-22  
**Prepared for:** DHL Collections POC review  
**Scope:** Estimated voice-agent runtime cost across OpenAI, ElevenLabs, and Anthropic options

---

## 0. Executive Summary

Using current published provider pricing and the active POC configuration, the **OpenAI Realtime Mini baseline** is the lowest-cost production-equivalent stack in this comparison.

| Option | Stack | Estimated cost / 4-min call | Estimated cost / min | Cost vs baseline | Best fit |
|---|---|---:|---:|---:|---|
| A | OpenAI Realtime Mini + GPT-4.1 | **$0.186** | **$0.047** | Baseline | Lowest cost with current quality profile |
| A-Optimized | OpenAI Realtime Mini + GPT-4.1 Mini | $0.098 | $0.025 | -47% | Cost-reduction path, subject to QA validation |
| B | OpenAI GPT Realtime + GPT-4.1 | $0.358 | $0.089 | +92% | Higher-fidelity OpenAI voice |
| C | ElevenLabs Flash + Scribe Realtime + GPT-4.1 | $0.300 | $0.075 | +61% | Broader voice catalog and cloning |
| C-Optimized | ElevenLabs Flash + Scribe Realtime + GPT-4.1 Mini | $0.212 | $0.053 | +14% | Lower-cost ElevenLabs variant |
| D | ElevenLabs Flash + Scribe Realtime + Claude Haiku 4.5 cached | $0.212 | $0.053 | +14% | ElevenLabs voice with alternative LLM |
| E | ElevenLabs Multilingual v2 + Scribe Realtime + GPT-4.1 | $0.480 | $0.120 | +158% | Premium multilingual voice quality |
| F | ElevenAgents hosting + Claude Haiku 4.5 cached | $0.342 | $0.085 | +84% | Managed voice-agent platform option |

For the POC baseline, Option A provides the strongest cost and simplicity profile. Option A-Optimized is the primary cost-reduction path to evaluate through QA. ElevenLabs options are best positioned where voice catalog, cloning, or regional-language realism is worth the cost premium.

---

## 1. Baseline Configuration

The baseline reflects the active POC voice-agent flow.

| Component | Provider / model | Cost treatment |
|---|---|---|
| Realtime voice transport and agent speech | OpenAI `gpt-realtime-mini` | Billed by text/audio tokens |
| Customer speech transcription | OpenAI `gpt-4o-mini-transcribe` | Billed by transcription usage |
| Conversation policy engine | OpenAI `gpt-4.1` | Billed by text input/output tokens |
| Supervisor checks | Rule-based application logic | No live model-call cost |
| Language coach | Rule-based application logic | No live model-call cost |
| Call summary | Rule-based application logic | No live model-call cost |

The comparison prices only the variable runtime components: voice transport, transcription, text reasoning, and platform hosting where applicable.

---

## 2. Workload Assumptions

The same representative 4-minute collections call is applied to every option.

| Parameter | Assumption |
|---|---:|
| Total call duration | 4 minutes |
| Agent speech | 2.4 minutes |
| Customer speech | 1.6 minutes |
| Customer turns | 12 |
| Agent speech volume for TTS pricing | 3,600 characters |
| OpenAI Realtime user-audio tokens | 960 |
| OpenAI Realtime assistant-audio tokens | 2,880 |
| OpenAI Realtime text input | 3,000 tokens |
| OpenAI Realtime text output | 1,000 tokens |
| Policy-engine input | 49,200 tokens |
| Policy-engine output | 1,440 tokens |

Actual invoices may vary with customer talk time, agent response length, prompt size, caching behavior, and provider-side tokenization.

---

## 3. Provider Rates Used

Rates below are public list prices checked on 2026-05-22.

### OpenAI

| Model | Text input | Cached text input | Text output | Audio input | Cached audio input | Audio output |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-realtime` | $4.00 / 1M | $0.40 / 1M | $16.00 / 1M | $32.00 / 1M | $0.40 / 1M | $64.00 / 1M |
| `gpt-realtime-mini` | $0.60 / 1M | $0.06 / 1M | $2.40 / 1M | $10.00 / 1M | $0.30 / 1M | $20.00 / 1M |
| `gpt-4.1` | $2.00 / 1M | $0.50 / 1M | $8.00 / 1M | n/a | n/a | n/a |
| `gpt-4.1-mini` | $0.40 / 1M | $0.10 / 1M | $1.60 / 1M | n/a | n/a | n/a |

| Transcription model | Text input | Text output | Audio input | Published minute estimate |
|---|---:|---:|---:|---:|
| `gpt-4o-transcribe` | $2.50 / 1M | $10.00 / 1M | $6.00 / 1M | $0.006 / min |
| `gpt-4o-mini-transcribe` | $1.25 / 1M | $5.00 / 1M | $3.00 / 1M | $0.003 / min |

### ElevenLabs

| Component | Unit | Price |
|---|---|---:|
| Flash / Turbo TTS | per 1,000 characters | $0.05 |
| Multilingual v2 / v3 TTS | per 1,000 characters | $0.10 |
| Scribe v1 / v2 STT | per audio hour | $0.22 |
| Scribe v2 Realtime STT | per audio hour | $0.39 |
| ElevenAgents / Speech Engine hosting | included/additional call minute | $0.080 |
| ElevenAgents burst pricing | per minute over concurrency burst | $0.160 |

### Anthropic

| Model | Input | 5-min cache write | 1-hour cache write | Cache read | Output |
|---|---:|---:|---:|---:|---:|
| Claude Haiku 4.5 | $1.00 / 1M | $1.25 / 1M | $2.00 / 1M | $0.10 / 1M | $5.00 / 1M |
| Claude Sonnet 4.6 | $3.00 / 1M | $3.75 / 1M | $6.00 / 1M | $0.30 / 1M | $15.00 / 1M |
| Claude Opus 4.7 | $5.00 / 1M | $6.25 / 1M | $10.00 / 1M | $0.50 / 1M | $25.00 / 1M |

---

## 4. Detailed Cost Build-Up

### Option A - OpenAI Realtime Mini + GPT-4.1

| Line item | Formula | Cost |
|---|---:|---:|
| Realtime audio input | 960 x $10.00 / 1M | $0.0096 |
| Realtime audio output | 2,880 x $20.00 / 1M | $0.0576 |
| Realtime text input | 3,000 x $0.60 / 1M | $0.0018 |
| Realtime text output | 1,000 x $2.40 / 1M | $0.0024 |
| Transcription | 1.6 min x $0.003 / min | $0.0048 |
| GPT-4.1 input | 49,200 x $2.00 / 1M | $0.0984 |
| GPT-4.1 output | 1,440 x $8.00 / 1M | $0.0115 |
| **Total / call** | | **$0.1861** |

### Option A-Optimized - OpenAI Realtime Mini + GPT-4.1 Mini

| Line item | Formula | Cost |
|---|---:|---:|
| OpenAI voice and transcription | Same as Option A | $0.0762 |
| GPT-4.1 Mini input | 49,200 x $0.40 / 1M | $0.0197 |
| GPT-4.1 Mini output | 1,440 x $1.60 / 1M | $0.0023 |
| **Total / call** | | **$0.0982** |

### Option B - OpenAI GPT Realtime + GPT-4.1

| Line item | Formula | Cost |
|---|---:|---:|
| Realtime audio input | 960 x $32.00 / 1M | $0.0307 |
| Realtime audio output | 2,880 x $64.00 / 1M | $0.1843 |
| Realtime text input | 3,000 x $4.00 / 1M | $0.0120 |
| Realtime text output | 1,000 x $16.00 / 1M | $0.0160 |
| Transcription | 1.6 min x $0.003 / min | $0.0048 |
| GPT-4.1 input/output | Same as Option A | $0.1099 |
| **Total / call** | | **$0.3578** |

### Option C - ElevenLabs Flash + Scribe Realtime + GPT-4.1

| Line item | Formula | Cost |
|---|---:|---:|
| Scribe v2 Realtime STT | 1.6 / 60 x $0.39 | $0.0104 |
| Flash/Turbo TTS | 3,600 x $0.05 / 1k chars | $0.1800 |
| GPT-4.1 input/output | Same as Option A | $0.1099 |
| **Total / call** | | **$0.3003** |

### Option C-Optimized - ElevenLabs Flash + Scribe Realtime + GPT-4.1 Mini

| Line item | Formula | Cost |
|---|---:|---:|
| Scribe v2 Realtime STT | 1.6 / 60 x $0.39 | $0.0104 |
| Flash/Turbo TTS | 3,600 x $0.05 / 1k chars | $0.1800 |
| GPT-4.1 Mini input/output | 49,200 in + 1,440 out | $0.0220 |
| **Total / call** | | **$0.2124** |

### Option D - ElevenLabs Flash + Scribe Realtime + Claude Haiku 4.5 Cached

This assumes the stable policy prompt benefits from Anthropic 5-minute prompt caching.

| Line item | Formula | Cost |
|---|---:|---:|
| Scribe v2 Realtime STT | 1.6 / 60 x $0.39 | $0.0104 |
| Flash/Turbo TTS | 3,600 x $0.05 / 1k chars | $0.1800 |
| Dynamic uncached input | 6,000 x $1.00 / 1M | $0.0060 |
| 5-min cache write | 3,600 x $1.25 / 1M | $0.0045 |
| Cache reads | 39,600 x $0.10 / 1M | $0.0040 |
| Claude Haiku 4.5 output | 1,440 x $5.00 / 1M | $0.0072 |
| **Total / call** | | **$0.2121** |

### Option E - ElevenLabs Multilingual v2 + Scribe Realtime + GPT-4.1

| Line item | Formula | Cost |
|---|---:|---:|
| Scribe v2 Realtime STT | 1.6 / 60 x $0.39 | $0.0104 |
| Multilingual v2/v3 TTS | 3,600 x $0.10 / 1k chars | $0.3600 |
| GPT-4.1 input/output | Same as Option A | $0.1099 |
| **Total / call** | | **$0.4803** |

### Option F - ElevenAgents Hosting + Claude Haiku 4.5 Cached

| Line item | Formula | Cost |
|---|---:|---:|
| ElevenAgents hosting | 4 min x $0.080 | $0.3200 |
| Claude Haiku cached policy | From Option D, LLM-only subtotal | $0.0217 |
| **Total / call** | | **$0.3417** |

---

## 5. Cost Driver Summary

| Option | Primary cost driver | Share of total |
|---|---|---:|
| A | GPT-4.1 policy input | ~53% |
| A-Optimized | Realtime Mini audio output | ~59% |
| B | GPT Realtime audio output | ~52% |
| C | ElevenLabs Flash/Turbo TTS | ~60% |
| D | ElevenLabs Flash/Turbo TTS | ~85% |
| E | ElevenLabs Multilingual TTS | ~75% |
| F | ElevenAgents hosting minutes | ~94% |

---

## 6. Sensitivity to Agent Talk Time

Agent speech length is one of the biggest controllable variables. The table below changes agent speech by 50% while holding the rest of the call model constant.

| Option | -50% agent speech | Baseline | +50% agent speech |
|---|---:|---:|---:|
| A | $0.150 | $0.186 | $0.222 |
| A-Optimized | $0.067 | $0.098 | $0.129 |
| B | $0.252 | $0.358 | $0.464 |
| C | $0.205 | $0.300 | $0.396 |
| D | $0.118 | $0.212 | $0.306 |
| E | $0.295 | $0.480 | $0.666 |
| F | $0.338 | $0.342 | $0.345 |

---

## 7. Volume Projection

Projection assumes 1,000 calls/day, 30,000 calls/month, and 365,000 calls/year.

| Option | $/call | Daily | Monthly | Annual |
|---|---:|---:|---:|---:|
| A | $0.186 | $186 | $5,584 | $67,934 |
| A-Optimized | $0.098 | $98 | $2,946 | $35,837 |
| B | $0.358 | $358 | $10,733 | $130,582 |
| C | $0.300 | $300 | $9,010 | $109,617 |
| C-Optimized | $0.212 | $212 | $6,372 | $77,520 |
| D | $0.212 | $212 | $6,362 | $77,402 |
| E | $0.480 | $480 | $14,410 | $175,317 |
| F | $0.342 | $342 | $10,250 | $124,706 |

---

## 8. Commercial Fit

| Priority | Best-fit option |
|---|---|
| Lowest cost with current quality profile | Option A |
| Further cost reduction after QA validation | Option A-Optimized |
| Best OpenAI voice quality | Option B |
| Larger voice catalog or voice cloning | Option C or D |
| Stronger regional/multilingual voice realism | Option E |
| Managed voice-agent platform model | Option F |
| Lowest operational complexity | Option A |

Option A is the commercial baseline for cost and operational simplicity. Option A-Optimized is the primary savings opportunity. ElevenLabs options are most relevant where voice catalog, cloning, or regional-language realism carries higher business value than the cost premium.

---

## 9. Sources

- OpenAI API pricing: https://platform.openai.com/docs/pricing/
- OpenAI Realtime cost guide: https://developers.openai.com/api/docs/guides/realtime-costs
- OpenAI `gpt-realtime` model page: https://platform.openai.com/docs/models/gpt-realtime
- OpenAI `gpt-realtime-mini` model page: https://platform.openai.com/docs/models/gpt-realtime-mini
- OpenAI `gpt-4.1-mini` model page: https://platform.openai.com/docs/models/gpt-4.1-mini
- ElevenLabs API pricing: https://elevenlabs.io/pricing/api
- ElevenLabs Agents pricing: https://elevenlabs.io/pricing/agents
- ElevenLabs Scribe v2 Realtime: https://elevenlabs.io/realtime-speech-to-text/
- Anthropic Claude API pricing: https://platform.claude.com/docs/en/about-claude/pricing
