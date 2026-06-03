import { useMemo, useState } from 'react'
import type { CostState, SupervisorBoardState, SupervisorIssue } from '../types'
import { formatTime } from '../lib/format'
import { StatusPill } from './StatusPill'

type SupervisorBoardProps = {
  board: SupervisorBoardState
  costs: CostState
  onMoveIssue: (
    issueId: string,
    status: 'new' | 'reviewing' | 'accepted' | 'dismissed',
  ) => Promise<void>
  hideCosts?: boolean
}

type StatusId = 'new' | 'reviewing' | 'accepted' | 'dismissed'
type FilterId = 'all' | StatusId

const STATUSES: StatusId[] = ['new', 'reviewing', 'accepted', 'dismissed']
const STATUS_TONE: Record<StatusId, 'warn' | 'neutral' | 'good'> = {
  new: 'warn',
  reviewing: 'neutral',
  accepted: 'good',
  dismissed: 'neutral',
}
const LABELS: Record<FilterId, string> = {
  all: 'All',
  new: 'New',
  reviewing: 'Reviewing',
  accepted: 'Accepted',
  dismissed: 'Dismissed',
}

function severityTone(severity: string): 'danger' | 'warn' | 'neutral' {
  if (severity === 'high') return 'danger'
  if (severity === 'medium') return 'warn'
  return 'neutral'
}

export function SupervisorBoard({ board, costs, onMoveIssue, hideCosts }: SupervisorBoardProps) {
  const [filter, setFilter] = useState<FilterId>('all')

  const allIssues = useMemo<(SupervisorIssue & { status: StatusId })[]>(
    () =>
      board.columns.flatMap((column) =>
        column.issues.map((issue) => ({ ...issue, status: column.id })),
      ),
    [board],
  )

  const counts = useMemo(() => {
    const map: Record<FilterId, number> = {
      all: allIssues.length,
      new: 0,
      reviewing: 0,
      accepted: 0,
      dismissed: 0,
    }
    for (const issue of allIssues) map[issue.status] += 1
    return map
  }, [allIssues])

  const visible = useMemo(() => {
    const sorted = [...allIssues].sort((a, b) => {
      if (a.turn_number !== b.turn_number) return b.turn_number - a.turn_number
      return (b.created_at || '').localeCompare(a.created_at || '')
    })
    if (filter === 'all') return sorted
    return sorted.filter((issue) => issue.status === filter)
  }, [allIssues, filter])

  const supervisorCost = costs.supervisor
  const langCost = costs.language_coach

  return (
    <div className="sup-shell">
      <div className="sup-header">
        <div>
          <div className="sup-eyebrow">Supervisor agent</div>
          <h2>Findings &amp; QA</h2>
          <p>
            {hideCosts
              ? 'Backend reviewer. Advisory only — never blocks the live caller.'
              : <>Backend reviewer running <code>{supervisorCost.model}</code>. Advisory only — never blocks the live caller.</>}
          </p>
        </div>
        <div className="sup-stats">
          <div className="sup-stat">
            <span>Findings</span>
            <strong>{allIssues.length}</strong>
          </div>
          <div className="sup-stat">
            <span>Reviews</span>
            <strong>{supervisorCost.events}</strong>
          </div>
          {hideCosts ? null : (
            <>
              <div className="sup-stat">
                <span>Supervisor $</span>
                <strong>${supervisorCost.estimated_cost_usd.toFixed(4)}</strong>
              </div>
              <div className="sup-stat">
                <span>+ Lang coach $</span>
                <strong>${langCost.estimated_cost_usd.toFixed(4)}</strong>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="sup-filters">
        {(['all', ...STATUSES] as FilterId[]).map((id) => (
          <button
            key={id}
            type="button"
            className={`sup-filter${filter === id ? ' sup-filter--on' : ''}`}
            onClick={() => setFilter(id)}
          >
            <span>{LABELS[id]}</span>
            <small>{counts[id]}</small>
          </button>
        ))}
      </div>

      <div className="sup-list">
        {visible.length === 0 ? (
          <div className="sup-empty">
            {allIssues.length === 0
              ? 'No findings yet. The supervisor reviews each agent turn after it finishes — flagged issues appear here.'
              : 'Nothing in this lane right now.'}
          </div>
        ) : null}

        {visible.map((issue) => (
          <article className={`sup-row sup-row--${issue.severity}`} key={issue.id}>
            <div className="sup-row__head">
              <div className="sup-row__lead">
                <span className={`sup-dot sup-dot--${issue.severity}`} aria-hidden />
                <div className="sup-row__main">
                  <div className="sup-row__title">{issue.title}</div>
                  <div className="sup-row__meta">
                    <StatusPill tone={severityTone(issue.severity)}>{issue.severity}</StatusPill>
                    <span>{issue.category}</span>
                    <span>turn {issue.turn_number}</span>
                    <span>{formatTime(issue.created_at)}</span>
                  </div>
                </div>
              </div>
              <StatusPill tone={STATUS_TONE[issue.status]}>{issue.status}</StatusPill>
            </div>

            {issue.evidence ? (
              <div className="sup-row__section">
                <span className="sup-row__label">Evidence</span>
                <div className="sup-row__body">{issue.evidence}</div>
              </div>
            ) : null}
            {issue.suggested_fix ? (
              <div className="sup-row__section sup-row__section--fix">
                <span className="sup-row__label">Fix</span>
                <p className="sup-row__fix">{issue.suggested_fix}</p>
              </div>
            ) : null}

            <div className="sup-row__actions">
              {STATUSES.map((status) => (
                <button
                  key={status}
                  type="button"
                  className={`mini-button${issue.status === status ? ' mini-button--active' : ''}`}
                  onClick={() => void onMoveIssue(issue.id, status)}
                >
                  {status}
                </button>
              ))}
            </div>
          </article>
        ))}
      </div>
    </div>
  )
}
