type StatusPillProps = {
  tone: 'neutral' | 'good' | 'warn' | 'danger'
  children: string
}

export function StatusPill({ tone, children }: StatusPillProps) {
  return <span className={`status-pill status-pill--${tone}`}>{children}</span>
}
