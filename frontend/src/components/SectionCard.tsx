import type { ReactNode } from 'react'

type SectionCardProps = {
  eyebrow?: string
  title: string
  subtitle?: string
  actions?: ReactNode
  children: ReactNode
}

export function SectionCard({
  eyebrow,
  title,
  subtitle,
  actions,
  children,
}: SectionCardProps) {
  return (
    <section className="section-card">
      <header className="section-card__header">
        <div>
          {eyebrow ? <div className="section-card__eyebrow">{eyebrow}</div> : null}
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {actions ? <div className="section-card__actions">{actions}</div> : null}
      </header>
      <div className="section-card__body">{children}</div>
    </section>
  )
}
