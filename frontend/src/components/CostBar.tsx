import { formatNumber, formatUsd } from '../lib/format'

type CostBarProps = {
  title: string
  subtitle: string
  model: string
  totalTokens: number
  estimatedCostUsd: number
  breakdown: Array<{ label: string; value: string }>
}

export function CostBar({
  title,
  subtitle,
  model,
  totalTokens,
  estimatedCostUsd,
  breakdown,
}: CostBarProps) {
  return (
    <div className="cost-bar">
      <div>
        <div className="cost-bar__eyebrow">{title}</div>
        <div className="cost-bar__headline">{subtitle}</div>
        <div className="cost-bar__meta">Model: {model}</div>
      </div>
      <div className="cost-bar__stats">
        <div className="cost-chip">
          <span>Tokens</span>
          <strong>{formatNumber(totalTokens)}</strong>
        </div>
        <div className="cost-chip cost-chip--accent">
          <span>Estimated Cost</span>
          <strong>{formatUsd(estimatedCostUsd)}</strong>
        </div>
      </div>
      <div className="cost-bar__breakdown">
        {breakdown.map((item) => (
          <div className="breakdown-chip" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
    </div>
  )
}
