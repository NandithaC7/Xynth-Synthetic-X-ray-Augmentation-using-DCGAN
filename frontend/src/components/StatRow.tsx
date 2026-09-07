type Props = {
  label: string
  value: string | number
  hint?: string
}

/** Single metric row — no card chrome. */
export function StatRow({ label, value, hint }: Props) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-b border-border py-3">
      <div>
        <div className="text-sm text-muted">{label}</div>
        {hint ? <div className="mt-0.5 text-xs text-muted">{hint}</div> : null}
      </div>
      <div className="font-mono text-lg text-text">{value}</div>
    </div>
  )
}
