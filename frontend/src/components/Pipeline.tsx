type Props = {
  steps: string[]
}

/** Horizontal text pipeline: Upload → Train → Generate → Evaluate */
export function Pipeline({ steps }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm text-text">
      {steps.map((step, i) => (
        <span key={step} className="flex items-center gap-2">
          <span className="border border-border px-3 py-1.5">{step}</span>
          {i < steps.length - 1 ? <span className="text-muted">→</span> : null}
        </span>
      ))}
    </div>
  )
}
