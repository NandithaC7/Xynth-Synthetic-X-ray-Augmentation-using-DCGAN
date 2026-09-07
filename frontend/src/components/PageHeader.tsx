import type { ReactNode } from 'react'

type Props = {
  title: string
  subtitle?: string
  children?: ReactNode
}

/** Page section header with generous whitespace. */
export function PageHeader({ title, subtitle, children }: Props) {
  return (
    <div className="mb-12 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold text-dark">{title}</h1>
        {subtitle ? <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">{subtitle}</p> : null}
      </div>
      {children}
    </div>
  )
}
