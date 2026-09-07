import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const COLORS = {
  primary: '#1E5EFF',
  dark: '#0A3D91',
  muted: '#6B7280',
  border: '#E5E7EB',
}

type LossChartProps = {
  gLosses: number[]
  dLosses: number[]
}

/** Generator / discriminator BCE loss curves. */
export function LossChart({ gLosses, dLosses }: LossChartProps) {
  const data = gLosses.map((g, i) => ({
    step: i,
    generator: Number(g.toFixed(4)),
    discriminator: Number((dLosses[i] ?? 0).toFixed(4)),
  }))

  if (data.length === 0) {
    return <EmptyChart label="No training loss data yet. Run DCGAN training first." />
  }

  return (
    <div className="h-72 w-full border border-border p-4">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" />
          <XAxis dataKey="step" tick={{ fontSize: 11, fill: COLORS.muted }} />
          <YAxis tick={{ fontSize: 11, fill: COLORS.muted }} />
          <Tooltip
            contentStyle={{ borderRadius: 0, border: `1px solid ${COLORS.border}` }}
          />
          <Legend />
          <Line type="monotone" dataKey="generator" stroke={COLORS.primary} dot={false} strokeWidth={1.5} />
          <Line type="monotone" dataKey="discriminator" stroke={COLORS.dark} dot={false} strokeWidth={1.5} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

type MetricsBarProps = {
  baseline: { accuracy: number; precision: number; recall: number; f1: number }
  augmented: { accuracy: number; precision: number; recall: number; f1: number }
}

/** Side-by-side baseline vs augmented metric bars. */
export function MetricsBarChart({ baseline, augmented }: MetricsBarProps) {
  const data = [
    { metric: 'Accuracy', baseline: baseline.accuracy, augmented: augmented.accuracy },
    { metric: 'Precision', baseline: baseline.precision, augmented: augmented.precision },
    { metric: 'Recall', baseline: baseline.recall, augmented: augmented.recall },
    { metric: 'F1', baseline: baseline.f1, augmented: augmented.f1 },
  ].map((row) => ({
    ...row,
    baseline: Number((row.baseline * 100).toFixed(2)),
    augmented: Number((row.augmented * 100).toFixed(2)),
  }))

  return (
    <div className="h-72 w-full border border-border p-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" />
          <XAxis dataKey="metric" tick={{ fontSize: 12, fill: COLORS.muted }} />
          <YAxis unit="%" tick={{ fontSize: 11, fill: COLORS.muted }} domain={[0, 100]} />
          <Tooltip
            contentStyle={{ borderRadius: 0, border: `1px solid ${COLORS.border}` }}
            formatter={(value) => [`${value}%`, '']}
          />
          <Legend />
          <Bar dataKey="baseline" fill={COLORS.dark} />
          <Bar dataKey="augmented" fill={COLORS.primary} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

type ClassBarProps = {
  classes: Record<string, number>
}

/** Dataset class count bars. */
export function ClassCountChart({ classes }: ClassBarProps) {
  const data = Object.entries(classes).map(([name, count]) => ({ name, count }))
  if (data.length === 0) {
    return <EmptyChart label="Dataset stats unavailable." />
  }
  return (
    <div className="h-64 w-full border border-border p-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 24 }}>
          <CartesianGrid stroke={COLORS.border} strokeDasharray="3 3" />
          <XAxis type="number" tick={{ fontSize: 11, fill: COLORS.muted }} />
          <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11, fill: COLORS.muted }} />
          <Tooltip contentStyle={{ borderRadius: 0, border: `1px solid ${COLORS.border}` }} />
          <Bar dataKey="count" fill={COLORS.primary} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function EmptyChart({ label }: { label: string }) {
  return (
    <div className="flex h-48 items-center justify-center border border-dashed border-border px-6 text-center text-sm text-muted">
      {label}
    </div>
  )
}
