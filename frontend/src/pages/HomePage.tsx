import { useState } from 'react'
import { api } from '../services/api'
import { Button } from '../components/Button'
import { ClassCountChart } from '../components/Charts'
import { PageHeader } from '../components/PageHeader'
import { Pipeline } from '../components/Pipeline'
import { StatRow } from '../components/StatRow'
import { useResults } from '../hooks/useResults'

/** Home — overview, pipeline, dataset summary, action triggers. */
export function HomePage() {
  const { data, error, loading, refresh } = useResults()
  const [busy, setBusy] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)

  async function run(action: 'dcgan' | 'generate' | 'classifier') {
    setBusy(action)
    setNote(null)
    try {
      if (action === 'dcgan') {
        // Short demo-friendly defaults; override via API body / .env for full runs
        const res = await api.trainDcgan({ epochs: 2 })
        setNote(res.message)
      } else if (action === 'generate') {
        const res = await api.generate({ num_images: 64, build_augmented: true })
        setNote(res.message)
      } else {
        const res = await api.trainClassifier({ epochs: 2, max_per_class: 200 })
        setNote(res.message)
      }
      await refresh()
    } catch (err) {
      setNote(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setBusy(null)
    }
  }

  if (loading) {
    return <p className="text-sm text-muted">Loading…</p>
  }

  const classes = data?.dataset.classes ?? {}

  return (
    <div>
      <PageHeader
        title="Project overview"
        subtitle="Xynth generates synthetic chest X-rays with a DCGAN to augment an imbalanced COVID-19 radiography dataset, then measures whether augmentation improves CNN classification."
      />

      <section className="mb-14">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted">Pipeline</h2>
        <Pipeline steps={['Upload', 'Train DCGAN', 'Generate', 'Evaluate']} />
      </section>

      <section className="mb-14">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted">Quick stats</h2>
        <div className="max-w-md">
          <StatRow label="Dataset classes" value={Object.keys(classes).length} />
          <StatRow label="Total images" value={(data?.dataset.total ?? 0).toLocaleString()} />
          <StatRow
            label="Target class"
            value={data?.dataset.target_class ?? '—'}
            hint={`${(data?.dataset.target_count ?? 0).toLocaleString()} images`}
          />
          <StatRow
            label="Generated images"
            value={data?.gallery?.length ? `${data.gallery.length}+` : '0'}
            hint="Shown in gallery (may be capped)"
          />
          <StatRow
            label="Best accuracy"
            value={
              data?.comparison
                ? `${(Math.max(data.comparison.baseline.accuracy, data.comparison.augmented.accuracy) * 100).toFixed(1)}%`
                : '—'
            }
          />
        </div>
      </section>

      <section className="mb-14">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted">
          Dataset distribution
        </h2>
        <ClassCountChart classes={classes} />
        {error ? <p className="mt-3 text-sm text-primary">{error}</p> : null}
      </section>

      <section className="mb-8">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted">
          Run experiments
        </h2>
        <p className="mb-4 max-w-2xl text-sm text-muted">
          Jobs run in the background on the API. Demo buttons use short epochs and a
          per-class cap so you can verify the pipeline quickly. For paper-quality runs,
          use the CLI with full epoch counts from <span className="font-mono">.env</span>.
        </p>
        <div className="flex flex-wrap gap-3">
          <Button disabled={busy !== null} onClick={() => void run('dcgan')}>
            {busy === 'dcgan' ? 'Starting…' : 'Train DCGAN'}
          </Button>
          <Button disabled={busy !== null} onClick={() => void run('generate')}>
            {busy === 'generate' ? 'Starting…' : 'Generate'}
          </Button>
          <Button disabled={busy !== null} onClick={() => void run('classifier')}>
            {busy === 'classifier' ? 'Starting…' : 'Train Classifier'}
          </Button>
          <Button variant="ghost" disabled={busy !== null} onClick={() => void refresh()}>
            Refresh
          </Button>
        </div>
        {note ? <p className="mt-4 text-sm text-muted">{note}</p> : null}
        {data?.jobs ? (
          <div className="mt-6 max-w-lg space-y-2 text-sm">
            {Object.entries(data.jobs).map(([name, job]) => (
              <div key={name} className="flex justify-between border-b border-border py-2">
                <span className="text-muted">{name}</span>
                <span className="font-mono text-text">
                  {job.status}
                  {job.message ? ` — ${job.message}` : ''}
                </span>
              </div>
            ))}
          </div>
        ) : null}
      </section>
    </div>
  )
}
