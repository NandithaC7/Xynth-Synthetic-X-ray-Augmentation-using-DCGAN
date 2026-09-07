import { MetricsBarChart } from '../components/Charts'
import { ConfusionMatrix } from '../components/ConfusionMatrix'
import { PageHeader } from '../components/PageHeader'
import { StatRow } from '../components/StatRow'
import { staticUrl } from '../services/api'
import { useResults } from '../hooks/useResults'

function pct(n: number | undefined) {
  if (n === undefined || Number.isNaN(n)) return '—'
  return `${(n * 100).toFixed(2)}%`
}

/** Classifier comparison dashboard. */
export function ResultsPage() {
  const { data, loading } = useResults()

  if (loading) return <p className="text-sm text-muted">Loading…</p>

  const comparison = data?.comparison
  const baseline = data?.baseline
  const augmented = data?.augmented

  return (
    <div>
      <PageHeader
        title="Classifier results"
        subtitle="Identical train/test protocol: only the training set changes between baseline (real only) and augmented (real + synthetic target class)."
      />

      {!comparison ? (
        <p className="text-sm text-muted">
          No comparison metrics yet. Train the classifier from Home after generating an
          augmented dataset.
        </p>
      ) : (
        <>
          <section className="mb-14 grid gap-10 md:grid-cols-2">
            <div>
              <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-muted">
                Baseline
              </h2>
              <StatRow label="Accuracy" value={pct(comparison.baseline.accuracy)} />
              <StatRow label="Precision" value={pct(comparison.baseline.precision)} />
              <StatRow label="Recall" value={pct(comparison.baseline.recall)} />
              <StatRow label="F1" value={pct(comparison.baseline.f1)} />
            </div>
            <div>
              <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-muted">
                Augmented
              </h2>
              <StatRow label="Accuracy" value={pct(comparison.augmented.accuracy)} />
              <StatRow label="Precision" value={pct(comparison.augmented.precision)} />
              <StatRow label="Recall" value={pct(comparison.augmented.recall)} />
              <StatRow label="F1" value={pct(comparison.augmented.f1)} />
            </div>
          </section>

          <section className="mb-14 max-w-md">
            <StatRow
              label="Δ Accuracy"
              value={`${comparison.delta_accuracy >= 0 ? '+' : ''}${(comparison.delta_accuracy * 100).toFixed(2)} pp`}
              hint="Augmented − baseline"
            />
          </section>

          <section className="mb-14">
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted">
              Metric comparison
            </h2>
            <MetricsBarChart baseline={comparison.baseline} augmented={comparison.augmented} />
          </section>

          <section className="mb-14 grid gap-10 lg:grid-cols-2">
            <div>
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted">
                Baseline confusion matrix
              </h2>
              {baseline?.confusion_matrix ? (
                <ConfusionMatrix
                  matrix={baseline.confusion_matrix}
                  classNames={baseline.class_names}
                />
              ) : null}
              {data?.plots?.confusion_baseline ? (
                <img
                  src={staticUrl(data.plots.confusion_baseline)}
                  alt="Baseline confusion matrix plot"
                  className="mt-4 max-w-full border border-border"
                />
              ) : null}
            </div>
            <div>
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted">
                Augmented confusion matrix
              </h2>
              {augmented?.confusion_matrix ? (
                <ConfusionMatrix
                  matrix={augmented.confusion_matrix}
                  classNames={augmented.class_names}
                />
              ) : null}
              {data?.plots?.confusion_augmented ? (
                <img
                  src={staticUrl(data.plots.confusion_augmented)}
                  alt="Augmented confusion matrix plot"
                  className="mt-4 max-w-full border border-border"
                />
              ) : null}
            </div>
          </section>
        </>
      )}
    </div>
  )
}
