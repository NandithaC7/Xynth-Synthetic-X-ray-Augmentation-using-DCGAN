import { LossChart } from '../components/Charts'
import { PageHeader } from '../components/PageHeader'
import { StatRow } from '../components/StatRow'
import { staticUrl } from '../services/api'
import { useResults } from '../hooks/useResults'

/** DCGAN page — loss curves, samples, checkpoint info. */
export function DcganPage() {
  const { data, loading } = useResults()

  if (loading) return <p className="text-sm text-muted">Loading…</p>

  const ckpt = data?.checkpoint
  const grids = data?.grids ?? []
  const lossPlot = data?.plots?.dcgan_loss

  return (
    <div>
      <PageHeader
        title="DCGAN"
        subtitle="Deep Convolutional GAN trained on the target minority class to synthesize 64×64 grayscale chest X-rays."
      />

      <section className="mb-14 max-w-md">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-muted">
          Checkpoint
        </h2>
        {ckpt ? (
          <>
            <StatRow label="Epoch" value={(ckpt.epoch ?? 0) + 1} />
            <StatRow label="Target class" value={ckpt.target_class} />
            <StatRow label="Latent dim" value={ckpt.latent_dim} />
            <StatRow label="Updated" value={new Date(ckpt.modified).toLocaleString()} />
          </>
        ) : (
          <p className="text-sm text-muted">No checkpoint yet. Train the DCGAN from Home.</p>
        )}
      </section>

      <section className="mb-14">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted">
          Training loss
        </h2>
        <LossChart
          gLosses={data?.losses?.g_losses ?? []}
          dLosses={data?.losses?.d_losses ?? []}
        />
        {lossPlot ? (
          <img
            src={staticUrl(lossPlot)}
            alt="DCGAN loss plot"
            className="mt-6 max-w-full border border-border"
          />
        ) : null}
      </section>

      <section>
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted">
          Generator samples
        </h2>
        {grids.length === 0 ? (
          <p className="text-sm text-muted">Sample grids appear after training or generation.</p>
        ) : (
          <div className="grid gap-6 sm:grid-cols-2">
            {grids.map((src) => (
              <img
                key={src}
                src={staticUrl(src)}
                alt="Generator sample grid"
                className="w-full border border-border object-contain transition-colors hover:border-primary"
              />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
