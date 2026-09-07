import { PageHeader } from '../components/PageHeader'
import { staticUrl } from '../services/api'
import { useResults } from '../hooks/useResults'

/** Generated images gallery with before/after style grids. */
export function GeneratedPage() {
  const { data, loading } = useResults()

  if (loading) return <p className="text-sm text-muted">Loading…</p>

  const gallery = data?.gallery ?? []
  const collage = data?.grids?.find((g) => g.includes('preview_collage'))
  const preview = data?.grids?.find((g) => g.includes('preview_4x4'))

  return (
    <div>
      <PageHeader
        title="Generated images"
        subtitle="Synthetic chest X-rays sampled from the trained generator. Only the target class is augmented; the original dataset is never modified."
      />

      <section className="mb-14">
        <div className="mb-8 border border-border bg-[#E8F0FF] px-6 py-10 text-center">
          <h2 className="text-xl font-semibold text-dark">Our Gallery</h2>
        </div>

        {gallery.length === 0 ? (
          <p className="text-sm text-muted">
            No samples yet. Generate images from the Home page after DCGAN training.
          </p>
        ) : (
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            {gallery.map((src) => (
              <img
                key={src}
                src={staticUrl(src)}
                alt="Synthetic X-ray"
                className="aspect-square w-full border border-border object-cover transition-colors hover:border-primary"
              />
            ))}
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted">
          Before / after grids
        </h2>
        <p className="mb-6 max-w-2xl text-sm text-muted">
          Preview collages summarize a batch of generator outputs used when building the
          augmented training set for the minority class.
        </p>
        <div className="grid gap-6 md:grid-cols-2">
          {preview ? (
            <figure>
              <img
                src={staticUrl(preview)}
                alt="4x4 preview"
                className="w-full border border-border transition-colors hover:border-primary"
              />
              <figcaption className="mt-2 text-xs text-muted">4×4 preview</figcaption>
            </figure>
          ) : null}
          {collage ? (
            <figure>
              <img
                src={staticUrl(collage)}
                alt="Preview collage"
                className="w-full border border-border transition-colors hover:border-primary"
              />
              <figcaption className="mt-2 text-xs text-muted">Preview collage</figcaption>
            </figure>
          ) : null}
          {!preview && !collage ? (
            <p className="text-sm text-muted">Grids will appear after generation.</p>
          ) : null}
        </div>
      </section>
    </div>
  )
}
