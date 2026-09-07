/** API client for the Xynth FastAPI backend. */

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Request failed: ${res.status}`)
  }
  return res.json() as Promise<T>
}

export type DatasetStats = {
  dataset_path: string
  exists: boolean
  classes: Record<string, number>
  total: number
  target_class: string
  target_count: number
}

export type MetricBlock = {
  accuracy: number
  precision: number
  recall: number
  f1: number
}

export type ResultsPayload = {
  dataset: DatasetStats
  comparison: {
    baseline: MetricBlock
    augmented: MetricBlock
    delta_accuracy: number
    class_names: string[]
    epochs: number
    device: string
  } | null
  baseline: (MetricBlock & {
    confusion_matrix: number[][]
    class_names: string[]
    confusion_matrix_plot?: string
  }) | null
  augmented: (MetricBlock & {
    confusion_matrix: number[][]
    class_names: string[]
    confusion_matrix_plot?: string
  }) | null
  dcgan: Record<string, unknown> | null
  losses: { g_losses: number[]; d_losses: number[]; total_steps: number } | null
  generation: Record<string, unknown> | null
  augmented_dataset: Record<string, unknown> | null
  checkpoint: {
    path: string
    epoch: number
    target_class: string
    latent_dim: number
    modified: string
  } | null
  gallery: string[]
  grids: string[]
  plots: Record<string, string>
  jobs: Record<string, { status: string; message: string; updated_at: string | null }>
  config: {
    target_class: string
    image_size: number
    latent_dim: number
    batch_size: number
    epochs: number
    classifier_epochs: number
    num_generated: number
  }
}

export const api = {
  health: () => request<{ status: string }>('/health'),
  datasetStats: () => request<DatasetStats>('/dataset/stats'),
  results: () => request<ResultsPayload>('/results'),
  trainDcgan: (body: { epochs?: number; batch_size?: number; resume?: boolean } = {}) =>
    request<{ status: string; message: string }>('/train/dcgan', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  generate: (body: { num_images?: number; build_augmented?: boolean } = {}) =>
    request<{ status: string; message: string }>('/generate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  trainClassifier: (body: { epochs?: number; max_per_class?: number } = {}) =>
    request<{ status: string; message: string }>('/train/classifier', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  job: (name: string) =>
    request<{ name: string; status: string; message: string }>(`/jobs/${name}`),
}

/** Resolve a backend static path for <img src>. */
export function staticUrl(path: string): string {
  if (path.startsWith('http')) return path
  return path
}
