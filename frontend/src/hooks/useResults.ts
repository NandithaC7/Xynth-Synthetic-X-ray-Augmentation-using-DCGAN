import { useCallback, useEffect, useState } from 'react'
import { api, type ResultsPayload } from '../services/api'

/** Fetch aggregated results and optionally poll while jobs are running. */
export function useResults(pollMs = 4000) {
  const [data, setData] = useState<ResultsPayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try {
      const payload = await api.results()
      setData(payload)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load results')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  useEffect(() => {
    const running = data
      ? Object.values(data.jobs ?? {}).some((j) => j.status === 'running' || j.status === 'queued')
      : false
    if (!running) return
    const id = window.setInterval(() => {
      void refresh()
    }, pollMs)
    return () => window.clearInterval(id)
  }, [data, pollMs, refresh])

  return { data, error, loading, refresh }
}
