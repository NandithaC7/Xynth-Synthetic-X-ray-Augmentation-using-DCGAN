import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { NavBar } from './components/NavBar'
import { DcganPage } from './pages/DcganPage'
import { GeneratedPage } from './pages/GeneratedPage'
import { HomePage } from './pages/HomePage'
import { ResultsPage } from './pages/ResultsPage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-bg text-text">
        <NavBar />
        <main className="mx-auto max-w-5xl px-6 py-12">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/dcgan" element={<DcganPage />} />
            <Route path="/generated" element={<GeneratedPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <footer className="mx-auto max-w-5xl border-t border-border px-6 py-8 text-xs text-muted">
          Xynth · DCGAN synthetic augmentation for COVID-19 chest X-ray classification
        </footer>
      </div>
    </BrowserRouter>
  )
}
