import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Home' },
  { to: '/dcgan', label: 'DCGAN' },
  { to: '/generated', label: 'Generated' },
  { to: '/results', label: 'Results' },
]

/** Top navigation — Linear / GitHub-style minimal bar. */
export function NavBar() {
  return (
    <header className="border-b border-border">
      <div className="mx-auto flex max-w-5xl items-baseline justify-between gap-8 px-6 py-6">
        <div>
          <NavLink to="/" className="text-xl font-semibold tracking-tight text-dark">
            Xynth
          </NavLink>
          <p className="mt-1 text-sm text-muted">
            Synthetic X-ray Augmentation using DCGAN
          </p>
        </div>
        <nav className="flex flex-wrap gap-1 text-sm">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) =>
                [
                  'border border-transparent px-3 py-1.5 transition-colors',
                  isActive
                    ? 'border-border text-primary'
                    : 'text-muted hover:border-border hover:text-text',
                ].join(' ')
              }
            >
              [ {link.label} ]
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
