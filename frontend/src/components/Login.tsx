import { useState, type FormEvent } from 'react'

const CREDENTIALS: Record<string, string> = {
  Akash: 'a1234',
  demo: 'fs1234',
  client: 'c1234',
}

const STORAGE_KEY = 'dhl_auth_user_v1'

export function getStoredUser(): string | null {
  if (typeof window === 'undefined') return null
  try {
    return window.sessionStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

export function clearStoredUser(): void {
  try {
    window.sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    // ignore
  }
}

type Props = {
  onAuthenticated: (username: string) => void
}

export function Login({ onAuthenticated }: Props) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError('')
    setSubmitting(true)
    const expected = CREDENTIALS[username.trim()]
    if (expected && expected === password) {
      try {
        window.sessionStorage.setItem(STORAGE_KEY, username.trim())
      } catch {
        // ignore
      }
      onAuthenticated(username.trim())
    } else {
      setError('Invalid username or password.')
    }
    setSubmitting(false)
  }

  return (
    <div className="login-shell">
      <div className="login-card">
        <div className="login-logos">
          <div className="brand-lockup" aria-label="DHL | Findability Sciences">
            <img className="brand-lockup__logo brand-lockup__logo--dhl" src="/logos/DHL.png" alt="DHL" />
            <span className="brand-lockup__x" aria-hidden>|</span>
            <img className="brand-lockup__logo brand-lockup__logo--fs" src="/logos/FSSML.png" alt="Findability Sciences" />
          </div>
        </div>
        <h1 className="login-title">Voice AI POC</h1>
        <p className="login-sub">Sign in to continue</p>
        <form onSubmit={handleSubmit} className="login-form">
          <label className="login-field">
            <span>Username</span>
            <input
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="username"
              required
            />
          </label>
          <label className="login-field">
            <span>Password</span>
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="password"
              required
            />
          </label>
          {error && <div className="login-error" role="alert">{error}</div>}
          <button type="submit" className="login-submit" disabled={submitting}>
            {submitting ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
        <div className="login-footer">
          <img src="/logos/FS.png" alt="FS" className="login-footer-mark" />
          <span>Internal demo. Authorised users only.</span>
        </div>
      </div>
    </div>
  )
}
