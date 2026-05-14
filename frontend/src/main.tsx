import { useState } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './App.css'
import App from './App.tsx'
import { Login, getStoredUser, clearStoredUser } from './components/Login'

function Root() {
  const [user, setUser] = useState<string | null>(() => getStoredUser())
  if (!user) {
    return <Login onAuthenticated={setUser} />
  }
  const handleLogout = () => {
    clearStoredUser()
    setUser(null)
  }
  return <App username={user} onLogout={handleLogout} />
}

createRoot(document.getElementById('root')!).render(<Root />)
