import { Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Providers from './pages/Providers'
import Keys from './pages/Keys'
import Usage from './pages/Usage'
import Setup from './pages/Setup'

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/providers', label: 'Providers' },
  { to: '/keys', label: 'Keys' },
  { to: '/usage', label: 'Usage' },
  { to: '/setup', label: 'Setup' },
]

function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <nav className="border-b border-gray-800 bg-gray-900">
        <div className="max-w-6xl mx-auto px-4 flex items-center h-14 gap-8">
          <span className="text-lg font-bold text-blue-400 tracking-wide">Louter</span>
          <div className="flex gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `px-3 py-2 rounded text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-gray-800 text-white'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-4 py-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/providers" element={<Providers />} />
          <Route path="/keys" element={<Keys />} />
          <Route path="/usage" element={<Usage />} />
          <Route path="/setup" element={<Setup />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
