import { NavLink, useNavigate, useLocation } from 'react-router-dom'
import {
    LayoutDashboard,
    Database,
    Cpu,
    Box,
    Sparkles,
    LogOut,
    ChevronLeft,
    ChevronRight,
    Settings,
} from 'lucide-react'
import { useState } from 'react'
import { useAuthStore } from '../store/authStore'
import TrainingMonitor from './TrainingMonitor'
import './Layout.css'
const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/datasets', icon: Database, label: 'Datasets' },
    { path: '/training', icon: Cpu, label: 'Training' },
    { path: '/models', icon: Box, label: 'Models' },
    { path: '/predictions', icon: Sparkles, label: 'Predictions' },
    { path: '/settings', icon: Settings, label: 'Settings' },
]

export default function Layout({ children }) {
    const [collapsed, setCollapsed] = useState(false)
    const { user, logout } = useAuthStore()
    const navigate = useNavigate()
    const location = useLocation()

    const handleLogout = () => {
        logout()
        navigate('/login')
    }

    // Determine page-specific class
    const getPageClass = () => {
        const path = location.pathname
        if (path === '/datasets') return 'page-datasets'
        return ''
    }

    return (
        <div className={`layout ${collapsed ? 'layout-collapsed' : ''}`}>
            {/* Sidebar */}
            <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
                <div className="sidebar-header">
                    <div className="logo">
                        <Sparkles className="logo-icon" />
                        {!collapsed && <span className="logo-text">InferX-ML</span>}
                    </div>
                    <button
                        className="collapse-btn"
                        onClick={() => setCollapsed(!collapsed)}
                    >
                        {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
                    </button>
                </div>

                <nav className="sidebar-nav">
                    {navItems.map((item) => (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            className={({ isActive }) =>
                                `nav-item ${isActive ? 'active' : ''}`
                            }
                            end={item.path === '/'}
                        >
                            <item.icon size={20} />
                            {!collapsed && <span>{item.label}</span>}
                        </NavLink>
                    ))}
                </nav>

                <div className="sidebar-footer">
                    {!collapsed && user && (
                        <div className="user-info">
                            <div className="avatar">
                                {user.username?.[0]?.toUpperCase() || 'U'}
                            </div>
                            <div className="user-details">
                                <span className="username">{user.username}</span>
                            </div>
                        </div>
                    )}
                    <button className="logout-btn" onClick={handleLogout}>
                        <LogOut size={20} />
                        {!collapsed && <span>Logout</span>}
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main className={`main-content ${getPageClass()}`}>
                {children}
            </main>

            {/* Global Background Monitors */}
            <TrainingMonitor />
        </div>
    )
}
