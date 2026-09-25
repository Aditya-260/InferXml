import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { useToast } from '../components/ui/use-toast'
import api from '../services/api'
import {
    User, Lock, CreditCard, Crown, Shield, ChevronRight,
    Eye, EyeOff, Check, X, Zap, Sparkles, LogOut, Mail, Bell,
    Download, Calendar, History, Wallet, Star, Pencil
} from 'lucide-react'
import './SettingsPage.css'

const API_BASE = import.meta.env.VITE_API_URL || '/api'


/* ─── Tab definitions ─── */
const TABS = [
    { id: 'profile',       icon: User,       label: 'Profile' },
    { id: 'security',      icon: Lock,       label: 'Security' },
    { id: 'notifications', icon: Bell,       label: 'Notifications' },
    { id: 'billing',       icon: CreditCard, label: 'Billing & Plan' },
]

/* ─── Default notification prefs ─── */
const DEFAULT_NOTIFS = {
    trainingComplete:  true,
    trainingFailed:    true,
    datasetUploaded:   false,
    rateLimitWarning:  true,
    weeklyReport:      false,
    securityAlerts:    true,
    productUpdates:    false,
}

const NOTIF_ITEMS = [
    { key: 'trainingComplete', label: 'Training completed',       desc: 'Get notified when a model finishes training.' },
    { key: 'trainingFailed',   label: 'Training failed',          desc: 'Alert me if a training job encounters an error.' },
    { key: 'datasetUploaded',  label: 'Dataset uploaded',         desc: 'Notify when a dataset upload succeeds.' },
    { key: 'rateLimitWarning', label: 'API rate-limit warning',   desc: 'Warn me when I\'m approaching the request limit.' },
    { key: 'weeklyReport',     label: 'Weekly usage report',      desc: 'Receive a weekly summary of your usage stats.' },
    { key: 'securityAlerts',   label: 'Security alerts',          desc: 'Get alerts for unusual sign-ins or password changes.' },
    { key: 'productUpdates',   label: 'Product updates',          desc: 'Hear about new features and platform improvements.' },
]

/* camelCase (frontend) ↔ snake_case (backend) mapping */
const KEY_TO_API = {
    trainingComplete: 'training_completed',
    trainingFailed:   'training_failed',
    datasetUploaded:  'dataset_uploaded',
    rateLimitWarning: 'api_rate_limit',
    weeklyReport:     'weekly_report',
    securityAlerts:   'security_alerts',
    productUpdates:   'product_updates',
}
const API_TO_KEY = Object.fromEntries(Object.entries(KEY_TO_API).map(([k, v]) => [v, k]))


export default function SettingsPage() {
    const [activeTab, setActiveTab] = useState('profile')
    const { user, logout, checkAuth } = useAuthStore()
    const { toast } = useToast()
    const navigate = useNavigate()

    /* ── Username edit state ── */
    const [editingUsername, setEditingUsername] = useState(false)
    const [newUsername, setNewUsername] = useState(user?.username || '')
    const [savingUsername, setSavingUsername] = useState(false)

    /* ── Password change state ── */
    const [currentPwd, setCurrentPwd] = useState('')
    const [newPwd, setNewPwd] = useState('')
    const [confirmPwd, setConfirmPwd] = useState('')
    const [showCurrent, setShowCurrent] = useState(false)
    const [showNew, setShowNew] = useState(false)
    const [changingPwd, setChangingPwd] = useState(false)

    /* ── Notification prefs (server-synced) ── */
    const [notifs, setNotifs] = useState(DEFAULT_NOTIFS)
    const [notifsLoading, setNotifsLoading] = useState(false)

    /* ── Billing History state ── */
    const [history, setHistory] = useState([])
    const [billingSummary, setBillingSummary] = useState(null)
    const [historyLoading, setHistoryLoading] = useState(false)
    const [historyPage, setHistoryPage] = useState(1)
    const [historyTotalPages, setHistoryTotalPages] = useState(1)
    const [billingConfig, setBillingConfig] = useState(null)

    /* Fetch prefs from backend when notifications tab is active */
    useEffect(() => {
        if (activeTab !== 'notifications') return
        let cancelled = false
        const fetchPrefs = async () => {
            setNotifsLoading(true)
            try {
                const res = await api.get('/notifications/preferences')
                if (!cancelled && res.data) {
                    const mapped = {}
                    for (const [apiKey, val] of Object.entries(res.data)) {
                        const uiKey = API_TO_KEY[apiKey]
                        if (uiKey) mapped[uiKey] = val
                    }
                    setNotifs(prev => ({ ...prev, ...mapped }))
                }
            } catch (err) {
                console.warn('Failed to load notification prefs:', err)
            } finally {
                if (!cancelled) setNotifsLoading(false)
            }
        }
        fetchPrefs()
        return () => { cancelled = true }
    }, [activeTab])

    const toggleNotif = async (key) => {
        const newVal = !notifs[key]
        /* Optimistic UI update */
        setNotifs(prev => ({ ...prev, [key]: newVal }))
        toast({
            title: newVal ? 'Enabled' : 'Disabled',
            description: `${NOTIF_ITEMS.find(n => n.key === key)?.label || key} ${newVal ? 'enabled' : 'disabled'}.`
        })
        try {
            await api.put('/notifications/preferences', { [KEY_TO_API[key]]: newVal })
        } catch (err) {
            /* Revert on failure */
            setNotifs(prev => ({ ...prev, [key]: !newVal }))
            toast({ title: 'Error', description: 'Failed to save preference.', variant: 'destructive' })
        }
    }

    /* ── Razorpay / Billing state ── */
    const [upgrading, setUpgrading] = useState(false)
    const [scriptLoaded, setScriptLoaded] = useState(false)

    /* Load Razorpay SDK lazily when billing tab opens */
    React.useEffect(() => {
        if (activeTab !== 'billing') return
        if (document.querySelector('script[src*="razorpay"]')) {
            setScriptLoaded(true)
            return
        }
        const s = document.createElement('script')
        s.src = 'https://checkout.razorpay.com/v1/checkout.js'
        s.async = true
        s.onload = () => setScriptLoaded(true)
        document.body.appendChild(s)
    }, [activeTab])

    /* Fetch Billing History */
    const fetchHistory = async (page = 1) => {
        if (activeTab !== 'billing') return
        setHistoryLoading(true)
        try {
            const res = await api.get(`/billing/history?page=${page}`)
            if (res.data) {
                setHistory(res.data.payments)
                setBillingSummary(res.data.summary)
                setHistoryTotalPages(res.data.pagination.pages)
                setHistoryPage(res.data.pagination.page)
            }
        } catch (err) {
            console.warn('Failed to load billing history:', err)
        } finally {
            setHistoryLoading(false)
        }
    }

    useEffect(() => {
        if (activeTab === 'billing') {
            fetchHistory(1)
            fetch(`${API_BASE}/billing/config`)
                .then(res => res.json())
                .then(data => setBillingConfig(data))
                .catch(err => console.error('Failed to load config', err))
        }
    }, [activeTab])

    /* ─────────── handlers ─────────── */

    const handleChangePassword = async (e) => {
        e.preventDefault()
        if (newPwd !== confirmPwd) {
            toast({ title: 'Mismatch', description: 'New passwords do not match.', variant: 'destructive' })
            return
        }
        setChangingPwd(true)
        try {
            await api.post('/auth/change-password', {
                current_password: currentPwd,
                new_password: newPwd
            })
            toast({ title: 'Success', description: 'Password changed successfully.' })
            setCurrentPwd(''); setNewPwd(''); setConfirmPwd('')
        } catch (err) {
            toast({
                title: 'Error',
                description: err.response?.data?.error || 'Failed to change password.',
                variant: 'destructive'
            })
        } finally {
            setChangingPwd(false)
        }
    }

    const handleUpgrade = async (planName) => {
        if (!scriptLoaded) {
            toast({ title: 'Error', description: 'Payment gateway is loading…', variant: 'destructive' })
            return
        }
        setUpgrading(planName)
        try {
            const authData = JSON.parse(localStorage.getItem('inferx-auth') || '{}')
            const token = authData?.state?.token
            if (!token) throw new Error('Not authenticated. Please log in again.')

            const orderRes = await fetch(`${API_BASE}/billing/create-order`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ plan: planName })
            })
            const orderData = await orderRes.json()
            if (!orderRes.ok) throw new Error(orderData.error || 'Failed to create order')

            const isAdvance = planName === 'advance'

            const options = {
                key: orderData.key_id,
                amount: orderData.amount,
                currency: orderData.currency,
                name: 'InferX-ML',
                description: isAdvance ? 'Upgrade to Advance — 1 Year' : 'Upgrade to Pro — 30 days',
                order_id: orderData.order_id,
                handler: async (response) => {
                    try {
                        const verifyRes = await fetch(`${API_BASE}/billing/verify-payment`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
                            body: JSON.stringify({
                                razorpay_payment_id: response.razorpay_payment_id,
                                razorpay_order_id: response.razorpay_order_id,
                                razorpay_signature: response.razorpay_signature
                            })
                        })
                        const verifyData = await verifyRes.json()
                        if (verifyRes.ok) {
                            toast({ title: '🎉 Success!', description: `You have been upgraded to ${isAdvance ? 'Advance' : 'Pro'}!` })
                            checkAuth()
                            fetchHistory(1)
                        } else throw new Error(verifyData.error || 'Payment verification failed')
                    } catch (err) {
                        toast({ title: 'Verification Failed', description: err.message, variant: 'destructive' })
                    }
                },
                prefill: { name: user?.username || '', email: user?.email || '' },
                theme: { color: isAdvance ? '#8b5cf6' : '#10b981' },
                modal: {
                    ondismiss: function () {
                        setUpgrading(false)
                    }
                }
            }
            const rzp = new window.Razorpay(options)
            rzp.on('payment.failed', (r) => toast({ title: 'Payment Failed', description: r.error.description, variant: 'destructive' }))
            rzp.open()
        } catch (err) {
            toast({ title: 'Error', description: err.message, variant: 'destructive' })
        } finally {
            setUpgrading(false)
        }
    }

    const handleDownloadInvoice = async (paymentId) => {
        try {
            const authData = JSON.parse(localStorage.getItem('inferx-auth') || '{}')
            const token = authData?.state?.token
            
            const response = await fetch(`${API_BASE}/billing/invoice/${paymentId}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            })
            
            if (!response.ok) throw new Error('Failed to download invoice')
            
            const blob = await response.blob()
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `Invoice_${paymentId}.pdf`
            document.body.appendChild(a)
            a.click()
            window.URL.revokeObjectURL(url)
            document.body.removeChild(a)
        } catch (err) {
            toast({ title: 'Download Failed', description: err.message, variant: 'destructive' })
        }
    }

    const handleSaveUsername = async () => {
        const trimmed = newUsername.trim()
        if (!trimmed || trimmed === user?.username) {
            setEditingUsername(false)
            setNewUsername(user?.username || '')
            return
        }
        setSavingUsername(true)
        try {
            const res = await api.put('/auth/update-username', { username: trimmed })
            // Update the user in zustand store
            useAuthStore.setState({ user: res.data.user })
            toast({ title: 'Success', description: 'Username updated.' })
            setEditingUsername(false)
        } catch (err) {
            toast({
                title: 'Error',
                description: err.response?.data?.error || 'Failed to update username.',
                variant: 'destructive'
            })
        } finally {
            setSavingUsername(false)
        }
    }

    const handleLogout = () => { logout(); navigate('/login') }

    const isFree = !user?.plan_type || user.plan_type === 'free'
    const isPro  = user?.plan_type === 'pro'
    const isAdvance = user?.plan_type === 'advance'

    /* ─────────── Profile Tab ─────────── */
    const renderProfile = () => (
        <div className="st-section">
            <h2 className="st-section-title">Profile Information</h2>
            <p className="st-section-desc">Your account details are shown below.</p>

            <div className="st-profile-card">
                <div className="st-avatar-big">
                    {user?.username?.[0]?.toUpperCase() || 'U'}
                </div>
                <div className="st-profile-fields">
                    <div className="st-field">
                        <label><User size={14} /> Username</label>
                        {editingUsername ? (
                            <div className="st-field-edit">
                                <input
                                    className="st-field-input"
                                    value={newUsername}
                                    onChange={e => setNewUsername(e.target.value)}
                                    placeholder="New username"
                                    maxLength={32}
                                    autoFocus
                                    onKeyDown={e => {
                                        if (e.key === 'Enter') handleSaveUsername()
                                        if (e.key === 'Escape') { setEditingUsername(false); setNewUsername(user?.username || '') }
                                    }}
                                />
                                <button
                                    className="st-btn-icon st-btn-save"
                                    onClick={handleSaveUsername}
                                    disabled={savingUsername}
                                    title="Save"
                                >
                                    {savingUsername ? <span className="st-spinner" /> : <Check size={16} />}
                                </button>
                                <button
                                    className="st-btn-icon st-btn-cancel"
                                    onClick={() => { setEditingUsername(false); setNewUsername(user?.username || '') }}
                                    title="Cancel"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        ) : (
                            <div className="st-field-value st-field-editable">
                                {user?.username || '—'}
                                <button
                                    className="st-btn-icon st-btn-edit"
                                    onClick={() => { setNewUsername(user?.username || ''); setEditingUsername(true) }}
                                    title="Edit username"
                                >
                                    <Pencil size={14} />
                                </button>
                            </div>
                        )}
                    </div>
                    <div className="st-field">
                        <label><Mail size={14} /> Email</label>
                        <div className="st-field-value">{user?.email || '—'}</div>
                    </div>
                    <div className="st-field">
                        <label><Crown size={14} /> Plan</label>
                        <div className="st-field-value">
                            <span className={`st-plan-tag ${isAdvance ? 'st-tag-advance' : isPro ? 'st-tag-pro' : ''}`}>
                                {isAdvance ? '⭐ Advance' : isPro ? '⚡ Pro' : 'Free'}
                            </span>
                        </div>
                    </div>
                    <div className="st-field">
                        <label><Shield size={14} /> Account status</label>
                        <div className="st-field-value">
                            <span className="st-status-active">● Active</span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="st-danger-zone">
                <h3>Danger Zone</h3>
                <button className="st-btn st-btn-danger" onClick={handleLogout}>
                    <LogOut size={16} /> Log out of this device
                </button>
            </div>
        </div>
    )

    /* ─────────── Security Tab ─────────── */
    const renderSecurity = () => (
        <div className="st-section">
            <h2 className="st-section-title">Change Password</h2>
            <p className="st-section-desc">
                Use at least 8 characters with a mix of letters and numbers.
            </p>

            <form className="st-form" onSubmit={handleChangePassword}>
                <div className="st-input-group">
                    <label>Current Password</label>
                    <div className="st-input-wrap">
                        <input
                            type={showCurrent ? 'text' : 'password'}
                            value={currentPwd}
                            onChange={e => setCurrentPwd(e.target.value)}
                            placeholder="Enter current password"
                            required
                        />
                        <button type="button" className="st-eye" onClick={() => setShowCurrent(!showCurrent)}>
                            {showCurrent ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                    </div>
                </div>

                <div className="st-input-group">
                    <label>New Password</label>
                    <div className="st-input-wrap">
                        <input
                            type={showNew ? 'text' : 'password'}
                            value={newPwd}
                            onChange={e => setNewPwd(e.target.value)}
                            placeholder="Enter new password"
                            required
                        />
                        <button type="button" className="st-eye" onClick={() => setShowNew(!showNew)}>
                            {showNew ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                    </div>
                </div>

                <div className="st-input-group">
                    <label>Confirm New Password</label>
                    <div className="st-input-wrap">
                        <input
                            type="password"
                            value={confirmPwd}
                            onChange={e => setConfirmPwd(e.target.value)}
                            placeholder="Re-enter new password"
                            required
                        />
                    </div>
                </div>

                {/* Password strength hints */}
                <div className="st-pwd-hints">
                    <span className={newPwd.length >= 8 ? 'st-hint-ok' : ''}>
                        {newPwd.length >= 8 ? <Check size={12} /> : <X size={12} />} 8+ characters
                    </span>
                    <span className={/[A-Za-z]/.test(newPwd) ? 'st-hint-ok' : ''}>
                        {/[A-Za-z]/.test(newPwd) ? <Check size={12} /> : <X size={12} />} Letters
                    </span>
                    <span className={/\d/.test(newPwd) ? 'st-hint-ok' : ''}>
                        {/\d/.test(newPwd) ? <Check size={12} /> : <X size={12} />} Numbers
                    </span>
                    <span className={newPwd && newPwd === confirmPwd ? 'st-hint-ok' : ''}>
                        {newPwd && newPwd === confirmPwd ? <Check size={12} /> : <X size={12} />} Match
                    </span>
                </div>

                <button
                    type="submit"
                    className="st-btn st-btn-primary"
                    disabled={changingPwd || !currentPwd || !newPwd || !confirmPwd}
                >
                    {changingPwd ? <><span className="st-spinner" /> Updating…</> : 'Update Password'}
                </button>
            </form>
        </div>
    )

    /* ─────────── Billing Tab ─────────── */
    const renderBilling = () => (
        <div className="st-section">
            <h2 className="st-section-title">Billing & Plan</h2>
            <p className="st-section-desc">
                You are on the <strong>{isAdvance ? 'Advance' : isPro ? 'Pro' : 'Free'}</strong> plan.
                {isFree && ' Upgrade to unlock all algorithms and unlimited training.'}
            </p>

            <div className="st-plan-summary">
                {isFree && (
                    <div className="st-summary-card">
                        <h3>Free Plan</h3>
                        <p>You are using the free tier.</p>
                        <button className="st-btn st-btn-primary" onClick={() => navigate('/pricing')}>View Plans & Upgrade</button>
                    </div>
                )}
                {isPro && (
                    <div className="st-summary-card st-summary-pro">
                        <h3>Pro Plan</h3>
                        <p>Active — {billingConfig ? (billingConfig.pro_plan_amount / 100) : '499'} / mo. Renews in {user?.plan_expires_at ? new Date(user.plan_expires_at).toLocaleDateString() : '—'}</p>
                        <button className="st-btn st-btn-upgrade" onClick={() => handleUpgrade('pro')} disabled={upgrading === 'pro' || !scriptLoaded}>
                            {upgrading === 'pro' ? <><span className="st-spinner"/> Processing...</> : 'Renew Pro Subscription'}
                        </button>
                        <button className="st-btn st-btn-advance-secondary" onClick={() => navigate('/pricing')}>Upgrade to Advance</button>
                    </div>
                )}
                {isAdvance && (
                    <div className="st-summary-card st-summary-advance">
                        <h3>Advance Plan</h3>
                        <p>Active — {billingConfig ? (billingConfig.advance_plan_amount / 100) : '4999'} / yr. Renews in {user?.plan_expires_at ? new Date(user.plan_expires_at).toLocaleDateString() : '—'}</p>
                        <button className="st-btn st-btn-advance" onClick={() => handleUpgrade('advance')} disabled={upgrading === 'advance' || !scriptLoaded}>
                            {upgrading === 'advance' ? <><span className="st-spinner"/> Processing...</> : 'Renew Advance Subscription'}
                        </button>
                    </div>
                )}
            </div>

            {/* Payment History Section */}
            <div className="st-history-section">
                <div className="st-history-header">
                    <h3><History size={18} /> Payment History</h3>
                    <div className="st-history-stats">
                        <div className="st-stat">
                            <label>Total Spent</label>
                            <span>{billingSummary?.total_spent_display || '₹0.00'}</span>
                        </div>
                        <div className="st-stat">
                            <label>Successful</label>
                            <span>{billingSummary?.successful_payments || 0}</span>
                        </div>
                    </div>
                </div>

                <div className="st-history-table-wrap">
                    <table className="st-history-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Order ID</th>
                                <th>Amount</th>
                                <th>Status</th>
                                <th>Invoice</th>
                            </tr>
                        </thead>
                        <tbody>
                            {historyLoading && history.length === 0 ? (
                                <tr><td colSpan="5" className="st-table-loading">Loading history…</td></tr>
                            ) : history.length === 0 ? (
                                <tr><td colSpan="5" className="st-table-empty">No transactions yet.</td></tr>
                            ) : history.map(item => (
                                <tr key={item.id}>
                                    <td>{new Date(item.created_at).toLocaleDateString()}</td>
                                    <td className="st-td-id">{item.razorpay_order_id}</td>
                                    <td>₹{(item.amount / 100).toFixed(2)}</td>
                                    <td>
                                        <span className={`st-status-tag st-status-${item.status}`}>
                                            {item.status}
                                        </span>
                                    </td>
                                    <td>
                                        {item.status === 'captured' ? (
                                            <button 
                                                className="st-btn-icon" 
                                                onClick={() => handleDownloadInvoice(item.id)}
                                                title="Download Invoice"
                                            >
                                                <Download size={16} />
                                            </button>
                                        ) : '—'}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {historyTotalPages > 1 && (
                    <div className="st-pagination">
                        <button 
                            disabled={historyPage === 1 || historyLoading}
                            onClick={() => fetchHistory(historyPage - 1)}
                        >
                            Previous
                        </button>
                        <span>Page {historyPage} of {historyTotalPages}</span>
                        <button 
                            disabled={historyPage === historyTotalPages || historyLoading}
                            onClick={() => fetchHistory(historyPage + 1)}
                        >
                            Next
                        </button>
                    </div>
                )}
            </div>
        </div>
    )

    /* ─────────── Notifications Tab ─────────── */
    const renderNotifications = () => (
        <div className="st-section">
            <h2 className="st-section-title">Notifications</h2>
            <p className="st-section-desc">
                Choose which events you want to be notified about. Preferences sync with your account.
            </p>

            <div className="st-notif-list">
                {NOTIF_ITEMS.map(item => (
                    <div className="st-notif-row" key={item.key}>
                        <div className="st-notif-info">
                            <span className="st-notif-label">{item.label}</span>
                            <span className="st-notif-desc">{item.desc}</span>
                        </div>
                        <button
                            className={`st-toggle ${notifs[item.key] ? 'st-toggle-on' : ''}`}
                            onClick={() => toggleNotif(item.key)}
                            aria-label={`Toggle ${item.label}`}
                        >
                            <span className="st-toggle-thumb" />
                        </button>
                    </div>
                ))}
            </div>
        </div>
    )

    /* ─────────── RENDER ─────────── */
    return (
        <div className="st-page">
            <div className="st-bg-glow st-glow1" />
            <div className="st-bg-glow st-glow2" />

            <header className="st-page-header">
                <h1 className="st-page-title">Settings</h1>
                <p className="st-page-sub">Manage your account, security, and subscription.</p>
            </header>

            <div className="st-layout">
                {/* Sidebar tabs */}
                <aside className="st-tabs">
                    {TABS.map(t => (
                        <button
                            key={t.id}
                            className={`st-tab ${activeTab === t.id ? 'st-tab-active' : ''}`}
                            onClick={() => setActiveTab(t.id)}
                        >
                            <t.icon size={18} />
                            <span>{t.label}</span>
                            <ChevronRight size={14} className="st-tab-arrow" />
                        </button>
                    ))}
                </aside>

                {/* Content panel */}
                <div className="st-panel">
                    {activeTab === 'profile'       && renderProfile()}
                    {activeTab === 'security'      && renderSecurity()}
                    {activeTab === 'notifications' && renderNotifications()}
                    {activeTab === 'billing'       && renderBilling()}
                </div>
            </div>
        </div>
    )
}
