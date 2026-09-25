import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
    ArrowRight,
    KeyRound,
    Lock,
    Mail,
    ShieldCheck,
    Sparkles,
    User,
    Cpu,
    Terminal,
} from 'lucide-react'
import { useAuthStore } from '../store/authStore'
import './Login.css'

const modeCopy = {
    login: {
        title: 'Welcome back',
        subtitle: 'Sign in to continue to your workspace.',
    },
    register: {
        title: 'Create your account',
        subtitle: 'Set up secure access and verify your email before your first session.',
    },
    verify: {
        title: 'Verify your email',
        subtitle: 'Enter the one-time code we sent to finish activating your account.',
    },
    forgot: {
        title: 'Reset your password',
        subtitle: 'We will send a one-time reset code to your email.',
    },
    reset: {
        title: 'Choose a new password',
        subtitle: 'Use the reset code from your email and set a fresh password.',
    },
}

export default function Login() {
    const navigate = useNavigate()
    const {
        login,
        register,
        verifyEmail,
        resendVerification,
        forgotPassword,
        resetPassword,
    } = useAuthStore()

    const [mode, setMode] = useState('login')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [notice, setNotice] = useState('')
    const [emailLocked, setEmailLocked] = useState(false)
    const [formData, setFormData] = useState({
        email: '',
        username: '',
        password: '',
        otp: '',
        nextPassword: '',
        confirmPassword: '',
    })

    const copy = useMemo(() => modeCopy[mode], [mode])

    const switchMode = (nextMode) => {
        setMode(nextMode)
        setError('')
        setNotice('')
        if (nextMode === 'login' || nextMode === 'register' || nextMode === 'forgot') {
            setEmailLocked(false)
        }
    }

    const handleChange = (e) => {
        const { name, value } = e.target
        setFormData((prev) => ({ ...prev, [name]: value }))
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError('')
        setNotice('')
        setLoading(true)

        try {
            if (mode === 'login') {
                const result = await login(formData.email, formData.password)
                if (result.success) {
                    navigate('/')
                    return
                }
                if (result.requiresVerification) {
                    setFormData((prev) => ({ ...prev, email: result.email || prev.email, otp: '' }))
                    setEmailLocked(true)
                    setMode('verify')
                    setNotice('Your account is almost ready. Enter the verification code from your email.')
                    return
                }
                setError(result.error)
                return
            }

            if (mode === 'register') {
                const result = await register(formData.email, formData.username, formData.password)
                if (result.success) {
                    setFormData((prev) => ({
                        ...prev,
                        email: result.data.email || prev.email,
                        otp: '',
                    }))
                    setEmailLocked(true)
                    setMode('verify')
                    setNotice(result.data.message)
                    return
                }
                setError(result.error)
                return
            }

            if (mode === 'verify') {
                const result = await verifyEmail(formData.email, formData.otp)
                if (result.success) {
                    navigate('/')
                    return
                }
                setError(result.error)
                return
            }

            if (mode === 'forgot') {
                const result = await forgotPassword(formData.email)
                if (result.success) {
                    setEmailLocked(true)
                    setMode('reset')
                    setNotice(result.message)
                } else {
                    setError(result.error)
                }
                return
            }

            if (mode === 'reset') {
                if (formData.nextPassword !== formData.confirmPassword) {
                    setError('Passwords do not match')
                    return
                }
                const result = await resetPassword(formData.email, formData.otp, formData.nextPassword)
                if (result.success) {
                    setMode('login')
                    setEmailLocked(false)
                    setFormData((prev) => ({
                        ...prev,
                        password: '',
                        otp: '',
                        nextPassword: '',
                        confirmPassword: '',
                    }))
                    setNotice(result.message)
                } else {
                    setError(result.error)
                }
            }
        } finally {
            setLoading(false)
        }
    }

    const handleResendVerification = async () => {
        setError('')
        setNotice('')
        setLoading(true)
        const result = await resendVerification(formData.email)
        setLoading(false)
        if (result.success) {
            setNotice(result.message)
        } else {
            setError(result.error)
        }
    }

    return (
        <div className="login-page">
            <div className="auth-background">
                <div className="auth-glow auth-glow-left" />
                <div className="auth-glow auth-glow-right" />
            </div>

            <div className="auth-modal">
                <div className="auth-side-panel">
                    <div className="auth-brand">
                        <div className="auth-brand-icon">
                            <Sparkles size={24} />
                        </div>
                        <div>
                            <h1>InferX-ML</h1>
                            <p>Next-Gen No-Code AI Platform</p>
                        </div>
                    </div>

                    <div className="auth-side-copy">
                        <span className="auth-side-badge">Enterprise Suite v2.4</span>
                        <h2>Empower your models. Scale your insights.</h2>
                        <p>
                            An integrated, secure workspace to train, evaluate, and orchestrate production-grade machine learning pipelines instantly.
                        </p>
                    </div>

                    <div className="auth-side-points">
                        <div className="auth-point">
                            <div className="auth-point-icon">
                                <Cpu size={18} />
                            </div>
                            <div className="auth-point-content">
                                <span className="auth-point-title">High-Performance Compute</span>
                                <span className="auth-point-desc">Spin up GPU nodes on-demand with automated model auto-scaling and ultra-low latency.</span>
                            </div>
                        </div>
                        <div className="auth-point">
                            <div className="auth-point-icon">
                                <ShieldCheck size={18} />
                            </div>
                            <div className="auth-point-content">
                                <span className="auth-point-title">Unified Access Gatekeeper</span>
                                <span className="auth-point-desc">Secure sign-in with Google, GitHub, or OTP verification to keep datasets isolated.</span>
                            </div>
                        </div>
                        <div className="auth-point">
                            <div className="auth-point-icon">
                                <Terminal size={18} />
                            </div>
                            <div className="auth-point-content">
                                <span className="auth-point-title">Explainable AI Sandbox</span>
                                <span className="auth-point-desc">Interact with live prediction engines and real-time visual interpretations (SHAP/LIME).</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="auth-form-panel">
                    <div className="auth-header">
                        {(mode === 'login' || mode === 'register') && (
                            <div className="auth-mode-switch">
                                <button
                                    type="button"
                                    className={`auth-mode-tab ${mode === 'login' ? 'active' : ''}`}
                                    onClick={() => switchMode('login')}
                                >
                                    Sign In
                                </button>
                                <button
                                    type="button"
                                    className={`auth-mode-tab ${mode === 'register' ? 'active' : ''}`}
                                    onClick={() => switchMode('register')}
                                >
                                    Create Account
                                </button>
                            </div>
                        )}

                        <h2>{copy.title}</h2>
                        <p>{copy.subtitle}</p>
                    </div>

                    {notice && <div className="auth-message auth-message-info">{notice}</div>}
                    {error && <div className="auth-message auth-message-error">{error}</div>}

                    {(mode === 'login' || mode === 'register') && (
                        <>
                            <div className="oauth-section">
                                <button
                                    type="button"
                                    className="oauth-btn oauth-google"
                                    onClick={() => {
                                        window.location.href = `${import.meta.env.VITE_API_URL || '/api'}/auth/google`
                                    }}
                                >
                                    <svg width="20" height="20" viewBox="0 0 24 24">
                                        <path
                                            d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
                                            fill="#4285F4"
                                        />
                                        <path
                                            d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                                            fill="#34A853"
                                        />
                                        <path
                                            d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                                            fill="#FBBC05"
                                        />
                                        <path
                                            d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                                            fill="#EA4335"
                                        />
                                    </svg>
                                    Continue with Google
                                </button>
                                <button
                                    type="button"
                                    className="oauth-btn oauth-github"
                                    onClick={() => {
                                        window.location.href = `${import.meta.env.VITE_API_URL || '/api'}/auth/github`
                                    }}
                                >
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                        <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                                    </svg>
                                    Continue with GitHub
                                </button>
                            </div>

                            <div className="auth-divider">
                                <span>or continue with email</span>
                            </div>
                        </>
                    )}

                    <form onSubmit={handleSubmit} className="login-form">
                        <div className="form-group">
                            <label className="label">Email</label>
                            <div className="input-wrapper">
                                <Mail size={18} className="input-icon" />
                                <input
                                    type="email"
                                    name="email"
                                    value={formData.email}
                                    onChange={handleChange}
                                    className="input"
                                    placeholder="you@company.com"
                                    required
                                    disabled={emailLocked}
                                />
                            </div>
                        </div>

                        {mode === 'register' && (
                            <div className="form-group">
                                <label className="label">Username</label>
                                <div className="input-wrapper">
                                    <User size={18} className="input-icon" />
                                    <input
                                        type="text"
                                        name="username"
                                        value={formData.username}
                                        onChange={handleChange}
                                        className="input"
                                        placeholder="team_owner"
                                        required
                                    />
                                </div>
                            </div>
                        )}

                        {(mode === 'login' || mode === 'register') && (
                            <div className="form-group">
                                <div className="label-row">
                                    <label className="label">Password</label>
                                    {mode === 'login' && (
                                        <button
                                            type="button"
                                            className="auth-link-button"
                                            onClick={() => switchMode('forgot')}
                                        >
                                            Forgot password?
                                        </button>
                                    )}
                                </div>
                                <div className="input-wrapper">
                                    <Lock size={18} className="input-icon" />
                                    <input
                                        type="password"
                                        name="password"
                                        value={formData.password}
                                        onChange={handleChange}
                                        className="input"
                                        placeholder="Enter your password"
                                        required
                                    />
                                </div>
                                {mode === 'register' && (
                                    <p className="field-hint">Use at least 8 characters with letters and numbers.</p>
                                )}
                            </div>
                        )}

                        {(mode === 'verify' || mode === 'reset') && (
                            <div className="form-group">
                                <div className="label-row">
                                    <label className="label">One-time code</label>
                                    {mode === 'verify' && (
                                        <button
                                            type="button"
                                            className="auth-link-button"
                                            onClick={handleResendVerification}
                                            disabled={loading}
                                        >
                                            Resend code
                                        </button>
                                    )}
                                </div>
                                <div className="input-wrapper">
                                    <KeyRound size={18} className="input-icon" />
                                    <input
                                        type="text"
                                        name="otp"
                                        value={formData.otp}
                                        onChange={handleChange}
                                        className="input auth-otp-input"
                                        placeholder="000000"
                                        inputMode="numeric"
                                        maxLength={6}
                                        required
                                    />
                                </div>
                            </div>
                        )}

                        {mode === 'reset' && (
                            <>
                                <div className="form-group">
                                    <label className="label">New password</label>
                                    <div className="input-wrapper">
                                        <Lock size={18} className="input-icon" />
                                        <input
                                            type="password"
                                            name="nextPassword"
                                            value={formData.nextPassword}
                                            onChange={handleChange}
                                            className="input"
                                            placeholder="Create a new password"
                                            required
                                        />
                                    </div>
                                </div>

                                <div className="form-group">
                                    <label className="label">Confirm password</label>
                                    <div className="input-wrapper">
                                        <Lock size={18} className="input-icon" />
                                        <input
                                            type="password"
                                            name="confirmPassword"
                                            value={formData.confirmPassword}
                                            onChange={handleChange}
                                            className="input"
                                            placeholder="Repeat your new password"
                                            required
                                        />
                                    </div>
                                </div>
                            </>
                        )}

                        <button type="submit" className="btn btn-primary submit-btn" disabled={loading}>
                            {loading ? (
                                <div className="spinner" style={{ width: 20, height: 20 }}></div>
                            ) : (
                                <>
                                    {mode === 'login' && 'Sign In'}
                                    {mode === 'register' && 'Create Account'}
                                    {mode === 'verify' && 'Verify Email'}
                                    {mode === 'forgot' && 'Send Reset Code'}
                                    {mode === 'reset' && 'Update Password'}
                                    <ArrowRight size={18} />
                                </>
                            )}
                        </button>
                    </form>

                    <div className="form-footer">
                        {mode === 'login' && (
                            <p>
                                New here?
                                <button type="button" className="toggle-btn" onClick={() => switchMode('register')}>
                                    Create an account
                                </button>
                            </p>
                        )}

                        {mode === 'register' && (
                            <p>
                                Already have an account?
                                <button type="button" className="toggle-btn" onClick={() => switchMode('login')}>
                                    Sign in
                                </button>
                            </p>
                        )}

                        {(mode === 'verify' || mode === 'forgot' || mode === 'reset') && (
                            <p>
                                Back to
                                <button type="button" className="toggle-btn" onClick={() => switchMode('login')}>
                                    Sign in
                                </button>
                            </p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
