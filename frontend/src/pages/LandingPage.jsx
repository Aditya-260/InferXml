import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, useInView, useMotionValue, useSpring, useTransform } from 'framer-motion'
import { BentoGrid, BentoCard } from '../components/ui/BentoGrid'
import { AnimatedTestimonials } from '../components/ui/AnimatedTestimonials'
import {
    Sparkles,
    Database,
    Cpu,
    Brain,
    Target,
    Terminal,
    Package,
    ArrowRight,
    Upload,
    BarChart3,
    Zap,
    Shield,
    Clock,
    Code2,
    ChevronRight,
    Github,
    BookOpen,
    ExternalLink,
    Layers,
    GitBranch,
    Workflow,
    TrendingUp,
    Atom,
    Network,
    Crown,
    Check,
    X,
    Star
} from 'lucide-react'
import './LandingPage.css'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

/* ═══════════════════════════════════
   Fade-in wrapper for scroll reveal
   ═══════════════════════════════════ */
function FadeIn({ children, delay = 0, className = '' }) {
    const ref = React.useRef(null)
    const isInView = useInView(ref, { once: true, margin: '-80px' })

    return (
        <motion.div
            ref={ref}
            initial={{ opacity: 0, y: 30 }}
            animate={isInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay, ease: 'easeOut' }}
            className={className}
        >
            {children}
        </motion.div>
    )
}

/* ═══════════════════════════════════
   FEATURE DATA
   ═══════════════════════════════════ */
const features = [
    {
        Icon: Database,
        name: 'Smart Data Profiling',
        description: 'Upload CSV, Excel, or images — instant schema detection, stats, and quality checks.',
        href: '#',
        cta: 'Learn more',
        className: 'bento-wide',
    },
    {
        Icon: Brain,
        name: 'AutoML Training',
        description: 'Automatic model selection, hyperparameter tuning, and cross-validation across 10+ algorithms.',
        href: '#',
        cta: 'Learn more',
        className: '',
    },
    {
        Icon: Sparkles,
        name: 'Explainable AI',
        description: 'Real-time learning timeline shows what the model learned, why it chose each step.',
        href: '#',
        cta: 'Learn more',
        className: '',
    },
    {
        Icon: Target,
        name: 'AI Goal Analysis',
        description: 'Describe your goal in plain English — AI suggests the right target and approach.',
        href: '#',
        cta: 'Learn more',
        className: '',
    },
    {
        Icon: Terminal,
        name: 'Live Developer Logs',
        description: 'Watch training in real time with streaming logs, progress bars, and status updates.',
        href: '#',
        cta: 'Learn more',
        className: 'bento-wide',
    },
    {
        Icon: Shield,
        name: 'Auto Preprocessing',
        description: 'Missing values, encoding, scaling, outlier removal — all handled automatically before training.',
        href: '#',
        cta: 'Learn more',
        className: '',
    },
    {
        Icon: Package,
        name: 'One-Click Export',
        description: 'Download trained models as production-ready packages. Deploy anywhere.',
        href: '#',
        cta: 'Learn more',
        className: '',
    },
]

/* ═══════════════════════════════════
   STEPS DATA
   ═══════════════════════════════════ */
const steps = [
    {
        num: '01',
        icon: Upload,
        title: 'Upload',
        desc: 'Drop your CSV, Excel, or image dataset. We profile it instantly.',
    },
    {
        num: '02',
        icon: Cpu,
        title: 'Train',
        desc: 'AI picks the best model, trains it, explains every decision.',
    },
    {
        num: '03',
        icon: BarChart3,
        title: 'Predict',
        desc: 'Use your model via the built-in Streamlit UI. No deployment needed.',
    },
]

/* ═══════════════════════════════════
   STATS DATA
   ═══════════════════════════════════ */
const stats = [
    { value: '10+', label: 'ML Models per run', icon: Zap },
    { value: '<5min', label: 'Avg training time', icon: Clock },
    { value: '100%', label: 'Open source', icon: Shield },
    { value: '0', label: 'Lines of code needed', icon: Code2 },
]

/* ═══════════════════════════════════
   TEAM DATA
   ═══════════════════════════════════ */
const team = [
    {
        quote: "Building a platform that makes machine learning accessible to everyone is a dream come true.",
        name: "Alok Gupta",
        designation: "Lead Developer",
        src: "/AlokGupta.png",
    },
    {
        quote: "Ensuring our AI models are explainable and transparent is key to user trust.",
        name: "Om Babar",
        designation: "AI Engineer",
        src: "/OmBabar.png",
    },
    {
        quote: "Optimizing the training pipeline for speed and efficiency was a fun challenge.",
        name: "Bharatkumar Gungoman",
        designation: "Backend Engineer",
        src: "/Bharat.jpeg",
    },
]

/* ═══════════════════════════════════════════════
   LANDING PAGE COMPONENT
   ═══════════════════════════════════════════════ */
export default function LandingPage() {
    const navigate = useNavigate()
    const [scrolled, setScrolled] = useState(false)
    const [billingConfig, setBillingConfig] = useState(null)

    useEffect(() => {
        // Fetch dynamic pricing from backend
        fetch(`${API_BASE}/billing/config`)
            .then(res => res.json())
            .then(data => setBillingConfig(data))
            .catch(err => console.error('Failed to fetch billing config:', err))
    }, [])

    /* ═══ Mouse-tracking tilt for terminal card ═══ */
    const terminalRef = React.useRef(null)
    const mouseX = useMotionValue(0)
    const mouseY = useMotionValue(0)

    const rotateX = useSpring(useTransform(mouseY, [-0.5, 0.5], [18, -18]), { stiffness: 80, damping: 12 })
    const rotateY = useSpring(useTransform(mouseX, [-0.5, 0.5], [-18, 18]), { stiffness: 80, damping: 12 })

    const handleTerminalMouseMove = (e) => {
        const rect = terminalRef.current?.getBoundingClientRect()
        if (!rect) return
        const x = (e.clientX - rect.left) / rect.width - 0.5
        const y = (e.clientY - rect.top) / rect.height - 0.5
        mouseX.set(x)
        mouseY.set(y)
    }

    const handleTerminalMouseLeave = () => {
        mouseX.set(0)
        mouseY.set(0)
    }

    useEffect(() => {
        const handleScroll = () => setScrolled(window.scrollY > 40)
        window.addEventListener('scroll', handleScroll)
        return () => window.removeEventListener('scroll', handleScroll)
    }, [])

    const scrollToSection = (id) => {
        document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    }

    return (
        <div className="landing-page">

            {/* ═══ NAVBAR ═══ */}
            <nav className={`landing-nav ${scrolled ? 'nav-scrolled' : ''}`}>
                <div className="nav-inner">
                    <div className="nav-brand" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
                        <img src="/logo.png" alt="InferX-ML" className="brand-logo" />
                        <span className="brand-text">InferX-ML</span>
                    </div>

                    <div className="nav-links">
                        <button onClick={() => scrollToSection('features')} className="nav-link">Features</button>
                        <button onClick={() => scrollToSection('how-it-works')} className="nav-link">How it Works</button>
                        <button onClick={() => scrollToSection('pricing')} className="nav-link">Pricing</button>
                    </div>

                    <div className="nav-actions">
                        <button onClick={() => navigate('/login')} className="nav-link">Login</button>
                        <button onClick={() => navigate('/login')} className="nav-cta">
                            Get Started
                            <ChevronRight size={16} />
                        </button>
                    </div>
                </div>
            </nav>

            {/* ═══ HERO — Text Left, Terminal Right ═══ */}
            <section className="hero-section">
                {/* Animated gradient orbs */}
                <div className="hero-orb orb-1" />
                <div className="hero-orb orb-2" />
                <div className="hero-orb orb-3" />
                <div className="hero-grid-bg" />

                {/* Floating ML icons */}
                {[
                    { Icon: Brain,      x: '8%',  y: '18%', size: 28, delay: 0,   dur: 5 },
                    { Icon: Network,    x: '85%', y: '12%', size: 24, delay: 0.8, dur: 6 },
                    { Icon: Layers,     x: '5%',  y: '72%', size: 22, delay: 1.5, dur: 4.5 },
                    { Icon: Atom,       x: '92%', y: '65%', size: 26, delay: 0.3, dur: 5.5 },
                    { Icon: GitBranch,  x: '15%', y: '88%', size: 20, delay: 2,   dur: 4 },
                    { Icon: Workflow,   x: '78%', y: '85%', size: 22, delay: 1.2, dur: 6.5 },
                    { Icon: TrendingUp, x: '50%', y: '8%',  size: 20, delay: 0.6, dur: 5 },
                ].map(({ Icon, x, y, size, delay, dur }, i) => (
                    <motion.div
                        key={i}
                        className="hero-float-icon"
                        style={{ left: x, top: y }}
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{
                            opacity: [0, 0.35, 0.2, 0.35],
                            scale: 1,
                            y: [0, -12, 0, 12, 0],
                        }}
                        transition={{
                            opacity: { delay, duration: dur, repeat: Infinity },
                            scale:   { delay, duration: 0.6 },
                            y:       { delay, duration: dur, repeat: Infinity, ease: 'easeInOut' },
                        }}
                    >
                        <Icon size={size} />
                    </motion.div>
                ))}

                <div className="hero-split">
                    {/* Left — Text */}
                    <motion.div
                        className="hero-text hero-text-left"
                        initial={{ opacity: 0, x: -40 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.7, ease: 'easeOut' }}
                    >
                        <motion.div
                            className="hero-badge"
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: 0.2, duration: 0.5 }}
                        >
                            <Zap size={14} />
                            <span>AI-Powered AutoML Engine</span>
                        </motion.div>
                        <h1 className="hero-title">
                            From Raw Data to
                            <br />
                            <span className="hero-gradient-text">Production Models</span>
                            <br />
                            <span className="hero-title-accent">in Minutes.</span>
                        </h1>
                        <p className="hero-subtitle">
                            Drop your CSV. Pick a target. InferX handles the rest — feature engineering,
                            model selection, hyperparameter tuning, and explainability.
                            <strong> Zero code. Full control.</strong>
                        </p>
                        <div className="hero-actions">
                            <button onClick={() => navigate('/login')} className="btn-hero-primary">
                                <Sparkles size={18} />
                                Start Building Free
                                <ArrowRight size={18} />
                            </button>
                            <button onClick={() => scrollToSection('how-it-works')} className="btn-hero-secondary">
                                See How It Works
                                <ExternalLink size={16} />
                            </button>
                        </div>
                        <div className="hero-stats">
                            <div className="hero-stat">
                                <span className="stat-value">10+</span>
                                <span className="stat-label">ML Algorithms</span>
                            </div>
                            <div className="hero-stat-divider" />
                            <div className="hero-stat">
                                <span className="stat-value">94%+</span>
                                <span className="stat-label">Avg Accuracy</span>
                            </div>
                            <div className="hero-stat-divider" />
                            <div className="hero-stat">
                                <span className="stat-value">&lt;5 min</span>
                                <span className="stat-label">To Deploy</span>
                            </div>
                        </div>
                    </motion.div>

                    {/* Right — Terminal Mock with 3D tilt */}
                    <motion.div
                        ref={terminalRef}
                        className="hero-terminal-wrap"
                        initial={{ opacity: 0, x: 40 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.7, delay: 0.15, ease: 'easeOut' }}
                        style={{
                            rotateX,
                            rotateY,
                            transformPerspective: 800,
                            transformStyle: 'preserve-3d',
                        }}
                        onMouseMove={handleTerminalMouseMove}
                        onMouseLeave={handleTerminalMouseLeave}
                    >
                        <div className="hero-dashboard-mock">
                            <div className="mock-header">
                                <div className="mock-dots">
                                    <span className="dot dot-red" />
                                    <span className="dot dot-yellow" />
                                    <span className="dot dot-green" />
                                </div>
                                <span className="mock-title">InferX-ML — Training</span>
                            </div>
                            <div className="mock-body">
                                <div className="mock-sidebar">
                                    <div className="mock-nav-item active"><Database size={14} /> Datasets</div>
                                    <div className="mock-nav-item"><Cpu size={14} /> Training</div>
                                    <div className="mock-nav-item"><Package size={14} /> Models</div>
                                    <div className="mock-nav-item"><BarChart3 size={14} /> Predictions</div>
                                </div>
                                <div className="mock-content">
                                    <div className="mock-terminal">
                                        <div className="term-line"><span className="term-green">✓</span> Preprocessing complete</div>
                                        <div className="term-line"><span className="term-blue">⟳</span> Training RandomForest...</div>
                                        <div className="term-line"><span className="term-blue">⟳</span> Training XGBoost...</div>
                                        <div className="term-line"><span className="term-green">✓</span> Best model: XGBoost — 94.7%</div>
                                        <div className="term-line term-dim">█████████████░░ 87%</div>
                                    </div>
                                    <div className="mock-score-ring">
                                        <svg viewBox="0 0 120 120" className="ring-svg">
                                            <circle cx="60" cy="60" r="52" className="ring-bg" />
                                            <circle cx="60" cy="60" r="52" className="ring-fill" />
                                        </svg>
                                        <div className="ring-label">
                                            <span className="ring-value">94.7%</span>
                                            <span className="ring-text">Accuracy</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* ═══ FEATURES (Bento Grid) ═══ */}
            <section id="features" className="features-section">
                <FadeIn>
                    <div className="section-label">
                        <Sparkles size={14} />
                        <span>Capabilities</span>
                    </div>
                    <h2 className="section-title">Everything you need to ship ML</h2>
                    <p className="section-subtitle">From raw data to production-ready models — without writing a single line of code.</p>
                </FadeIn>

                <FadeIn delay={0.1}>
                    <BentoGrid className="bento-3-rows">
                        {features.map((feature) => (
                            <BentoCard key={feature.name} {...feature} />
                        ))}
                    </BentoGrid>
                </FadeIn>
            </section>

            {/* ═══ HOW IT WORKS ═══ */}
            <section id="how-it-works" className="steps-section">
                <FadeIn>
                    <div className="section-label">
                        <Zap size={14} />
                        <span>Workflow</span>
                    </div>
                    <h2 className="section-title">Three steps. That's it.</h2>
                    <p className="section-subtitle">No pipelines to configure. No infrastructure to manage.</p>
                </FadeIn>

                <div className="steps-row">
                    {steps.map((s, i) => (
                        <React.Fragment key={s.num}>
                            <FadeIn delay={i * 0.15}>
                                <div className="step-card">
                                    <span className="step-num">{s.num}</span>
                                    <div className="step-icon-wrap">
                                        <s.icon size={28} />
                                    </div>
                                    <h3>{s.title}</h3>
                                    <p>{s.desc}</p>
                                </div>
                            </FadeIn>
                            {i < steps.length - 1 && (
                                <div className="step-connector">
                                    <ArrowRight size={20} />
                                </div>
                            )}
                        </React.Fragment>
                    ))}
                </div>
            </section>

            {/* ═══ STATS STRIP ═══ */}
            <section className="stats-section">
                <div className="stats-row">
                    {stats.map((s, i) => (
                        <FadeIn key={s.label} delay={i * 0.1}>
                            <div className="stat-card">
                                <s.icon size={20} className="stat-icon" />
                                <span className="stat-value">{s.value}</span>
                                <span className="stat-label">{s.label}</span>
                            </div>
                        </FadeIn>
                    ))}
                </div>
            </section>

            {/* ═══ PRICING ═══ */}
            <section id="pricing" className="pricing-section">
                <FadeIn>
                    <div className="section-label">
                        <Crown size={14} />
                        <span>Pricing</span>
                    </div>
                    <h2 className="section-title">Simple, transparent pricing</h2>
                    <p className="section-subtitle">Start free. Upgrade when you need more power.</p>
                </FadeIn>

                <div className="pricing-grid">
                    {/* Free Plan */}
                    <FadeIn delay={0.1}>
                        <div className="pricing-card">
                            <div className="pricing-card-header">
                                <span className="pricing-plan-name">Free</span>
                                <div className="pricing-price">
                                    <span className="pricing-currency">₹</span>
                                    <span className="pricing-amount">0</span>
                                    <span className="pricing-period">/mo</span>
                                </div>
                                <p className="pricing-desc">Perfect for learning &amp; small experiments</p>
                            </div>
                            <ul className="pricing-features">
                                <li><Check size={16} className="feature-check" /> 1 training per day</li>
                                <li><Check size={16} className="feature-check" /> 4 ML algorithms</li>
                                <li><Check size={16} className="feature-check" /> 20 API requests / min</li>
                                <li><Check size={16} className="feature-check" /> Basic model export</li>
                                <li className="feature-disabled"><X size={16} className="feature-x" /> XGBoost &amp; LightGBM</li>
                                <li className="feature-disabled"><X size={16} className="feature-x" /> Advanced deep learning</li>
                                <li className="feature-disabled"><X size={16} className="feature-x" /> Priority support</li>
                            </ul>
                            <div className="pricing-card-footer">
                                <button onClick={() => navigate('/login')} className="pricing-btn pricing-btn-outline">
                                    Get Started Free
                                </button>
                            </div>
                        </div>
                    </FadeIn>

                    {/* Pro Plan */}
                    <FadeIn delay={0.2}>
                        <div className="pricing-card pricing-card-pro">
                            <div className="pricing-popular-badge">
                                <Zap size={12} />
                                Most Popular
                            </div>
                            <div className="pricing-card-header">
                                <span className="pricing-plan-name">Pro</span>
                                <div className="pricing-price">
                                    <span className="pricing-currency">₹</span>
                                    <span className="pricing-amount">{billingConfig ? (billingConfig.pro_plan_amount / 100) : '499'}</span>
                                    <span className="pricing-period">/mo</span>
                                </div>
                                <p className="pricing-desc">For serious practitioners &amp; production use</p>
                            </div>
                            <ul className="pricing-features">
                                <li><Check size={16} className="feature-check pro" /> Unlimited training jobs</li>
                                <li><Check size={16} className="feature-check pro" /> 10+ ML algorithms</li>
                                <li><Check size={16} className="feature-check pro" /> 200 API requests / min</li>
                                <li><Check size={16} className="feature-check pro" /> Advanced model export</li>
                                <li><Check size={16} className="feature-check pro" /> XGBoost &amp; LightGBM</li>
                                <li><Check size={16} className="feature-check pro" /> YOLOv8 deep learning</li>
                                <li><Check size={16} className="feature-check pro" /> Priority support</li>
                            </ul>
                            <div className="pricing-card-footer">
                                <button onClick={() => navigate('/login')} className="pricing-btn pricing-btn-primary">
                                    <Crown size={16} />
                                    Upgrade to Pro
                                </button>
                            </div>
                        </div>
                    </FadeIn>

                    {/* Advance Plan */}
                    <FadeIn delay={0.3}>
                        <div className="pricing-card pricing-card-advance" style={{ borderColor: '#8b5cf6', boxShadow: '0 0 40px -10px rgba(139, 92, 246, 0.15)', background: 'linear-gradient(to bottom, rgba(30, 30, 40, 0.8), rgba(20, 20, 28, 0.9))' }}>
                            <div className="pricing-card-header">
                                <span className="pricing-plan-name" style={{ color: '#8b5cf6' }}>Advance</span>
                                <div className="pricing-price">
                                    <span className="pricing-currency">₹</span>
                                    <span className="pricing-amount">{billingConfig ? (billingConfig.advance_plan_amount / 100) : '4999'}</span>
                                    <span className="pricing-period">/yr</span>
                                </div>
                                <p className="pricing-desc">Best value for long-term production workloads</p>
                            </div>
                            <ul className="pricing-features">
                                <li><Check size={16} className="feature-check pro" style={{ color: '#8b5cf6' }} /> Everything in Pro</li>
                                <li><Check size={16} className="feature-check pro" style={{ color: '#8b5cf6' }} /> 1 Year Full Access</li>
                                <li><Check size={16} className="feature-check pro" style={{ color: '#8b5cf6' }} /> Priority API access</li>
                                <li><Check size={16} className="feature-check pro" style={{ color: '#8b5cf6' }} /> Early access features</li>
                                <li><Check size={16} className="feature-check pro" style={{ color: '#8b5cf6' }} /> Dedicated support</li>
                                <li><Check size={16} className="feature-check pro" style={{ color: '#8b5cf6' }} /> Custom integrations</li>
                            </ul>
                            <div className="pricing-card-footer">
                                <button onClick={() => navigate('/login')} className="pricing-btn pricing-btn-primary" style={{ background: 'linear-gradient(135deg, #8b5cf6, #a78bfa)', boxShadow: '0 8px 20px -8px rgba(139, 92, 246, 0.5)' }}>
                                    <Star size={16} />
                                    Get Advance Plan
                                </button>
                            </div>
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* ═══ FINAL CTA ═══ */}
            <section className="cta-section">
                <FadeIn>
                    <div className="cta-content">
                        <h2>Ready to build smarter models?</h2>
                        <p>Start training in under 5 minutes. No credit card required.</p>
                        <button onClick={() => navigate('/login')} className="btn-hero-primary btn-lg">
                            <Sparkles size={20} />
                            Get Started Free
                            <ArrowRight size={20} />
                        </button>
                    </div>
                </FadeIn>
                <div className="cta-orb cta-orb-1" />
                <div className="cta-orb cta-orb-2" />
            </section>

            {/* ═══ DEVELOPED BY ═══ */}
            {/* <section className="team-section">
                <FadeIn>
                    <div className="section-label">
                        <Code2 size={14} />
                        <span>The Team</span>
                    </div>
                    <h2 className="section-title">Developed By</h2>
                    <p className="section-subtitle">The minds behind InferX-ML.</p>
                </FadeIn>

                <FadeIn delay={0.2}>
                    <AnimatedTestimonials testimonials={team} autoplay={false} />
                </FadeIn>
            </section> */}

            {/* ═══ FOOTER ═══ */}
            {/* ═══ FOOTER ═══ */}
            <footer className="landing-footer">
                <div className="footer-content">
                    {/* Brand Column */}
                    <div className="footer-col brand-col">
                        <div className="footer-brand">
                            <img src="/logo.png" alt="InferX-ML" className="brand-logo footer-logo" />
                            <span>InferX-ML</span>
                        </div>
                        <p className="footer-tagline">
                            Empowering developers and businesses to build, train, and deploy AI models without writing code.
                        </p>
                        <div className="footer-socials">
                            <a href="#" aria-label="Github"><Github size={18} /></a>
                            <a href="#" aria-label="Twitter"><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zl-1.161 17.52h1.833L7.084 4.126H5.117z"></path></svg></a>
                            <a href="#" aria-label="Discord"><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-1.28 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006 1.28 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" /></svg></a>
                        </div>
                    </div>

                    {/* Product */}
                    <div className="footer-col">
                        <h4>Product</h4>
                        <a href="#">Features</a>
                        <a href="#">Integrations</a>
                        <a href="#">Pricing</a>
                        <a href="#">Changelog</a>
                    </div>

                    {/* Resources */}
                    <div className="footer-col">
                        <h4>Resources</h4>
                        <a href="#">Documentation</a>
                        <a href="#">API Reference</a>
                        <a href="#">Community</a>
                        <a href="#">Blog</a>
                    </div>

                    {/* Legal */}
                    <div className="footer-col">
                        <h4>Legal</h4>
                        <a href="#">Privacy Policy</a>
                        <a href="#">Terms of Service</a>
                        <a href="#">Security</a>
                        <a href="#">Cookie Policy</a>
                    </div>
                </div>

                <div className="footer-bottom">
                    <span className="footer-copy">© 2026 InferX-ML. All rights reserved.</span>
                    <div className="footer-status">
                        <div className="status-dot"></div>
                        <span>All Systems Operational</span>
                    </div>
                </div>
            </footer>
        </div>
    )
}
