import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { useState, useRef, useEffect, useMemo } from 'react'
import {
    Box,
    Download,
    Trash2,
    TrendingUp,
    Clock,
    BarChart3,
    Loader2,
    Sparkles,
    Target,
    Layers,
    Trophy,
    ChevronDown,
    X,
    Activity,
    Shield,
    Zap,
    Brain,
    ArrowRight,
    Info,
    FlaskConical,
    CheckCircle,
    ImageIcon,
    Search
} from 'lucide-react'
import { modelsApi } from '../services/api'
import './Models.css'

/* ── tiny bar-chart drawn in pure SVG ── */
function MiniBarChart({ data, width = 220, height = 100 }) {
    if (!data || data.length === 0) return null
    const max = Math.max(...data.map(d => d.value))
    const barW = Math.max(12, Math.min(28, (width - data.length * 4) / data.length))
    const totalW = data.length * (barW + 4)
    return (
        <svg width={totalW} height={height + 22} className="mini-bar-chart">
            {data.map((d, i) => {
                const barH = max > 0 ? (d.value / max) * height : 0
                const isBest = d.best
                return (
                    <g key={i} transform={`translate(${i * (barW + 4)}, 0)`}>
                        <rect
                            y={height - barH}
                            width={barW}
                            height={barH}
                            rx={4}
                            fill={isBest ? 'url(#bestGrad)' : 'rgba(16,185,129,0.35)'}
                            className="bar-rect"
                        />
                        <text
                            x={barW / 2}
                            y={height + 14}
                            textAnchor="middle"
                            fontSize={8}
                            fill="var(--color-gray-500)"
                        >
                            {d.label.length > 5 ? d.label.slice(0, 5) : d.label}
                        </text>
                    </g>
                )
            })}
            <defs>
                <linearGradient id="bestGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#34d399" />
                    <stop offset="100%" stopColor="#10b981" />
                </linearGradient>
            </defs>
        </svg>
    )
}

/* ── Radial Score Ring ── */
function ScoreRing({ score, size = 64 }) {
    const pct = (score || 0) * 100
    const r = (size - 8) / 2
    const circ = 2 * Math.PI * r
    const offset = circ - (pct / 100) * circ
    const color = pct >= 80 ? '#10b981' : pct >= 60 ? '#f59e0b' : '#ef4444'
    return (
        <svg width={size} height={size} className="score-ring">
            <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={5} />
            <circle
                cx={size / 2} cy={size / 2} r={r} fill="none"
                stroke={color} strokeWidth={5}
                strokeDasharray={circ} strokeDashoffset={offset}
                strokeLinecap="round"
                transform={`rotate(-90 ${size / 2} ${size / 2})`}
                style={{ transition: 'stroke-dashoffset 1s ease' }}
            />
            <text x={size / 2} y={size / 2 + 1} textAnchor="middle" dominantBaseline="middle"
                fontSize={size > 50 ? 14 : 11} fontWeight={700} fill="#e5e7eb">
                {pct.toFixed(0)}%
            </text>
        </svg>
    )
}

/* ── Full Analysis Modal ── */
function AnalysisModal({ model, onClose, onDelete }) {
    const modalRef = useRef(null)
    const allModels = model.results?.all_models || []
    const explanationData = model.explanation_data || {}
    const phases = explanationData.phases || []
    const models = explanationData.models || []
    const trust = explanationData.trust || {}
    const insights = explanationData.insights || []

    const { data: graphsData, isLoading: isLoadingGraphs } = useQuery({
        queryKey: ['model-graphs', model.id],
        queryFn: () => modelsApi.getGraphs(model.id),
        enabled: !!model.id,
        retry: 1
    })
    const graphs = graphsData?.data?.graphs || {}
    const hasGraphs = Object.keys(graphs).length > 0

    // Build chart data from all_models
    const chartData = useMemo(() => {
        return allModels
            .sort((a, b) => b.score - a.score)
            .map(m => ({
                label: m.model,
                value: Math.max(0, m.score * 100),
                best: m.model === model.best_model_name
            }))
    }, [allModels, model.best_model_name])

    // close on ESC
    useEffect(() => {
        const handler = (e) => { if (e.key === 'Escape') onClose() }
        window.addEventListener('keydown', handler)
        return () => window.removeEventListener('keydown', handler)
    }, [onClose])

    // close on backdrop click
    const handleBackdrop = (e) => {
        if (e.target === modalRef.current) onClose()
    }

    const bestModel = allModels.find(m => m.model === model.best_model_name)

    return (
        <div className="analysis-backdrop" ref={modalRef} onClick={handleBackdrop}>
            <div className="analysis-modal">
                {/* Header */}
                <div className="analysis-header">
                    <div className="analysis-header-left">
                        <div className="analysis-icon-wrap">
                            <Activity size={20} />
                        </div>
                        <div>
                            <h2>Model Analysis</h2>
                            <p className="analysis-subtitle">{model.name}</p>
                        </div>
                    </div>
                    <div className="analysis-header-actions">
                        {onDelete && (
                            <button
                                className="btn-icon btn-danger-icon"
                                onClick={() => {
                                    if (window.confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
                                        onDelete()
                                        onClose()
                                    }
                                }}
                                title="Delete Model"
                            >
                                <Trash2 size={18} />
                            </button>
                        )}
                        <button className="close-btn" onClick={onClose}>
                            <X size={20} />
                        </button>
                    </div>
                </div>

                <div className="analysis-body">
                    {/* ── Overview Cards Row ── */}
                    <div className="overview-cards">
                        <div className="overview-card">
                            <div className="ov-icon ov-purple"><Trophy size={18} /></div>
                            <div className="ov-info">
                                <span className="ov-label">Best Model</span>
                                <span className="ov-value">{model.best_model_name || '—'}</span>
                            </div>
                        </div>
                        <div className="overview-card">
                            <div className="ov-icon ov-green"><Target size={18} /></div>
                            <div className="ov-info">
                                <span className="ov-label">Accuracy</span>
                                <span className="ov-value">{model.best_score ? `${(model.best_score * 100).toFixed(1)}%` : '—'}</span>
                            </div>
                        </div>
                        <div className="overview-card">
                            <div className="ov-icon ov-blue"><FlaskConical size={18} /></div>
                            <div className="ov-info">
                                <span className="ov-label">Models Tested</span>
                                <span className="ov-value">{allModels.length}</span>
                            </div>
                        </div>
                        <div className="overview-card">
                            <div className="ov-icon ov-amber"><Zap size={18} /></div>
                            <div className="ov-info">
                                <span className="ov-label">Problem Type</span>
                                <span className="ov-value capitalize">{model.problem_type || '—'}</span>
                            </div>
                        </div>
                    </div>

                    {/* ── Model Comparison Chart ── */}
                    {chartData.length > 0 && (
                        <div className="analysis-section">
                            <h3><BarChart3 size={16} /> Model Performance Comparison</h3>
                            <div className="chart-container">
                                <div className="chart-wrapper">
                                    {chartData.map((d, i) => (
                                        <div key={i} className={`chart-bar-col ${d.best ? 'best' : ''}`}>
                                            <span className="chart-value">{d.value.toFixed(1)}%</span>
                                            <div className="chart-bar-track">
                                                <div
                                                    className="chart-bar-fill"
                                                    style={{ height: `${d.value}%` }}
                                                />
                                            </div>
                                            <span className="chart-label">{d.label}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* ── Detailed Metrics Table ── */}
                    {allModels.length > 0 && (
                        <div className="analysis-section">
                            <h3><Layers size={16} /> Detailed Metrics</h3>
                            <div className="metrics-table-wrap">
                                <table className="metrics-table">
                                    <thead>
                                        <tr>
                                            <th>Model</th>
                                            <th>Score</th>
                                            {Object.keys(allModels[0]?.metrics || {}).map(key => (
                                                <th key={key}>{key.replace('_', ' ')}</th>
                                            ))}
                                            <th>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {allModels.sort((a, b) => b.score - a.score).map((m, i) => (
                                            <tr key={i} className={m.model === model.best_model_name ? 'best-row' : ''}>
                                                <td className="model-name-cell">
                                                    {m.model === model.best_model_name && <Trophy size={12} className="trophy-mini" />}
                                                    {m.model}
                                                </td>
                                                <td className="score-cell">{(m.score * 100).toFixed(2)}%</td>
                                                {Object.values(m.metrics || {}).map((val, j) => (
                                                    <td key={j}>{typeof val === 'number' ? val.toFixed(4) : val}</td>
                                                ))}
                                                <td>
                                                    <span className={`mini-badge ${m.model === model.best_model_name ? 'winner' : ''}`}>
                                                        {m.model === model.best_model_name ? '🏆 Best' : 'Tested'}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* ── AI Insights ── */}
                    {insights.length > 0 && (
                        <div className="analysis-section">
                            <h3><Brain size={16} /> AI Insights</h3>
                            <div className="insights-list">
                                {insights.slice(0, 8).map((insight, i) => (
                                    <div key={i} className="insight-item">
                                        <span className="insight-dot" />
                                        <p>{insight.message}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* ── Training Phases ── */}
                    {phases.length > 0 && (
                        <div className="analysis-section">
                            <h3><Activity size={16} /> Training Pipeline</h3>
                            <div className="phases-timeline">
                                {phases.map((phase, i) => (
                                    <div key={i} className="phase-item">
                                        <div className="phase-connector-line" />
                                        <div className="phase-dot">
                                            <CheckCircle size={14} />
                                        </div>
                                        <div className="phase-info">
                                            <span className="phase-icon">{phase.icon}</span>
                                            <div>
                                                <strong>{phase.title}</strong>
                                                <p>{phase.description}</p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* ── Trust & Safety ── */}
                    {trust.data_size && (
                        <div className="analysis-section">
                            <h3><Shield size={16} /> Trust & Reliability</h3>
                            <div className="trust-grid">
                                <div className="trust-item">
                                    <span className="trust-label">Dataset Size</span>
                                    <span className="trust-value">{trust.data_size?.toLocaleString()} rows</span>
                                </div>
                                <div className="trust-item">
                                    <span className="trust-label">Training Split</span>
                                    <span className="trust-value">{trust.train_ratio || '80%'}</span>
                                </div>
                                <div className="trust-item">
                                    <span className="trust-label">Test Split</span>
                                    <span className="trust-value">{trust.test_ratio || '20%'}</span>
                                </div>
                                <div className="trust-item">
                                    <span className="trust-label">Confidence</span>
                                    <span className={`trust-badge trust-${trust.confidence || 'medium'}`}>
                                        {trust.confidence || 'medium'}
                                    </span>
                                </div>
                            </div>
                            {trust.confidence_reasons && (
                                <ul className="trust-reasons">
                                    {trust.confidence_reasons.map((r, i) => (
                                        <li key={i}>{r}</li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    )}

                    {/* ── Evaluation Graphs ── */}
                    {hasGraphs && (
                        <div className="analysis-section graphs-section">
                            <h3><ImageIcon size={16} /> Evaluation Graphs</h3>
                            <div className="graphs-grid">
                                {Object.entries(graphs).map(([name, src]) => (
                                    <div key={name} className="graph-item card">
                                        <div className="graph-header">
                                            <h4>{name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</h4>
                                        </div>
                                        <div className="graph-img-wrap">
                                            <img src={src} alt={name} loading="lazy" />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

/* ══════════════════════════════════════
   Main Models Page
   ══════════════════════════════════════ */
export default function Models() {
    const queryClient = useQueryClient()
    const [downloadingId, setDownloadingId] = useState(null)
    const [analysisModel, setAnalysisModel] = useState(null)
    const [currentPage, setCurrentPage] = useState(1)
    const [searchTerm, setSearchTerm] = useState('')

    const MODELS_PER_PAGE = 12

    const { data, isLoading } = useQuery({
        queryKey: ['models'],
        queryFn: () => modelsApi.list()
    })

    const rawModels = data?.data?.models || []

    // 1. Sort models by created_at descending (latest first)
    const sortedModels = useMemo(() => {
        return [...rawModels].sort((a, b) => {
            return new Date(b.created_at) - new Date(a.created_at)
        })
    }, [rawModels])

    // 2. Filter models by search term (case-insensitive model name search)
    const filteredModels = useMemo(() => {
        if (!searchTerm.trim()) return sortedModels
        const term = searchTerm.toLowerCase()
        return sortedModels.filter(m => m.name && m.name.toLowerCase().includes(term))
    }, [sortedModels, searchTerm])

    const totalPages = Math.max(1, Math.ceil(filteredModels.length / MODELS_PER_PAGE))

    const paginatedModels = useMemo(() => {
        const start = (currentPage - 1) * MODELS_PER_PAGE
        return filteredModels.slice(start, start + MODELS_PER_PAGE)
    }, [filteredModels, currentPage])

    useEffect(() => {
        if (currentPage > totalPages) {
            setCurrentPage(totalPages)
        }
    }, [currentPage, totalPages])

    const deleteMutation = useMutation({
        mutationFn: (id) => modelsApi.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries(['models'])
        },
        onError: (error) => {
            console.error('Delete failed:', error)
            alert('Failed to delete model. Please try again.')
        }
    })

    const handleDownload = async (modelId, modelName) => {
        setDownloadingId(modelId)
        try {
            const response = await modelsApi.download(modelId)
            const blob = new Blob([response.data], { type: 'application/zip' })
            const url = window.URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `${modelName.replace(/\s+/g, '_')}_model.zip`
            document.body.appendChild(a)
            a.click()
            window.URL.revokeObjectURL(url)
            document.body.removeChild(a)
        } catch (error) {
            console.error('Download failed:', error)
            alert('Download failed. Model package may not be available.')
        } finally {
            setDownloadingId(null)
        }
    }

    const getProblemTypeColor = (type) => {
        if (type?.includes('classification')) return 'purple'
        if (type === 'regression') return 'blue'
        if (type === 'clustering') return 'amber'
        if (type === 'timeseries') return 'green'
        return 'gray'
    }

    return (
        <div className="models-page">
            {/* Hero header */}
            <div className="models-hero">
                <div className="hero-glow" />
                <div className="hero-content">
                    <span className="hero-badge"><Sparkles size={12} /> MODEL VAULT</span>
                    <h1>Trained Models</h1>
                    <p className="hero-subtitle">
                        View, analyze, and deploy your trained machine learning models.
                        Each model includes full performance metrics and AI-powered analysis.
                    </p>
                </div>
            </div>

            {/* Stats strip & Search Bar */}
            {rawModels.length > 0 && (
                <div className="models-toolbar">
                    <div className="stats-strip">
                        <div className="stat-chip">
                            <Layers size={14} />
                            <span><strong>{rawModels.length}</strong> Models</span>
                        </div>
                        <div className="stat-chip">
                            <Trophy size={14} />
                            <span>Best: <strong>
                                {rawModels.length > 0 ? `${(Math.max(...rawModels.map(m => m.best_score || 0)) * 100).toFixed(1)}%` : '—'}
                            </strong></span>
                        </div>
                        <div className="stat-chip">
                            <Target size={14} />
                            <span>Types: <strong>
                                {[...new Set(rawModels.map(m => m.problem_type).filter(Boolean))].join(', ') || '—'}
                            </strong></span>
                        </div>
                    </div>

                    <div className="models-search-bar">
                        <div className="search-input-wrap">
                            <Search size={16} className="search-icon" />
                            <input
                                type="text"
                                placeholder="Search models by name..."
                                value={searchTerm}
                                onChange={(e) => {
                                    setSearchTerm(e.target.value)
                                    setCurrentPage(1)
                                }}
                                className="search-input"
                            />
                            {searchTerm && (
                                <button className="clear-search-btn" onClick={() => setSearchTerm('')} title="Clear search">
                                    <X size={14} />
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Model list */}
            {isLoading ? (
                <div className="loading-state">
                    <div className="spinner" />
                    <p>Loading models...</p>
                </div>
            ) : rawModels.length > 0 ? (
                filteredModels.length > 0 ? (
                    <>
                    <div className="models-grid">
                        {paginatedModels.map((model) => {
                            const allResults = model.results?.all_models || []
                            const chartData = allResults
                                .sort((a, b) => b.score - a.score)
                                .slice(0, 6)
                                .map(m => ({
                                    label: m.model,
                                    value: Math.max(0, m.score * 100),
                                    best: m.model === model.best_model_name
                                }))

                            return (
                                <div key={model.id} className="model-card card">
                                    {/* Card top — badge + score ring */}
                                    <div className="model-card-top">
                                        <div className="mc-left">
                                            <span className={`problem-badge pb-${getProblemTypeColor(model.problem_type)}`}>
                                                {model.problem_type?.replace('_', ' ') || 'unknown'}
                                            </span>
                                            <h3 className="model-name">{model.name}</h3>
                                        </div>
                                        <div className="mc-right">
                                            <ScoreRing score={model.best_score} size={68} />
                                        </div>
                                    </div>

                                    {/* Quick stats */}
                                    <div className="model-quick-stats">
                                        <div className="qs-item">
                                            <Trophy size={13} className="qs-icon" />
                                            <span className="qs-label">Best</span>
                                            <span className="qs-val">{model.best_model_name || '—'}</span>
                                        </div>
                                        <div className="qs-item">
                                            <Target size={13} className="qs-icon" />
                                            <span className="qs-label">Target</span>
                                            <span className="qs-val mono">{model.target_column || '—'}</span>
                                        </div>
                                        <div className="qs-item">
                                            <Clock size={13} className="qs-icon" />
                                            <span className="qs-label">Trained</span>
                                            <span className="qs-val">{new Date(model.created_at).toLocaleDateString()}</span>
                                        </div>
                                    </div>

                                    {/* Mini chart preview */}
                                    {chartData.length > 0 && (
                                        <div className="mini-chart-preview">
                                            <MiniBarChart data={chartData} height={50} />
                                        </div>
                                    )}

                                    {/* Actions */}
                                    <div style={{
                                        display: 'flex',
                                        flexDirection: 'column',
                                        gap: '8px',
                                        marginTop: 'auto',
                                        paddingTop: '12px',
                                        borderTop: '1px solid rgba(255,255,255,0.1)'
                                    }}>
                                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                                            <button
                                                onClick={() => setAnalysisModel(model)}
                                                style={{
                                                    flex: 1,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    gap: '6px',
                                                    padding: '8px 12px',
                                                    background: 'rgba(16,185,129,0.12)',
                                                    border: '1px solid rgba(16,185,129,0.35)',
                                                    borderRadius: '8px',
                                                    color: '#6ee7b7',
                                                    fontSize: '0.82rem',
                                                    fontWeight: 600,
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                <BarChart3 size={14} />
                                                Analyze
                                            </button>
                                            <Link
                                                to={`/predictions/${model.id}`}
                                                style={{
                                                    flex: 1,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    gap: '6px',
                                                    padding: '8px 12px',
                                                    background: 'linear-gradient(135deg, #059669, #14b8a6)',
                                                    border: 'none',
                                                    borderRadius: '8px',
                                                    color: '#fff',
                                                    fontSize: '0.82rem',
                                                    fontWeight: 600,
                                                    textDecoration: 'none',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                <TrendingUp size={14} />
                                                Predict
                                            </Link>
                                        </div>
                                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                                            <button
                                                onClick={() => handleDownload(model.id, model.name)}
                                                disabled={downloadingId === model.id || !model.has_package}
                                                title={!model.has_package ? 'Package not available' : 'Download model'}
                                                style={{
                                                    flex: 1,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    gap: '6px',
                                                    padding: '8px 12px',
                                                    background: 'rgba(16,185,129,0.15)',
                                                    border: '2px solid rgba(16,185,129,0.5)',
                                                    borderRadius: '8px',
                                                    color: '#6ee7b7',
                                                    fontSize: '0.8rem',
                                                    fontWeight: 600,
                                                    cursor: (!model.has_package) ? 'not-allowed' : 'pointer',
                                                    opacity: (!model.has_package) ? 0.4 : 1
                                                }}
                                            >
                                                {downloadingId === model.id ? (
                                                    <Loader2 size={14} className="spin" />
                                                ) : (
                                                    <Download size={14} />
                                                )}
                                                Download
                                            </button>
                                            <button
                                                disabled={deleteMutation.isPending}
                                                title="Delete Model"
                                                onClick={() => {
                                                    if (window.confirm('Delete this model?')) {
                                                        deleteMutation.mutate(model.id)
                                                    }
                                                }}
                                                style={{
                                                    flex: 1,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    gap: '6px',
                                                    padding: '8px 12px',
                                                    background: 'rgba(239,68,68,0.15)',
                                                    border: '2px solid rgba(239,68,68,0.5)',
                                                    borderRadius: '8px',
                                                    color: '#fca5a5',
                                                    fontSize: '0.8rem',
                                                    fontWeight: 600,
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                <Trash2 size={14} />
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            )
                        })}
                    </div>

                    {totalPages > 1 && (
                        <div className="models-pagination">
                            <button
                                className="btn btn-outline models-pagination-btn"
                                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                                disabled={currentPage === 1}
                            >
                                Previous
                            </button>
                            <span className="pagination-meta">
                                Page {currentPage} of {totalPages}
                            </span>
                            <button
                                className="btn btn-outline models-pagination-btn"
                                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                                disabled={currentPage === totalPages}
                            >
                                Next
                            </button>
                        </div>
                    )}
                    </>
                ) : (
                    <div className="empty-state card">
                        <Box size={48} className="empty-icon" />
                        <h3>No matching models found</h3>
                        <p>Try adjusting your search criteria to find what you're looking for.</p>
                        <button className="btn btn-primary" onClick={() => setSearchTerm('')}>
                            Clear Search Query
                        </button>
                    </div>
                )
            ) : (
                <div className="empty-state card">
                    <Box size={48} className="empty-icon" />
                    <h3>No models yet</h3>
                    <p>Train your first model to see it here</p>
                    <Link to="/training" className="btn btn-primary">
                        <Zap size={16} />
                        Start Training
                    </Link>
                </div>
            )}

            {/* Analysis modal */}
            {analysisModel && (
                <AnalysisModal
                    model={analysisModel}
                    onClose={() => setAnalysisModel(null)}
                    onDelete={() => deleteMutation.mutate(analysisModel.id)}
                />
            )}
        </div>
    )
}
