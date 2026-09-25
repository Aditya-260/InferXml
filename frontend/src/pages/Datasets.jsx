import { useState, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { datasetsApi } from '../services/api'
import {
    Search, Upload, Trash2, X, Database, Download,
    FileText, ChevronDown, ExternalLink, Loader2,
    FolderUp, CheckCircle, AlertCircle, Info, ArrowRight,
    Wand2, Plus, Minus, Sparkles, Eye, Save, RotateCcw,
    MessageSquare, Table, RefreshCw, Lock
} from 'lucide-react'
import { useAuthStore } from '../store/authStore'
import './Datasets.css'

/* ─── helpers ─── */
const formatFileSize = (bytes) => {
    if (!bytes) return '—'
    const units = ['B', 'KB', 'MB', 'GB']
    let i = 0
    let size = bytes
    while (size >= 1024 && i < units.length - 1) { size /= 1024; i++ }
    return `${size.toFixed(i ? 1 : 0)} ${units[i]}`
}

const getStatusBadge = (status) => {
    const map = {
        completed: { label: 'Profiled', cls: 'badge-ok' },
        pending:   { label: 'Pending',  cls: 'badge-pending' },
        failed:    { label: 'Failed',   cls: 'badge-fail' },
    }
    const s = map[status] || map.pending
    return <span className={`ds-badge ${s.cls}`}>{s.label}</span>
}

const KAG_CATEGORIES = [
    "Agriculture", "Arts & Entertainment", "Biology", "Business", "Climate",
    "Computer Science", "Crime", "Economics", "Education", "Energy",
    "Environment", "Fashion", "Finance", "Food", "Games",
    "Geography", "Healthcare", "History", "Humanities", "Internet",
    "Journalism", "Law", "Linguistics", "Literature", "Mathematics",
    "Military", "Music", "Nature", "Physics", "Politics",
    "Psychology", "Religion", "Science", "Social Science", "Sports",
    "Technology", "Transportation", "Urban Planning", "Weather", "Web Development"
]

/* ══════════════════════════════════════ */
export default function Datasets() {
    const queryClient = useQueryClient()
    const fileInputRef = useRef(null)
    const { user } = useAuthStore()
    const isFree = !user?.plan_type || user.plan_type === 'free'

    /* ── local state ── */
    const [activeTab, setActiveTab] = useState('library')   // library | search | generate
    const [showUpload, setShowUpload] = useState(false)
    const [uploading, setUploading] = useState(false)
    const [uploadProgress, setUploadProgress] = useState(0)
    const [selectedDataset, setSelectedDataset] = useState(null)
    const [toast, setToast] = useState(null)

    // kaggle search state
    const [kaggleQuery, setKaggleQuery] = useState('')
    const [kaggleSearchTerm, setKaggleSearchTerm] = useState('')
    const [savingRef, setSavingRef] = useState(null)

    // AI synthetic generation state
    const [aiStage, setAiStage] = useState('idea')  // idea | prompting | schema | generating | preview | saved
    const [aiPrompt, setAiPrompt] = useState('')
    const [aiSchema, setAiSchema] = useState(null)   // { name, description, columns }
    const [aiRows, setAiRows] = useState(1000)
    const [aiPreview, setAiPreview] = useState(null)  // { preview: [...], total_rows, columns }
    const [aiLoading, setAiLoading] = useState(false)
    const [aiError, setAiError] = useState(null)
    const [aiRefinePrompt, setAiRefinePrompt] = useState('')

    /* ── queries ── */
    const { data: datasetsRes, isLoading } = useQuery({
        queryKey: ['datasets'],
        queryFn: () => datasetsApi.list()
    })
    const datasets = datasetsRes?.data?.datasets || datasetsRes?.data || []

    const { data: kaggleRes, isLoading: kaggleLoading, isFetching: kaggleFetching } = useQuery({
        queryKey: ['kaggle-search', kaggleSearchTerm],
        queryFn: () => datasetsApi.searchKaggle(kaggleSearchTerm),
        enabled: !!kaggleSearchTerm,
    })
    const kaggleResults = kaggleRes?.data?.results || kaggleRes?.data?.datasets || []
    const kaggleIsLimited = kaggleRes?.data?.is_limited || false

    /* ── mutations ── */
    const showToast = (type, message) => {
        setToast({ type, message })
        setTimeout(() => setToast(null), 4000)
    }

    const uploadMutation = useMutation({
        mutationFn: (formData) => {
            setUploading(true)
            return datasetsApi.upload(formData, setUploadProgress)
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['datasets'] })
            setShowUpload(false)
            setUploading(false)
            setUploadProgress(0)
            showToast('success', 'Dataset uploaded successfully')
        },
        onError: (err) => {
            setUploading(false)
            setUploadProgress(0)
            showToast('error', err?.response?.data?.error || 'Upload failed')
        }
    })

    const deleteMutation = useMutation({
        mutationFn: (id) => datasetsApi.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['datasets'] })
            setSelectedDataset(null)
            showToast('success', 'Dataset deleted')
        }
    })

    const kaggleDownloadMutation = useMutation({
        mutationFn: ({ ref, name }) => datasetsApi.downloadKaggle(ref, name),
        onSuccess: (res) => {
            queryClient.invalidateQueries({ queryKey: ['datasets'] })
            setSavingRef(null)
            showToast('success', res?.data?.message || 'Dataset saved to your library')
        },
        onError: (err) => {
            setSavingRef(null)
            showToast('error', err?.response?.data?.error || 'Download failed')
        }
    })

    const handleFileSelect = (e) => {
        const file = e.target.files[0]
        if (!file) return
        const formData = new FormData()
        formData.append('file', file)
        uploadMutation.mutate(formData)
    }

    const handleKaggleSearch = (e) => {
        if (e) e.preventDefault()
        if (kaggleQuery.trim()) setKaggleSearchTerm(kaggleQuery.trim())
    }

    const handleCategoryClick = (cat) => {
        setKaggleQuery(cat)
        setKaggleSearchTerm(cat)
    }

    const handleSearchReset = () => {
        setKaggleQuery('')
        setKaggleSearchTerm('')
    }

    /* ── AI synthetic helpers ── */
    const handleAiSchema = async (prompt) => {
        if (!prompt?.trim()) return
        setAiLoading(true)
        setAiError(null)
        setAiStage('prompting')
        try {
            const res = await datasetsApi.aiSchema(prompt.trim())
            const schema = res?.data?.schema
            if (schema?.error) throw new Error(schema.error)
            setAiSchema(schema)
            setAiStage('schema')
        } catch (err) {
            setAiError(err?.response?.data?.error || err.message || 'Schema generation failed')
            setAiStage('idea')
        } finally {
            setAiLoading(false)
        }
    }

    const handleAiGenerate = async (save = false) => {
        if (!aiSchema?.columns) return
        setAiLoading(true)
        setAiError(null)
        setAiStage('generating')
        try {
            const res = await datasetsApi.aiGenerate({
                name: aiSchema.name || 'ai_dataset',
                num_rows: aiRows,
                columns: aiSchema.columns,
                save
            })
            if (save && res?.data?.saved) {
                queryClient.invalidateQueries({ queryKey: ['datasets'] })
                setAiStage('saved')
                showToast('success', res?.data?.message || 'Dataset saved!')
            } else {
                setAiPreview(res?.data)
                setAiStage('preview')
            }
        } catch (err) {
            setAiError(err?.response?.data?.error || 'Generation failed')
            setAiStage('schema')
        } finally {
            setAiLoading(false)
        }
    }

    const handleAiReset = () => {
        setAiStage('idea')
        setAiPrompt('')
        setAiSchema(null)
        setAiPreview(null)
        setAiError(null)
        setAiRefinePrompt('')
    }

    const handleAiRefine = () => {
        const newPrompt = aiRefinePrompt.trim() || aiPrompt
        setAiRefinePrompt('')
        handleAiSchema(newPrompt)
    }

    const handleSaveKaggle = (ds) => {
        setSavingRef(ds.ref)
        kaggleDownloadMutation.mutate({ ref: ds.ref, name: ds.title })
    }

    /* ════════════════════════════════════════════════
       RENDER
       ════════════════════════════════════════════════ */
    return (
        <div className="ds-page">

            {/* ── Header ── */}
            <header className="ds-header">
                <div className="ds-header-left">
                    <Database size={22} />
                    <h1>Datasets</h1>
                </div>
                <div className="ds-header-actions">
                    <button className="ds-btn ds-btn-secondary" onClick={() => setShowUpload(true)}>
                        <Upload size={16} /> Upload
                    </button>
                </div>
            </header>

            {/* ── Tabs ── */}
            <nav className="ds-tabs">
                <button
                    className={`ds-tab ${activeTab === 'library' ? 'active' : ''}`}
                    onClick={() => setActiveTab('library')}
                >
                    <FolderUp size={15} /> My Library
                    {datasets.length > 0 && <span className="ds-tab-count">{datasets.length}</span>}
                </button>
                <button
                    className={`ds-tab ${activeTab === 'search' ? 'active' : ''}`}
                    onClick={() => setActiveTab('search')}
                >
                    <Search size={15} /> Search
                </button>
                <button
                    className={`ds-tab ${activeTab === 'generate' ? 'active' : ''}`}
                    onClick={() => setActiveTab('generate')}
                >
                    <Wand2 size={15} /> Generate
                </button>
            </nav>

            {/* ════════ MY LIBRARY TAB ════════ */}
            {activeTab === 'library' && (
                <section className="ds-section">
                    {isLoading ? (
                        <div className="ds-loading">
                            <Loader2 size={28} className="spin" />
                            <p>Loading datasets…</p>
                        </div>
                    ) : datasets.length > 0 ? (
                        <div className="ds-table-wrap">
                            <table className="ds-table">
                                <thead>
                                    <tr>
                                        <th>Name</th>
                                        <th>Type</th>
                                        <th>Size</th>
                                        <th>Rows</th>
                                        <th>Status</th>
                                        <th></th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {datasets.map((ds) => (
                                        <tr key={ds.id} onClick={() => setSelectedDataset(ds)}>
                                            <td className="ds-cell-name">
                                                <FileText size={18} />
                                                <span>{ds.name}</span>
                                            </td>
                                            <td><span className="ds-badge badge-type">{ds.file_type?.toUpperCase()}</span></td>
                                            <td>{formatFileSize(ds.file_size)}</td>
                                            <td>{ds.num_rows ? ds.num_rows.toLocaleString() : '—'}</td>
                                            <td>{getStatusBadge(ds.profile_status)}</td>
                                            <td>
                                                <button
                                                    className="ds-icon-btn"
                                                    title="Delete"
                                                    onClick={(e) => { e.stopPropagation(); deleteMutation.mutate(ds.id) }}
                                                >
                                                    <Trash2 size={14} />
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ) : (
                        <div className="ds-empty">
                            <FolderUp size={40} />
                            <h3>No datasets yet</h3>
                            <p>Upload a file or search online to get started.</p>
                            <div className="ds-empty-actions">
                                <button className="ds-btn ds-btn-primary" onClick={() => setShowUpload(true)}>
                                    <Upload size={16} /> Upload
                                </button>
                                <button className="ds-btn ds-btn-secondary" onClick={() => setActiveTab('search')}>
                                    <Search size={16} /> Search Online
                                </button>
                            </div>
                        </div>
                    )}
                </section>
            )}

            {/* ════════ KAGGLE SEARCH TAB ════════ */}
            {activeTab === 'search' && (
                <section className="ds-section">
                    <form className="ds-search-bar" onSubmit={handleKaggleSearch}>
                        <Search size={16} className="ds-search-icon" />
                        <input
                            type="text"
                            placeholder="Search datasets…"
                            value={kaggleQuery}
                            onChange={(e) => setKaggleQuery(e.target.value)}
                            autoFocus
                        />
                        {kaggleSearchTerm && (
                            <button type="button" className="ds-btn ds-btn-ghost" onClick={handleSearchReset} title="Reset search">
                                <X size={15} />
                            </button>
                        )}
                        <button type="submit" className="ds-btn ds-btn-primary" disabled={!kaggleQuery.trim() || kaggleFetching}>
                            {kaggleFetching ? <Loader2 size={15} className="spin" /> : 'Search'}
                        </button>
                    </form>

                    <div className="ds-search-results-area">
                        {kaggleLoading || kaggleFetching ? (
                            <div className="ds-loading">
                                <Loader2 size={28} className="spin" />
                                <p>Searching datasets…</p>
                            </div>
                        ) : kaggleSearchTerm && kaggleResults.length === 0 ? (
                            <div className="ds-empty">
                                <Search size={40} />
                                <h3>No results</h3>
                                <p>Try a different search term.</p>
                            </div>
                        ) : kaggleResults.length > 0 ? (
                            <div className="ds-kaggle-grid-wrapper">
                                {kaggleIsLimited && (
                                    <div className="ds-limit-banner">
                                        <Info size={16} />
                                        <span>Showing top 5 results on the <strong>Free Tier</strong>. Upgrade to see all results.</span>
                                        <button className="ds-btn ds-btn-secondary ds-btn-sm" onClick={() => window.location.href='/pricing'}>Upgrade</button>
                                    </div>
                                )}
                                <div className="ds-kaggle-grid">
                                    {kaggleResults.map((ds) => (
                                    <div key={ds.ref} className="ds-kaggle-card">
                                        <div className="ds-kaggle-card-body">
                                            <h3 className="ds-kaggle-title">{ds.title}</h3>
                                            {ds.subtitle && <p className="ds-kaggle-sub">{ds.subtitle}</p>}
                                            <div className="ds-kaggle-meta">
                                                <span>{formatFileSize(ds.totalBytes)}</span>
                                                <span>{(ds.downloadCount || 0).toLocaleString()} downloads</span>
                                                <span>★ {ds.usabilityRating}</span>
                                            </div>
                                        </div>
                                        <div className="ds-kaggle-card-actions">
                                            <a href={ds.url} target="_blank" rel="noopener noreferrer" className="ds-btn ds-btn-ghost">
                                                <ExternalLink size={14} /> View
                                            </a>
                                            <button
                                                className="ds-btn ds-btn-primary"
                                                onClick={() => handleSaveKaggle(ds)}
                                                disabled={savingRef === ds.ref}
                                            >
                                                {savingRef === ds.ref
                                                    ? <><Loader2 size={14} className="spin" /> Saving…</>
                                                    : <><Download size={14} /> Save to Library</>}
                                            </button>
                                        </div>
                                    </div>
                                ))}
                                </div>
                            </div>
                        ) : (
                            <div className="ds-search-hint">
                                <Search size={32} />
                                <p>Search for any dataset and save it directly to your library.</p>

                                <div className="ds-categories-section">
                                    <span className="ds-categories-title">Quick Discovery</span>
                                    <div className="ds-categories-grid">
                                        {KAG_CATEGORIES.map(cat => (
                                            <div
                                                key={cat}
                                                className="ds-category-chip"
                                                onClick={() => handleCategoryClick(cat)}
                                            >
                                                {cat}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </section>
            )}

            {/* ════════ AI GENERATE TAB ════════ */}
            {activeTab === 'generate' && (
                <section className="ds-section">
                    {isFree ? (
                        <div className="ds-empty">
                            <Lock size={48} style={{ color: 'var(--text-muted)' }} />
                            <h3>Data Generation Locked</h3>
                            <p>Synthetic AI data generation is only available on Pro and Advance tiers.</p>
                            <button className="ds-btn ds-btn-primary" onClick={() => window.location.href='/pricing'}>
                                Upgrade Plan
                            </button>
                        </div>
                    ) : (
                    <div className="ds-ai-container">
                        {/* ── Step Indicator ── */}
                        <div className="ds-ai-steps">
                            {[['idea', 'Describe', MessageSquare], ['schema', 'Schema', Table], ['preview', 'Preview', Eye], ['saved', 'Saved', CheckCircle]].map(([key, label, Icon], idx) => (
                                <div key={key} className={`ds-ai-step ${aiStage === key || (['prompting'].includes(aiStage) && key === 'idea') || (['generating'].includes(aiStage) && key === 'schema') ? 'active' : ''} ${['schema', 'preview', 'saved'].indexOf(aiStage) >= ['schema', 'preview', 'saved'].indexOf(key) && ['schema', 'preview', 'saved'].includes(aiStage) && ['schema', 'preview', 'saved'].includes(key) ? 'done' : ''}`}>
                                    <div className="ds-ai-step-dot"><Icon size={14} /></div>
                                    <span>{label}</span>
                                    {idx < 3 && <div className="ds-ai-step-line" />}
                                </div>
                            ))}
                        </div>

                        {/* ── Error Banner ── */}
                        {aiError && (
                            <div className="ds-ai-error">
                                <AlertCircle size={16} />
                                <span>{aiError}</span>
                                <button onClick={() => setAiError(null)}><X size={14} /></button>
                            </div>
                        )}

                        {/* ═══ STAGE 1: IDEA ═══ */}
                        {(aiStage === 'idea' || aiStage === 'prompting') && (
                            <div className="ds-ai-idea">
                                <div className="ds-ai-idea-header">
                                    <Sparkles size={24} />
                                    <div>
                                        <h2>Describe Your Dataset</h2>
                                        <p>Tell the AI what kind of data you need — it will design the schema for you.</p>
                                    </div>
                                </div>

                                <div className="ds-ai-prompt-box">
                                    <textarea
                                        value={aiPrompt}
                                        onChange={e => setAiPrompt(e.target.value)}
                                        placeholder="e.g. Student performance dataset with demographics, study habits, and exam scores for 500 students across different departments..."
                                        rows={4}
                                        disabled={aiStage === 'prompting'}
                                    />
                                    <div className="ds-ai-prompt-footer">
                                        <div className="ds-ai-row-config">
                                            <label>Rows:</label>
                                            <input
                                                type="number"
                                                value={aiRows}
                                                onChange={e => setAiRows(Math.max(10, Math.min(100000, Number(e.target.value))))}
                                                min={10} max={100000}
                                            />
                                        </div>
                                        <button
                                            className="ds-btn ds-btn-primary"
                                            onClick={() => handleAiSchema(aiPrompt)}
                                            disabled={!aiPrompt.trim() || aiStage === 'prompting'}
                                        >
                                            {aiStage === 'prompting'
                                                ? <><Loader2 size={15} className="spin" /> AI is thinking…</>
                                                : <><Sparkles size={15} /> Generate Schema</>}
                                        </button>
                                    </div>
                                </div>

                                {/* Quick ideas */}
                                <div className="ds-ai-suggestions">
                                    <span className="ds-ai-suggestions-label">Quick Ideas:</span>
                                    {[
                                        'E-commerce customer transactions with product categories',
                                        'Hospital patient records with diagnosis and treatment',
                                        'Employee performance data with department and salary',
                                        'Student exam results with demographics and study hours',
                                        'IoT sensor data with temperature humidity and pressure',
                                        'Real estate listings with price location and features'
                                    ].map(idea => (
                                        <button key={idea} className="ds-ai-idea-chip" onClick={() => { setAiPrompt(idea); handleAiSchema(idea) }}>
                                            {idea}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* ═══ STAGE 2: SCHEMA REVIEW ═══ */}
                        {(aiStage === 'schema' || aiStage === 'generating') && aiSchema && (
                            <div className="ds-ai-schema">
                                <div className="ds-ai-schema-header">
                                    <div>
                                        <h2>{aiSchema.name || 'Generated Schema'}</h2>
                                        <p>{aiSchema.description}</p>
                                    </div>
                                    <span className="ds-ai-col-count">{aiSchema.columns?.length} columns</span>
                                </div>

                                <div className="ds-ai-schema-grid">
                                    {aiSchema.columns?.map((col, i) => (
                                        <div key={i} className="ds-ai-schema-card">
                                            <div className="ds-ai-schema-card-top">
                                                <span className="ds-ai-col-name">{col.name}</span>
                                                <span className={`ds-ai-col-type-badge type-${col.type}`}>{col.type}</span>
                                            </div>
                                            <p className="ds-ai-col-desc">{col.description}</p>
                                            {col.sample_values && (
                                                <div className="ds-ai-samples">
                                                    {col.sample_values.slice(0, 3).map((v, j) => (
                                                        <code key={j}>{String(v)}</code>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>

                                {/* Refine prompt */}
                                <div className="ds-ai-refine">
                                    <input
                                        type="text"
                                        value={aiRefinePrompt}
                                        onChange={e => setAiRefinePrompt(e.target.value)}
                                        placeholder="Modify: add a GPA column, remove age, change categories..."
                                        onKeyDown={e => e.key === 'Enter' && handleAiRefine()}
                                    />
                                    <button className="ds-btn ds-btn-secondary" onClick={handleAiRefine} disabled={aiLoading}>
                                        <RefreshCw size={14} /> Re-generate
                                    </button>
                                </div>

                                <div className="ds-ai-schema-actions">
                                    <button className="ds-btn ds-btn-ghost" onClick={handleAiReset}>
                                        <RotateCcw size={15} /> Start Over
                                    </button>
                                    <div className="ds-ai-schema-actions-right">
                                        <div className="ds-ai-row-config">
                                            <label>Rows:</label>
                                            <input
                                                type="number"
                                                value={aiRows}
                                                onChange={e => setAiRows(Math.max(10, Math.min(100000, Number(e.target.value))))}
                                                min={10} max={100000}
                                            />
                                        </div>
                                        <button
                                            className="ds-btn ds-btn-primary"
                                            onClick={() => handleAiGenerate(false)}
                                            disabled={aiLoading}
                                        >
                                            {aiStage === 'generating'
                                                ? <><Loader2 size={15} className="spin" /> Generating data…</>
                                                : <><Wand2 size={15} /> Generate Preview</>}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* ═══ STAGE 3: PREVIEW ═══ */}
                        {aiStage === 'preview' && aiPreview && (
                            <div className="ds-ai-preview">
                                <div className="ds-ai-preview-header">
                                    <div>
                                        <h2>Data Preview</h2>
                                        <p>Showing first {aiPreview.preview?.length || 0} of {aiPreview.total_rows?.toLocaleString()} rows</p>
                                    </div>
                                    <div className="ds-ai-preview-actions">
                                        <button className="ds-btn ds-btn-ghost" onClick={() => setAiStage('schema')}>
                                            <RotateCcw size={14} /> Back to Schema
                                        </button>
                                        <button className="ds-btn ds-btn-secondary" onClick={() => handleAiGenerate(false)} disabled={aiLoading}>
                                            <RefreshCw size={14} /> Re-generate
                                        </button>
                                        <button className="ds-btn ds-btn-primary" onClick={() => handleAiGenerate(true)} disabled={aiLoading}>
                                            {aiLoading ? <><Loader2 size={15} className="spin" /> Saving…</> : <><Save size={15} /> Save to Library</>}
                                        </button>
                                    </div>
                                </div>

                                <div className="ds-ai-table-wrap">
                                    <table className="ds-ai-table">
                                        <thead>
                                            <tr>
                                                {aiPreview.columns?.map(col => <th key={col}>{col}</th>)}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {aiPreview.preview?.map((row, i) => (
                                                <tr key={i}>
                                                    {aiPreview.columns?.map(col => (
                                                        <td key={col}>{row[col] === true ? '✓' : row[col] === false ? '✗' : String(row[col] ?? '')}</td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>

                                {/* Refine prompt for data */}
                                <div className="ds-ai-refine">
                                    <input
                                        type="text"
                                        value={aiRefinePrompt}
                                        onChange={e => setAiRefinePrompt(e.target.value)}
                                        placeholder="Not happy? Describe changes: make values more realistic, add more variety..."
                                        onKeyDown={e => { if (e.key === 'Enter') { handleAiRefine() } }}
                                    />
                                    <button className="ds-btn ds-btn-secondary" onClick={handleAiRefine} disabled={aiLoading}>
                                        <RefreshCw size={14} /> Modify Schema
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* ═══ STAGE 4: SAVED ═══ */}
                        {aiStage === 'saved' && (
                            <div className="ds-ai-saved">
                                <CheckCircle size={48} />
                                <h2>Dataset Saved!</h2>
                                <p>Your AI-generated dataset is now in your library.</p>
                                <div className="ds-ai-saved-actions">
                                    <button className="ds-btn ds-btn-secondary" onClick={handleAiReset}>
                                        <Sparkles size={15} /> Create Another
                                    </button>
                                    <button className="ds-btn ds-btn-primary" onClick={() => { handleAiReset(); setActiveTab('library') }}>
                                        <ArrowRight size={15} /> Go to Library
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                    )}
                </section>
            )}

            {/* ════════ Upload Modal ════════ */}
            {showUpload && (
                <div className="ds-overlay" onClick={() => !uploading && setShowUpload(false)}>
                    <div className="ds-modal" onClick={e => e.stopPropagation()}>
                        <div className="ds-modal-header">
                            <h2>Upload Dataset</h2>
                            <button className="ds-icon-btn" onClick={() => !uploading && setShowUpload(false)} disabled={uploading}>
                                <X size={18} />
                            </button>
                        </div>
                        <div className="ds-modal-body">
                            <div
                                className="ds-upload-zone"
                                onClick={() => !uploading && fileInputRef.current?.click()}
                            >
                                {uploading ? (
                                    <div className="ds-upload-progress">
                                        <div className="ds-progress-track">
                                            <div className="ds-progress-fill" style={{ width: `${uploadProgress}%` }}></div>
                                        </div>
                                        <p>{uploadProgress}% uploaded</p>
                                    </div>
                                ) : (
                                    <>
                                        <Upload size={36} />
                                        <h3>Drop files here or click to upload</h3>
                                        <p>CSV, Excel, Images, ZIP</p>
                                    </>
                                )}
                            </div>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".csv,.xlsx,.xls,.jpg,.jpeg,.png,.zip"
                                onChange={handleFileSelect}
                                style={{ display: 'none' }}
                            />
                        </div>
                    </div>
                </div>
            )}

            {/* ════════ Dataset Details Modal ════════ */}
            {selectedDataset && (
                <div className="ds-overlay" onClick={() => setSelectedDataset(null)}>
                    <div className="ds-modal ds-modal-lg" onClick={e => e.stopPropagation()}>
                        <div className="ds-modal-header">
                            <h2>{selectedDataset.name}</h2>
                            <button className="ds-icon-btn" onClick={() => setSelectedDataset(null)}>
                                <X size={18} />
                            </button>
                        </div>
                        <div className="ds-modal-body">
                            <div className="ds-detail-grid">
                                <div className="ds-detail-row"><span>Type</span><span>{selectedDataset.file_type?.toUpperCase()}</span></div>
                                <div className="ds-detail-row"><span>Size</span><span>{formatFileSize(selectedDataset.file_size)}</span></div>
                                <div className="ds-detail-row"><span>Rows</span><span>{selectedDataset.num_rows?.toLocaleString() || '—'}</span></div>
                                <div className="ds-detail-row"><span>Columns</span><span>{selectedDataset.num_columns || '—'}</span></div>
                                <div className="ds-detail-row"><span>Status</span>{getStatusBadge(selectedDataset.profile_status)}</div>
                                <div className="ds-detail-row"><span>Uploaded</span><span>{new Date(selectedDataset.created_at).toLocaleDateString()}</span></div>
                            </div>

                            {selectedDataset.column_info && (
                                <div className="ds-columns-section">
                                    <h4>Columns</h4>
                                    <div className="ds-columns-list">
                                        {Object.entries(selectedDataset.column_info).map(([name, info]) => (
                                            <div key={name} className="ds-column-item">
                                                <span>{name}</span>
                                                <code>{info.dtype}</code>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                        <div className="ds-modal-footer">
                            <button className="ds-btn ds-btn-danger" onClick={() => deleteMutation.mutate(selectedDataset.id)}>
                                <Trash2 size={15} /> Delete
                            </button>
                            <button
                                className="ds-btn btn-outline"
                                onClick={() => window.open(`/datasets/${selectedDataset.id}/view`, '_blank')}
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                    border: '1px solid rgba(16, 185, 129, 0.3)',
                                    color: '#10b981'
                                }}
                            >
                                <ExternalLink size={16} />
                                View Dataset
                            </button>
                            <button className="ds-btn ds-btn-primary" onClick={() => {
                                window.location.href = `/training?dataset=${selectedDataset.id}`
                            }}>
                                Train Model <ArrowRight size={15} />
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* ════════ Toast ════════ */}
            {toast && (
                <div className={`ds-toast ds-toast-${toast.type}`}>
                    {toast.type === 'success' && <CheckCircle size={16} />}
                    {toast.type === 'info' && <Info size={16} />}
                    {toast.type === 'error' && <AlertCircle size={16} />}
                    <span>{toast.message}</span>
                    <button className="ds-toast-close" onClick={() => setToast(null)}>
                        <X size={12} />
                    </button>
                </div>
            )}
        </div>
    )
}
