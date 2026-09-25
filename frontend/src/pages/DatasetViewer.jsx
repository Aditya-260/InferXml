import { useState, useEffect, useMemo } from 'react'
import { useParams } from 'react-router-dom'
import {
    Database,
    Search,
    ChevronUp,
    ChevronDown,
    ChevronLeft,
    ChevronRight,
    Hash,
    Type,
    AlertCircle,
    Loader,
    FileSpreadsheet,
    ArrowUpDown,
    Info,
    BarChart3,
    LayoutList
} from 'lucide-react'
import { datasetsApi } from '../services/api'
import './DatasetViewer.css'

const ROWS_PER_PAGE = 50

export default function DatasetViewer() {
    const { datasetId } = useParams()
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [searchQuery, setSearchQuery] = useState('')
    const [sortColumn, setSortColumn] = useState(null)
    const [sortDirection, setSortDirection] = useState('asc')
    const [currentPage, setCurrentPage] = useState(1)
    const [showStats, setShowStats] = useState(false)

    useEffect(() => {
        document.title = 'Dataset Viewer — InferX-ML'
        const fetchData = async () => {
            try {
                setLoading(true)
                const res = await datasetsApi.preview(datasetId)
                setData(res.data)
                document.title = `${res.data.name} — InferX-ML`
            } catch (err) {
                setError(err?.response?.data?.error || 'Failed to load dataset')
            } finally {
                setLoading(false)
            }
        }
        fetchData()
    }, [datasetId])

    // Filter rows by search query
    const filteredRows = useMemo(() => {
        if (!data?.rows) return []
        if (!searchQuery.trim()) return data.rows
        const q = searchQuery.toLowerCase()
        return data.rows.filter(row =>
            Object.values(row).some(val =>
                val !== null && val !== undefined && String(val).toLowerCase().includes(q)
            )
        )
    }, [data?.rows, searchQuery])

    // Sort rows
    const sortedRows = useMemo(() => {
        if (!sortColumn) return filteredRows
        return [...filteredRows].sort((a, b) => {
            const aVal = a[sortColumn]
            const bVal = b[sortColumn]
            if (aVal === null || aVal === undefined) return 1
            if (bVal === null || bVal === undefined) return -1
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return sortDirection === 'asc' ? aVal - bVal : bVal - aVal
            }
            const cmp = String(aVal).localeCompare(String(bVal))
            return sortDirection === 'asc' ? cmp : -cmp
        })
    }, [filteredRows, sortColumn, sortDirection])

    // Pagination
    const totalPages = Math.max(1, Math.ceil(sortedRows.length / ROWS_PER_PAGE))
    const paginatedRows = sortedRows.slice(
        (currentPage - 1) * ROWS_PER_PAGE,
        currentPage * ROWS_PER_PAGE
    )

    useEffect(() => { setCurrentPage(1) }, [searchQuery, sortColumn, sortDirection])

    const handleSort = (colName) => {
        if (sortColumn === colName) {
            setSortDirection(d => d === 'asc' ? 'desc' : 'asc')
        } else {
            setSortColumn(colName)
            setSortDirection('asc')
        }
    }

    const getColumnTypeIcon = (col) => {
        if (col.type === 'numeric') return <Hash size={13} className="dv-col-type-icon dv-type-numeric" />
        if (col.type === 'categorical') return <Type size={13} className="dv-col-type-icon dv-type-categorical" />
        return <Type size={13} className="dv-col-type-icon" />
    }

    const formatCellValue = (val) => {
        if (val === null || val === undefined) return <span className="dv-null">null</span>
        if (typeof val === 'number') {
            if (Number.isInteger(val)) return val.toLocaleString()
            return val.toLocaleString(undefined, { maximumFractionDigits: 4 })
        }
        return String(val)
    }

    if (loading) {
        return (
            <div className="dv-page">
                <div className="dv-loading">
                    <Loader size={36} className="dv-spinner" />
                    <p>Loading dataset…</p>
                </div>
            </div>
        )
    }

    if (error) {
        return (
            <div className="dv-page">
                <div className="dv-error">
                    <AlertCircle size={36} />
                    <h2>Could not load dataset</h2>
                    <p>{error}</p>
                </div>
            </div>
        )
    }

    if (!data) return null

    return (
        <div className="dv-page">
            {/* ── Header ── */}
            <header className="dv-header">
                <div className="dv-header-left">
                    <div className="dv-header-icon">
                        <FileSpreadsheet size={24} />
                    </div>
                    <div>
                        <h1 className="dv-title">{data.name}</h1>
                        <div className="dv-meta">
                            <span className="dv-badge dv-badge-type">{data.file_type?.toUpperCase()}</span>
                            <span><LayoutList size={14} /> {data.total_rows.toLocaleString()} rows</span>
                            <span><Database size={14} /> {data.total_columns} columns</span>
                            {data.truncated && (
                                <span className="dv-badge dv-badge-warn">
                                    <Info size={12} /> Showing first 500 rows
                                </span>
                            )}
                        </div>
                    </div>
                </div>
                <div className="dv-header-right">
                    <button
                        className={`dv-stats-toggle ${showStats ? 'dv-stats-active' : ''}`}
                        onClick={() => setShowStats(s => !s)}
                    >
                        <BarChart3 size={16} />
                        Column Stats
                    </button>
                </div>
            </header>

            {/* ── Column Stats Panel ── */}
            {showStats && (
                <div className="dv-stats-panel">
                    <h3 className="dv-stats-title">Column Statistics</h3>
                    <div className="dv-stats-grid">
                        {data.columns.map(col => (
                            <div key={col.name} className="dv-stat-card">
                                <div className="dv-stat-header">
                                    {getColumnTypeIcon(col)}
                                    <span className="dv-stat-name">{col.name}</span>
                                    <span className="dv-stat-dtype">{col.dtype}</span>
                                </div>
                                <div className="dv-stat-body">
                                    <div className="dv-stat-row">
                                        <span>Nulls</span>
                                        <span>{col.null_count}</span>
                                    </div>
                                    <div className="dv-stat-row">
                                        <span>Unique</span>
                                        <span>{col.unique_count}</span>
                                    </div>
                                    {col.type === 'numeric' && (
                                        <>
                                            <div className="dv-stat-row">
                                                <span>Min</span>
                                                <span>{col.min?.toLocaleString(undefined, { maximumFractionDigits: 4 }) ?? '—'}</span>
                                            </div>
                                            <div className="dv-stat-row">
                                                <span>Max</span>
                                                <span>{col.max?.toLocaleString(undefined, { maximumFractionDigits: 4 }) ?? '—'}</span>
                                            </div>
                                            <div className="dv-stat-row">
                                                <span>Mean</span>
                                                <span>{col.mean?.toLocaleString(undefined, { maximumFractionDigits: 4 }) ?? '—'}</span>
                                            </div>
                                        </>
                                    )}
                                    {col.type === 'categorical' && col.top_values && (
                                        <div className="dv-stat-top">
                                            <span className="dv-stat-top-label">Top values</span>
                                            {Object.entries(col.top_values).slice(0, 3).map(([val, count]) => (
                                                <div key={val} className="dv-stat-row dv-stat-row-sm">
                                                    <span className="dv-stat-val-name">{val}</span>
                                                    <span>{count}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* ── Toolbar ── */}
            <div className="dv-toolbar">
                <div className="dv-search-wrap">
                    <Search size={16} className="dv-search-icon" />
                    <input
                        type="text"
                        className="dv-search"
                        placeholder="Search across all columns…"
                        value={searchQuery}
                        onChange={e => setSearchQuery(e.target.value)}
                    />
                    {searchQuery && (
                        <span className="dv-search-count">
                            {filteredRows.length.toLocaleString()} match{filteredRows.length !== 1 ? 'es' : ''}
                        </span>
                    )}
                </div>
                <div className="dv-page-info">
                    Page {currentPage} of {totalPages}
                </div>
            </div>

            {/* ── Data Table ── */}
            <div className="dv-table-wrap">
                <table className="dv-table">
                    <thead>
                        <tr>
                            <th className="dv-th-index">#</th>
                            {data.columns.map(col => (
                                <th
                                    key={col.name}
                                    className={`dv-th ${sortColumn === col.name ? 'dv-th-sorted' : ''}`}
                                    onClick={() => handleSort(col.name)}
                                >
                                    <div className="dv-th-content">
                                        {getColumnTypeIcon(col)}
                                        <span>{col.name}</span>
                                        {sortColumn === col.name ? (
                                            sortDirection === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />
                                        ) : (
                                            <ArrowUpDown size={12} className="dv-sort-hint" />
                                        )}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {paginatedRows.length > 0 ? paginatedRows.map((row, idx) => (
                            <tr key={idx}>
                                <td className="dv-td-index">{(currentPage - 1) * ROWS_PER_PAGE + idx + 1}</td>
                                {data.columns.map(col => (
                                    <td key={col.name} className={col.type === 'numeric' ? 'dv-td-num' : ''}>
                                        {formatCellValue(row[col.name])}
                                    </td>
                                ))}
                            </tr>
                        )) : (
                            <tr>
                                <td colSpan={data.columns.length + 1} className="dv-no-results">
                                    No rows match your search
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>

            {/* ── Pagination ── */}
            {totalPages > 1 && (
                <div className="dv-pagination">
                    <button
                        className="dv-page-btn"
                        disabled={currentPage <= 1}
                        onClick={() => setCurrentPage(1)}
                    >
                        First
                    </button>
                    <button
                        className="dv-page-btn"
                        disabled={currentPage <= 1}
                        onClick={() => setCurrentPage(p => p - 1)}
                    >
                        <ChevronLeft size={16} /> Prev
                    </button>
                    <div className="dv-page-numbers">
                        {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                            let page
                            if (totalPages <= 5) {
                                page = i + 1
                            } else if (currentPage <= 3) {
                                page = i + 1
                            } else if (currentPage >= totalPages - 2) {
                                page = totalPages - 4 + i
                            } else {
                                page = currentPage - 2 + i
                            }
                            return (
                                <button
                                    key={page}
                                    className={`dv-page-num ${page === currentPage ? 'dv-page-current' : ''}`}
                                    onClick={() => setCurrentPage(page)}
                                >
                                    {page}
                                </button>
                            )
                        })}
                    </div>
                    <button
                        className="dv-page-btn"
                        disabled={currentPage >= totalPages}
                        onClick={() => setCurrentPage(p => p + 1)}
                    >
                        Next <ChevronRight size={16} />
                    </button>
                    <button
                        className="dv-page-btn"
                        disabled={currentPage >= totalPages}
                        onClick={() => setCurrentPage(totalPages)}
                    >
                        Last
                    </button>
                </div>
            )}
        </div>
    )
}
