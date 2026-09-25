import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
    Sparkles,
    ExternalLink,
    ArrowLeft,
    Box,
    FileSpreadsheet
} from 'lucide-react'
import { modelsApi } from '../services/api'
import './Predictions.css'

// Streamlit URL - runs on port 8501
const STREAMLIT_URL = window.location.hostname === 'localhost'
    ? 'http://localhost:8501'
    : `http://${window.location.hostname}:8501`

export default function Predictions() {
    const { modelId } = useParams()
    const navigate = useNavigate()
    const [selectedModel, setSelectedModel] = useState(modelId || '')
    const [iframeLoading, setIframeLoading] = useState(true)
    const [currentPage, setCurrentPage] = useState(1)

    const MODELS_PER_PAGE = 12

    const { data: modelsData, isLoading } = useQuery({
        queryKey: ['models'],
        queryFn: () => modelsApi.list()
    })

    const models = modelsData?.data?.models || []
    const packagedModels = models.filter((m) => m.has_package)
    const totalPages = Math.max(1, Math.ceil(packagedModels.length / MODELS_PER_PAGE))
    const pageStart = (currentPage - 1) * MODELS_PER_PAGE
    const paginatedModels = packagedModels.slice(pageStart, pageStart + MODELS_PER_PAGE)

    // Get selected model details
    const currentModel = models.find(m => String(m.id) === String(selectedModel))

    // When modelId from URL changes, update selectedModel
    useEffect(() => {
        if (modelId) {
            setSelectedModel(modelId)
        }
    }, [modelId])

    useEffect(() => {
        if (currentPage > totalPages) {
            setCurrentPage(totalPages)
        }
    }, [currentPage, totalPages])

    const handleModelSelect = (id) => {
        setSelectedModel(id)
        setIframeLoading(true)
        // Update URL
        navigate(`/predictions/${id}`)
    }

    // If no model selected, show model selector
    if (!selectedModel) {
        return (
            <div className="predictions-page">
                <div className="page-header">
                    <h1>Make Predictions</h1>
                    <p>Select a trained model to make predictions using the Streamlit UI</p>
                </div>

                {isLoading ? (
                    <div className="loading-state">
                        <div className="spinner"></div>
                        <p>Loading models...</p>
                    </div>
                ) : models.length > 0 ? (
                    <div className="model-selector card">
                        <h2>Select a Model</h2>
                        <div className="predictions-models-grid">
                            {paginatedModels.map((model) => (
                                <div
                                    key={model.id}
                                    className="model-option"
                                    onClick={() => handleModelSelect(model.id)}
                                >
                                    <div className="model-option-icon">
                                        <Sparkles size={24} />
                                    </div>
                                    <div className="model-option-info">
                                        <h3>{model.name}</h3>
                                        <span className="badge badge-info">
                                            {model.problem_type?.replace('_', ' ')}
                                        </span>
                                        {model.best_score && (
                                            <span className="model-score">
                                                Score: {(model.best_score * 100).toFixed(1)}%
                                            </span>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {totalPages > 1 && (
                            <div className="predictions-pagination">
                                <button
                                    className="btn btn-outline predictions-pagination-btn"
                                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                                    disabled={currentPage === 1}
                                >
                                    Previous
                                </button>
                                <span className="pagination-info">Page {currentPage} of {totalPages}</span>
                                <button
                                    className="btn btn-outline predictions-pagination-btn"
                                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                                    disabled={currentPage === totalPages}
                                >
                                    Next
                                </button>
                            </div>
                        )}

                        {packagedModels.length === 0 && (
                            <div className="no-models-message">
                                <Box size={48} className="empty-icon" />
                                <p>No models with prediction packages available.</p>
                                <p>Train a new model to enable predictions.</p>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="empty-state card">
                        <Box size={48} className="empty-icon" />
                        <h3>No models available</h3>
                        <p>Train your first model to start making predictions</p>
                    </div>
                )}
            </div>
        )
    }

    // Model is selected - show Streamlit iframe
    return (
        <div className="predictions-page predictions-fullscreen">
            <div className="prediction-header">
                <button
                    className="btn btn-secondary back-btn"
                    onClick={() => {
                        setSelectedModel('')
                        navigate('/predictions')
                    }}
                >
                    <ArrowLeft size={18} />
                    Back to Models
                </button>

                <div className="current-model-info">
                    <Sparkles size={20} />
                    <span>{currentModel?.name || `Model #${selectedModel}`}</span>
                    {currentModel?.best_score && (
                        <span className="badge badge-primary">
                            {(currentModel.best_score * 100).toFixed(1)}%
                        </span>
                    )}
                </div>

                <div className="prediction-header-actions">
                    {currentModel?.dataset_id && (
                        <button
                            className="btn btn-outline"
                            onClick={() => window.open(`/datasets/${currentModel.dataset_id}/view`, '_blank')}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                border: '1px solid rgba(16, 185, 129, 0.3)',
                                color: '#10b981',
                                fontSize: '0.85rem'
                            }}
                        >
                            <FileSpreadsheet size={15} />
                            View Dataset
                        </button>
                    )}
                    <a
                        href={`${STREAMLIT_URL}?model=${selectedModel}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn btn-outline open-external"
                    >
                        <ExternalLink size={16} />
                        Open in New Tab
                    </a>
                </div>
            </div>

            <div className="streamlit-container">
                {iframeLoading && (
                    <div className="iframe-loading">
                        <div className="spinner"></div>
                        <p>Loading prediction interface...</p>
                    </div>
                )}
                <iframe
                    src={`${STREAMLIT_URL}?model=${selectedModel}`}
                    title="Streamlit Predictions"
                    width="100%"
                    height="100%"
                    style={{
                        border: 'none',
                        borderRadius: '8px',
                        display: iframeLoading ? 'none' : 'block'
                    }}
                    onLoad={() => setIframeLoading(false)}
                />
            </div>
        </div>
    )
}
