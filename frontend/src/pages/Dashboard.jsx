import { useQuery } from '@tanstack/react-query'
import {
    Database,
    Cpu,
    Box,
    TrendingUp,
    ArrowUpRight,
    Clock,
    Sparkles,
    Trophy,
    Layers,
    Zap,
    ChevronRight
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { datasetsApi, modelsApi } from '../services/api'
import './Dashboard.css'

export default function Dashboard() {
    const { data: datasetsData, isLoading: isDatasetsLoading } = useQuery({
        queryKey: ['datasets'],
        queryFn: () => datasetsApi.list()
    })

    const { data: modelsData, isLoading: isModelsLoading } = useQuery({
        queryKey: ['models'],
        queryFn: () => modelsApi.list()
    })

    const datasets = datasetsData?.data?.datasets || []
    const models = modelsData?.data?.models || []

    // Sort models by latest created first
    const sortedModels = [...models].sort((a, b) => new Date(b.created_at) - new Date(a.created_at))

    // Calculate dynamic stats
    const bestScore = models.length > 0
        ? Math.max(...models.map(m => m.best_score || 0))
        : 0

    const stats = [
        {
            label: 'Total Datasets',
            value: datasets.length,
            subtitle: 'Uploaded data assets',
            icon: Database,
            color: 'blue',
            link: '/datasets'
        },
        {
            label: 'Trained Models',
            value: models.length,
            subtitle: 'Production ready',
            icon: Box,
            color: 'purple',
            link: '/models'
        },
        {
            label: 'Best Accuracy',
            value: models.length > 0 ? `${(bestScore * 100).toFixed(1)}%` : '—',
            subtitle: 'Peak model score',
            icon: Trophy,
            color: 'amber',
            link: '/models'
        },
        {
            label: 'System Status',
            value: 'Active',
            subtitle: 'AutoML node ready',
            icon: Cpu,
            color: 'green',
            link: '/training',
            isStatus: true
        }
    ]

    const getProblemTypeColor = (type) => {
        if (type?.includes('classification')) return 'purple'
        if (type === 'regression') return 'blue'
        if (type === 'clustering') return 'amber'
        if (type === 'timeseries') return 'green'
        return 'gray'
    }

    const formatProblemType = (type) => {
        if (!type) return 'Unknown'
        return type
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ')
    }

    const isLoading = isDatasetsLoading || isModelsLoading

    return (
        <div className="dashboard">
            {/* Ambient Hero Header */}
            <div className="dashboard-hero">
                <div className="hero-glow" />
                <div className="hero-content">
                    <span className="hero-badge">
                        <Sparkles size={12} /> ENTERPRISE AUTO-ML WORKSPACE
                    </span>
                    <h1>InferX-ML Workspace</h1>
                    <p className="hero-subtitle">
                        Train state-of-the-art machine learning models serverlessly, explore dataset profiles, 
                        and generate real-time predictive endpoints.
                    </p>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="stats-grid">
                {stats.map((stat) => {
                    const CardComponent = stat.isStatus ? 'div' : Link
                    const cardProps = stat.isStatus ? {} : { to: stat.link }
                    
                    return (
                        <CardComponent
                            {...cardProps}
                            key={stat.label}
                            className={`stat-card stat-${stat.color}`}
                        >
                            <div className="stat-card-header">
                                <div className="stat-icon-wrap">
                                    <stat.icon size={20} />
                                </div>
                                {!stat.isStatus && <ArrowUpRight className="stat-arrow" size={16} />}
                            </div>
                            <div className="stat-card-body">
                                <span className="stat-value">
                                    {stat.isStatus && <span className="status-dot-pulse" />}
                                    {stat.value}
                                </span>
                                <span className="stat-label">{stat.label}</span>
                                <span className="stat-subtitle">{stat.subtitle}</span>
                            </div>
                        </CardComponent>
                    )
                })}
            </div>

            {/* Main sections container */}
            <div className="dashboard-layout">
                <div className="dashboard-left">
                    <div className="section">
                        <div className="section-header">
                            <div className="section-title-wrap">
                                <Zap size={16} className="text-primary" />
                                <h2>Quick Engine Actions</h2>
                            </div>
                        </div>
                        <div className="actions-grid">
                            <Link to="/datasets" className="action-card">
                                <div className="action-card-left">
                                    <div className="action-avatar bg-blue-trans">
                                        <Database size={18} />
                                    </div>
                                    <div className="action-details">
                                        <h4>Upload Dataset</h4>
                                        <p className="text-gray-500">Ingest tabular or image files to auto-profile features.</p>
                                    </div>
                                </div>
                                <div className="action-card-right">
                                    <span className="action-badge-btn ab-blue">
                                        <span>Proceed</span>
                                        <ChevronRight size={12} />
                                    </span>
                                </div>
                            </Link>

                            <Link to="/training" className="action-card">
                                <div className="action-card-left">
                                    <div className="action-avatar bg-purple-trans">
                                        <Cpu size={18} />
                                    </div>
                                    <div className="action-details">
                                        <h4>Train Model</h4>
                                        <p className="text-gray-500">Initiate SMOTE-balanced automated model searches.</p>
                                    </div>
                                </div>
                                <div className="action-card-right">
                                    <span className="action-badge-btn ab-purple">
                                        <span>Configure</span>
                                        <ChevronRight size={12} />
                                    </span>
                                </div>
                            </Link>

                            <Link to="/predictions" className="action-card">
                                <div className="action-card-left">
                                    <div className="action-avatar bg-green-trans">
                                        <TrendingUp size={18} />
                                    </div>
                                    <div className="action-details">
                                        <h4>Inference API</h4>
                                        <p className="text-gray-500">Expose instant REST prediction endpoints from models.</p>
                                    </div>
                                </div>
                                <div className="action-card-right">
                                    <span className="action-badge-btn ab-green">
                                        <span>Execute</span>
                                        <ChevronRight size={12} />
                                    </span>
                                </div>
                            </Link>
                        </div>
                    </div>
                </div>

                {/* Right Panel: Recent Model Catalog */}
                <div className="dashboard-right">
                    <div className="section">
                        <div className="section-header">
                            <div className="section-title-wrap">
                                <Layers size={16} className="text-primary" />
                                <h2>Recent Model Catalog</h2>
                            </div>
                            {models.length > 0 && (
                                <Link to="/models" className="view-all">
                                    <span>View Catalog</span>
                                    <ChevronRight size={14} />
                                </Link>
                            )}
                        </div>

                        {isLoading ? (
                            <div className="loading-placeholder card">
                                <div className="spinner" />
                                <p>Updating engine status...</p>
                            </div>
                        ) : models.length > 0 ? (
                            <div className="models-list">
                                {sortedModels.slice(0, 3).map((model) => (
                                    <div key={model.id} className="model-item card">
                                        <div className="model-item-left">
                                            <div className="model-avatar">
                                                <Box size={18} />
                                            </div>
                                            <div className="model-details">
                                                <h4>{model.name}</h4>
                                                <div className="model-sub-details">
                                                    <span className={`problem-badge pb-${getProblemTypeColor(model.problem_type)}`}>
                                                        {formatProblemType(model.problem_type)}
                                                    </span>
                                                    <span className="dot-divider" />
                                                    <span className="best-model-name text-gray-500">
                                                        {model.best_model_name || 'Unsupervised'}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="model-item-right">
                                            <span className="badge badge-success">
                                                {(model.best_score * 100).toFixed(1)}%
                                            </span>
                                            <div className="model-time text-gray-500">
                                                <Clock size={12} />
                                                <span>{new Date(model.created_at).toLocaleDateString()}</span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="empty-state card">
                                <div className="empty-icon-wrap">
                                    <Box size={32} />
                                </div>
                                <h3>Model Vault is Empty</h3>
                                <p>You have not trained any models yet. Feed the AutoML engine a dataset to begin.</p>
                                <Link to="/training" className="btn btn-primary">
                                    <Cpu size={16} />
                                    Launch Training Session
                                </Link>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
