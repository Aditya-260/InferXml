import { useState, useRef, useEffect, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import {
    Cpu,
    Play,
    Target,
    CheckCircle,
    Clock,
    AlertCircle,
    Sparkles,
    Wand2,
    Brain,
    ChevronDown,
    Database,
    Zap,
    BarChart3,
    Terminal,
    ImageIcon,
    Settings,
    Download,
    TableProperties,
    RotateCcw
} from 'lucide-react'
import { datasetsApi, trainingApi, modelsApi } from '../services/api'
import LearningTimeline from '../components/LearningTimeline'
import { useTrainingStore } from '../store/trainingStore'
import './Training.css'

export default function Training() {
    const queryClient = useQueryClient()
    const [searchParams] = useSearchParams()
    const preselectedDataset = searchParams.get('dataset')

    const {
        currentJob, isTraining, setCurrentJob, setIsTraining,
        lastStatus, setLastStatus,
        experimentName, setExperimentName,
        selectedDataset, setSelectedDataset,
        targetColumn, setTargetColumn,
        goalDescription, setGoalDescription,
        configData, setConfigData,
        imageProblemType, setImageProblemType,
        aiPrompt, setAiPrompt,
        resetTraining
    } = useTrainingStore()

    useEffect(() => {
        if (preselectedDataset && preselectedDataset !== selectedDataset) {
            setSelectedDataset(preselectedDataset)
        }
    }, [preselectedDataset, selectedDataset, setSelectedDataset])

    const logsEndRef = useRef(null)

    // Advanced Configuration State
    const [showAdvanced, setShowAdvanced] = useState(false)
    const [activeRightTab, setActiveRightTab] = useState('progress') // progress | console

    // Resizable Panels State
    const [leftWidth, setLeftWidth] = useState(380) // Default 380px
    const [isDragging, setIsDragging] = useState(false)
    const layoutRef = useRef(null)

    const startDragging = useCallback((e) => {
        e.preventDefault()
        setIsDragging(true)
    }, [])

    useEffect(() => {
        if (!isDragging) return

        const handleMove = (e) => {
            if (!layoutRef.current) return
            const containerRect = layoutRef.current.getBoundingClientRect()
            const clientX = e.touches ? e.touches[0].clientX : e.clientX
            let newWidth = clientX - containerRect.left
            
            // Bounds check (min 300px, max viewport container width minus 320px)
            const minWidth = 300
            const maxWidth = Math.max(300, containerRect.width - 320)
            if (newWidth < minWidth) newWidth = minWidth
            if (newWidth > maxWidth) newWidth = maxWidth
            
            setLeftWidth(newWidth)
        }

        const handleUp = () => {
            setIsDragging(false)
        }

        window.addEventListener('mousemove', handleMove)
        window.addEventListener('mouseup', handleUp)
        window.addEventListener('touchmove', handleMove, { passive: true })
        window.addEventListener('touchend', handleUp)

        document.body.classList.add('is-dragging-split')

        return () => {
            window.removeEventListener('mousemove', handleMove)
            window.removeEventListener('mouseup', handleUp)
            window.removeEventListener('touchmove', handleMove)
            window.removeEventListener('touchend', handleUp)
            document.body.classList.remove('is-dragging-split')
        }
    }, [isDragging])

    // AI Prompt Analysis State
    const [aiAnalysis, setAiAnalysis] = useState(null)
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [analysisError, setAnalysisError] = useState(null)

    // Imbalance Detection State
    const [imbalanceDetected, setImbalanceDetected] = useState(false)
    const [autoBalance, setAutoBalance] = useState(false)
    const [isEnsembling, setIsEnsembling] = useState(false)
    const [headerInputs, setHeaderInputs] = useState([])
    const [headerFixError, setHeaderFixError] = useState(null)
    const [trainingStartError, setTrainingStartError] = useState(null)

    const { data: datasetsData } = useQuery({
        queryKey: ['datasets'],
        queryFn: () => datasetsApi.list()
    })

    const datasets = datasetsData?.data?.datasets || []
    const selectedDatasetInfo = datasets.find(d => d.id.toString() === selectedDataset)

    const { data: profileData } = useQuery({
        queryKey: ['dataset-profile', selectedDataset],
        queryFn: () => datasetsApi.getProfile(selectedDataset),
        enabled: !!selectedDataset
    })

    const {
        data: headerDetectionData,
        isFetching: isDetectingHeaders,
        refetch: refetchHeaderDetection
    } = useQuery({
        queryKey: ['dataset-header-detection', selectedDataset],
        queryFn: () => datasetsApi.detectHeaders(selectedDataset),
        enabled: !!selectedDataset && !!selectedDatasetInfo && selectedDatasetInfo.data_type !== 'image',
        refetchOnMount: 'always',
        retry: 1
    })

    const columns = profileData?.data?.column_info
        ? Object.keys(profileData.data.column_info)
        : []

    // Detect if this is an image dataset
    const isImageDataset = selectedDatasetInfo?.data_type === 'image'
    const imageClasses = isImageDataset && selectedDatasetInfo?.column_info
        ? selectedDatasetInfo.column_info
        : null
    const headerDetection = headerDetectionData?.data
    const needsHeaderFix = !!selectedDataset && !isImageDataset && headerDetection?.has_headers === false

    // Polling for status
    const { data: statusData, refetch: refetchStatus } = useQuery({
        queryKey: ['training-status', currentJob?.id],
        queryFn: () => trainingApi.getStatus(currentJob.id),
        enabled: !!currentJob,
        refetchInterval: isTraining ? 2000 : false
    })
    const cachedStatus = lastStatus?.experiment?.id === currentJob?.id ? lastStatus : null
    const latestTrainingData = statusData?.data || cachedStatus

    useEffect(() => {
        if (statusData?.data) {
            setLastStatus(statusData.data)
        }
    }, [statusData, setLastStatus])

    // Polling for logs
    const { data: logsData, refetch: refetchLogs } = useQuery({
        queryKey: ['training-logs', currentJob?.id],
        queryFn: () => trainingApi.getLogs(currentJob.id),
        enabled: !!currentJob,
        refetchInterval: isTraining ? 2000 : false
    })

    // Polling for thinking logs
    const { data: thinkingData, refetch: refetchThinking } = useQuery({
        queryKey: ['training-thinking', currentJob?.id],
        queryFn: () => trainingApi.getThinkingLogs(currentJob.id),
        enabled: !!currentJob,
        refetchInterval: isTraining ? 1500 : false
    })

    const thinkingMessages = thinkingData?.data?.thinking_logs || []

    // Get explanation data from status response
    const explanationData = latestTrainingData?.experiment?.explanation_data || null

    // Fetch graphs if training completed
    const { data: graphsData } = useQuery({
        queryKey: ['model-graphs', currentJob?.id],
        queryFn: () => modelsApi.getGraphs(currentJob.id),
        enabled: !!currentJob && latestTrainingData?.experiment?.status === 'completed',
        retry: 1
    })
    const graphs = graphsData?.data?.graphs || {}
    const hasGraphs = Object.keys(graphs).length > 0

    const startTrainingMutation = useMutation({
        mutationFn: (data) => trainingApi.start(data),
        onMutate: () => {
            setTrainingStartError(null)
        },
        onSuccess: (response) => {
            setCurrentJob(response.data.experiment)
            setLastStatus({ experiment: response.data.experiment, jobs: [] })
            setIsTraining(true)
        },
        onError: (error) => {
            const data = error.response?.data || {}
            setTrainingStartError({
                message: data.error || 'Training could not be started',
                suggestion: data.suggestion || data.hint || null,
                availableColumns: data.available_columns || null
            })
        }
    })

    const ensembleMutation = useMutation({
        mutationFn: (experimentId) => trainingApi.ensemble(experimentId),
        onSuccess: () => {
            setIsTraining(true)
            setIsEnsembling(false)
        },
        onError: () => {
            setIsEnsembling(false)
        }
    })

    const setHeadersMutation = useMutation({
        mutationFn: (headers) => datasetsApi.setHeaders(selectedDataset, headers),
        onSuccess: async () => {
            setHeaderFixError(null)
            await Promise.allSettled([
                queryClient.invalidateQueries({ queryKey: ['datasets'] }),
                queryClient.invalidateQueries({ queryKey: ['dataset-profile', selectedDataset] }),
                queryClient.invalidateQueries({ queryKey: ['dataset-header-detection', selectedDataset] })
            ])
            await Promise.allSettled([
                refetchHeaderDetection(),
                queryClient.refetchQueries({ queryKey: ['dataset-profile', selectedDataset] })
            ])
        },
        onError: (error) => {
            setHeaderFixError(error.response?.data?.error || 'Failed to update dataset headers')
        }
    })

    const handleCombineTop3 = () => {
        if (!currentJob?.id) return
        setIsEnsembling(true)
        ensembleMutation.mutate(currentJob.id)
    }

    useEffect(() => {
        setHeaderFixError(null)
        setHeaderInputs([])
        setTargetColumn('')
        setTrainingStartError(null)
    }, [selectedDataset])

    useEffect(() => {
        if (headerDetection?.has_headers === false) {
            setHeaderInputs(headerDetection.suggested_headers || [])
        }
    }, [selectedDataset, headerDetection?.has_headers, headerDetection?.suggested_headers])

    const handleHeaderInputChange = (index, value) => {
        setHeaderInputs((current) => current.map((header, i) => (i === index ? value : header)))
    }

    const handleConfirmHeaders = () => {
        const cleanHeaders = headerInputs.map((header) => header.trim())
        if (!cleanHeaders.length || cleanHeaders.some((header) => !header)) {
            setHeaderFixError('Every column needs a header before continuing')
            return
        }
        setHeadersMutation.mutate(cleanHeaders)
    }

    // Check for completion after one final refresh so the UI receives 100% progress.
    useEffect(() => {
        const status = latestTrainingData?.experiment?.status
        const isTerminal = ['completed', 'failed', 'cancelled'].includes(status)

        if (!isTraining || !isTerminal) return

        let isCurrent = true

        const fetchFinalState = async () => {
            await Promise.allSettled([
                refetchStatus(),
                refetchLogs(),
                refetchThinking()
            ])

            if (isCurrent) {
                setIsTraining(false)
            }
        }

        fetchFinalState()

        return () => {
            isCurrent = false
        }
    }, [isTraining, latestTrainingData?.experiment?.status, refetchStatus, refetchLogs, refetchThinking, setIsTraining])

    // AI Prompt Analysis Handler
    const handleAnalyzeWithAI = async () => {
        if (!selectedDataset || !aiPrompt.trim()) return

        setIsAnalyzing(true)
        setAnalysisError(null)
        setAiAnalysis(null)

        try {
            const response = await trainingApi.analyzePrompt(parseInt(selectedDataset), aiPrompt)
            if (response.data.success) {
                setAiAnalysis(response.data.analysis)
                if (response.data.analysis.suggested_target) {
                    setTargetColumn(response.data.analysis.suggested_target)
                }
                setGoalDescription(aiPrompt)
            } else {
                setAnalysisError(response.data.error || 'Analysis failed')
            }
        } catch (error) {
            setAnalysisError(error.response?.data?.error || 'Failed to analyze with AI')
        } finally {
            setIsAnalyzing(false)
        }
    }

    // Effect to check imbalance when target column changes
    useEffect(() => {
        if (!targetColumn || !profileData?.data?.column_info) {
            setImbalanceDetected(false)
            setAutoBalance(false)
            return
        }
        const colInfo = profileData.data.column_info[targetColumn]
        if (colInfo && colInfo?.top_values) {
            const counts = Object.values(colInfo.top_values).map(Number)
            if (counts.length >= 2) {
                const max = Math.max(...counts)
                const min = Math.min(...counts)
                // Extreme imbalance (minority class is less than 20% of majority)
                if (max > 0 && min / max <= 0.2) {
                    setImbalanceDetected(true)
                } else {
                    setImbalanceDetected(false)
                    setAutoBalance(false)
                }
            } else {
                setImbalanceDetected(false)
                setAutoBalance(false)
            }
        } else {
            setImbalanceDetected(false)
            setAutoBalance(false)
        }
    }, [targetColumn, profileData])

    const handleStartTraining = () => {
        if (!selectedDataset || !experimentName.trim() || needsHeaderFix) return

        // Filter out empty config values natively instead of sending nulls to backend
        const cleanConfig = {}
        Object.entries(configData).forEach(([key, val]) => {
            if (val !== '') {
                cleanConfig[key] = key === 'learning_rate' ? parseFloat(val) : parseInt(val, 10)
            }
        })

        startTrainingMutation.mutate({
            dataset_id: parseInt(selectedDataset),
            name: experimentName.trim(),
            target_column: isImageDataset ? null : (targetColumn || null),
            goal_description: isImageDataset ? (imageProblemType === 'object_detection' ? 'Object Detection' : 'Image classification') : goalDescription,
            config: { ...cleanConfig, problem_type: isImageDataset ? imageProblemType : undefined, auto_balance: autoBalance }
        })
    }

    // Combine all logs
    const allLogs = logsData?.data?.logs || []
    const combinedLogs = allLogs.map(job =>
        `--- ${job.model_name} ---\n${job.logs || ''}`
    ).join('\n')

    // Extract trained jobs for Leaderboard
    const trainedJobs = latestTrainingData?.jobs || []
    // Sort jobs by score descendant (accuracy or r2_score)
    const sortedJobs = [...trainedJobs].sort((a, b) => {
        const scoreA = a.metrics?.accuracy || a.metrics?.r2_score || 0
        const scoreB = b.metrics?.accuracy || b.metrics?.r2_score || 0
        return scoreB - scoreA
    })

    // Auto-scroll to bottom of logs
    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
        }
    }, [combinedLogs])

    const handleDownloadLogs = () => {
        if (!currentJob) return;

        let logContent = `INFERX-ML TRAINING LOG\n`;
        logContent += `======================\n`;
        logContent += `Experiment: ${currentJob.name}\n`;
        logContent += `Status: ${latestTrainingData?.experiment?.status || 'Unknown'}\n`;
        logContent += `Timestamp: ${new Date().toLocaleString()}\n\n`;

        logContent += `DEVELOPER LOGS\n`;
        logContent += `--------------\n`;
        logContent += combinedLogs || 'No developer logs available.';
        logContent += `\n\n`;

        if (thinkingMessages && thinkingMessages.length > 0) {
            logContent += `THINKING LOGS (AI INNER MONOLOGUE)\n`;
            logContent += `----------------------------------\n`;
            thinkingMessages.forEach(msg => {
                const time = msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : 'N/A';
                logContent += `[${time}] ${msg.message}\n`;
            });
            logContent += `\n\n`;
        }

        if (explanationData && explanationData.insights) {
            logContent += `ACTIVITY INSIGHTS (WHAT MODEL DID)\n`;
            logContent += `----------------------------------\n`;
            explanationData.insights.forEach(insight => {
                const time = insight.timestamp ? new Date(insight.timestamp).toLocaleTimeString() : 'N/A';
                logContent += `[${time}] [${insight.phase || 'General'}] ${insight.message}\n`;
            });
        }

        const blob = new Blob([logContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `training_log_${currentJob.name.replace(/\s+/g, '_')}_${currentJob.id}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const experimentStatus = latestTrainingData?.experiment?.status

    return (
        <div className="training-page">
            {/* ── Compact Header ── */}
            <div className="training-header">
                <div className="header-left">
                    <Cpu size={20} className="header-icon" />
                    <h1>Model Training</h1>
                </div>
                <div className="header-actions" style={{ display: 'flex', gap: '0.75rem' }}>
                    <button
                        className="btn btn-outline"
                        onClick={resetTraining}
                        disabled={isTraining}
                        title="Reset all settings to start a new training"
                        style={{ padding: '0.5rem 1rem' }}
                    >
                        <RotateCcw size={16} />
                        Reset
                    </button>
                    <button
                        className="btn btn-primary start-btn"
                        onClick={handleStartTraining}
                        disabled={!experimentName.trim() || !selectedDataset || needsHeaderFix || isDetectingHeaders || (isTraining && experimentStatus !== 'completed')}
                    >
                        {isTraining ? (
                            <>
                                <div className="spinner" style={{ width: 16, height: 16 }}></div>
                                Training...
                            </>
                        ) : (
                            <>
                                <Play size={16} />
                                {experimentStatus === 'completed' ? 'New Training' : 'Start Training'}
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* ── Two-Column Layout ── */}
            <div className="training-layout" ref={layoutRef}>

                {/* ═══ LEFT PANEL: User Inputs ═══ */}
                <div className="panel panel-left" style={{ width: `${leftWidth}px`, flexShrink: 0 }}>
                    <div className="panel-header">
                        <Zap size={16} />
                        <span>Configuration</span>
                    </div>

                    {trainingStartError && (
                        <div className="training-start-error">
                            <div className="training-start-error-title">
                                <AlertCircle size={15} />
                                <span>{trainingStartError.message}</span>
                            </div>
                            {trainingStartError.suggestion && (
                                <p>{trainingStartError.suggestion}</p>
                            )}
                            {trainingStartError.availableColumns?.length > 0 && (
                                <div className="training-start-columns">
                                    {trainingStartError.availableColumns.map((column) => (
                                        <span key={column}>{column}</span>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* 1. Experiment Name (Required) */}
                    <div className="form-group">
                        <label className="label">
                            Experiment Name <span className="required">*</span>
                        </label>
                        <input
                            type="text"
                            className={`input ${!experimentName.trim() ? 'input-required' : ''}`}
                            value={experimentName}
                            onChange={(e) => setExperimentName(e.target.value)}
                            placeholder="Enter experiment name..."
                            disabled={isTraining}
                            required
                        />
                    </div>

                    {/* 2. Dataset */}
                    <div className="form-group">
                        <label className="label">
                            <Database size={14} />
                            Dataset <span className="required">*</span>
                        </label>
                        <select
                            className="input"
                            value={selectedDataset}
                            onChange={(e) => setSelectedDataset(e.target.value)}
                            disabled={isTraining}
                        >
                            <option value="">Select a dataset...</option>
                            {datasets.map((dataset) => (
                                <option key={dataset.id} value={dataset.id}>
                                    {dataset.name} ({dataset.data_type === 'image'
                                        ? `${dataset.num_rows?.toLocaleString()} images · ${dataset.num_columns} classes`
                                        : `${dataset.num_rows?.toLocaleString()} rows`})
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* ── Image Dataset Config ── */}
                    {isImageDataset && (
                        <div className="form-group">
                            <label className="label">
                                <Target size={14} />
                                Vision Task <span className="required">*</span>
                            </label>
                            <select
                                className="input"
                                value={imageProblemType}
                                onChange={(e) => setImageProblemType(e.target.value)}
                                disabled={isTraining}
                            >
                                <option value="image_classification">Image Classification (Predict Categories)</option>
                                <option value="object_detection">Object Detection (Find Bounding Boxes via YOLOv8)</option>
                            </select>
                        </div>
                    )}

                    {/* ── Image Dataset: Class Gallery ── */}
                    {isImageDataset && imageClasses && imageProblemType === 'image_classification' && (
                        <div className="image-classes-panel">
                            <label className="label">
                                <ImageIcon size={14} />
                                Image Classes Detected
                            </label>
                            <div className="image-class-gallery">
                                {imageClasses.classes?.map((cls) => {
                                    const count = imageClasses.counts?.[cls] || 0
                                    const maxCount = Math.max(...Object.values(imageClasses.counts || {}))
                                    const pct = maxCount > 0 ? (count / maxCount) * 100 : 0
                                    return (
                                        <div key={cls} className="image-class-card">
                                            <div className="class-info">
                                                <span className="class-name">{cls}</span>
                                                <span className="class-count">{count} images</span>
                                            </div>
                                            <div className="class-bar-track">
                                                <div className="class-bar-fill" style={{ width: `${pct}%` }} />
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                            <div className="image-summary">
                                <span>📊 {imageClasses.total_images} total images · {imageClasses.num_classes} classes</span>
                                <span>🧠 Models: MobileNetV2, EfficientNet-B0</span>
                            </div>
                        </div>
                    )}

                    {/* ── Tabular Dataset Config ── */}
                    {!isImageDataset && (
                        <>
                            {selectedDataset && (
                                <div className={`header-fix-panel ${needsHeaderFix ? 'needs-review' : ''}`}>
                                    <div className="header-fix-title">
                                        <TableProperties size={15} />
                                        <span>Dataset Headers</span>
                                    </div>

                                    {isDetectingHeaders && (
                                        <div className="header-fix-status">
                                            <div className="spinner" style={{ width: 14, height: 14 }}></div>
                                            <span>Checking column headers...</span>
                                        </div>
                                    )}

                                    {!isDetectingHeaders && headerDetection?.has_headers === true && (
                                        <div className="header-fix-ok">
                                            <CheckCircle size={14} />
                                            <span>Headers look ready for training.</span>
                                        </div>
                                    )}

                                    {!isDetectingHeaders && needsHeaderFix && (
                                        <>
                                            <div className="header-fix-alert">
                                                <AlertCircle size={15} />
                                                <span>The first row looks like data, so training is paused until headers are confirmed.</span>
                                            </div>

                                            <div className="raw-preview">
                                                {(headerDetection.raw_rows || []).slice(0, 3).map((row, rowIndex) => (
                                                    <div className="raw-preview-row" key={rowIndex}>
                                                        {row.map((value, colIndex) => (
                                                            <span className="raw-preview-cell" key={`${rowIndex}-${colIndex}`}>
                                                                {String(value ?? '')}
                                                            </span>
                                                        ))}
                                                    </div>
                                                ))}
                                            </div>

                                            <div className="header-input-grid">
                                                {headerInputs.map((header, index) => (
                                                    <label className="header-input-item" key={index}>
                                                        <span>Column {index + 1}</span>
                                                        <input
                                                            className="input"
                                                            value={header}
                                                            onChange={(e) => handleHeaderInputChange(index, e.target.value)}
                                                            disabled={setHeadersMutation.isPending || isTraining}
                                                        />
                                                    </label>
                                                ))}
                                            </div>

                                            {headerFixError && (
                                                <div className="ai-error">
                                                    <AlertCircle size={14} />
                                                    <span>{headerFixError}</span>
                                                </div>
                                            )}

                                            <button
                                                type="button"
                                                className="btn btn-primary header-confirm-btn"
                                                onClick={handleConfirmHeaders}
                                                disabled={setHeadersMutation.isPending || isTraining}
                                            >
                                                {setHeadersMutation.isPending ? (
                                                    <div className="spinner" style={{ width: 14, height: 14 }}></div>
                                                ) : (
                                                    <CheckCircle size={14} />
                                                )}
                                                {setHeadersMutation.isPending ? 'Saving...' : 'Confirm Headers'}
                                            </button>
                                        </>
                                    )}
                                </div>
                            )}

                            <div className={`ai-section ${(!selectedDataset || needsHeaderFix) ? 'section-disabled' : ''}`}>
                                <label className="label">
                                    <Sparkles size={14} />
                                    AI Goal Analysis
                                </label>
                                <div className="ai-input-row">
                                    <textarea
                                        className="input textarea"
                                        value={aiPrompt}
                                        onChange={(e) => setAiPrompt(e.target.value)}
                                        placeholder="e.g., Predict which products need restocking..."
                                        rows={2}
                                        disabled={!selectedDataset || needsHeaderFix || isTraining || isAnalyzing}
                                    />
                                    <button
                                        className="btn btn-secondary ai-analyze-btn"
                                        onClick={handleAnalyzeWithAI}
                                        disabled={!selectedDataset || needsHeaderFix || !aiPrompt.trim() || isAnalyzing || isTraining}
                                    >
                                        {isAnalyzing ? (
                                            <div className="spinner" style={{ width: 14, height: 14 }}></div>
                                        ) : (
                                            <Wand2 size={14} />
                                        )}
                                        {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                                    </button>
                                </div>
                                <p className="form-hint">Let AI suggest the best target column</p>

                                {/* AI Analysis Results */}
                                {aiAnalysis && (
                                    <div className="ai-analysis-result">
                                        <div className="ai-result-header">
                                            <Sparkles size={16} className="text-primary" />
                                            <span>AI Recommendation</span>
                                        </div>
                                        <div className="ai-result-content">
                                            <div className="ai-result-item">
                                                <strong>Target:</strong>
                                                <span className="ai-target-badge">{aiAnalysis.suggested_target}</span>
                                            </div>
                                            <div className="ai-result-item">
                                                <strong>Type:</strong>
                                                <span className="ai-problem-badge">{aiAnalysis.problem_type}</span>
                                            </div>
                                            {aiAnalysis.confidence && (
                                                <div className="ai-result-item">
                                                    <strong>Confidence:</strong>
                                                    <span>{(aiAnalysis.confidence * 100).toFixed(0)}%</span>
                                                </div>
                                            )}
                                            <div className="ai-reasoning">
                                                <strong>Reasoning:</strong>
                                                <p>{aiAnalysis.reasoning}</p>
                                            </div>
                                            {aiAnalysis.preprocessing_suggestions?.length > 0 && (
                                                <div className="ai-suggestions">
                                                    <strong>Tips:</strong>
                                                    <ul>
                                                        {aiAnalysis.preprocessing_suggestions.map((tip, i) => (
                                                            <li key={i}>{tip}</li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}

                                {analysisError && (
                                    <div className="ai-error">
                                        <AlertCircle size={14} />
                                        <span>{analysisError}</span>
                                    </div>
                                )}
                            </div>

                            {/* 4. Target Column (Optional — not needed for all training types) */}
                            <div className="form-group">
                                <label className="label">
                                    <Target size={14} />
                                    Target Column <span className="optional-tag">Optional</span>
                                </label>
                                <select
                                    className="input"
                                    value={targetColumn}
                                    onChange={(e) => setTargetColumn(e.target.value)}
                                    disabled={!selectedDataset || needsHeaderFix || isTraining}
                                >
                                    <option value="">{selectedDataset ? 'None (auto-detect or unsupervised)' : 'Select a dataset first...'}</option>
                                    {columns.map((col) => (
                                        <option key={col} value={col}>{col}</option>
                                    ))}
                                </select>
                                <p className="form-hint">Select only for supervised learning (classification / regression)</p>
                            </div>

                            {/* Imbalance Detection Alert */}
                            {imbalanceDetected && !isImageDataset && targetColumn && (
                                <div className="imbalance-alert" style={{ marginTop: '1rem', padding: '1rem', backgroundColor: 'rgba(239, 68, 68, 0.05)', border: '1px solid rgba(239, 68, 68, 0.4)', borderRadius: '8px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#ef4444', marginBottom: '0.5rem', fontWeight: 'bold' }}>
                                        <AlertCircle size={16} />
                                        <span>Extreme Imbalance Detected</span>
                                    </div>
                                    <p style={{ fontSize: '0.9rem', marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>
                                        Your data is imbalanced. Would you like InferX to generate synthetic 'Fraud' records to balance the training?
                                    </p>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.9rem' }}>
                                        <input 
                                            type="checkbox" 
                                            checked={autoBalance} 
                                            onChange={(e) => setAutoBalance(e.target.checked)}
                                            disabled={isTraining}
                                            style={{ margin: 0 }}
                                        />
                                        <span style={{ color: 'var(--text-color)' }}>Yes, automatically balance with SMOTE</span>
                                    </label>
                                </div>
                            )}
                        </>
                    )}

                    {/* 5. Advanced Settings (Visible for both types) */}
                    {selectedDataset && (
                        <div className="advanced-settings-section" style={{ marginTop: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                            <button
                                type="button"
                                className="btn btn-outline"
                                style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                                onClick={() => setShowAdvanced(!showAdvanced)}
                                disabled={isTraining}
                            >
                                <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <Settings size={14} />
                                    Advanced Configuration
                                </span>
                                <ChevronDown size={14} style={{ transform: showAdvanced ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
                            </button>

                            {showAdvanced && (
                                <div className="advanced-settings-content" style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem', padding: '1rem', backgroundColor: 'var(--bg-secondary)', borderRadius: '6px' }}>
                                    {isImageDataset ? (
                                        <>
                                            <div className="form-group" style={{ marginBottom: 0 }}>
                                                <label className="label">Epochs</label>
                                                <input type="number" min="1" max="1000" className="input" value={configData.epochs} onChange={(e) => setConfigData({ ...configData, epochs: e.target.value })} placeholder={imageProblemType === 'object_detection' ? "Default: 30" : "Default: 15"} disabled={isTraining} />
                                            </div>
                                            <div className="form-group" style={{ marginBottom: 0 }}>
                                                <label className="label">Batch Size</label>
                                                <input type="number" min="1" max="1024" className="input" value={configData.batch_size} onChange={(e) => setConfigData({ ...configData, batch_size: e.target.value })} placeholder={imageProblemType === 'object_detection' ? "Default: 16" : "Default: 32"} disabled={isTraining} />
                                            </div>
                                            {imageProblemType === 'object_detection' && (
                                                <div className="form-group" style={{ marginBottom: 0 }}>
                                                    <label className="label">Image Size</label>
                                                    <select className="input" value={configData.img_size} onChange={(e) => setConfigData({ ...configData, img_size: e.target.value })} disabled={isTraining}>
                                                        <option value="">Default (640)</option>
                                                        <option value="320">320 (Faster Training)</option>
                                                        <option value="640">640 (Balanced)</option>
                                                        <option value="1280">1280 (Higher Accuracy)</option>
                                                    </select>
                                                </div>
                                            )}
                                            <div className="form-group" style={{ marginBottom: 0 }}>
                                                <label className="label">Learning Rate</label>
                                                <input type="number" step="0.001" min="0.0001" max="1.0" className="input" value={configData.learning_rate} onChange={(e) => setConfigData({ ...configData, learning_rate: e.target.value })} placeholder={imageProblemType === 'object_detection' ? "Default: 0.01" : "Default: 0.001"} disabled={isTraining} />
                                            </div>
                                        </>
                                    ) : (
                                        <>
                                            <div className="form-group" style={{ marginBottom: 0 }}>
                                                <label className="label">Epochs (For Neural Networks)</label>
                                                <input type="number" min="1" max="1000" className="input" value={configData.epochs} onChange={(e) => setConfigData({ ...configData, epochs: e.target.value })} placeholder="Default: 50" disabled={isTraining} />
                                            </div>
                                            <div className="form-group" style={{ marginBottom: 0 }}>
                                                <label className="label">Number of Estimators (Trees)</label>
                                                <input type="number" min="1" max="5000" className="input" value={configData.n_estimators} onChange={(e) => setConfigData({ ...configData, n_estimators: e.target.value })} placeholder="Default: 100" disabled={isTraining} />
                                            </div>
                                            <div className="form-group" style={{ marginBottom: 0 }}>
                                                <label className="label">Max Depth</label>
                                                <input type="number" min="1" max="100" className="input" value={configData.max_depth} onChange={(e) => setConfigData({ ...configData, max_depth: e.target.value })} placeholder="Default: Auto (-1)" disabled={isTraining} />
                                            </div>
                                        </>
                                    )}
                                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0, marginTop: '0.5rem' }}>
                                        Leave blank to use optimal industry defaults for our selected algorithms.
                                    </p>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* ═══ RESIZE DIVIDER HANDLE ═══ */}
                <div 
                    className={`training-resize-handle ${isDragging ? 'dragging' : ''}`}
                    onMouseDown={startDragging}
                    onTouchStart={startDragging}
                />

                {/* ═══ RIGHT PANEL: Logs + Model Progress with Dual Tabs ═══ */}
                <div className="panel panel-right" style={{ flexGrow: 1, minWidth: 0 }}>
                    
                    {/* Tab Navigation Headers */}
                    <div className="right-tabs-navigation">
                        <button 
                            className={`right-tab-btn ${activeRightTab === 'progress' ? 'active' : ''}`}
                            onClick={() => setActiveRightTab('progress')}
                        >
                            <Brain size={16} />
                            <span>Progress & Insights</span>
                        </button>
                        <button 
                            className={`right-tab-btn ${activeRightTab === 'console' ? 'active' : ''}`}
                            onClick={() => setActiveRightTab('console')}
                        >
                            <Terminal size={16} />
                            <span>Developer Console</span>
                            {isTraining && <div className="live-dot" />}
                        </button>
                    </div>

                    {/* Tab Content Area */}
                    <div className="right-tab-content-wrapper">
                        
                        {/* TAB 1: PROGRESS & ANALYTICS */}
                        {activeRightTab === 'progress' && (
                            <div className="right-section model-section">
                                <div className="panel-header">
                                    <Brain size={16} />
                                    <span>What the Model is Doing</span>
                                    {currentJob && (
                                        <button 
                                            className="btn-download-logs" 
                                            onClick={handleDownloadLogs}
                                            title="Download Full Training Log"
                                        >
                                            <Download size={14} />
                                            <span>Download Log</span>
                                        </button>
                                    )}
                                </div>

                                {currentJob ? (
                                    <div className="model-progress-content">
                                        {/* Status Bar */}
                                        <div className="status-row">
                                            <span className="exp-name">{currentJob.name}</span>
                                            <span className={`status-badge ${experimentStatus}`}>
                                                {experimentStatus || 'Starting...'}
                                            </span>
                                        </div>

                                        {/* Pipeline Steps */}
                                        <div className="pipeline-steps">
                                            <div className={`pipeline-step ${['training', 'completed'].includes(experimentStatus) ? 'completed' : 'active'}`}>
                                                <div className="pipeline-dot">
                                                    <CheckCircle size={14} />
                                                </div>
                                                <span>Preprocessing</span>
                                            </div>
                                            <div className="pipeline-line" />
                                            <div className={`pipeline-step ${experimentStatus === 'training' ? 'active' : experimentStatus === 'completed' ? 'completed' : ''}`}>
                                                <div className="pipeline-dot">
                                                    <Clock size={14} className={experimentStatus === 'training' ? 'animate-pulse' : ''} />
                                                </div>
                                                <span>Training</span>
                                            </div>
                                            <div className="pipeline-line" />
                                            <div className={`pipeline-step ${experimentStatus === 'completed' ? 'completed' : ''}`}>
                                                <div className="pipeline-dot">
                                                    <BarChart3 size={14} />
                                                </div>
                                                <span>Evaluation</span>
                                            </div>
                                        </div>

                                        {/* LIVE LEADERBOARD */}
                                        {sortedJobs.length > 0 && (
                                            <div className="leaderboard-section">
                                                <div className="panel-header" style={{ marginBottom: '1rem', marginTop: '1.5rem', paddingLeft: 0, paddingRight: 0, borderBottom: 'none' }}>
                                                    <Target size={16} />
                                                    <span>Live Model Leaderboard</span>
                                                    {isTraining && <div className="live-dot" />}
                                                </div>
                                                <div className="leaderboard-grid">
                                                    {sortedJobs.map((job, index) => {
                                                        const score = job.metrics?.accuracy || job.metrics?.r2_score || 0
                                                        const scoreFmt = (score * 100).toFixed(2) + '%'
                                                        const isEnsemble = job.model_name.toLowerCase().includes('ensemble')
                                                        const isBest = index === 0 && job.status === 'completed' && score > 0
                                                        return (
                                                            <div key={job.id} className={`leaderboard-card ${isBest ? 'top-model' : ''} ${isEnsemble ? 'ensemble-model' : ''} ${job.status}`}>
                                                                <div className="lb-rank">#{index + 1}</div>
                                                                <div className="lb-info">
                                                                    <div className="lb-name">
                                                                        {job.model_name}
                                                                        {isEnsemble && <Sparkles size={12} className="ensemble-icon" />}
                                                                    </div>
                                                                    <div className="lb-metrics">
                                                                        {job.status === 'completed' && score > 0 ? (
                                                                            <span className="lb-score">Score: <strong>{scoreFmt}</strong></span>
                                                                        ) : job.status === 'failed' ? (
                                                                            <span className="lb-failed">Failed</span>
                                                                        ) : job.status === 'training' ? (
                                                                            <span className="lb-training-badge">Training...</span>
                                                                        ) : (
                                                                            <span className="lb-pending">Waiting...</span>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                                {isBest && <div className="lb-crown">👑</div>}
                                                            </div>
                                                        )
                                                    })}
                                                </div>

                                                {/* Ensemble Action Banner if completed */}
                                                {experimentStatus === 'completed' && sortedJobs.length >= 2 && !sortedJobs.some(j => j.model_name.toLowerCase().includes('ensemble')) && (
                                                    <div className="ensemble-action-banner">
                                                        <div className="ensemble-text">
                                                            <Sparkles size={18} className="ensemble-icon" />
                                                            <div>
                                                                <span>Supercharge accuracy by ensembling top models!</span>
                                                            </div>
                                                        </div>
                                                        <button 
                                                            className="btn ensemble-action-btn"
                                                            onClick={handleCombineTop3}
                                                            disabled={isEnsembling}
                                                        >
                                                            {isEnsembling ? (
                                                                <div className="spinner" style={{ width: 14, height: 14 }}></div>
                                                            ) : (
                                                                <Zap size={14} />
                                                            )}
                                                            {isEnsembling ? 'Ensembling...' : 'Combine Models'}
                                                        </button>
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        {/* Learning Timeline */}
                                        <LearningTimeline
                                            explanationData={explanationData}
                                            isTraining={isTraining}
                                        />

                                        {/* Training Success */}
                                        {experimentStatus === 'completed' && (
                                            <div className="success-message">
                                                <CheckCircle size={20} className="text-success" />
                                                <h3>Training Complete!</h3>
                                                {latestTrainingData?.experiment?.best_model_name && (
                                                    <div className="result-summary">
                                                        <div className="result-item">
                                                            <span className="result-label">🏆 Best Model</span>
                                                            <span className="result-value">{latestTrainingData.experiment.best_model_name}</span>
                                                        </div>
                                                        <div className="result-item">
                                                            <span className="result-label">📊 Score</span>
                                                            <span className="result-value">
                                                                {(latestTrainingData.experiment.best_score * 100).toFixed(2)}%
                                                            </span>
                                                        </div>
                                                    </div>
                                                )}
                                                <a href="/models" className="btn btn-sm btn-outline">View Models</a>
                                            </div>
                                        )}

                                        {/* Training Graphs */}
                                        {experimentStatus === 'completed' && hasGraphs && (
                                            <div className="training-graphs-section" style={{ marginTop: '1.5rem' }}>
                                                <div className="panel-header" style={{ marginBottom: '1rem', paddingLeft: 0, paddingRight: 0, borderBottom: 'none' }}>
                                                    <ImageIcon size={16} />
                                                    <span>Evaluation Graphs</span>
                                                </div>
                                                <div className="graphs-grid">
                                                    {Object.entries(graphs).map(([name, src]) => (
                                                        <div key={name} className="graph-item card">
                                                            <div className="graph-header">
                                                                <h5 style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                                                    {name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                                                                </h5>
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
                                ) : (
                                    <div className="model-empty">
                                        <p>Configure and start training to see model progress here.</p>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* TAB 2: DEVELOPER CONSOLE LOGS */}
                        {activeRightTab === 'console' && (
                            <div className="right-section logs-section" style={{ height: '100%' }}>
                                <div className="panel-header">
                                    <Terminal size={16} />
                                    <span>Developer Logs</span>
                                    {isTraining && <div className="live-dot" />}
                                </div>
                                <pre className="terminal-content" style={{ height: 'calc(100% - 41px)' }}>
                                    {combinedLogs || 'Waiting for training to start...'}
                                    <div ref={logsEndRef} />
                                </pre>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
