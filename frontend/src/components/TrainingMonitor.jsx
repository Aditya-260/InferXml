import React, { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useLocation } from 'react-router-dom'
import { trainingApi } from '../services/api'
import { useTrainingStore } from '../store/trainingStore'
import { ArrowRight, CheckCircle, Sparkles, XCircle, X } from 'lucide-react'
import './TrainingMonitor.css'

export default function TrainingMonitor() {
    const { currentJob, isTraining, lastStatus, setIsTraining, setLastStatus } = useTrainingStore()
    const navigate = useNavigate()
    const location = useLocation()
    const [notification, setNotification] = useState(null)

    const { data: statusData } = useQuery({
        queryKey: ['training-status', currentJob?.id],
        queryFn: () => trainingApi.getStatus(currentJob.id),
        enabled: !!currentJob && isTraining,
        refetchInterval: isTraining ? 2000 : false
    })

    useEffect(() => {
        if (!isTraining || !currentJob || !statusData) return

        const payload = statusData.data
        const status = payload.experiment.status

        setLastStatus(payload)

        if (status === 'completed' || status === 'failed') {
            // Turn off training mode so we stop polling
            setIsTraining(false)

            // Only show the global popup if we are NOT already on the training page
            // because the training page already shows completion details inline.
            if (location.pathname !== '/training') {
                setNotification({
                    id: currentJob.id,
                    status,
                    name: currentJob.name,
                    modelName: payload.experiment.best_model_name,
                    score: payload.experiment.best_score
                })

                // Auto dismiss after 10 seconds
                setTimeout(() => {
                    setNotification(null)
                }, 10000)
            }
        }
    }, [statusData, isTraining, currentJob, setIsTraining, setLastStatus, location.pathname])

    const shouldShowWidget = isTraining && currentJob && location.pathname !== '/training'
    const cachedStatus = lastStatus?.experiment?.id === currentJob?.id ? lastStatus : null
    const widgetStatus = statusData?.data || cachedStatus

    if (!notification && !shouldShowWidget) return null

    return (
        <>
            {shouldShowWidget && (
                <PersistentTrainingWidget
                    currentJob={currentJob}
                    statusPayload={widgetStatus}
                    onBackToTraining={() => navigate('/training')}
                />
            )}

            {notification && (
                <div className="training-notification">
                    <button className="close-btn" onClick={() => setNotification(null)}>
                        <X size={16} />
                    </button>
                    <div className="notification-content">
                        {notification.status === 'completed' ? (
                            <CheckCircle size={24} className="text-success" />
                        ) : (
                            <XCircle size={24} className="text-danger" />
                        )}
                        <div className="notification-info">
                            <h4>{notification.status === 'completed' ? 'Training Complete!' : 'Training Failed'}</h4>
                            <p>Dataset: {notification.name}</p>
                            {notification.status === 'completed' && notification.modelName && (
                                <p className="model-score">Best: {notification.modelName} ({(notification.score * 100).toFixed(2)}%)</p>
                            )}
                        </div>
                    </div>
                    {notification.status === 'completed' && (
                        <div className="notification-actions">
                            <button
                                className="btn btn-primary btn-sm"
                                onClick={() => {
                                    setNotification(null)
                                    navigate(`/predictions/${notification.id}`)
                                }}
                                style={{ width: '100%' }}
                            >
                                View & Predict
                            </button>
                        </div>
                    )}
                </div>
            )}
        </>
    )
}

function PersistentTrainingWidget({ currentJob, statusPayload, onBackToTraining }) {
    const experiment = statusPayload?.experiment
    const explanationData = experiment?.explanation_data || {}
    const insights = explanationData.insights || []
    const latestInsight = insights[insights.length - 1]
    const activeJob = statusPayload?.jobs?.find((job) => job.status === 'training')
    const progress = Math.max(0, Math.min(100, Number(explanationData.progress || activeJob?.progress || 0)))
    const detail = latestInsight?.message || activeJob?.model_name || explanationData.current_phase || 'Preparing training run...'

    return (
        <div className="persistent-training-widget" role="status" aria-live="polite">
            <div className="ptw-header">
                <div className="ptw-orb">
                    <Sparkles size={16} />
                </div>
                <div className="ptw-title-group">
                    <span className="ptw-kicker">Training Live</span>
                    <h4>{experiment?.name || currentJob.name}</h4>
                </div>
                <span className="ptw-percent">{Math.round(progress)}%</span>
            </div>

            <div className="ptw-progress-track">
                <div className="ptw-progress-fill" style={{ width: `${progress}%` }} />
            </div>

            <p className="ptw-detail">{detail}</p>

            <button className="ptw-action" onClick={onBackToTraining}>
                <span>Back to Training</span>
                <ArrowRight size={15} />
            </button>
        </div>
    )
}
