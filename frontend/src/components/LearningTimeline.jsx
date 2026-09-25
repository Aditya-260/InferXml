import React, { useState, useEffect, useRef } from 'react';
import { ChevronDown, ChevronRight, CheckCircle, Loader, Circle, Shield, Activity, Clock } from 'lucide-react';
import './LearningTimeline.css';

/**
 * Format a timestamp to a short relative or clock time
 */
const formatTime = (isoString) => {
    if (!isoString) return '';
    try {
        const d = new Date(isoString);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch { return ''; }
};

/**
 * Calculate elapsed duration between two ISO timestamps
 */
const formatDuration = (start, end) => {
    if (!start) return '';
    const s = new Date(start);
    const e = end ? new Date(end) : new Date();
    const diff = Math.max(0, Math.round((e - s) / 1000));
    if (diff < 60) return `${diff}s`;
    const mins = Math.floor(diff / 60);
    const secs = diff % 60;
    return `${mins}m ${secs}s`;
};

// ─── Phase Card: Dynamic, insight-driven ───
const PhaseCard = ({ phase, isActive, isComplete, insights, isExpanded, onToggle }) => {
    const statusIcon = isComplete
        ? <CheckCircle size={18} className="lt-phase-icon lt-complete" />
        : isActive
            ? <Loader size={18} className="lt-phase-icon lt-active lt-spin" />
            : <Circle size={18} className="lt-phase-icon lt-pending" />;

    // Dynamic subtitle: show the latest insight for this phase, or the phase description
    const latestInsight = insights.length > 0 ? insights[insights.length - 1].message : null;
    const subtitle = isActive && latestInsight
        ? latestInsight
        : isComplete && latestInsight
            ? latestInsight
            : phase.description;

    return (
        <div className={`lt-phase-card ${isComplete ? 'lt-card-complete' : ''} ${isActive ? 'lt-card-active' : ''}`}>
            <div className="lt-phase-header" onClick={onToggle}>
                <div className="lt-phase-left">
                    {statusIcon}
                    <span className="lt-phase-emoji">{phase.icon}</span>
                    <div className="lt-phase-title-group">
                        <h4 className="lt-phase-title">{phase.title}</h4>
                        <p className="lt-phase-subtitle">{subtitle}</p>
                    </div>
                </div>
                <div className="lt-phase-right">
                    {phase.started_at && (
                        <span className="lt-phase-time">
                            <Clock size={12} />
                            {formatDuration(phase.started_at, phase.completed_at)}
                        </span>
                    )}
                    {insights.length > 0 && (
                        isExpanded
                            ? <ChevronDown size={16} className="lt-expand-icon" />
                            : <ChevronRight size={16} className="lt-expand-icon" />
                    )}
                </div>
            </div>

            {isExpanded && insights.length > 0 && (
                <div className="lt-phase-details">
                    <div className="lt-insight-feed">
                        {insights.map((insight, idx) => (
                            <div key={idx} className="lt-insight-row">
                                <span className="lt-insight-time">{formatTime(insight.timestamp)}</span>
                                <span className="lt-insight-dot" />
                                <span className="lt-insight-msg">{insight.message}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

// ─── Trust Panel ───
const TrustPanel = ({ trust }) => {
    if (!trust || !trust.data_size) return null;

    return (
        <div className="lt-trust">
            <div className="lt-trust-header">
                <Shield size={16} />
                <span>Validation & Trust</span>
            </div>
            <div className="lt-trust-grid">
                <div className="lt-trust-stat">
                    <span className="lt-trust-val">{trust.data_size?.toLocaleString()}</span>
                    <span className="lt-trust-lbl">Training Samples</span>
                </div>
                <div className="lt-trust-stat">
                    <span className="lt-trust-val">{trust.test_size?.toLocaleString()}</span>
                    <span className="lt-trust-lbl">Test Holdout ({trust.test_ratio})</span>
                </div>
                <div className="lt-trust-stat">
                    <span className={`lt-trust-badge lt-conf-${trust.confidence}`}>{trust.confidence}</span>
                    <span className="lt-trust-lbl">Confidence Level</span>
                </div>
            </div>
            {trust.confidence_reasons?.length > 0 && (
                <div className="lt-trust-reasons">
                    {trust.confidence_reasons.map((reason, idx) => (
                        <div key={idx} className="lt-trust-reason">
                            <CheckCircle size={12} className="lt-trust-check" />
                            <span>{reason}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

// ─── Final Summary: Clean, no ensemble upsell ───
const FinalSummary = ({ summary }) => {
    if (!summary) return null;

    return (
        <div className="lt-summary">
            <div className="lt-summary-banner">
                <span className="lt-summary-icon">🎉</span>
                <div>
                    <h3 className="lt-summary-title">Model Training Complete</h3>
                    <p className="lt-summary-sub">Your AI model is ready for predictions</p>
                </div>
            </div>

            <div className="lt-summary-result">
                <div className="lt-summary-model">
                    <span className="lt-summary-label">Best Model</span>
                    <span className="lt-summary-value">{summary.best_model}</span>
                </div>
                <div className="lt-summary-score">
                    <span className="lt-summary-label">Accuracy</span>
                    <span className="lt-summary-value lt-score-highlight">{summary.score_display}</span>
                </div>
            </div>

            <div className="lt-summary-explain">
                <div className="lt-explain-block">
                    <h5>Why This Model?</h5>
                    <p>{summary.why_chosen}</p>
                </div>
                <div className="lt-explain-block">
                    <h5>What This Means</h5>
                    <p>{summary.what_it_means}</p>
                </div>
            </div>

            {(summary.model_strengths?.length > 0 || summary.model_limitations?.length > 0) && (
                <div className="lt-summary-traits">
                    {summary.model_strengths?.length > 0 && (
                        <div className="lt-trait-col lt-trait-good">
                            <h5>✔ Strengths</h5>
                            <ul>
                                {summary.model_strengths.map((s, i) => <li key={i}>{s}</li>)}
                            </ul>
                        </div>
                    )}
                    {summary.model_limitations?.length > 0 && (
                        <div className="lt-trait-col lt-trait-warn">
                            <h5>⚠ Considerations</h5>
                            <ul>
                                {summary.model_limitations.map((l, i) => <li key={i}>{l}</li>)}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

// ═══════════════════════════════════════════
// Main Learning Timeline Component
// ═══════════════════════════════════════════
const LearningTimeline = ({ explanationData, isTraining }) => {
    const [expandedPhases, setExpandedPhases] = useState({});
    const feedRef = useRef(null);

    // Auto-expand the currently active phase
    useEffect(() => {
        if (explanationData?.current_phase) {
            setExpandedPhases(prev => ({ ...prev, [explanationData.current_phase]: true }));
        }
    }, [explanationData?.current_phase]);

    // Auto-scroll feed when new insights appear
    useEffect(() => {
        if (feedRef.current) {
            feedRef.current.scrollTop = feedRef.current.scrollHeight;
        }
    }, [explanationData?.insights?.length]);

    if (!explanationData || !explanationData.phases?.length) {
        return (
            <div className="lt-container lt-empty">
                <div className="lt-header">
                    <Activity size={18} />
                    <span>Training Activity</span>
                </div>
                <p className="lt-empty-msg">Start training to see live AI progress here…</p>
            </div>
        );
    }

    const { phases, insights, trust, summary, current_phase } = explanationData;
    const progress = explanationData.progress || 0;

    const togglePhase = (phaseId) => {
        setExpandedPhases(prev => ({ ...prev, [phaseId]: !prev[phaseId] }));
    };

    return (
        <div className="lt-container">
            {/* Header */}
            <div className="lt-header">
                <Activity size={18} />
                <span>Training Activity</span>
                {isTraining && (
                    <div className="lt-live-indicator">
                        <span className="lt-live-dot" />
                        <span>LIVE</span>
                    </div>
                )}
                <span className="lt-progress-pct">{progress}%</span>
            </div>

            {/* Progress Bar */}
            <div className="lt-progress-wrap">
                <div className="lt-progress-track">
                    <div
                        className={`lt-progress-fill ${progress >= 100 ? 'lt-progress-done' : ''}`}
                        style={{ width: `${progress}%` }}
                    />
                </div>
            </div>

            {/* Phase Timeline */}
            <div className="lt-phases" ref={feedRef}>
                {phases.map((phase) => (
                    <PhaseCard
                        key={phase.id}
                        phase={phase}
                        isActive={phase.id === current_phase && !phase.completed_at}
                        isComplete={!!phase.completed_at}
                        insights={insights?.filter(i => i.phase === phase.id) || []}
                        isExpanded={expandedPhases[phase.id]}
                        onToggle={() => togglePhase(phase.id)}
                    />
                ))}
            </div>

            <TrustPanel trust={trust} />

            {summary && <FinalSummary summary={summary} />}
        </div>
    );
};

export default LearningTimeline;
