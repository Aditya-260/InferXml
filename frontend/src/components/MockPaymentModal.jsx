import React, { useState, useEffect, useRef } from 'react'
import { Check, CreditCard, Shield, Loader2 } from 'lucide-react'
import './MockPaymentModal.css'

const MockPaymentModal = ({ open, onClose, onSuccess, amount, currency, user }) => {
    const [step, setStep] = useState('confirm') // confirm → processing → done
    const timerRef = useRef(null)

    useEffect(() => {
        if (!open) setStep('confirm')
        return () => clearTimeout(timerRef.current)
    }, [open])

    if (!open) return null

    const handlePay = () => {
        setStep('processing')
        timerRef.current = setTimeout(() => {
            setStep('done')
            timerRef.current = setTimeout(() => {
                onSuccess()
                onClose()
            }, 1200)
        }, 1800)
    }

    return (
        <div className="mock-pay-overlay" onClick={onClose}>
            <div className="mock-pay-modal" onClick={e => e.stopPropagation()}>
                {step === 'confirm' && (
                    <>
                        <div className="mock-pay-header">
                            <Shield size={28} className="mock-pay-shield" />
                            <h3>Confirm Payment</h3>
                            <span className="mock-pay-tag">Demo Mode</span>
                        </div>
                        <div className="mock-pay-body">
                            <div className="mock-pay-amount">
                                <span className="mock-pay-currency">{currency === 'INR' ? '₹' : '$'}</span>
                                <span className="mock-pay-value">{(amount / 100).toFixed(0)}</span>
                            </div>
                            <p className="mock-pay-desc">InferX-ML Pro — 30 days</p>
                            <div className="mock-pay-details">
                                <div className="mock-pay-row">
                                    <span>Account</span>
                                    <span>{user?.email || 'user@example.com'}</span>
                                </div>
                                <div className="mock-pay-row">
                                    <span>Method</span>
                                    <span><CreditCard size={14} /> •••• 4242 (Test)</span>
                                </div>
                            </div>
                        </div>
                        <div className="mock-pay-actions">
                            <button className="mock-pay-btn-cancel" onClick={onClose}>Cancel</button>
                            <button className="mock-pay-btn-pay" onClick={handlePay}>
                                <CreditCard size={16} />
                                Pay {currency === 'INR' ? '₹' : '$'}{(amount / 100).toFixed(0)}
                            </button>
                        </div>
                    </>
                )}
                {step === 'processing' && (
                    <div className="mock-pay-status">
                        <Loader2 size={40} className="mock-pay-spinner" />
                        <h3>Processing Payment...</h3>
                        <p>Simulating secure payment gateway</p>
                    </div>
                )}
                {step === 'done' && (
                    <div className="mock-pay-status mock-pay-success">
                        <div className="mock-pay-check-circle">
                            <Check size={32} />
                        </div>
                        <h3>Payment Successful!</h3>
                        <p>Upgrading your account to Pro</p>
                    </div>
                )}
            </div>
        </div>
    )
}

export default MockPaymentModal
