import React, { useState, useEffect } from 'react'
import { useAuthStore } from '../store/authStore'
import { useToast } from '../components/ui/use-toast'
import { Crown, Check, X, Zap, Sparkles, Star } from 'lucide-react'
import './PricingPage.css'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

export default function PricingPage() {
    const { user, checkAuth } = useAuthStore()
    const { toast } = useToast()
    const [loading, setLoading] = useState(false)
    const [scriptLoaded, setScriptLoaded] = useState(false)
    const [billingConfig, setBillingConfig] = useState(null)

    // Load Razorpay checkout SDK
    useEffect(() => {
        // Fetch dynamic pricing from backend
        fetch(`${API_BASE}/billing/config`)
            .then(res => res.json())
            .then(data => setBillingConfig(data))
            .catch(err => console.error('Failed to fetch billing config:', err))

        if (document.querySelector('script[src*="razorpay"]')) {
            setScriptLoaded(true)
            return
        }
        const script = document.createElement('script')
        script.src = 'https://checkout.razorpay.com/v1/checkout.js'
        script.async = true
        script.onload = () => setScriptLoaded(true)
        script.onerror = () => {
            console.error('Failed to load Razorpay SDK')
            toast({ title: 'Error', description: 'Payment gateway failed to load. Please refresh.', variant: 'destructive' })
        }
        document.body.appendChild(script)

        return () => {
            if (document.body.contains(script)) {
                document.body.removeChild(script)
            }
        }
    }, [])

    const handleUpgrade = async (planName) => {
        if (!scriptLoaded) {
            toast({
                title: "Error",
                description: "Payment gateway is still loading. Please try again in a few seconds.",
                variant: "destructive"
            })
            return
        }

        setLoading(planName)
        try {
            const authData = JSON.parse(localStorage.getItem('inferx-auth') || '{}')
            const token = authData?.state?.token
            if (!token) throw new Error('Not authenticated. Please log in again.')

            // 1. Create order on backend
            const orderRes = await fetch(`${API_BASE}/billing/create-order`, {
                method: 'POST',
                headers: { 
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ plan: planName })
            })

            const orderData = await orderRes.json()
            if (!orderRes.ok) throw new Error(orderData.error || 'Failed to create order')

            const isAdvance = planName === 'advance'

            // 2. Open Razorpay checkout
            const options = {
                key: orderData.key_id,
                amount: orderData.amount,
                currency: orderData.currency,
                name: "InferX-ML",
                description: isAdvance ? "Upgrade to Advance — 1 Year" : "Upgrade to Pro — 30 days",
                order_id: orderData.order_id,
                handler: async function (response) {
                    // 3. Verify payment on backend
                    try {
                        const verifyRes = await fetch(`${API_BASE}/billing/verify-payment`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Authorization': `Bearer ${token}`
                            },
                            body: JSON.stringify({
                                razorpay_payment_id: response.razorpay_payment_id,
                                razorpay_order_id: response.razorpay_order_id,
                                razorpay_signature: response.razorpay_signature
                            })
                        })

                        const verifyData = await verifyRes.json()
                        if (verifyRes.ok) {
                            toast({ title: "🎉 Success!", description: `You have been upgraded to the ${isAdvance ? 'Advance' : 'Pro'} plan!` })
                            checkAuth()
                        } else {
                            throw new Error(verifyData.error || 'Payment verification failed')
                        }
                    } catch (err) {
                        toast({ title: "Verification Failed", description: err.message, variant: "destructive" })
                    }
                },
                prefill: {
                    name: user?.username || "",
                    email: user?.email || ""
                },
                theme: { color: isAdvance ? "#8b5cf6" : "#10b981" },
                modal: {
                    ondismiss: function () {
                        setLoading(false)
                    }
                }
            }

            const rzp = new window.Razorpay(options)
            rzp.on('payment.failed', function (response) {
                toast({
                    title: "Payment Failed",
                    description: response.error.description || 'Something went wrong with the payment.',
                    variant: "destructive"
                })
            })
            rzp.open()

        } catch (error) {
            toast({ title: "Error", description: error.message, variant: "destructive" })
        } finally {
            setLoading(false)
        }
    }

    const freePlan = [
        { text: '1 training per day', included: true },
        { text: '4 ML algorithms', included: true },
        { text: '20 API requests / min', included: true },
        { text: 'Basic model export', included: true },
        { text: 'XGBoost & LightGBM', included: false },
        { text: 'Advanced deep learning', included: false },
        { text: 'Priority support', included: false },
    ]

    const proPlan = [
        { text: 'Unlimited training jobs', included: true },
        { text: '10+ ML algorithms', included: true },
        { text: '200 API requests / min', included: true },
        { text: 'Advanced model export', included: true },
        { text: 'XGBoost & LightGBM', included: true },
        { text: 'YOLOv8 deep learning', included: true },
        { text: 'Priority support', included: true },
    ]

    const advancePlan = [
        { text: 'Everything in Pro', included: true },
        { text: '1 Year Full Access', included: true },
        { text: 'Priority API access', included: true },
        { text: 'Early access features', included: true },
        { text: 'Dedicated support', included: true },
        { text: 'Custom integrations', included: true },
        { text: 'Onboarding session', included: true },
    ]

    const isFree = user?.plan_type === 'free'
    const isPro = user?.plan_type === 'pro'
    const isAdvance = user?.plan_type === 'advance'

    return (
        <div className="pp-page">
            {/* Background decoration */}
            <div className="pp-bg-orb pp-orb-1" />
            <div className="pp-bg-orb pp-orb-2" />

            <div className="pp-header">
                <div className="pp-badge">
                    <Crown size={14} />
                    <span>Pricing</span>
                </div>
                <h1 className="pp-title">Choose your plan</h1>
                <p className="pp-subtitle">
                    Start free. Upgrade when you need more power for production workloads.
                </p>
            </div>

            <div className="pp-grid pp-grid-3">
                {/* ── Free Plan ── */}
                <div className={`pp-card ${isFree ? 'pp-card-active' : ''}`}>
                    {isFree && (
                        <div className="pp-current-badge">Current Plan</div>
                    )}
                    <div className="pp-card-top">
                        <span className="pp-plan-label">Free</span>
                        <div className="pp-price-row">
                            <span className="pp-currency">₹</span>
                            <span className="pp-amount">0</span>
                            <span className="pp-period">/month</span>
                        </div>
                        <p className="pp-plan-desc">Perfect for learning & small experiments</p>
                    </div>

                    <div className="pp-divider" />

                    <ul className="pp-features">
                        {freePlan.map((f, i) => (
                            <li key={i} className={f.included ? '' : 'pp-disabled'}>
                                {f.included
                                    ? <Check size={16} className="pp-icon-check" />
                                    : <X size={16} className="pp-icon-x" />
                                }
                                {f.text}
                            </li>
                        ))}
                    </ul>

                    <button className="pp-btn pp-btn-ghost" disabled>
                        {isFree ? '✓ Active' : 'Free Tier'}
                    </button>
                </div>

                {/* ── Pro Plan (Monthly) ── */}
                <div className={`pp-card pp-card-pro ${isPro ? 'pp-card-active' : ''}`}>
                    <div className="pp-popular-tag">
                        <Zap size={12} />
                        Most Popular
                    </div>
                    {isPro && (
                        <div className="pp-current-badge pp-current-pro">Current Plan</div>
                    )}
                    <div className="pp-card-top">
                        <span className="pp-plan-label pp-label-pro">Pro</span>
                        <div className="pp-price-row">
                            <span className="pp-currency">₹</span>
                            <span className="pp-amount pp-amount-pro">
                                {billingConfig ? (billingConfig.pro_plan_amount / 100) : '499'}
                            </span>
                            <span className="pp-period">/month</span>
                        </div>
                        <p className="pp-plan-desc">For serious ML practitioners & production use</p>
                    </div>

                    <div className="pp-divider pp-divider-pro" />

                    <ul className="pp-features">
                        {proPlan.map((f, i) => (
                            <li key={i}>
                                <Check size={16} className="pp-icon-check pp-check-pro" />
                                {f.text}
                            </li>
                        ))}
                    </ul>

                    <button
                        className="pp-btn pp-btn-primary"
                        onClick={() => handleUpgrade('pro')}
                        disabled={loading || isPro || isAdvance || !scriptLoaded}
                    >
                        {loading === 'pro' ? (
                            <>
                                <span className="pp-spinner" />
                                Processing...
                            </>
                        ) : isPro ? (
                            <>
                                <Sparkles size={16} />
                                Pro Activated
                            </>
                        ) : (
                            <>
                                <Crown size={16} />
                                Upgrade to Pro
                            </>
                        )}
                    </button>
                </div>

                {/* ── Advance Plan (Yearly) ── */}
                <div className={`pp-card pp-card-advance ${isAdvance ? 'pp-card-active' : ''}`}>
                    {isAdvance && (
                        <div className="pp-current-badge pp-current-advance">Current Plan</div>
                    )}
                    <div className="pp-card-top">
                        <span className="pp-plan-label pp-label-advance">Advance</span>
                        <div className="pp-price-row">
                            <span className="pp-currency">₹</span>
                            <span className="pp-amount pp-amount-advance">
                                {billingConfig ? (billingConfig.advance_plan_amount / 100) : '4999'}
                            </span>
                            <span className="pp-period">/year</span>
                        </div>
                        <p className="pp-plan-desc">Best value for long-term production workloads</p>
                    </div>

                    <div className="pp-divider pp-divider-advance" />

                    <ul className="pp-features">
                        {advancePlan.map((f, i) => (
                            <li key={i}>
                                <Check size={16} className="pp-icon-check pp-check-advance" />
                                {f.text}
                            </li>
                        ))}
                    </ul>

                    <button
                        className="pp-btn pp-btn-advance"
                        onClick={() => handleUpgrade('advance')}
                        disabled={loading || isAdvance || !scriptLoaded}
                    >
                        {loading === 'advance' ? (
                            <>
                                <span className="pp-spinner" />
                                Processing...
                            </>
                        ) : isAdvance ? (
                            <>
                                <Star size={16} />
                                Advance Activated
                            </>
                        ) : (
                            <>
                                <Star size={16} />
                                Get Advance Plan
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Bottom note */}
            <p className="pp-footnote">
                All plans include secure model storage, real-time training logs, and AI-powered insights.
            </p>
        </div>
    )
}
