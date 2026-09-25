import { useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'

export default function AuthCallback() {
    const navigate = useNavigate()
    const [params] = useSearchParams()
    const { handleOAuthCallback } = useAuthStore()

    useEffect(() => {
        const token = params.get('token')
        const user = params.get('user')

        if (token && user) {
            handleOAuthCallback(token, user)
            navigate('/', { replace: true })
        } else {
            navigate('/login', { replace: true })
        }
    }, [])

    return (
        <div
            style={{
                height: '100vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#09090b',
            }}
        >
            <div className="spinner" />
        </div>
    )
}
