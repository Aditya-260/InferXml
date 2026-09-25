import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import api from '../services/api'
import queryClient from '../queryClient'

export const useAuthStore = create(
    persist(
        (set, get) => ({
            user: null,
            token: null,
            isAuthenticated: false,

            applyAuth: ({ access_token, user }) => {
                set({
                    user,
                    token: access_token,
                    isAuthenticated: true
                })
                api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`
            },

            login: async (email, password) => {
                try {
                    const response = await api.post('/auth/login', { email, password })
                    get().applyAuth(response.data)

                    return { success: true }
                } catch (error) {
                    const data = error.response?.data || {}
                    return {
                        success: false,
                        error: data.error || 'Login failed',
                        code: data.code,
                        requiresVerification: data.requires_verification,
                        email: data.email
                    }
                }
            },

            register: async (email, username, password) => {
                try {
                    const response = await api.post('/auth/register', {
                        email,
                        username,
                        password
                    })

                    return { success: true, data: response.data }
                } catch (error) {
                    return {
                        success: false,
                        error: error.response?.data?.error || 'Registration failed'
                    }
                }
            },

            verifyEmail: async (email, otp) => {
                try {
                    const response = await api.post('/auth/verify-email', { email, otp })
                    get().applyAuth(response.data)
                    return { success: true }
                } catch (error) {
                    return {
                        success: false,
                        error: error.response?.data?.error || 'Verification failed'
                    }
                }
            },

            resendVerification: async (email) => {
                try {
                    const response = await api.post('/auth/resend-verification', { email })
                    return { success: true, message: response.data.message }
                } catch (error) {
                    return {
                        success: false,
                        error: error.response?.data?.error || 'Unable to resend code'
                    }
                }
            },

            forgotPassword: async (email) => {
                try {
                    const response = await api.post('/auth/forgot-password', { email })
                    return { success: true, message: response.data.message }
                } catch (error) {
                    return {
                        success: false,
                        error: error.response?.data?.error || 'Unable to send reset code'
                    }
                }
            },

            resetPassword: async (email, otp, password) => {
                try {
                    const response = await api.post('/auth/reset-password', { email, otp, password })
                    return { success: true, message: response.data.message }
                } catch (error) {
                    return {
                        success: false,
                        error: error.response?.data?.error || 'Unable to reset password'
                    }
                }
            },

            handleOAuthCallback: (token, user) => {
                const parsed = typeof user === 'string' ? JSON.parse(user) : user
                set({
                    user: parsed,
                    token,
                    isAuthenticated: true,
                })
                api.defaults.headers.common['Authorization'] = `Bearer ${token}`
            },

            logout: () => {
                set({
                    user: null,
                    token: null,
                    isAuthenticated: false
                })
                delete api.defaults.headers.common['Authorization']
                // Clear all react-query cached data so the next user starts fresh
                queryClient.clear()
            },

            checkAuth: () => {
                const { token } = get()
                if (token) {
                    api.defaults.headers.common['Authorization'] = `Bearer ${token}`
                    set({ isAuthenticated: true })
                }
            }
        }),
        {
            name: 'inferx-auth',
            partialize: (state) => ({
                token: state.token,
                user: state.user
            })
        }
    )
)
