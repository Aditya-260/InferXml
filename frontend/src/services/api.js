import axios from 'axios'

const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || '/api',
    timeout: 30000,
    headers: {
        'Content-Type': 'application/json'
    }
})

// Request interceptor
api.interceptors.request.use(
    (config) => {
        // Get token from localStorage (persisted by zustand)
        const authData = localStorage.getItem('inferx-auth')
        if (authData) {
            try {
                const parsed = JSON.parse(authData)
                const token = parsed?.state?.token
                if (token) {
                    config.headers.Authorization = `Bearer ${token}`
                }
            } catch (e) {
                console.error('Failed to parse auth data:', e)
            }
        }
        return config
    },
    (error) => {
        return Promise.reject(error)
    }
)

// Response interceptor
api.interceptors.response.use(
    (response) => response,
    (error) => {
        // Handle 401 - Unauthorized
        if (error.response?.status === 401) {
            // Clear auth and redirect to login
            localStorage.removeItem('inferx-auth')
            window.location.href = '/login'
        }
        return Promise.reject(error)
    }
)

export default api

// API Methods
export const datasetsApi = {
    list: () => api.get('/datasets'),
    get: (id) => api.get(`/datasets/${id}`),
    upload: (formData, onProgress) =>
        api.post('/datasets/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            onUploadProgress: (e) => onProgress?.(Math.round((e.loaded * 100) / e.total))
        }),
    delete: (id) => api.delete(`/datasets/${id}`),
    getProfile: (id) => api.get(`/datasets/${id}/profile`),
    detectHeaders: (id) => api.get(`/datasets/${id}/detect-headers`, { timeout: 45000 }),
    setHeaders: (id, headers) => api.post(`/datasets/${id}/set-headers`, { headers }, { timeout: 120000 }),
    useSample: (sampleId) => api.post('/datasets/use-sample', { sample_id: sampleId }),
    searchKaggle: (query, page = 1) => api.get('/datasets/kaggle/search', { params: { query, page } }),
    downloadKaggle: (ref, name) => api.post('/datasets/kaggle/download', { ref, name }, { timeout: 120000 }),
    generateSynthetic: (config) => api.post('/datasets/generate-synthetic', config, { timeout: 60000 }),
    aiSchema: (prompt) => api.post('/datasets/ai/schema', { prompt }, { timeout: 30000 }),
    aiGenerate: (config) => api.post('/datasets/ai/generate', config, { timeout: 120000 }),
    preview: (id) => api.get(`/datasets/${id}/preview`),
}

export const trainingApi = {
    start: (data) => api.post('/training/start', data),
    getStatus: (jobId) => api.get(`/training/${jobId}/status`),
    getLogs: (jobId) => api.get(`/training/${jobId}/logs`),
    getThinkingLogs: (jobId) => api.get(`/training/${jobId}/thinking`),
    cancel: (jobId) => api.post(`/training/${jobId}/cancel`),
    ensembleTop3: (jobId) => api.post(`/training/${jobId}/ensemble`),
    analyzePrompt: (datasetId, prompt) => api.post('/training/analyze-prompt', { dataset_id: datasetId, prompt }),
    ensemble: (experimentId) => api.post(`/training/${experimentId}/ensemble`)
}

export const modelsApi = {
    list: () => api.get('/models'),
    get: (id) => api.get(`/models/${id}`),
    delete: (id) => api.delete(`/models/${id}`),
    download: (id) => api.get(`/models/${id}/download`, { responseType: 'blob' }),
    getSchema: (id) => api.get(`/models/${id}/schema`),
    getGraphs: (id) => api.get(`/models/${id}/graphs`)
}

export const predictionsApi = {
    predict: (modelId, input) => api.post(`/predict/${modelId}`, { input }),
    batchPredict: (modelId, formData) =>
        api.post(`/predict/${modelId}/batch`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        }),
    explain: (modelId, input) => api.post(`/predict/${modelId}/explain`, { input })
}

export const ordersApi = {
    list: (status) => api.get('/orders', { params: { status } }),
    get: (id) => api.get(`/orders/${id}`),
    create: (data) => api.post('/orders', data),
    updateItems: (id, items) => api.put(`/orders/${id}/items`, { items }),
    approve: (id) => api.post(`/orders/${id}/approve`),
    reject: (id, reason) => api.post(`/orders/${id}/reject`, { reason }),
    getPending: () => api.get('/orders/pending')
}

