import { useState, useCallback } from 'react'

let toastId = 0

export function useToast() {
  const [toasts, setToasts] = useState([])

  const toast = useCallback(({ title, description, variant = "default", duration = 4000 }) => {
    const id = ++toastId

    // Create and show a DOM toast notification
    const container = document.getElementById('toast-container') || createToastContainer()

    const el = document.createElement('div')
    el.id = `toast-${id}`
    el.style.cssText = `
      padding: 16px 20px;
      border-radius: 10px;
      margin-bottom: 10px;
      font-family: 'Inter', system-ui, sans-serif;
      font-size: 14px;
      color: #fff;
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.1);
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
      animation: toast-in 0.3s ease-out;
      max-width: 380px;
      word-wrap: break-word;
      background: ${variant === 'destructive' ? 'rgba(220, 38, 38, 0.9)' : 'rgba(30, 30, 46, 0.95)'};
    `

    el.innerHTML = `
      ${title ? `<div style="font-weight:600;margin-bottom:${description ? '4px' : '0'}">${title}</div>` : ''}
      ${description ? `<div style="opacity:0.85;font-size:13px">${description}</div>` : ''}
    `

    container.appendChild(el)

    setTimeout(() => {
      el.style.animation = 'toast-out 0.3s ease-in forwards'
      setTimeout(() => el.remove(), 300)
    }, duration)

    setToasts(prev => [...prev, { id, title, description, variant }])
  }, [])

  const dismiss = useCallback((id) => {
    const el = document.getElementById(`toast-${id}`)
    if (el) el.remove()
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  return { toast, toasts, dismiss }
}

function createToastContainer() {
  const container = document.createElement('div')
  container.id = 'toast-container'
  container.style.cssText = `
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 99999;
    display: flex;
    flex-direction: column-reverse;
    pointer-events: none;
  `

  // Inject keyframe animations
  const style = document.createElement('style')
  style.textContent = `
    @keyframes toast-in {
      from { opacity: 0; transform: translateY(16px) scale(0.96); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes toast-out {
      from { opacity: 1; transform: translateY(0) scale(1); }
      to   { opacity: 0; transform: translateY(16px) scale(0.96); }
    }
    #toast-container > div { pointer-events: auto; }
  `
  document.head.appendChild(style)
  document.body.appendChild(container)
  return container
}
