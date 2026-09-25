import express from 'express'
import nodemailer from 'nodemailer'

const app = express()
app.use(express.json())

const port = Number(process.env.MAILER_PORT || 4000)
const fromEmail = process.env.SMTP_FROM || process.env.SMTP_USER || 'no-reply@inferx.local'
const productName = process.env.MAILER_PRODUCT_NAME || 'InferX-ML'

const transporter = nodemailer.createTransport({
    host: process.env.SMTP_HOST,
    port: Number(process.env.SMTP_PORT || 587),
    secure: String(process.env.SMTP_SECURE || 'false').toLowerCase() === 'true',
    auth: process.env.SMTP_USER
        ? {
            user: process.env.SMTP_USER,
            pass: process.env.SMTP_PASS,
        }
        : undefined,
})

function renderTemplate(type, payload = {}) {
    const expires = payload.expiresInMinutes || 10
    if (type === 'verification') {
        return {
            subject: `Verify your ${productName} account`,
            html: `
                <div style="font-family: Inter, Arial, sans-serif; max-width: 560px; margin: 0 auto; color: #111827;">
                    <h2 style="margin-bottom: 12px;">Verify your email</h2>
                    <p style="margin-bottom: 20px;">Hi ${payload.username || 'there'}, use this one-time code to verify your account.</p>
                    <div style="font-size: 32px; font-weight: 700; letter-spacing: 8px; padding: 18px 24px; background: #f3f4f6; border-radius: 12px; text-align: center; margin-bottom: 20px;">${payload.otp}</div>
                    <p style="margin-bottom: 8px;">This code expires in ${expires} minutes.</p>
                    <p style="color: #6b7280; font-size: 14px;">If you did not request this, you can safely ignore this email.</p>
                </div>
            `,
        }
    }

    if (type === 'password_reset') {
        return {
            subject: `Reset your ${productName} password`,
            html: `
                <div style="font-family: Inter, Arial, sans-serif; max-width: 560px; margin: 0 auto; color: #111827;">
                    <h2 style="margin-bottom: 12px;">Reset your password</h2>
                    <p style="margin-bottom: 20px;">Hi ${payload.username || 'there'}, use this one-time code to reset your password.</p>
                    <div style="font-size: 32px; font-weight: 700; letter-spacing: 8px; padding: 18px 24px; background: #f3f4f6; border-radius: 12px; text-align: center; margin-bottom: 20px;">${payload.otp}</div>
                    <p style="margin-bottom: 8px;">This code expires in ${expires} minutes.</p>
                    <p style="color: #6b7280; font-size: 14px;">If you did not request this, reset your password from the app and review your account activity.</p>
                </div>
            `,
        }
    }

    if (type === 'training_completed') {
        return {
            subject: `✅ Training complete — ${payload.experiment_name || 'Your model'}`,
            html: `
                <div style="font-family: Inter, Arial, sans-serif; max-width: 560px; margin: 0 auto; color: #111827;">
                    <h2 style="margin-bottom: 12px;">Training Completed 🎉</h2>
                    <p style="margin-bottom: 20px;">Hi ${payload.username || 'there'}, your model <strong>${payload.experiment_name || ''}</strong> has finished training successfully.</p>
                    <div style="padding: 16px 20px; background: #ecfdf5; border-left: 4px solid #10b981; border-radius: 8px; margin-bottom: 20px;">
                        <p style="margin: 0; font-weight: 600; color: #065f46;">Ready to use</p>
                        <p style="margin: 4px 0 0; color: #047857; font-size: 14px;">Head to the Models page to view metrics, download, or start making predictions.</p>
                    </div>
                    <p style="color: #6b7280; font-size: 13px;">You can manage notification preferences in Settings → Notifications.</p>
                </div>
            `,
        }
    }

    if (type === 'training_failed') {
        return {
            subject: `❌ Training failed — ${payload.experiment_name || 'Your model'}`,
            html: `
                <div style="font-family: Inter, Arial, sans-serif; max-width: 560px; margin: 0 auto; color: #111827;">
                    <h2 style="margin-bottom: 12px;">Training Failed</h2>
                    <p style="margin-bottom: 20px;">Hi ${payload.username || 'there'}, the training job for <strong>${payload.experiment_name || ''}</strong> encountered an error.</p>
                    <div style="padding: 16px 20px; background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 8px; margin-bottom: 20px;">
                        <p style="margin: 0; font-weight: 600; color: #991b1b;">Error Details</p>
                        <p style="margin: 4px 0 0; color: #b91c1c; font-size: 14px; word-break: break-word;">${payload.error_message || 'An unexpected error occurred.'}</p>
                    </div>
                    <p style="color: #6b7280; font-size: 13px;">Check the training logs for more details, or try adjusting your configuration and retraining.</p>
                </div>
            `,
        }
    }

    if (type === 'dataset_uploaded') {
        return {
            subject: `📂 Dataset uploaded — ${payload.dataset_name || 'New dataset'}`,
            html: `
                <div style="font-family: Inter, Arial, sans-serif; max-width: 560px; margin: 0 auto; color: #111827;">
                    <h2 style="margin-bottom: 12px;">Dataset Uploaded</h2>
                    <p style="margin-bottom: 20px;">Hi ${payload.username || 'there'}, your dataset <strong>${payload.dataset_name || ''}</strong> has been uploaded and profiled successfully.</p>
                    <div style="padding: 16px 20px; background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 8px; margin-bottom: 20px;">
                        <p style="margin: 0; font-weight: 600; color: #1e40af;">Next step</p>
                        <p style="margin: 4px 0 0; color: #1d4ed8; font-size: 14px;">Head to the Training page to start building a model with this dataset.</p>
                    </div>
                    <p style="color: #6b7280; font-size: 13px;">You can manage notification preferences in Settings → Notifications.</p>
                </div>
            `,
        }
    }

    if (type === 'rate_limit_warning') {
        return {
            subject: `⚠️ API rate limit warning — ${productName}`,
            html: `
                <div style="font-family: Inter, Arial, sans-serif; max-width: 560px; margin: 0 auto; color: #111827;">
                    <h2 style="margin-bottom: 12px;">Rate Limit Warning</h2>
                    <p style="margin-bottom: 20px;">Hi ${payload.username || 'there'}, you've used <strong>${payload.current_usage || '?'}</strong> of your <strong>${payload.limit || '?'}</strong> allowed API requests.</p>
                    <div style="padding: 16px 20px; background: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 8px; margin-bottom: 20px;">
                        <p style="margin: 0; font-weight: 600; color: #92400e;">Approaching limit</p>
                        <p style="margin: 4px 0 0; color: #b45309; font-size: 14px;">Consider upgrading to Pro for higher rate limits, or wait for your quota to reset.</p>
                    </div>
                    <p style="color: #6b7280; font-size: 13px;">You can manage notification preferences in Settings → Notifications.</p>
                </div>
            `,
        }
    }

    if (type === 'security_alert') {
        return {
            subject: `🔒 Security alert — ${payload.event_type || 'Account activity'}`,
            html: `
                <div style="font-family: Inter, Arial, sans-serif; max-width: 560px; margin: 0 auto; color: #111827;">
                    <h2 style="margin-bottom: 12px;">Security Alert</h2>
                    <p style="margin-bottom: 20px;">Hi ${payload.username || 'there'}, we detected the following activity on your account:</p>
                    <div style="padding: 16px 20px; background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 8px; margin-bottom: 20px;">
                        <p style="margin: 0; font-weight: 600; color: #991b1b;">${payload.event_type || 'Unusual activity'}</p>
                        <p style="margin: 4px 0 0; color: #b91c1c; font-size: 14px;">${payload.details || 'If this was you, no action is needed.'}</p>
                    </div>
                    <p style="color: #6b7280; font-size: 13px;">If this wasn't you, change your password immediately from Settings → Security.</p>
                </div>
            `,
        }
    }

    if (type === 'subscription_expired') {
        return {
            subject: `⏰ Your ${productName} Pro plan has expired`,
            html: `
                <div style="font-family: Inter, Arial, sans-serif; max-width: 560px; margin: 0 auto; color: #111827;">
                    <h2 style="margin-bottom: 12px;">Pro Plan Expired</h2>
                    <p style="margin-bottom: 20px;">Hi ${payload.username || 'there'}, your Pro subscription expired on <strong>${payload.expiry_date || 'recently'}</strong>.</p>
                    <div style="padding: 16px 20px; background: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 8px; margin-bottom: 20px;">
                        <p style="margin: 0; font-weight: 600; color: #92400e;">Your account has been moved to the Free plan</p>
                        <p style="margin: 4px 0 0; color: #b45309; font-size: 14px;">Training limits and algorithm access have been reduced. Renew anytime to restore full Pro features.</p>
                    </div>
                    <a href="${process.env.APP_URL || 'http://localhost:3000'}/pricing" style="display: inline-block; padding: 12px 24px; background: #10b981; color: #fff; border-radius: 8px; text-decoration: none; font-weight: 600;">Renew Pro Plan</a>
                    <p style="color: #6b7280; font-size: 13px; margin-top: 20px;">You can manage notification preferences in Settings → Notifications.</p>
                </div>
            `,
        }
    }

    if (type === 'subscription_renewed') {
        return {
            subject: `✅ ${productName} Pro plan renewed successfully`,
            html: `
                <div style="font-family: Inter, Arial, sans-serif; max-width: 560px; margin: 0 auto; color: #111827;">
                    <h2 style="margin-bottom: 12px;">Pro Plan Renewed 🎉</h2>
                    <p style="margin-bottom: 20px;">Hi ${payload.username || 'there'}, your Pro subscription has been renewed successfully.</p>
                    <div style="padding: 16px 20px; background: #ecfdf5; border-left: 4px solid #10b981; border-radius: 8px; margin-bottom: 20px;">
                        <p style="margin: 0; font-weight: 600; color: #065f46;">Plan: ${(payload.plan_name || 'pro').charAt(0).toUpperCase() + (payload.plan_name || 'pro').slice(1)}</p>
                        <p style="margin: 4px 0 0; color: #047857; font-size: 14px;">Valid until <strong>${payload.new_expiry_date || '—'}</strong>. Enjoy unlimited training, all algorithms, and priority support.</p>
                    </div>
                    <p style="color: #6b7280; font-size: 13px;">You can manage notification preferences in Settings → Notifications.</p>
                </div>
            `,
        }
    }

    throw new Error('Unsupported email template')
}

app.get('/health', (_req, res) => {
    res.json({ status: 'ok', service: 'inferx-mailer' })
})

app.post('/send', async (req, res) => {
    const { type, to, payload } = req.body || {}
    if (!type || !to) {
        return res.status(400).json({ error: 'type and to are required' })
    }

    try {
        const template = renderTemplate(type, payload)
        await transporter.sendMail({
            from: fromEmail,
            to,
            subject: template.subject,
            html: template.html,
        })
        return res.json({ message: 'sent' })
    } catch (error) {
        console.error('Mailer send failed', error)
        return res.status(502).json({ error: 'send_failed' })
    }
})

app.listen(port, '0.0.0.0', () => {
    console.log(`InferX mailer listening on ${port}`)
})
