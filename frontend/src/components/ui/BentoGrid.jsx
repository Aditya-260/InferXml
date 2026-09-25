import React from 'react'
import { ArrowRight } from 'lucide-react'

/**
 * BentoGrid + BentoCard — adapted from shadcn/aceternity to vanilla CSS.
 * No Tailwind, no TypeScript, no extra deps.
 */

const BentoGrid = ({ children, className = '' }) => {
    return (
        <div className={`bento-grid ${className}`}>
            {children}
        </div>
    )
}

const BentoCard = ({ name, className = '', background, Icon, description, href, cta }) => {
    return (
        <div className={`bento-card ${className}`}>
            {background && <div className="bento-bg">{background}</div>}
            <div className="bento-content">
                <Icon className="bento-icon" size={28} />
                <h3 className="bento-title">{name}</h3>
                <p className="bento-desc">{description}</p>
            </div>
            {cta && (
                <div className="bento-cta">
                    <a href={href || '#'} className="bento-cta-link">
                        {cta}
                        <ArrowRight size={14} />
                    </a>
                </div>
            )}
            <div className="bento-hover-overlay" />
        </div>
    )
}

export { BentoGrid, BentoCard }
