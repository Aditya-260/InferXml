import React, { useRef, useState, useEffect } from 'react'
import { useScroll, useTransform, motion } from 'framer-motion'

export const ContainerScroll = ({ titleComponent, children }) => {
    const containerRef = useRef(null)
    const { scrollYProgress } = useScroll({
        target: containerRef,
    })
    const [viewport, setViewport] = useState({ isMobile: false, isTablet: false })

    useEffect(() => {
        const updateViewport = () => {
            const width = window.innerWidth
            setViewport({
                isMobile: width <= 640,
                isTablet: width <= 980,
            })
        }
        updateViewport()
        window.addEventListener('resize', updateViewport)
        return () => window.removeEventListener('resize', updateViewport)
    }, [])

    const { isMobile, isTablet } = viewport
    const height = isMobile ? '42rem' : isTablet ? '46rem' : '54rem'
    const outerPadding = isMobile ? '0.75rem' : isTablet ? '1rem' : '1.5rem'
    const innerPadding = isMobile ? '3.5rem 0 1rem' : isTablet ? '4.5rem 0 1.5rem' : '3rem 0'

    const rotate = useTransform(scrollYProgress, [0, 1], [isMobile ? 0 : isTablet ? 10 : 16, 0])
    const scale = useTransform(
        scrollYProgress,
        [0, 1],
        isMobile ? [1, 1] : isTablet ? [0.92, 0.98] : [0.98, 1]
    )
    const translate = useTransform(scrollYProgress, [0, 1], [0, isMobile ? -18 : isTablet ? -54 : -100])

    return (
        <div
            ref={containerRef}
            style={{
                height,
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'center',
                position: 'relative',
                padding: outerPadding,
                width: '100%',
            }}
        >
            <div
                style={{
                    padding: innerPadding,
                    width: '100%',
                    position: 'relative',
                    perspective: isMobile ? 'none' : '1000px',
                }}
            >
                <Header translate={translate} titleComponent={titleComponent} />
                <Card rotate={rotate} scale={scale} isMobile={isMobile}>
                    {children}
                </Card>
            </div>
        </div>
    )
}

const Header = ({ translate, titleComponent }) => {
    return (
        <motion.div
            style={{ translateY: translate }}
            className="scroll-header"
        >
            {titleComponent}
        </motion.div>
    )
}

const Card = ({ rotate, scale, children, isMobile }) => {
    return (
        <motion.div
            style={{
                rotateX: rotate,
                scale,
                transformStyle: isMobile ? 'flat' : 'preserve-3d',
                boxShadow: isMobile
                    ? '0 18px 36px rgba(0, 0, 0, 0.22)'
                    : '0 0 #0000004d, 0 9px 20px #0000004a, 0 37px 37px #00000042, 0 84px 50px #00000026, 0 149px 60px #0000000a, 0 233px 65px #00000003',
            }}
            className="scroll-card"
        >
            <div className="scroll-card-inner">
                {children}
            </div>
        </motion.div>
    )
}

export default ContainerScroll
