import { ArrowLeft, ArrowRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState } from 'react';
import './AnimatedTestimonials.css';

export const AnimatedTestimonials = ({
    testimonials,
    autoplay = false,
    className = '',
}) => {
    const [active, setActive] = useState(0);

    const handleNext = () => {
        setActive((prev) => (prev + 1) % testimonials.length);
    };

    const handlePrev = () => {
        setActive((prev) => (prev - 1 + testimonials.length) % testimonials.length);
    };

    const isActive = (index) => {
        return index === active;
    };

    useEffect(() => {
        if (autoplay) {
            const interval = setInterval(handleNext, 5000);
            return () => clearInterval(interval);
        }
    }, [autoplay]);

    const randomRotateY = () => {
        return Math.floor(Math.random() * 21) - 10;
    };

    return (
        <div className={`animated-testimonials-container ${className}`}>
            <div className="at-grid">

                {/* Image Section */}
                <div className="at-image-container">
                    <AnimatePresence>
                        {testimonials.map((testimonial, index) => (
                            <motion.div
                                key={testimonial.src} // Use unique key if src is not unique
                                initial={{
                                    opacity: 0,
                                    scale: 0.9,
                                    z: -100,
                                    rotate: randomRotateY(),
                                }}
                                animate={{
                                    opacity: isActive(index) ? 1 : 0.7,
                                    scale: isActive(index) ? 1 : 0.95,
                                    z: isActive(index) ? 0 : -100,
                                    rotate: isActive(index) ? 0 : randomRotateY(),
                                    zIndex: isActive(index)
                                        ? 999
                                        : testimonials.length + 2 - index,
                                    y: isActive(index) ? [0, -80, 0] : 0,
                                }}
                                exit={{
                                    opacity: 0,
                                    scale: 0.9,
                                    z: 100,
                                    rotate: randomRotateY(),
                                }}
                                transition={{
                                    duration: 0.4,
                                    ease: 'easeInOut',
                                }}
                                className="at-image-wrapper"
                            >
                                <img
                                    src={testimonial.src}
                                    alt={testimonial.name}
                                    draggable={false}
                                    className="at-image"
                                />
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>

                {/* Content Section */}
                <div className="at-content">
                    <motion.div
                        key={active}
                        initial={{
                            y: 20,
                            opacity: 0,
                        }}
                        animate={{
                            y: 0,
                            opacity: 1,
                        }}
                        exit={{
                            y: -20,
                            opacity: 0,
                        }}
                        transition={{
                            duration: 0.2,
                            ease: 'easeInOut',
                        }}
                    >
                        <h3 className="at-name">
                            {testimonials[active].name}
                        </h3>
                        <p className="at-designation">
                            {testimonials[active].designation}
                        </p>
                        <motion.p className="at-quote">
                            {testimonials[active].quote.split(' ').map((word, index) => (
                                <motion.span
                                    key={index}
                                    initial={{
                                        filter: 'blur(10px)',
                                        opacity: 0,
                                        y: 5,
                                    }}
                                    animate={{
                                        filter: 'blur(0px)',
                                        opacity: 1,
                                        y: 0,
                                    }}
                                    transition={{
                                        duration: 0.2,
                                        ease: 'easeInOut',
                                        delay: 0.02 * index,
                                    }}
                                    className="at-word"
                                >
                                    {word}&nbsp;
                                </motion.span>
                            ))}
                        </motion.p>
                    </motion.div>

                    <div className="at-controls">
                        <button
                            onClick={handlePrev}
                            className="at-btn group/button"
                        >
                            <ArrowLeft className="arrow-left" size={20} />
                        </button>
                        <button
                            onClick={handleNext}
                            className="at-btn group/button"
                        >
                            <ArrowRight className="arrow-right" size={20} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
