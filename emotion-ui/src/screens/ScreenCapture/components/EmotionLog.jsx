import React, { useEffect, useRef } from 'react';
import { Box, Button, Typography } from '@mui/material';
import '../../../styles/animation.css';

const EmotionLog = ({ logs, onStart, onStop }) => {
    const logContainerRef = useRef(null);

    const emotionColorMap = {
        Happy: '#ffc107',
        Sad: '#6c757d',
        Anger: '#dc3545',
        Surprise: '#0dcaf0',
        Neutral: '#adb5bd',
        Disgust: '#198754',
        Fear: '#6610f2',
    };

    // Auto-scroll solo si ya está abajo
    useEffect(() => {
        if (logContainerRef.current) {
            // Esperar al siguiente ciclo del render para asegurar que el contenido esté montado
            const scrollToBottom = () => {
                logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
            };
            // Dos técnicas combinadas para asegurar ejecución después del render
            requestAnimationFrame(() => setTimeout(scrollToBottom, 0));
        }
    }, [logs]);

    return (
        <Box
            id="emociones-columna"
            sx={{
                display: 'flex',
                flexDirection: 'column',
                width: '320px',
                height: '576px',
            }}
        >
            <Box
                id="emociones"
                ref={logContainerRef}
                sx={{
                    flexGrow: 1,
                    overflowY: 'auto',
                    padding: '1rem',
                    backgroundColor: '#ffffff',
                    border: '2px solid #007BFF',
                    borderRadius: '8px',
                    boxShadow: '0 4px 10px rgba(0, 123, 255, 0.2)',
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    marginBottom: '1rem',
                    display: 'flex',
                    flexDirection: 'column',
                }}
            >
                {logs.map((item, i) => (
                    <Box
                        key={`${item.rostro}-${i}-${item.hora}`}
                        className="fade-in-log"
                        sx={{
                            marginBottom: '0.5rem',
                            padding: '0.5rem 0.8rem',
                            backgroundColor: '#eaf4ff',
                            borderLeft: `4px solid ${emotionColorMap[item.emocion] || '#007BFF'}`,
                            borderRadius: '4px',
                            fontWeight: 'bold',
                            color: '#000',
                            transition: 'background-color 0.3s',
                        }}
                    >
                        {item.rostro} - {item.emocion} - <span style={{ fontSize: '0.75rem' }}>{item.hora}</span>
                    </Box>
                ))}
            </Box>

            <Box
                id="botones"
                sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: '1rem',
                    height: '15%',
                }}
            >
                <Button
                    fullWidth
                    variant="contained"
                    onClick={onStart}
                    sx={{
                        backgroundColor: '#007BFF',
                        color: '#fff',
                        '&:hover': { backgroundColor: '#0056b3' },
                    }}
                >
                    Iniciar captura
                </Button>
                <Button
                    fullWidth
                    variant="contained"
                    onClick={onStop}
                    sx={{
                        backgroundColor: '#dc3545',
                        color: '#fff',
                        '&:hover': { backgroundColor: '#b02a37' },
                    }}
                >
                    Pausar captura
                </Button>
            </Box>
        </Box>
    );
};

export default EmotionLog;
