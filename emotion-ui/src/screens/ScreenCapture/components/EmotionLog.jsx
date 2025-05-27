import React, { useEffect, useRef } from 'react';
import { Box, Button, Paper, Typography } from '@mui/material';

const EmotionLog = ({ logs, onStart, onStop }) => {
    const logContainerRef = useRef(null);
    const emotionColorMap = {
        Happy: '#ffc107',
        Sad: '#6c757d',
        Anger: '#dc3545',
        Surprise: '#0dcaf0',
        Neutral: '#adb5bd',
        Disgust: '#198754',
        Fear: '#6610f2'
    };

    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
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
                    <Paper
                        key={`${item.rostro}-${i}-${item.hora}`}
                        elevation={2}
                        className="fade-in-log"
                        sx={{
                            padding: '0.4rem 0.6rem',
                            marginBottom: '0.4rem',
                            borderLeft: `5px solid ${emotionColorMap[item.emocion] || '#007BFF'}`,
                            backgroundColor: '#f9f9f9',
                            fontFamily: 'monospace',
                            fontSize: '0.82rem',
                            lineHeight: 1.2,
                            transition: 'all 0.3s ease-out',
                        }}
                    >
                        <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                            {item.rostro}: {item.emocion} <span style={{ float: 'right', fontSize: '0.75rem' }}>{item.hora}</span>
                        </Typography>
                    </Paper>
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
                    Detener captura
                </Button>
            </Box>
        </Box>
    );
};

export default EmotionLog;
