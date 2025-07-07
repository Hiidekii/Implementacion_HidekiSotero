import React, { useRef, useEffect, useState } from 'react';
import { Box, Typography, Button, TextField } from '@mui/material';
import EmotionLog from './components/EmotionLog';
import EmotionPieChart from './components/EmotionPieChart';
import EmotionWaveChart from './components/EmotionWaveChart';

const ScreenCap = () => {
    const videoRef = useRef(null);
    const overlayRef = useRef(null);
    const [logs, setLogs] = useState([]);
    const [frameEmotions, setFrameEmotions] = useState([]);
    const [timelineData, setTimelineData] = useState([]);
    const [intervalId, setIntervalId] = useState(null);
    const [markerText, setMarkerText] = useState('');
    const [markers, setMarkers] = useState([]);

    const iniciarCaptura = async () => {
        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: { cursor: 'always' },
                audio: false,
            });
            videoRef.current.srcObject = stream;
            const id = setInterval(() => enviarFrame(), 3000);
            setIntervalId(id);
        } catch (error) {
            alert('Error al capturar pantalla');
            console.error(error);
        }
    };

    const pausarCaptura = () => {
        if (intervalId) {
            clearInterval(intervalId);
            setIntervalId(null);
        }

        const tracks = videoRef.current?.srcObject?.getTracks();
        tracks?.forEach(track => track.stop());
        videoRef.current.srcObject = null;

        const ctx = overlayRef.current?.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
    };

    const enviarFrame = async () => {
        const video = videoRef.current;
        if (!video || video.videoWidth === 0 || video.videoHeight === 0) return;

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const base64 = canvas.toDataURL('image/jpeg');

        try {
            const response = await fetch('http://localhost:5000/detect_emotion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: base64,
                    width: video.videoWidth,
                    height: video.videoHeight,
                }),
            });

            const data = await response.json();
            if (data.result) {
                setLogs(prev => [...prev, ...data.result]);
                setFrameEmotions(data.result);

                const total = data.result.length;
                const emotionFrame = {
                    timestamp: new Date().toLocaleTimeString(),
                    Happy: 0,
                    Sad: 0,
                    Anger: 0,
                    Surprise: 0,
                    Neutral: 0,
                    Disgust: 0,
                    Fear: 0
                };

                data.result.forEach(({ emocion }) => {
                    if (emotionFrame.hasOwnProperty(emocion)) {
                        emotionFrame[emocion]++;
                    }
                });

                Object.keys(emotionFrame).forEach(key => {
                    if (key !== 'timestamp') {
                        emotionFrame[key] = total > 0 ? emotionFrame[key] / total : 0;
                    }
                });

                setTimelineData(prev => [...prev, emotionFrame]);
                dibujarBoundingBoxes(data.result);
            }
        } catch (error) {
            console.error('Error al enviar frame:', error);
        }
    };

    const dibujarBoundingBoxes = (lista) => {
        const overlay = overlayRef.current;
        const ctx = overlay.getContext('2d');
        overlay.width = videoRef.current.videoWidth;
        overlay.height = videoRef.current.videoHeight;
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        lista.forEach(item => {
            const { x, y, w, h } = item.box;
            ctx.strokeStyle = '#00FF00';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            const label = `${item.emocion}`;
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.fillRect(x, y - 30, w, 24);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 20px sans-serif';
            ctx.fillText(label, x + 5, y - 10);
        });
    };

    return (
        <Box id="contenido" sx={{
            display: 'flex',
            alignItems: 'stretch',
            justifyContent: 'center',
            gap: '2rem',
            padding: '0.5rem 1rem',
            backgroundColor: '#ffffff',
            minHeight: '100vh',
            flexWrap: 'wrap'
        }}>
            <Box id="contenedor-video" sx={{
                position: 'relative',
                flexGrow: 1,
                aspectRatio: '16 / 9',
                maxWidth: '1280px',
                maxHeight: '720px',
                width: '100%',
            }}>
                <video ref={videoRef} autoPlay playsInline id="video" style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'fill',
                    border: '3px solid black',
                    boxShadow: '0 0 10px rgba(0,0,0,0.3)',
                    zIndex: 1,
                }}></video>
                <canvas ref={overlayRef} id="overlay" style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                    zIndex: 2,
                }}></canvas>
            </Box>

            <EmotionLog logs={logs} onStart={iniciarCaptura} onStop={pausarCaptura} />

            <Typography
                variant="h6"
                sx={{ width: '100%', textAlign: 'left' }}
            >
                Evolución de las emociones en tiempo real
            </Typography>

            <Box sx={{
                display: 'flex',
                flexDirection: 'row',
                justifyContent: 'center',
                gap: '2rem',
                width: '100%',
                flexWrap: 'wrap'
            }}>
                <Box sx={{ flex: 1, minWidth: 600, maxWidth: 500 }}>
                    <EmotionPieChart frameEmotions={frameEmotions} />
                    <Box display="flex" alignItems="center" gap={1} mt={30}>
                        <TextField
                            label="Agregar etiqueta temporal"
                            value={markerText}
                            onChange={(e) => setMarkerText(e.target.value)}
                            size="small"
                            fullWidth
                        />
                        <Button
                            variant="outlined"
                            onClick={() => {
                                const timestamp = new Date().toLocaleTimeString();

                                setTimelineData(prev => {
                                    const updated = [...prev];
                                    if (updated.length > 0) {
                                        updated[updated.length - 1] = {
                                            ...updated[updated.length - 1],
                                            marker: markerText
                                        };
                                    }
                                    return updated;
                                });

                                setMarkers(prev => [...prev, { timestamp, label: markerText }]);
                                setMarkerText('');
                            }}
                        >
                            Marcar
                        </Button>
                    </Box>
                </Box>

                <Box sx={{ flex: 2, minWidth: 500 }}>
                    <EmotionWaveChart timelineData={timelineData} />
                </Box>
            </Box>

            <Button
                variant="contained"
                color="error"
                sx={{ mt: 4 }}
                onClick={() => {
                    localStorage.setItem('emotionTimeline', JSON.stringify(timelineData));
                    window.location.href = '/resumen';
                }}
            >
                Finalizar captura
            </Button>
        </Box>
    );
};

export default ScreenCap;
