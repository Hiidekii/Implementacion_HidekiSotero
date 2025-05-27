import React, { useRef, useEffect, useState } from 'react';
import { Box } from '@mui/material';
import EmotionLog from './components/EmotionLog';

const ScreenCap = () => {
    const videoRef = useRef(null);
    const overlayRef = useRef(null);
    const [logs, setLogs] = useState([]);
    const [intervalId, setIntervalId] = useState(null);

    const iniciarCaptura = async () => {
        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: { cursor: 'always' },
                audio: false,
            });
            videoRef.current.srcObject = stream;
            const id = setInterval(() => enviarFrame(), 1000);
            setIntervalId(id);
        } catch (error) {
            alert('Error al capturar pantalla');
            console.error(error);
        }
    };

    const detenerCaptura = () => {
        if (intervalId) {
            clearInterval(intervalId);
            setIntervalId(null);
        }

        const tracks = videoRef.current?.srcObject?.getTracks();
        tracks?.forEach(track => track.stop());
        videoRef.current.srcObject = null;

        const ctx = overlayRef.current?.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
        setLogs([]);
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

            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.fillRect(x, y - 25, w, 20);
            ctx.fillStyle = '#ffffff';
            ctx.font = '14px sans-serif';
            ctx.fillText(`${item.emocion}`, x + 5, y - 10);
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
        }}>
            {/* Contenedor de video */}
            <Box id="contenedor-video" sx={{
                position: 'relative',
                flexGrow: 1,
                aspectRatio: '16 / 9',
                maxWidth: '1024px',
                maxHeight: '576px',
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

            {/* Columna de logs */}
            <EmotionLog logs={logs} onStart={iniciarCaptura} onStop={detenerCaptura} />
        </Box>
    );
};

export default ScreenCap;