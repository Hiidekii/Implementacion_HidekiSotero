import React from 'react';

const VideoOverlay = () => {
    return (
        <div
            id="contenedor-video"
            style={{
                position: 'relative',
                flexGrow: 1,
                aspectRatio: '16 / 9',
                maxWidth: '1024px',
                maxHeight: '576px',
                width: '100%',
            }}
        >
            <video
                id="video"
                autoPlay
                playsInline
                style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'fill',
                    border: '3px solid #000',
                    boxShadow: '0 0 10px rgba(0, 0, 0, 0.3)',
                    zIndex: 1,
                }}
            ></video>
            <canvas
                id="overlay"
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                    zIndex: 2,
                }}
            ></canvas>
        </div>
    );
};

export default VideoOverlay;
