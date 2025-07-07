import React, { useEffect, useState } from 'react';
import { ReferenceLine } from 'recharts';
import { Box, Typography } from '@mui/material';
import {
    LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid,
    PieChart, Pie, Cell
} from 'recharts';

const COLORS = {
    Happy: '#ffc107',
    Sad: '#6c757d',
    Anger: '#dc3545',
    Surprise: '#0dcaf0',
    Neutral: '#adb5bd',
    Disgust: '#198754',
    Fear: '#6610f2',
};

const EMOTION_EMOJIS = {
    Happy: '😊',
    Sad: '😔',
    Anger: '😡',
    Surprise: '😲',
    Neutral: '😐',
    Disgust: '🤢',
    Fear: '😨',
};

const Resumen = () => {
    const [data, setData] = useState([]);
    const [pieData, setPieData] = useState([]);
    const [hoverDominantEmotion, setHoverDominantEmotion] = useState(null);
    const [markers, setMarkers] = useState([]);
    const [selectedMarker, setSelectedMarker] = useState(null);

    useEffect(() => {
        const stored = localStorage.getItem('emotionTimeline');
        if (stored) {
            const timeline = JSON.parse(stored);
            setData(timeline);

            const totals = {};
            Object.keys(COLORS).forEach((e) => (totals[e] = 0));
            timeline.forEach(frame => {
                Object.keys(COLORS).forEach(emotion => {
                    totals[emotion] += frame[emotion] || 0;
                });
            });

            const detectedMarkers = timeline
                .filter(item => item.marker)
                .map(item => ({ timestamp: item.timestamp, label: item.marker }));

            setMarkers(detectedMarkers);

            const pie = Object.entries(totals).map(([name, value]) => ({
                name,
                value: Number(value.toFixed(2)),
            }));

            setPieData(pie);
        }
    }, []);

    const handleMouseMove = (state) => {
        if (state && state.activePayload && state.activePayload.length > 0) {
            const payload = state.activePayload[0].payload;
            const entries = Object.entries(payload).filter(([key]) => Object.keys(COLORS).includes(key));
            if (entries.length > 0) {
                const [maxEmotion] = entries.reduce((prev, curr) => (curr[1] > prev[1] ? curr : prev));
                setHoverDominantEmotion(maxEmotion);
            }
        }
    };

    return (
        <Box p={4}>
            <Typography variant="h5" mb={3}>Resumen de emociones durante la sesión</Typography>

            <ResponsiveContainer width="100%" height={400}>
                <LineChart data={data} onMouseMove={handleMouseMove}>
                    <CartesianGrid stroke="#e0e0e0" strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Legend />

                    {markers.map(({ timestamp, label }, idx) => (
                        <ReferenceLine
                            key={idx}
                            x={timestamp}
                            stroke="red"
                            strokeDasharray="3 3"
                            strokeWidth={selectedMarker === timestamp ? 3 : 1}
                        />
                    ))}

                    {Object.keys(COLORS).map((emotion) => (
                        <Line
                            key={emotion}
                            type="monotone"
                            dataKey={emotion}
                            stroke={COLORS[emotion]}
                            strokeWidth={3}
                            dot={false}
                        />
                    ))}
                </LineChart>
            </ResponsiveContainer>

            <Box
                mt={5}
                display="flex"
                flexDirection={{ xs: 'column', md: 'row' }}
                justifyContent="space-between"
                alignItems="stretch"
                gap={4}
            >
                <Box flex={1} display="flex" flexDirection="column" alignItems="center">
                    <Typography variant="h6" mb={2} textAlign="center">
                        Distribución total de emociones
                    </Typography>
                    <ResponsiveContainer width="100%" height={350}>
                        <PieChart>
                            <Pie
                                dataKey="value"
                                data={pieData}
                                cx="50%"
                                cy="50%"
                                outerRadius={120}
                                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                            >
                                {pieData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[entry.name]} />
                                ))}
                            </Pie>
                            <Tooltip />
                        </PieChart>
                    </ResponsiveContainer>
                </Box>

                <Box
                    flex={1}
                    display="flex"
                    flexDirection="column"
                    alignItems="center"
                    justifyContent="center"
                >
                    <Typography variant="h6" mb={2} textAlign="center">
                        Emoción destacada
                    </Typography>
                    <Box fontSize={140}>
                        {hoverDominantEmotion ? EMOTION_EMOJIS[hoverDominantEmotion] : '❔'}
                    </Box>
                    <Typography variant="h5" color="textSecondary" mt={1}>
                        {hoverDominantEmotion || 'Ninguna'}
                    </Typography>

                    <Box mt={3} width="100%" maxHeight={200} overflow="auto">
                        <Typography variant="subtitle1" gutterBottom>
                            Marcadores:
                        </Typography>
                        {markers.map(({ timestamp, label }, idx) => (
                            <Box
                                key={idx}
                                onClick={() => setSelectedMarker(timestamp)}
                                sx={{
                                    cursor: 'pointer',
                                    padding: '4px 8px',
                                    borderRadius: '4px',
                                    backgroundColor: selectedMarker === timestamp ? 'rgba(255,0,0,0.1)' : 'transparent',
                                    '&:hover': {
                                        backgroundColor: 'rgba(0,0,0,0.05)',
                                    }
                                }}
                            >
                                <Typography variant="body2">{`${label} (${timestamp})`}</Typography>
                            </Box>
                        ))}
                    </Box>
                </Box>
            </Box>
        </Box>
    );
};

export default Resumen;
