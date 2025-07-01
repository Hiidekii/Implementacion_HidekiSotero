import React, { useEffect, useState } from 'react';
import { Box, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';

const COLORS = {
    Happy: '#ffc107',
    Sad: '#6c757d',
    Anger: '#dc3545',
    Surprise: '#0dcaf0',
    Neutral: '#adb5bd',
    Disgust: '#198754',
    Fear: '#6610f2',
};

const Resumen = () => {
    const [data, setData] = useState([]);

    useEffect(() => {
        const stored = localStorage.getItem('emotionTimeline');
        if (stored) {
            setData(JSON.parse(stored));
        }
    }, []);

    return (
        <Box p={4}>
            <Typography variant="h5" mb={3}>Resumen de emociones durante la sesión</Typography>
            <ResponsiveContainer width="100%" height={400}>
                <LineChart data={data}>
                    <CartesianGrid stroke="#e0e0e0" strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Legend />
                    {Object.keys(COLORS).map((emotion) => (
                        <Line
                            key={emotion}
                            type="monotone"
                            dataKey={emotion}
                            stroke={COLORS[emotion]}
                            strokeWidth={3} // más grueso
                            dot={false}
                        />
                    ))}
                </LineChart>
            </ResponsiveContainer>
        </Box>
    );
};

export default Resumen;