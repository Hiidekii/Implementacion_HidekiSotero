import React from 'react';
import { Box, Typography } from '@mui/material';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const EMOTIONS = [
    { label: 'Happy', emoji: '😊', color: '#ffc107' },
    { label: 'Sad', emoji: '😔', color: '#6c757d' },
    { label: 'Anger', emoji: '😡', color: '#dc3545' },
    { label: 'Surprise', emoji: '😲', color: '#0dcaf0' },
    { label: 'Neutral', emoji: '😐', color: '#adb5bd' },
    { label: 'Disgust', emoji: '🤢', color: '#198754' },
    { label: 'Fear', emoji: '😨', color: '#6610f2' },
];

const EmotionWaveChart = ({ timelineData }) => {
    const data = timelineData.map((frame, index) => {
        const entry = { time: index };
        EMOTIONS.forEach(({ label }) => {
            entry[label] = frame[label] || 0;
        });
        return entry;
    });

    return (
        <Box width="100%" mt={4}>
            <Box display="flex" flexDirection="column" gap={1}>
                {EMOTIONS.map(({ label, emoji, color }) => (
                    <Box key={label} display="flex" alignItems="center">
                        <Box width={30} textAlign="center" fontSize={24} mr={1}>{emoji}</Box>
                        <Box flexGrow={1} height={100}>
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                                    <XAxis dataKey="time" hide />
                                    <YAxis hide domain={[0, 'auto']} />
                                    <Tooltip />
                                    <Area type="monotone" dataKey={label} stroke={color} fill={color} isAnimationActive={false} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </Box>
                    </Box>
                ))}
            </Box>
        </Box>
    );
};

export default EmotionWaveChart;
