import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import { PieChart } from '@mui/x-charts/PieChart';

const EmotionPieChart = ({ frameEmotions }) => {
    // Mapeo de colores
    const emotionColorMap = {
        Happy: '#ffc107',
        Sad: '#6c757d',
        Anger: '#dc3545',
        Surprise: '#0dcaf0',
        Neutral: '#adb5bd',
        Disgust: '#198754',
        Fear: '#6610f2'
    };

    // Calcular frecuencias de emociones del frame
    const chartData = useMemo(() => {
        const counts = {};
        frameEmotions.forEach(item => {
            counts[item.emocion] = (counts[item.emocion] || 0) + 1;
        });

        return Object.entries(counts).map(([emocion, value]) => ({
            id: emocion,
            value,
            label: emocion,
            color: emotionColorMap[emocion] || '#888',
        }));
    }, [frameEmotions]);

    return (
        <Box sx={{ mt: 4 }}>
            <PieChart
                series={[
                    {
                        data: chartData,
                        innerRadius: 30,
                        outerRadius: 100,
                        paddingAngle: 5,
                        cornerRadius: 5
                    },
                ]}
                width={320}
                height={250}
            />
        </Box>
    );
};

export default EmotionPieChart;