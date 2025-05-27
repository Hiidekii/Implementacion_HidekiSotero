import React from 'react';
import { Box, Typography } from '@mui/material';

const VideoAnalysis = () => {
    return (
        <Box sx={{ padding: '2rem' }}>
            <Typography variant="h4" gutterBottom>
                Análisis de Video
            </Typography>
            <Typography variant="body1">
                Aquí se mostrará el análisis de expresiones y emociones a partir de un video cargado.
                Próximamente podrás subir un archivo y visualizar los resultados de detección.
            </Typography>
        </Box>
    );
};

export default VideoAnalysis;
