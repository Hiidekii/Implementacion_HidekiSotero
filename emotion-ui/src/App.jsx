import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ResponsiveAppBar from './common/ResponsiveAppBar';
import ScreenCap from './screens/ScreenCapture/ScreenCap';
import VideoAnalysis from './screens/VideoAnalysis/VideoAnalysis';

function App() {
  return (
    <Router>
      <ResponsiveAppBar />
      <Routes>
        <Route path="/" element={<ScreenCap />} />
        <Route path="/captura" element={<ScreenCap />} />
        <Route path="/video" element={<VideoAnalysis />} />
      </Routes>
    </Router>
  );
}

export default App;
