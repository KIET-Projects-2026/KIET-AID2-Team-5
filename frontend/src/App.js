import React, { useState } from 'react';
import NewsForm from './NewsForm';
import VideoPlayer from './VideoPlayer';
import './App.css';

function App() {
  const [video, setVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleVideoGenerated = (videoBlob) => {
    const videoUrl = URL.createObjectURL(videoBlob);
    setVideo(videoUrl);
    setLoading(false);
  };

  const handleError = (errorMessage) => {
    setError(errorMessage);
    setLoading(false);
  };

  const handleClose = () => {
    if (video) {
      URL.revokeObjectURL(video);
    }
    setVideo(null);
    setError(null);
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <h1>🎬 AI News Reporter</h1>
        <p>Generate Professional News Videos with AI</p>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {!video && !loading && (
          <div className="welcome-section">
            <div className="hero-content">
              <h2>Transform Any News Topic Into Professional Videos</h2>
              <p>Our AI fetches the latest news, creates engaging scripts, and generates realistic news presenter videos with lip-sync technology.</p>
              
              <div className="features">
                <div className="feature">
                  <span className="feature-icon">📰</span>
                  <span>Real-time News</span>
                </div>
                <div className="feature">
                  <span className="feature-icon">🤖</span>
                  <span>AI Avatar</span>
                </div>
                <div className="feature">
                  <span className="feature-icon">🎤</span>
                  <span>Natural Speech</span>
                </div>
                <div className="feature">
                  <span className="feature-icon">🎬</span>
                  <span>Professional Video</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {!video ? (
          <NewsForm 
            onVideoGenerated={handleVideoGenerated}
            onError={handleError}
            loading={loading}
            setLoading={setLoading}
            error={error}
          />
        ) : (
          <VideoPlayer 
            video={video} 
            onClose={handleClose}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>Powered by AI • News Reporter Technology</p>
      </footer>
    </div>
  );
}

export default App;
