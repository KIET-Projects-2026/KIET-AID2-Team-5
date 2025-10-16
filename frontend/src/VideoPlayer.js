import React, { useState } from 'react';

function VideoPlayer({ video, onClose }) {
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownload = async () => {
    try {
      setIsDownloading(true);
      
      // Create download link
      const link = document.createElement('a');
      link.href = video;
      link.download = `ai-news-report-${Date.now()}.mp4`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      setTimeout(() => setIsDownloading(false), 2000);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
      setIsDownloading(false);
    }
  };

  const handleShare = async () => {
    try {
      if (navigator.share) {
        await navigator.share({
          title: 'AI Generated News Report',
          text: 'Check out this AI-generated news video!',
          url: window.location.href
        });
      } else {
        // Fallback - copy to clipboard
        await navigator.clipboard.writeText(window.location.href);
        alert('Link copied to clipboard!');
      }
    } catch (error) {
      console.error('Sharing failed:', error);
    }
  };

  return (
    <div className="video-player-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="video-player-container">
        {/* Header */}
        <div className="video-header">
          <h2>🎥 Your AI News Report is Ready!</h2>
          <button className="close-btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        {/* Video */}
        <div className="video-wrapper">
          <video 
            controls 
            autoPlay 
            muted={false}
            className="generated-video"
            poster="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='720' height='720'%3E%3Crect width='100%25' height='100%25' fill='%23667eea'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' fill='white' font-size='24' font-family='Arial'%3E📺 News Report%3C/text%3E%3C/svg%3E"
          >
            <source src={video} type="video/mp4" />
            <p>Your browser does not support video playback.</p>
          </video>
        </div>

        {/* Video Info */}
        <div className="video-info">
          <div className="info-section">
            <h3>📊 Video Information</h3>
            <div className="info-grid">
              <div className="info-item">
                <span className="info-label">Format:</span>
                <span className="info-value">MP4 Video</span>
              </div>
              <div className="info-item">
                <span className="info-label">Quality:</span>
                <span className="info-value">720p HD</span>
              </div>
              <div className="info-item">
                <span className="info-label">Audio:</span>
                <span className="info-value">AI Generated Speech</span>
              </div>
              <div className="info-item">
                <span className="info-label">Features:</span>
                <span className="info-value">Lip-sync Animation</span>
              </div>
            </div>
          </div>

          <div className="features-used">
            <h4>✨ AI Features Used:</h4>
            <ul>
              <li>📰 Real-time news fetching</li>
              <li>✍️ AI script generation</li>
              <li>🎤 Text-to-speech synthesis</li>
              <li>👤 Avatar lip-sync animation</li>
              <li>🎬 Professional video composition</li>
            </ul>
          </div>
        </div>

        {/* Actions */}
        <div className="video-actions">
          <button 
            onClick={handleDownload}
            disabled={isDownloading}
            className="download-btn"
          >
            {isDownloading ? (
              <>⏳ Downloading...</>
            ) : (
              <>⬇️ Download Video</>
            )}
          </button>

          <button onClick={handleShare} className="share-btn">
            📤 Share
          </button>

          <button onClick={onClose} className="generate-another-btn">
            🔄 Generate Another
          </button>
        </div>

        {/* Tips */}
        <div className="video-tips">
          <p>💡 <strong>Tip:</strong> Right-click the video for additional options like picture-in-picture mode!</p>
        </div>
      </div>
    </div>
  );
}

export default VideoPlayer;
