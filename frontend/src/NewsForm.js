import React, { useState } from 'react';
import axios from 'axios';

function NewsForm({ onVideoGenerated, onError, loading, setLoading, error }) {
  const [topic, setTopic] = useState('');
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');

  const steps = [
    { id: 'fetching', label: 'Fetching News', duration: 15 },
    { id: 'script', label: 'Generating Script', duration: 20 },
    { id: 'speech', label: 'Creating Speech', duration: 30 },
    { id: 'video', label: 'Generating Video', duration: 60 },
    { id: 'graphics', label: 'Adding Graphics', duration: 15 }
  ];

  const exampleTopics = [
    'Latest technology news',
    'Climate change updates',
    'Sports news today',
    'Stock market analysis', 
    'AI developments',
    'Space exploration news'
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!topic.trim()) {
      onError('Please enter a news topic');
      return;
    }

    if (topic.trim().length < 3) {
      onError('Please enter at least 3 characters');
      return;
    }

    setLoading(true);
    setProgress(0);
    setCurrentStep('Initializing...');
    
    // Simulate progress updates
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + Math.random() * 10;
      });
    }, 2000);

    // Simulate step updates
    let stepIndex = 0;
    const stepInterval = setInterval(() => {
      if (stepIndex < steps.length) {
        setCurrentStep(steps[stepIndex].label);
        stepIndex++;
      } else {
        clearInterval(stepInterval);
      }
    }, 25000); // Update every 25 seconds

    try {
const response = await axios.post('http://localhost:5000/api/generate',
    { topic: topic.trim() },
    {
      responseType: 'blob',
      timeout: 900000, // 5 minutes timeout
      onUploadProgress: (progressEvent) => {
        // Handle upload progress if needed
      }
    }
);


      clearInterval(progressInterval);
      clearInterval(stepInterval);
      setProgress(100);
      setCurrentStep('Complete!');
      
      setTimeout(() => {
        onVideoGenerated(response.data);
      }, 1000);

    } catch (err) {
      clearInterval(progressInterval);
      clearInterval(stepInterval);
      
      console.error('Error generating video:', err);
      
      let errorMessage = 'Failed to generate video. Please try again.';
      
      if (err.code === 'ECONNABORTED') {
        errorMessage = 'Request timed out. Video generation is taking longer than expected.';
      } else if (err.response?.status === 404) {
        errorMessage = 'No news found for this topic. Please try a different topic.';
      } else if (err.response?.status === 500) {
        errorMessage = 'Server error occurred. Please try again later.';
      } else if (err.response?.data?.error) {
        errorMessage = err.response.data.error;
      } else if (!navigator.onLine) {
        errorMessage = 'No internet connection. Please check your connection.';
      }
      
      onError(errorMessage);
    }
  };

  const handleExampleClick = (example) => {
    if (!loading) {
      setTopic(example);
    }
  };

  return (
    <div className="news-form-container">
      {!loading ? (
        <div className="form-wrapper">
          <h2>What news would you like to see?</h2>
          
          <form onSubmit={handleSubmit} className="news-form">
            <div className="input-group">
              <label htmlFor="topic">Enter News Topic:</label>
              <input
                id="topic"
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g., latest technology news, climate change updates..."
                maxLength={200}
                className={error ? 'error' : ''}
              />
              <div className="input-info">
                <span className="char-count">{topic.length}/200</span>
                {topic.length >= 3 && <span className="valid-indicator">✓</span>}
              </div>
            </div>

            {error && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                <span>{error}</span>
              </div>
            )}

            <div className="example-topics">
              <p>Try these examples:</p>
              <div className="example-grid">
                {exampleTopics.map((example, index) => (
                  <button
                    key={index}
                    type="button"
                    className="example-topic"
                    onClick={() => handleExampleClick(example)}
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>

            <button 
              type="submit" 
              className="generate-btn"
              disabled={!topic.trim() || topic.trim().length < 3}
            >
              🎬 Generate News Video
            </button>
          </form>
        </div>
      ) : (
        <div className="loading-container">
          <div className="loading-header">
            <h2>🎥 Creating Your News Video</h2>
            <p>Please wait while we generate your personalized news report...</p>
          </div>

          <div className="progress-section">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="progress-info">
              <span className="progress-text">{Math.round(progress)}% Complete</span>
              <span className="current-step">{currentStep}</span>
            </div>
          </div>

          <div className="steps-indicator">
            {steps.map((step, index) => (
              <div 
                key={step.id}
                className={`step-item ${
                  currentStep.includes(step.label) ? 'active' : 
                  index < steps.findIndex(s => currentStep.includes(s.label)) ? 'completed' : 'pending'
                }`}
              >
                <div className="step-number">{index + 1}</div>
                <div className="step-label">{step.label}</div>
              </div>
            ))}
          </div>

          <div className="loading-tips">
            <h4>💡 Did you know?</h4>
            <p>Our AI processes multiple news sources and creates professional scripts with realistic lip-sync animation!</p>
          </div>

          <div className="estimated-time">
            <p>⏱️ Estimated time remaining: {Math.max(1, 5 - Math.floor(progress / 20))} minutes</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default NewsForm;
