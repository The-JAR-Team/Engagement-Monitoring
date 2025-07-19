import React, { useRef, useState, useEffect } from 'react';
import useFaceMesh from '../hooks/useFaceMesh';
import '../styles/EngagementMonitor.css';
import { initializeOnnxModel, predictEngagement, getCurrentModelInfo } from '../services/engagementOnnxService';
import ModelSelector from './ModelSelector';

const EngagementMonitor = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [faceMeshStatus, setFaceMeshStatus] = useState('Initializing...');
  const [isActive, setIsActive] = useState(true);
  const [engagementScore, setEngagementScore] = useState(null);
  const [engagementClass, setEngagementClass] = useState('');
  const [errorMessage, setErrorMessage] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [onnxModelReady, setOnnxModelReady] = useState(false);
  const [onnxStatus, setOnnxStatus] = useState('Loading ONNX model...');
  const onnxErrorCountRef = useRef(0);
  
  // Buffer for landmark data
  const landmarkBufferRef = useRef([]);
  
  // Dynamic model configuration
  const [requiredFrames, setRequiredFrames] = useState(100); // Default fallback
  
  // For FPS calculation
  const frameCountRef = useRef(0);
  const lastFpsLogTimeRef = useRef(Date.now());
  
  // Constants for data collection
  const FPS = 10;
  const FRAME_INTERVAL = 100; // Collect frame every 100ms
  const SEND_INTERVAL = 1000; // Send data every 1 second
  
  // References for intervals
  const trackingIntervalRef = useRef(null);
  const sendingIntervalRef = useRef(null);

  // Initialize ONNX model
  useEffect(() => {
    const initModel = async () => {
      try {
        setOnnxStatus('Starting ONNX model initialization...');
        const initialized = await initializeOnnxModel();
        setOnnxModelReady(initialized);
        if (!initialized) {
          setOnnxStatus('Failed to initialize ONNX model');
          setErrorMessage("Failed to initialize ONNX model");
        } else {
          setOnnxStatus('ONNX model initialized successfully');
          console.log("ONNX model initialized successfully");
        }
      } catch (error) {
        setOnnxStatus(`Error initializing ONNX model: ${error.message}`);
        console.error("Error initializing ONNX model:", error);
        setErrorMessage("Error initializing ONNX model: " + (error.message || "Unknown error"));
      }
    };
    
    initModel();
    lastFpsLogTimeRef.current = Date.now(); // Initialize FPS log time
  }, []);

  // Handle FaceMesh results
  const handleResults = (results) => {
    // FPS Calculation
    frameCountRef.current++;
    const now = Date.now();
    if (now - lastFpsLogTimeRef.current >= 5000) { // 5 seconds
      const elapsedSeconds = (now - lastFpsLogTimeRef.current) / 1000;
      const fps = frameCountRef.current / elapsedSeconds;
      console.log(`🎥 Current Camera FPS: ${fps.toFixed(2)}`);
      frameCountRef.current = 0;
      lastFpsLogTimeRef.current = now;
    }

    // Hide loading state when we get first results
    if (isLoading) {
      setIsLoading(false);
    }

    // Clear any error when we get valid results
    if (errorMessage) {
      setErrorMessage(null);
    }
    
    if (!canvasRef.current || !results || !results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Get video dimensions
    const videoWidth = videoRef.current.videoWidth;
    const videoHeight = videoRef.current.videoHeight;

    // Update canvas dimensions to match video
    canvas.width = videoWidth;
    canvas.height = videoHeight;

    // Draw video frame on canvas
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    try {
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    } catch (e) {
      console.error("Error drawing video to canvas:", e);
      return;
    }

    // Get bounding box from face landmarks
    const landmarks = results.multiFaceLandmarks[0];
    
    // Store the landmark data
    if (isActive && landmarks) {
      // Store only the landmarks data, not the timestamp
      landmarkBufferRef.current.push(landmarks);
      
      // Debug: Log landmark collection every 50 frames to avoid spam
      if (landmarkBufferRef.current.length % 50 === 0) {
        console.log(`📥 Collected ${landmarkBufferRef.current.length} landmark frames (${landmarks.length} landmarks per frame)`);
      }
      
      // Keep only the most recent frames needed
      if (landmarkBufferRef.current.length > requiredFrames) {
        landmarkBufferRef.current.shift();
      }
    }

    // Calculate the face bounding box
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    landmarks.forEach(landmark => {
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    });

    // Add some padding to the bounding box
    const padding = 20;
    minX = Math.max(0, minX - padding);
    minY = Math.max(0, minY - padding);
    maxX = Math.min(canvas.width, maxX + padding);
    maxY = Math.min(canvas.height, maxY + padding);

    // Determine color based on engagement score
    let boxColor = '#FFCC00'; // Default yellow
    let shortLabel = 'Neutral'; // Default label
    
    if (engagementScore !== null) {
      if (engagementScore > 0.88) {
        boxColor = '#00FF00'; // Extra green for very engaged (>0.88)
        shortLabel = 'very engaged';
      } else if (engagementScore > 0.65) {
        boxColor = '#66CC00'; // Green for engaged (0.65-0.88)
        shortLabel = 'engaged';
      } else if (engagementScore < 0.45) {
        boxColor = '#FF0000'; // Extra red for very disengaged (<0.45)
        shortLabel = 'very disengaged';
      } else if (engagementScore < 0.65) {
        boxColor = '#FF6600'; // Red-orange for disengaged (0.45-0.65)
        shortLabel = 'disengaged';
      }
    }

    // Draw face bounding box
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.rect(minX, minY, maxX - minX, maxY - minY);
    ctx.stroke();

    // Display engagement score if available
    if (engagementScore !== null) {
      // Calculate width of box based on face size
      const boxWidth = maxX - minX;
      
      // Create background for engagement score
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      // Increase background height for larger font sizes
      ctx.fillRect(minX, minY - 48, boxWidth, 48);
      
      // Display label on top with larger dynamic font
      const labelFontSize = Math.max(18, Math.min(24, boxWidth / 14));
      ctx.font = `bold ${labelFontSize}px Arial`;
      ctx.fillStyle = boxColor;
      ctx.textAlign = "center";
      ctx.fillText(`${shortLabel}`, minX + (boxWidth / 2), minY - 32);
      
      // Display score underneath with even larger font
      const scoreFontSize = Math.max(20, Math.min(28, boxWidth / 18));
      ctx.font = `bold ${scoreFontSize}px Arial`;
      ctx.fillText(`${engagementScore.toFixed(2)}`, minX + (boxWidth / 2), minY - 12);
    }

    ctx.restore();
  };

  // Handle status updates from FaceMesh
  const handleFaceMeshStatus = (status) => {
    setFaceMeshStatus(status);
    
    // Check for errors in status messages
    if (status.toLowerCase().includes('error') || 
        status.toLowerCase().includes('failed') ||
        status.toLowerCase().includes('denied')) {
      setErrorMessage(status);
      setIsLoading(false);
    }
    
    // When FaceMesh is ready, clear loading state
    if (status === 'FaceMesh Ready') {
      setErrorMessage(null);
      setIsLoading(false);
    }
  };

  // Initialize data collection and sending intervals
  useEffect(() => {
    if (isActive) {
      // Start collecting frames at regular intervals
      trackingIntervalRef.current = setInterval(() => {
        // This is just to ensure the buffer doesn't grow too large
        // Actual collection happens in handleResults
      }, FRAME_INTERVAL);

      // Start processing engagement data at regular intervals
      sendingIntervalRef.current = setInterval(async () => {
        if (landmarkBufferRef.current.length < requiredFrames) {
          console.log(`⚠️ Not enough frames yet (${landmarkBufferRef.current.length}/${requiredFrames})`);
          return;
        }
        
        try {
          // Get the most recent frames from the buffer
          const relevantLandmarks = landmarkBufferRef.current
            .slice(-requiredFrames);

          // Check if ONNX model is ready
          if (!onnxModelReady) {
            console.log("⚠️ ONNX model not ready yet");
            return;
          }

          // Process landmarks with ONNX model
          console.log(`Processing ${relevantLandmarks.length} landmarks with ONNX model`);
          
          const result = await predictEngagement(relevantLandmarks);
            
          if (!result) {
            console.error('❌ ONNX prediction failed');
            setErrorMessage("ONNX prediction failed");
            return;
          }
            
          // Debug: Log ONNX prediction result
          console.log('🎯 DEBUG: ONNX prediction result:', {
            score: result.score,
            name: result.name,
            timestamp: new Date().toISOString()
          });
          
          setEngagementScore(result.score);
          setEngagementClass(result.name);
          
          console.log(`✅ Processed ${requiredFrames} frames with ONNX model. Result:`, result.score, result.name);
        } catch (error) {
          console.error('❌ Error processing landmarks with ONNX:', error);
          setErrorMessage("Error processing landmarks: " + (error.message || "Unknown error"));
        }
      }, SEND_INTERVAL);
    }

    return () => {
      // Clean up intervals when component unmounts or isActive changes
      if (trackingIntervalRef.current) {
        clearInterval(trackingIntervalRef.current);
      }
      if (sendingIntervalRef.current) {
        clearInterval(sendingIntervalRef.current);
      }
    };
  }, [isActive, onnxModelReady, requiredFrames]);

  // Update required frames based on current model
  useEffect(() => {
    const updateModelConfig = () => {
      try {
        const modelInfo = getCurrentModelInfo();
        if (modelInfo && modelInfo.inputFormat) {
          const newRequiredFrames = modelInfo.inputFormat.sequenceLength;
          setRequiredFrames(newRequiredFrames);
          console.log(`📏 Updated required frames to ${newRequiredFrames} for model: ${modelInfo.name}`);
        }
      } catch (error) {
        console.warn('Failed to get model info, using default required frames:', error);
      }
    };
    
    // Update immediately
    updateModelConfig();
    
    // Also update when ONNX model becomes ready
    if (onnxModelReady) {
      updateModelConfig();
    }
  }, [onnxModelReady]);

  // Toggle active state
  const toggleActive = () => {
    setIsActive(!isActive);
    if (!isActive) {
      landmarkBufferRef.current = [];
    }
  };

  // Handle retry when FaceMesh fails
  const handleRetry = async () => {
    setErrorMessage(null);
    setIsLoading(true);
    setFaceMeshStatus('Initializing...');
    
    // Clear buffers
    landmarkBufferRef.current = [];
    
    // Re-initialize ONNX model
    try {
      setOnnxModelReady(false);
      const initialized = await initializeOnnxModel();
      setOnnxModelReady(initialized);
      if (!initialized) {
        setErrorMessage("Failed to initialize ONNX model");
      }
    } catch (error) {
      console.error("Error re-initializing ONNX model:", error);
      setErrorMessage("Error initializing ONNX model: " + (error.message || "Unknown error"));
    }
    
    // Force a browser repaint/reflow to help with webcam issues
    if (videoRef.current) {
      videoRef.current.style.display = 'none';
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.style.display = 'block';
        }
      }, 50);
    }
  };

  const getEngagementClass = () => {
    if (!engagementClass) return '';
    return engagementClass.toLowerCase().replace(/\s+/g, '-');
  };

  return (
    <div className="engagement-monitor">
      {/* Video Container */}
      <div className="video-container">
        <video 
          ref={videoRef} 
          className="webcam" 
          muted 
          playsInline 
          autoPlay
          style={{ objectFit: 'contain' }} 
        />
        <canvas 
          ref={canvasRef} 
          className="overlay" 
          style={{ objectFit: 'contain' }}
        />
        
        {/* Show a loading overlay when initializing */}
        {isLoading && (
          <div className="loading-spinner">
            <div className="spinner"></div>
            <div className="loading-text">{faceMeshStatus}</div>
          </div>
        )}
      </div>

      {/* HUD Overlay */}
      <div className="hud-overlay">
        {/* Top HUD */}
        <div className="top-hud">
          {/* Engagement Score Display */}
          <div className={`engagement-score-display ${getEngagementClass()}`}>
            <div className="score-value">
              {engagementScore !== null ? engagementClass : '---'}
            </div>
            <div className="score-label">Engagement Level</div>
            {engagementScore !== null && (
              <div className="score-percentage">
                {(engagementScore * 100).toFixed(1)}%
              </div>
            )}
          </div>

          {/* System Status */}
          <div className="system-status">
            <div className="status-item">
              <span className="status-label">FaceMesh</span>
              <span className="status-value">{faceMeshStatus}</span>
              <div className={`status-indicator ${faceMeshStatus === 'FaceMesh Ready' ? 'ready' : 'loading'}`}></div>
            </div>
            <div className="status-item">
              <span className="status-label">ONNX Model</span>
              <span className="status-value">{onnxModelReady ? 'Ready' : 'Loading...'}</span>
              <div className={`status-indicator ${onnxModelReady ? 'ready' : (errorMessage ? 'error' : 'loading')}`}></div>
            </div>
            <div className="status-item">
              <span className="status-label">Frames</span>
              <span className="status-value">{landmarkBufferRef.current.length}/{requiredFrames}</span>
              <div className={`status-indicator ${landmarkBufferRef.current.length >= requiredFrames ? 'ready' : 'loading'}`}></div>
            </div>
            <div className="status-item">
              <span className="status-label">Status</span>
              <span className="status-value">{isActive ? 'Active' : 'Paused'}</span>
              <div className={`status-indicator ${isActive ? 'ready' : 'loading'}`}></div>
            </div>
          </div>
        </div>

        {/* Model Selector */}
        <div className="model-selector">
          <ModelSelector />
        </div>

        {/* Bottom Controls */}
        <div className="bottom-controls">
          <button 
            className={`control-button ${isActive ? 'active' : ''}`}
            onClick={toggleActive}
          >
            {isActive ? 'Pause' : 'Resume'}
          </button>
          {errorMessage && (
            <button className="control-button" onClick={handleRetry}>
              Retry
            </button>
          )}
        </div>
      </div>

      {/* Error Overlay */}
      {errorMessage && !isLoading && (
        <div className="error-overlay">
          <div className="error-message">{errorMessage}</div>
          <button className="retry-button" onClick={handleRetry}>
            Retry Initialization
          </button>
        </div>
      )}

      {/* FaceMesh hook - with separate status handler */}
      {useFaceMesh(true, videoRef, handleResults, handleFaceMeshStatus)}
    </div>
  );
};

export default EngagementMonitor;
