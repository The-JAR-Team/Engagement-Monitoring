import React, { useEffect, useState } from 'react';
import { initializeOnnxModel } from '../services/engagementOnnxService';
import { getOnnxModelUri } from '../onnxModelLoader';
import { fetchModelFromHooks } from '../onnxHooksFallback';

// This component will preload the ONNX model when the app starts
const OnnxModelPreloader = () => {
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelError, setModelError] = useState(null);

  useEffect(() => {
    const preloadModel = async () => {
      try {
        console.log('OnnxModelPreloader: Starting model preload...');
        
        // First, try the standard initialization
        let initialized = await initializeOnnxModel();
        
        if (!initialized) {
          console.log('OnnxModelPreloader: Standard initialization failed, trying URL loader...');
          
          // If that fails, try the URL loader
          const modelUri = await getOnnxModelUri('v1.onnx');
          if (modelUri) {
            console.log('OnnxModelPreloader: URL loader succeeded');
            setModelLoaded(true);
            return;
          }
          
          console.log('OnnxModelPreloader: URL loader failed, trying hooks fallback...');
          
          // If that fails, try the hooks fallback
          const hooksModelUri = await fetchModelFromHooks('v1.onnx');
          if (hooksModelUri) {
            console.log('OnnxModelPreloader: Hooks fallback succeeded');
            setModelLoaded(true);
            return;
          }
          
          // If we get here, all methods failed
          throw new Error('All model loading methods failed');
        } else {
          console.log('OnnxModelPreloader: Standard initialization succeeded');
          setModelLoaded(true);
        }
      } catch (error) {
        console.error('OnnxModelPreloader: Error preloading model:', error);
        setModelError(error.message);
      }
    };
    
    preloadModel();
  }, []);

  // This component doesn't render anything visible
  return null;
};

export default OnnxModelPreloader;
