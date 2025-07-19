import * as ort from 'onnxruntime-web';
import { getOnnxModelUri } from '../onnxModelLoader';
import { fetchModelFromHooks } from '../onnxHooksFallback';
import { getActiveModelConfig, getModelPaths, getAllModelConfigs, setActiveModel as setModel, getModelConfig } from '../config/modelConfig';

// Reference landmark indices for distance normalization
const NOSE_TIP_IDX = 1;
const LEFT_EYE_OUTER_IDX = 33;
const RIGHT_EYE_OUTER_IDX = 263;

// ONNX model session - will be initialized once
let onnxSession = null;
let currentModelConfig = null;

// Helper function to get model dimensions
const getModelDimensions = () => {
  const config = currentModelConfig || getActiveModelConfig();
  return {
    SEQ_LEN: config.inputFormat.sequenceLength,
    NUM_LANDMARKS: config.inputFormat.numLandmarks,
    NUM_COORDS: config.inputFormat.numCoords
  };
};

/**
 * Initialize the ONNX model session
 * @param {string} modelId - Optional model ID to use (if not provided, uses active model)
 * @returns {Promise<boolean>} Success status
 */
export const initializeOnnxModel = async (modelId = null) => {
  try {
    // Get the model configuration
    currentModelConfig = getActiveModelConfig();
    if (modelId) {
      if (setModel(modelId)) {
        currentModelConfig = getActiveModelConfig();
      }
    }
      console.log(`Initializing ONNX model: ${currentModelConfig.name}`);
    
    // Set up ONNX runtime options from model config
    const options = {
      executionProviders: currentModelConfig.processingOptions.executionProviders,
      graphOptimizationLevel: currentModelConfig.processingOptions.graphOptimizationLevel
    };
    
    // Get possible paths for the current model
    const possiblePaths = getModelPaths(currentModelConfig.filename);
    
    let modelLoaded = false;
    let lastError = null;
    
    // Method 1: Try the model loader to get a blob URL
    try {
      const modelUri = await getOnnxModelUri(currentModelConfig.filename);
      if (modelUri) {
        onnxSession = await ort.InferenceSession.create(modelUri, options);
        console.log('ONNX model loaded successfully');
        return true;
      }
    } catch (error) {
      console.warn('Failed to load model from blob URL:', error.message);
      lastError = error;
    }
    
    // Method 2: Try fetching directly from hooks folder
    try {
      const hooksModelUri = await fetchModelFromHooks(currentModelConfig.filename);
      if (hooksModelUri) {
        onnxSession = await ort.InferenceSession.create(hooksModelUri, options);
        console.log('ONNX model loaded successfully');
        return true;
      }
    } catch (error) {
      console.warn('Failed to load model from hooks blob URL:', error.message);
      lastError = error;
    }
    
    // Method 3: Try each path directly
    for (const path of possiblePaths) {
      try {
        onnxSession = await ort.InferenceSession.create(path, options);
        console.log(`ONNX model loaded successfully from: ${path}`);
        modelLoaded = true;
        break;
      } catch (error) {
        console.warn(`Failed to load model from ${path}:`, error.message);
        lastError = error;
      }
    }
    
    if (!modelLoaded) {
      throw lastError || new Error('Failed to load model from any of the possible paths');
    }
    
    return true;
  } catch (error) {
    console.error('Failed to initialize ONNX model:', error);
    return false;
  }
};

/**
 * Preprocess landmarks to the format expected by the model
 * @param {Array} landmarksSequence - Array of landmark frames
 * @returns {Float32Array|null} - Preprocessed landmarks as a typed array
 */
export const preprocessLandmarks = (landmarksSequence) => {
  if (!Array.isArray(landmarksSequence)) {
    console.error("Error: Input landmarksSequence must be an array.");
    return null;
  }

  // Get current model dimensions
  const { SEQ_LEN, NUM_LANDMARKS, NUM_COORDS } = getModelDimensions();

  // First, prepare 3D array with shape [SEQ_LEN, NUM_LANDMARKS, NUM_COORDS]
  const processedFrames = [];
  
  // Process each frame of landmarks
  for (let frameIdx = 0; frameIdx < Math.min(landmarksSequence.length, SEQ_LEN); frameIdx++) {
    const frameLandmarks = landmarksSequence[frameIdx];
    
    // Create frame array with shape [NUM_LANDMARKS, NUM_COORDS]
    const frameArray = Array(NUM_LANDMARKS).fill().map(() => Array(NUM_COORDS).fill(-1.0));
    
    // Fill with actual landmark data if available
    if (frameLandmarks && Array.isArray(frameLandmarks)) {
      for (let i = 0; i < Math.min(frameLandmarks.length, NUM_LANDMARKS); i++) {
        const landmark = frameLandmarks[i];
        if (landmark && typeof landmark.x === 'number' && 
            typeof landmark.y === 'number' && 
            typeof landmark.z === 'number') {
          frameArray[i][0] = landmark.x;
          frameArray[i][1] = landmark.y;
          frameArray[i][2] = landmark.z;
        }
      }
    }
    
    processedFrames.push(frameArray);
  }
  
  // Pad with empty frames if needed
  if (processedFrames.length < SEQ_LEN) {
    const padFrameTemplate = Array(NUM_LANDMARKS).fill().map(() => Array(NUM_COORDS).fill(-1.0));
    const numPaddingFrames = SEQ_LEN - processedFrames.length;
    
    for (let i = 0; i < numPaddingFrames; i++) {
      processedFrames.push(padFrameTemplate);
    }
  }
  
  // Apply distance normalization if required
  const config = currentModelConfig || getActiveModelConfig();
  let finalFrames = processedFrames;
  if (config.inputFormat.requiresNormalization) {
    finalFrames = applyDistanceNormalization(processedFrames);
  }
  // Flatten the 3D array to 1D for ONNX input
  const flattenedArray = [];
  for (const frame of finalFrames) {
    for (const landmark of frame) {
      flattenedArray.push(...landmark);
    }
  }
  
  return new Float32Array(flattenedArray);
};

/**
 * Apply distance normalization to landmarks
 * @param {Array} landmarksFrames - Array of landmark frames
 * @returns {Array} - Normalized landmarks
 */
export const applyDistanceNormalization = (landmarksFrames) => {
  const normalizedFrames = [];
  
  for (let frameIdx = 0; frameIdx < landmarksFrames.length; frameIdx++) {
    const frameLandmarks = landmarksFrames[frameIdx];
    
    // Check if entire frame consists of -1 values (skip normalization for these frames)
    const isEntireFrameInvalid = frameLandmarks.every(coords => coords.every(val => val === -1.0));
    
    if (isEntireFrameInvalid) {
      normalizedFrames.push(frameLandmarks);
      continue;
    }
    
    try {
      const centerLandmarkCoords = frameLandmarks[NOSE_TIP_IDX];
      const p1Coords = frameLandmarks[LEFT_EYE_OUTER_IDX];
      const p2Coords = frameLandmarks[RIGHT_EYE_OUTER_IDX];
      
      // Check if reference landmarks are valid (not part of padding)
      const isInvalid = 
        centerLandmarkCoords.some(val => val === -1.0) ||
        p1Coords.some(val => val === -1.0) ||
        p2Coords.some(val => val === -1.0);
      
      if (isInvalid) {
        normalizedFrames.push(frameLandmarks);
        continue;
      }
      
      // Translate landmarks relative to nose tip
      const translatedLandmarks = frameLandmarks.map(coords => {
        // Skip normalization for padding landmarks (value = -1.0)
        if (coords.some(val => val === -1.0)) {
          return coords; // Keep padding landmarks as is
        }
        
        return [
          coords[0] - centerLandmarkCoords[0],
          coords[1] - centerLandmarkCoords[1],
          coords[2] - centerLandmarkCoords[2]
        ];
      });
      
      // Calculate scale factor using X and Y coordinates of eye landmarks
      const scaleDistance = Math.sqrt(
        Math.pow(p1Coords[0] - p2Coords[0], 2) +
        Math.pow(p1Coords[1] - p2Coords[1], 2)
      );
      
      let scaledLandmarks;
      if (scaleDistance < 1e-6) {
        // Avoid division by zero or very small numbers
        scaledLandmarks = translatedLandmarks;
      } else {
        // Scale landmarks by the distance between eyes
        scaledLandmarks = translatedLandmarks.map(coords => {
          // Skip normalization for padding landmarks (value = -1.0)
          if (coords.some(val => val === -1.0)) {
            return coords; // Keep padding landmarks as is
          }
          
          // Check for extremely far landmarks (outliers)
          const distanceFromCenter = Math.sqrt(
            Math.pow(coords[0], 2) + 
            Math.pow(coords[1], 2) + 
            Math.pow(coords[2], 2)
          );
          
          // Use distance threshold of 5x the eye distance as a heuristic
          // to identify outliers that could be noise
          const distanceThreshold = 5.0 * scaleDistance;
          
          if (distanceFromCenter > distanceThreshold) {
            console.warn('Detected far-away landmark, applying special normalization');
            // For far-away landmarks, preserve direction but cap the distance
            const scaleFactor = distanceThreshold / distanceFromCenter;
            return [
              coords[0] * scaleFactor / scaleDistance,
              coords[1] * scaleFactor / scaleDistance,
              coords[2] * scaleFactor / scaleDistance
            ];
          }
          
          // Normal normalization for regular landmarks
          return [
            coords[0] / scaleDistance,
            coords[1] / scaleDistance,
            coords[2] / scaleDistance
          ];
        });
      }
      
      normalizedFrames.push(scaledLandmarks);
    } catch (error) {
      console.error('Error normalizing landmarks:', error);
      normalizedFrames.push(frameLandmarks);
    }
  }
  
  return normalizedFrames;
};

/**
 * Map a continuous regression score to a discrete class
 * @param {number} score - Engagement score (0.0 to 1.0)
 * @param {Object} classLabels - Custom class labels mapping (optional)
 * @returns {Object} - Class details including index and name
 */
export const mapScoreToClassDetails = (score, classLabels = null) => {
  const details = { index: -1, name: "Prediction Failed", score };
  
  if (score === null || score === undefined) {
    return details;
  }
  
  // Use provided class labels or fallback to default
  const labelMap = classLabels || {
    0: 'Not Engaged',
    1: 'Barely Engaged',
    2: 'Engaged',
    3: 'Highly Engaged',
    4: 'SNP'
  };
  
  let classIndex = -1;
  
  if (!(0.0 <= score && score <= 1.0)) {
    details.name = "Invalid Score Range";
  } else if (0.0 <= score && score < 0.175) {
    classIndex = 4;  // SNP
  } else if (0.175 <= score && score < 0.40) {
    classIndex = 0;  // Not Engaged
  } else if (0.40 <= score && score < 0.60) {
    classIndex = 1;  // Barely Engaged
  } else if (0.60 <= score && score < 0.825) {
    classIndex = 2;  // Engaged
  } else if (0.825 <= score && score <= 1.0) {
    classIndex = 3;  // Highly Engaged
  } else {
    classIndex = -1;
    details.name = "Score Mapping Error";
  }
  
  if (classIndex !== -1) {
    details.index = classIndex;
    details.name = labelMap[classIndex] || "Unknown Index";
  }
  
  return details;
};

/**
 * Process classification logits to get class details
 * @param {Array} logits - Raw logits from model
 * @returns {Object} - Classification details
 */
export const mapClassificationLogitsToClassDetails = (logits, classLabels = null) => {
  const details = { 
    index: -1, 
    name: "Classification Failed", 
    raw_logits: null 
  };
  
  if (!logits || !Array.isArray(logits)) {
    return details;
  }
  
  // Use provided class labels or fallback to default
  const labelMap = classLabels || {
    0: 'Not Engaged',
    1: 'Barely Engaged',
    2: 'Engaged',
    3: 'Highly Engaged',
    4: 'SNP'
  };
  
  details.raw_logits = logits;
  
  // Apply softmax to get probabilities
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = expLogits.reduce((sum, val) => sum + val, 0);
  const probabilities = expLogits.map(exp => exp / sumExp);
  
  // Find the class with highest probability
  let maxProb = -Infinity;
  let classIndex = -1;
  
  for (let i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      classIndex = i;
    }
  }
  
  const className = labelMap[classIndex] || "Unknown Index";
  
  details.index = classIndex;
  details.name = className;
  details.probabilities = probabilities;
  
  return details;
};

/**
 * Run ONNX model inference on preprocessed landmarks
 * @param {Array} landmarksData - Array of landmarks frames
 * @returns {Object|null} - Prediction result with engagement score and class
 */
export const predictEngagement = async (landmarksData) => {
  try {    // Make sure model is initialized
    if (!onnxSession) {
      const initialized = await initializeOnnxModel();
      if (!initialized) {
        throw new Error('ONNX model initialization failed');
      }
    }
    
    // Get current model configuration
    const config = currentModelConfig || getActiveModelConfig();
    const { SEQ_LEN, NUM_LANDMARKS, NUM_COORDS } = getModelDimensions();
    
    // Preprocess landmarks
    const preprocessedLandmarks = preprocessLandmarks(landmarksData);
    
    if (!preprocessedLandmarks) {
      throw new Error('Failed to preprocess landmarks');
    }
    
    // Create tensor with configured shape
    const inputTensor = new ort.Tensor(
      'float32', 
      preprocessedLandmarks, 
      config.inputFormat.tensorShape
    );
    
    // Run inference
    const feeds = {};
    feeds[onnxSession.inputNames[0]] = inputTensor;
    const results = await onnxSession.run(feeds);// Extract results based on model configuration
    const modelConfig = currentModelConfig || getActiveModelConfig();
    
    // Handle different output formats based on model
    let regressionScore, predictionDetailsReg, predictionDetailsCls;
      if (modelConfig.outputFormat.outputType === 'classification' && onnxSession.outputNames.length >= 2) {
      // Multi-output model (like v4 and v4_v2 with regression + classification)
      
      // For v4_v2 model, use named outputs if available
      let regressionScores, classificationLogits;
        if (modelConfig.id === 'v4_v2' && modelConfig.outputFormat.outputNames) {
        // New v4_v2 model with named outputs
        regressionScores = results['regression_scores']?.data || results[onnxSession.outputNames[0]].data;
        classificationLogits = results['classification_logits']?.data || results[onnxSession.outputNames[1]].data;
      } else {
        // Original v4 model with positional outputs
        regressionScores = results[onnxSession.outputNames[0]].data;
        classificationLogits = results[onnxSession.outputNames[1]].data;
      }
      
      // Get and clamp regression score
      const rawRegressionScore = regressionScores[0];
      regressionScore = Math.max(0.0, Math.min(1.0, rawRegressionScore));
      
      // Map score to class details using model's class labels
      predictionDetailsReg = mapScoreToClassDetails(regressionScore, modelConfig.outputFormat.classLabels);
      
      // Process classification output
      predictionDetailsCls = mapClassificationLogitsToClassDetails(
        Array.from(classificationLogits), modelConfig.outputFormat.classLabels
      );
        // Create result object similar to server response
      const result = {
        score: regressionScore,
        name: predictionDetailsReg.name,
        index: predictionDetailsReg.index,
        classification_head_name: predictionDetailsCls.name,
        classification_head_index: predictionDetailsCls.index,
        classification_head_probabilities: predictionDetailsCls.probabilities,
        raw_regression_score: rawRegressionScore,
        raw_classification_logits: predictionDetailsCls.raw_logits,
        model_used: modelConfig.name,
        model_id: modelConfig.id
      };
      
      return result;    } else {
      // Single output model (classification only)
      const outputData = results[onnxSession.outputNames[0]].data;
      
      if (outputData.length === 1) {
        // Single regression score
        const rawScore = outputData[0];
        regressionScore = Math.max(0.0, Math.min(1.0, rawScore));
        predictionDetailsReg = mapScoreToClassDetails(regressionScore, modelConfig.outputFormat.classLabels);
        
        const result = {
          score: regressionScore,
          name: predictionDetailsReg.name,
          index: predictionDetailsReg.index,
          raw_regression_score: rawScore,
          model_used: modelConfig.name,
          model_id: modelConfig.id
        };
        
        return result;
      } else {
        // Classification logits
        predictionDetailsCls = mapClassificationLogitsToClassDetails(
          Array.from(outputData), modelConfig.outputFormat.classLabels
        );
          const result = {
          score: predictionDetailsCls.probabilities ? Math.max(...predictionDetailsCls.probabilities) : 0,
          name: predictionDetailsCls.name,
          index: predictionDetailsCls.index,
          classification_head_name: predictionDetailsCls.name,
          classification_head_index: predictionDetailsCls.index,
          classification_head_probabilities: predictionDetailsCls.probabilities,
          raw_classification_logits: predictionDetailsCls.raw_logits,
          model_used: modelConfig.name,
          model_id: modelConfig.id
        };
        
        return result;
      }
    }
  } catch (error) {
    console.error('ONNX prediction error:', error);
    return null;
  }
};

/**
 * Get the currently active model information
 * @returns {Object} Current model configuration
 */
export const getCurrentModelInfo = () => {
  return currentModelConfig || getActiveModelConfig();
};

/**
 * Switch to a different ONNX model
 * @param {string} modelId - ID of the model to switch to
 * @returns {Promise<boolean>} Success status
 */
export const switchModel = async (modelId) => {
  try {
    
    // Validate model exists
    const newModelConfig = getModelConfig(modelId);
    if (!newModelConfig) {
      console.error(`Model '${modelId}' not found`);
      return false;
    }
    
    // Set the new active model
    if (!setModel(modelId)) {
      return false;
    }
      // Clear current session to force re-initialization
    onnxSession = null;
    currentModelConfig = null;
    
    console.log(`Switched to model: ${newModelConfig.name}`);
    
    return true;
  } catch (error) {
    console.error('Failed to switch model:', error);
    return false;
  }
};

/**
 * Get available models for selection
 * @returns {Promise<Object>} Available models configuration
 */
export const getAvailableModels = async () => {
  try {
    return getAllModelConfigs();
  } catch (error) {
    console.error('Failed to get available models:', error);
    return {};
  }
};

/**
 * Check if a model is currently loaded and ready
 * @returns {boolean} Whether model is loaded
 */
export const isModelLoaded = () => {
  return onnxSession !== null;
};

/**
 * Force reload the current model
 * @returns {Promise<boolean>} Success status
 */
export const reloadCurrentModel = async () => {
  try {
    // Clear current session
    onnxSession = null;
    currentModelConfig = null;
    
    // Re-initialize with current active model
    return await initializeOnnxModel();
  } catch (error) {
    console.error('Failed to reload model:', error);
    return false;
  }
};
