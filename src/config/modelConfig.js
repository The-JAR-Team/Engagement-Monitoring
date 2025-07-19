// Model configuration for ONNX engagement models

const models = {
  default: {
    id: 'default',
    name: 'GRU Engagement Model V1',
    filename: 'v1.onnx',
    inputFormat: {
      sequenceLength: 100,
      numLandmarks: 478,
      numCoords: 3,
      requiresNormalization: true,
      tensorShape: [1, 100, 478, 3]
    },
    processingOptions: {
      executionProviders: ['cpu'],
      graphOptimizationLevel: 'all'
    },
    outputFormat: {
      outputType: 'classification',
      classLabels: {
        0: 'Not Engaged',
        1: 'Barely Engaged',
        2: 'Engaged',
        3: 'Highly Engaged',
        4: 'SNP'
      }
    }
  }
};

let activeModelId = 'default';

export function getActiveModelConfig() {
  return models[activeModelId];
}

export function setActiveModel(id) {
  if (models[id]) {
    activeModelId = id;
    return true;
  }
  return false;
}

export function getModelConfig(id) {
  return models[id] || null;
}

export function getAllModelConfigs() {
  return models;
}

export function getModelPaths(filename) {
  // Return possible URLs/paths for model file
  return [
    `/models/${filename}`,
    `/models/${filename}` // fallback to same
  ];
}
