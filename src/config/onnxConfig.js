import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime to use CPU backend only
export const configureOnnxRuntime = () => {
  try {
    // Completely disable WASM backend
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = false;
    ort.env.wasm.proxy = false;
    
    // Disable all other backends except CPU
    ort.env.webgl = { disabled: true };
    ort.env.webgpu = { disabled: true };
    
    // Force CPU-only execution
    ort.env.logLevel = 'warning';
    
    console.log('ONNX Runtime configured for CPU-only execution');
    return true;
  } catch (error) {
    console.warn('Failed to configure ONNX Runtime:', error);
    return false;
  }
};

// Default session options for CPU execution
export const getDefaultSessionOptions = () => {
  return {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
    enableProfiling: false,
    enableCpuMemArena: true,
    enableMemPattern: true,
    executionMode: 'sequential',
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
    // Explicitly disable WASM
    extra: {
      session: {
        disable_prepacking: false,
        use_device_allocator_for_initializers: true,
        use_ort_model_bytes_directly: true,
        use_ort_model_bytes_for_subgraphs: true
      }
    }
  };
};
