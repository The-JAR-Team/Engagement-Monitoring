// Load ONNX model directly using fetch with no external dependencies
export const loadOnnxModelFromUrl = async (url) => {
  try {
    console.log(`Attempting to fetch ONNX model from: ${url}`);
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/octet-stream',
        'Content-Type': 'application/octet-stream'
      },
      cache: 'no-cache',
      mode: 'cors',
      credentials: 'same-origin'
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }
    
    console.log(`Successfully fetched model from: ${url}`);
    return await response.arrayBuffer();
  } catch (error) {
    console.error(`Error fetching model from ${url}:`, error);
    return null;
  }
};

// Function to load the ONNX model and return its ArrayBuffer
export const getOnnxModelBuffer = async (modelFilename = 'engagement_multitask_v4.onnx') => {
  // Try loading from URL first with various paths that might work
  const baseUrl = window.location.origin;
  const basePath = window.location.pathname.endsWith('/') 
    ? window.location.pathname 
    : window.location.pathname + '/';
  
  const urls = [
    `./models/${modelFilename}`,
    `/models/${modelFilename}`,
    `/focus-flow-client/models/${modelFilename}`,
    baseUrl + `/models/${modelFilename}`,
    baseUrl + `/focus-flow-client/models/${modelFilename}`,
    baseUrl + basePath + `models/${modelFilename}`,
    new URL(`models/${modelFilename}`, baseUrl + basePath).href
  ];
  
  // Log all URLs we're going to try
  console.log(`Attempting to load ONNX model buffer '${modelFilename}' from the following URLs:`, urls);
  
  for (const url of urls) {
    const modelBuffer = await loadOnnxModelFromUrl(url);
    if (modelBuffer) {
      console.log(`Successfully loaded ONNX model buffer from ${url}`);
      return modelBuffer;
    }
  }
  
  // If none of the URLs work, return null
  console.error('Failed to load ONNX model buffer from any URL');
  return null;
};
