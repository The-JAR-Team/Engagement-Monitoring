import { useEffect, useRef } from 'react';
import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

let globalFaceMeshInstance = null;
let globalFaceMeshInitialized = false;
let globalInitializationInProgress = false;
let globalCameraInstance = null;

export default function useFaceMesh(enabled, videoRef, onResults, onStatusChange, onErrorState) {
  const errorCount = useRef(0);
  const streamRef = useRef(null);
  const initializedRef = useRef(false);
  const resultCallbackRef = useRef(onResults);
  const statusCallbackRef = useRef(onStatusChange);
  // Provide a no-op default if onErrorState is not a function
  const errorStateCallbackRef = useRef(typeof onErrorState === 'function' ? onErrorState : () => {});
  const MAX_ERRORS = 10;

  useEffect(() => {
    resultCallbackRef.current = onResults;
    statusCallbackRef.current = onStatusChange;
    // Update error state callback only if function, else keep no-op
    errorStateCallbackRef.current = typeof onErrorState === 'function' ? onErrorState : () => {};
  }, [onResults, onStatusChange, onErrorState]);

  useEffect(() => {
    if (!enabled) return;
    let isComponentMounted = true;
    const MAX_INIT_RETRIES = 3;
    const INIT_RETRY_DELAY_MS = 1000;

    const updateStatus = (status) => {
      if (isComponentMounted && statusCallbackRef.current) {
        statusCallbackRef.current(status);
      }
    };

    const retryInitialization = async (fn) => {
      let lastError;
      for (let i = 1; i <= MAX_INIT_RETRIES; i++) {
        try {
          return await fn();
        } catch (err) {
          lastError = err;
          updateStatus(`Initializing FaceMesh… retry ${i}/${MAX_INIT_RETRIES}`);
          await new Promise(r => setTimeout(r, INIT_RETRY_DELAY_MS));
        }
      }
      throw lastError;
    };

    const initializeWebcam = async () => {
      try {
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });
        if (videoRef.current && isComponentMounted) {
          videoRef.current.srcObject = stream;
          streamRef.current = stream;
          await new Promise(resolve => { videoRef.current.onloadedmetadata = () => resolve(); });
          videoRef.current.play().catch(() => {});
          return true;
        }
      } catch (err) {
        updateStatus('Camera access denied');
        return false;
      }
      return false;
    };

    const initializeFaceMesh = async () => {
      errorCount.current = 0;
      const webcamReady = await initializeWebcam();
      if (!webcamReady || !isComponentMounted) return false;
      if (globalInitializationInProgress) {
        let wait = 0;
        while (globalInitializationInProgress && wait < 50) {
          await new Promise(r => setTimeout(r, 100));
          wait++;
        }
        if (globalFaceMeshInitialized && globalFaceMeshInstance) {
          globalFaceMeshInstance.onResults(results => {
            if (!isComponentMounted) return;
            if (!initializedRef.current) {
              initializedRef.current = true;
              updateStatus('FaceMesh Ready');
            }
            resultCallbackRef.current(results);
            errorCount.current = 0;
            errorStateCallbackRef.current(false);
          });
          if (!globalCameraInstance && videoRef.current) {
            globalCameraInstance = new Camera(videoRef.current, {
              onFrame: async () => {
                try { await globalFaceMeshInstance.send({ image: videoRef.current }); }
                catch (err) {
                  errorCount.current++;
                  errorStateCallbackRef.current(errorCount.current > MAX_ERRORS);
                  if (errorCount.current > MAX_ERRORS) updateStatus('Too many errors - Click Retry');
                }
              }, width: 640, height: 480
            });
            await globalCameraInstance.start();
          }
          return true;
        }
      }
      globalInitializationInProgress = true;
      try {
        if (!globalFaceMeshInstance || !globalFaceMeshInitialized) {
          updateStatus('Loading FaceMesh model...');
          const fm = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
          fm.setOptions({ maxNumFaces:1, refineLandmarks:true, minDetectionConfidence:0.5, minTrackingConfidence:0.5 });
          fm.onResults(results => {
            if (!isComponentMounted) return;
            if (!initializedRef.current) { initializedRef.current = true; updateStatus('FaceMesh Ready'); }
            resultCallbackRef.current(results);
            errorCount.current = 0;
            errorStateCallbackRef.current(false);
          });
          await fm.initialize();
          globalFaceMeshInstance = fm;
          globalFaceMeshInitialized = true;
        } else {
          globalFaceMeshInstance.onResults(resultCallbackRef.current);
        }
        if (!globalCameraInstance && videoRef.current) {
          globalCameraInstance = new Camera(videoRef.current, {
            onFrame: async () => {
              try { await globalFaceMeshInstance.send({ image: videoRef.current }); }
              catch (err) {
                errorCount.current++;
                errorStateCallbackRef.current(errorCount.current > MAX_ERRORS);
                if (errorCount.current > MAX_ERRORS) updateStatus('Too many errors - Click Retry');
              }
            }, width:640, height:480
          });
          await globalCameraInstance.start();
        }
        return true;
      } catch {
        updateStatus('Initialization Error - Click Retry');
        return false;
      } finally {
        globalInitializationInProgress = false;
      }
    };

    retryInitialization(initializeFaceMesh).catch(() => updateStatus('Initialization Failed - Click Retry'));
    return () => {
      isComponentMounted = false;
      if (streamRef.current) streamRef.current = null;
    };
  }, [enabled, videoRef]);
}
