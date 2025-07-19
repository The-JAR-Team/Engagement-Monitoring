// Fallback hook fetch from public hooks folder
export async function fetchModelFromHooks(modelFilename) {
  const url = `${window.location.origin}/hooks/${modelFilename}`;
  return url;
}
