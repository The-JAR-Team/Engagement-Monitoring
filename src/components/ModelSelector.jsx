import React, { useEffect, useState } from 'react';
import { getAvailableModels, switchModel } from '../services/engagementOnnxService';
import { getCurrentModelInfo } from '../services/engagementOnnxService';

const ModelSelector = () => {
  const [models, setModels] = useState([]);
  const [selected, setSelected] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getAvailableModels().then(cfgs => {
      const list = Object.values(cfgs);
      setModels(list);
      const active = getCurrentModelInfo();
      setSelected(active.id);
    });
  }, []);

  const handleChange = async (e) => {
    const id = e.target.value;
    setSelected(id);
    setLoading(true);
    try {
      await switchModel(id);
    } catch (error) {
      console.error('Failed to switch model:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <select 
      value={selected} 
      onChange={handleChange} 
      disabled={loading}
      className="model-dropdown"
    >
      {loading && <option>Switching model...</option>}
      {!loading && models.map(m => (
        <option key={m.id} value={m.id}>{m.name}</option>
      ))}
    </select>
  );
};

export default ModelSelector;
