import React from 'react'
import './App.css'
import EngagementMonitor from './components/EngagementMonitor'
import OnnxModelPreloader from './components/OnnxModelPreloader'

function App() {
  return (
    <>
      <OnnxModelPreloader />
      <EngagementMonitor />
    </>
  )
}

export default App
