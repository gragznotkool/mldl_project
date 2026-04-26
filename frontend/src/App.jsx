import React from 'react';
import Dashboard from './components/Dashboard';

function App() {
  return (
    <div className="app-container">
      <header className="header">
        <div className="logo-text">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{color: '#60a5fa'}}>
            <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
            <path d="M2 17l10 5 10-5"></path>
            <path d="M2 12l10 5 10-5"></path>
          </svg>
          Traffic Congestion AI
        </div>
        <div style={{display: 'flex', gap: '1rem'}}>
           {/* Add dummy buttons for aesthetic realism */}
           <button className="btn btn-secondary" style={{padding: '0.4rem 1rem', fontSize: '0.85rem'}}>Docs</button>
           <div style={{width: '32px', height: '32px', borderRadius: '50%', background: 'linear-gradient(135deg, #a78bfa, #3b82f6)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: 'white', fontSize: '0.8rem'}}>AD</div>
        </div>
      </header>
      
      <main className="main-content">
        <div style={{marginBottom: '2rem'}}>
          <h1 style={{fontSize: '2.5rem', marginBottom: '0.5rem', fontWeight: '800'}}>Network Analysis</h1>
          <p style={{color: 'var(--text-secondary)', fontSize: '1.1rem', maxWidth: '600px'}}>Predict urban traffic congestion levels using advanced LSTM neural network models.</p>
        </div>
        
        <Dashboard />
      </main>
    </div>
  );
}

export default App;
