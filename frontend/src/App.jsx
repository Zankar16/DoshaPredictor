import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Sparkles, Activity, User, Info, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';
import './App.css';

const API_BASE = import.meta.env.MODE === 'production' ? '/api' : 'http://localhost:5000/api';

function App() {
  const [features, setFeatures] = useState({});
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchFeatures();
  }, []);

  const fetchFeatures = async () => {
    try {
      const resp = await axios.get(`${API_BASE}/features`);
      setFeatures(resp.data);
      // Initialize form with first option of each feature
      const initialData = {};
      Object.keys(resp.data).forEach(key => {
        initialData[key] = resp.data[key][0];
      });
      setFormData(initialData);
    } catch (err) {
      console.error(err);
      setError('Failed to load Ayurvedic features.');
    }
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const resp = await axios.post(`${API_BASE}/predict`, formData);
      setPrediction(resp.data.dosha);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || 'A prediction error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo">
          <Sparkles className="icon-gold" />
          <h1>AyurPredict</h1>
        </div>
        <p>Discover your unique Ayurvedic Dosha profile</p>
      </header>

      <main className="app-main">
        {!prediction && !error ? (
          <section className="form-section fade-in">
            <div className="card">
              <div className="card-header">
                <User className="card-icon" />
                <h2>Personal Attributes</h2>
              </div>
              <form onSubmit={handleSubmit}>
                <div className="grid-container">
                  {Object.keys(features).map((key) => (
                    <div key={key} className="form-group">
                      <label htmlFor={key}>{key}</label>
                      <select name={key} id={key} value={formData[key]} onChange={handleChange}>
                        {features[key].map((opt) => (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ))}
                      </select>
                    </div>
                  ))}
                </div>
                <button type="submit" disabled={loading} className="btn-predict">
                  {loading ? <RefreshCw className="spin" /> : 'Analyze My Dosha'}
                </button>
              </form>
            </div>
          </section>
        ) : (
          <section className="result-section fade-in">
            <div className={`card result-card ${prediction ? 'success' : 'error'}`}>
              {prediction ? (
                <>
                  <CheckCircle2 className="result-icon success" />
                  <h2>Your Dominant Dosha</h2>
                  <div className="prediction-box">
                    <span className="dosha-name">{prediction}</span>
                  </div>
                  <p className="dosha-description">
                    Based on your physical and behavioral attributes, your constitution leans towards <strong>{prediction}</strong>. 
                  </p>
                  <button onClick={resetForm} className="btn-secondary">New Analysis</button>
                </>
              ) : (
                <>
                  <AlertCircle className="result-icon error" />
                  <h2>Analysis Failed</h2>
                  <p>{error}</p>
                  <button onClick={resetForm} className="btn-secondary">Try Again</button>
                </>
              )}
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <p>&copy; 2026 AyurPredict. Wisdom of Wellness.</p>
      </footer>
    </div>
  );
}

export default App;
