// frontend/assets/js/api.js
// Clean REST API client for WeatherCast AI backend

const ApiClient = {
  async getHealth() {
    try {
      const res = await fetch('/api/health/');
      if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error('API Health Error:', err);
      throw err;
    }
  },

  async getModelInfo() {
    try {
      const res = await fetch('/api/model-info/');
      if (!res.ok) throw new Error(`Model info fetch failed: HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error('API Model Info Error:', err);
      throw err;
    }
  },

  async predictTemperature(payload) {
    try {
      const res = await fetch('/api/predict/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok || data.status !== 'success') {
        const msg = data.message || data.details || 'Inference error';
        const err = new Error(msg);
        err.code = data.code;
        err.details = data.details;
        throw err;
      }
      return data;
    } catch (err) {
      console.error('API Prediction Error:', err);
      throw err;
    }
  },

  async fetchOpenMeteoHistory(lat, lon) {
    const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=temperature_2m&past_days=7&forecast_days=1&timezone=UTC`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Open-Meteo returned HTTP ${res.status}`);
    return await res.json();
  }
};
