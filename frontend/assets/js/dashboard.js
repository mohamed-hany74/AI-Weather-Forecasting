// frontend/assets/js/dashboard.js
// Main UI Controller for WeatherCast AI

let current168History = [];
let currentTimestamps = [];
let currentReferenceTimeUTC = null; // Single source of truth for t (UTC ISO string)

document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  checkSystemStatus();
  
  // Initialize with current time & realistic Cairo series
  const now = new Date();
  setReferenceTime(now);
  generateCairoSampleSeries(now);
});

// Standard HTML5 datetime-local format: "YYYY-MM-DDTHH:mm" (in user local time)
function formatLocalForPicker(date) {
  const pad = (n) => String(n).padStart(2, '0');
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  return `${year}-${month}-${day}T${hours}:${minutes}`;
}

function updateActiveUtcDisplay(date) {
  const displayEl = document.getElementById('activeUtcDisplay');
  if (displayEl && date && !isNaN(date.getTime())) {
    displayEl.textContent = `${date.toUTCString().replace('GMT', 'UTC')}`;
  }
}

function setReferenceTime(date) {
  if (!date || isNaN(date.getTime())) {
    date = new Date();
  }
  currentReferenceTimeUTC = date.toISOString();
  
  const picker = document.getElementById('timeInput');
  if (picker) {
    picker.value = formatLocalForPicker(date);
  }
  updateActiveUtcDisplay(date);
}

function setupEventListeners() {
  // Theme toggle
  const themeBtn = document.getElementById('themeToggleBtn');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      document.body.classList.toggle('light-theme');
      ForecastChart.render(current168History, currentTimestamps, 168);
    });
  }

  // Now button
  const setNowBtn = document.getElementById('setNowBtn');
  if (setNowBtn) {
    setNowBtn.addEventListener('click', () => {
      const now = new Date();
      setReferenceTime(now);
      generateCairoSampleSeries(now);
      showToast('Timestamp snapped to current time', 'info');
    });
  }

  // City chips
  document.querySelectorAll('.city-chips-grid .chip').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.city-chips-grid .chip').forEach(c => c.classList.remove('active'));
      e.currentTarget.classList.add('active');
      document.getElementById('latInput').value = e.currentTarget.dataset.lat;
      document.getElementById('lonInput').value = e.currentTarget.dataset.lon;
      showToast(`Selected ${e.currentTarget.dataset.name}`, 'info');
    });
  });

  // Time picker change event (supports partial and complete edits)
  const timeInput = document.getElementById('timeInput');
  if (timeInput) {
    timeInput.addEventListener('input', handleTimePickerChange);
    timeInput.addEventListener('change', handleTimePickerChange);
  }

  // Action Buttons
  const sampleBtn = document.getElementById('sampleDataBtn');
  if (sampleBtn) {
    sampleBtn.addEventListener('click', () => {
      const t = currentReferenceTimeUTC ? new Date(currentReferenceTimeUTC) : new Date();
      generateCairoSampleSeries(t);
      showToast('Loaded 168h Cairo sample series', 'success');
    });
  }

  const autoFetchBtn = document.getElementById('autoFetchBtn');
  if (autoFetchBtn) {
    autoFetchBtn.addEventListener('click', handleAutoFetchOpenMeteo);
  }

  const forecastBtn = document.getElementById('generateForecastBtn');
  if (forecastBtn) {
    forecastBtn.addEventListener('click', handleGenerateForecast);
  }

  // Chart range selector
  document.querySelectorAll('.range-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      e.currentTarget.classList.add('active');
      const hours = parseInt(e.currentTarget.dataset.range, 10);
      ForecastChart.render(current168History, currentTimestamps, hours);
    });
  });
}

function handleTimePickerChange(e) {
  const val = e.target.value;
  if (val) {
    const parsed = new Date(val);
    if (!isNaN(parsed.getTime())) {
      currentReferenceTimeUTC = parsed.toISOString();
      updateActiveUtcDisplay(parsed);
      generateCairoSampleSeries(parsed);
    }
  }
}

async function checkSystemStatus() {
  try {
    const [health, modelInfo] = await Promise.all([
      ApiClient.getHealth(),
      ApiClient.getModelInfo()
    ]);

    if (health.status === 'ok') {
      document.getElementById('apiStatusBadge').innerHTML = '<span class="status-dot"></span> API Online';
      document.getElementById('healthApiStatus').textContent = 'PASS';
      document.getElementById('healthApiStatus').className = 'health-badge health-badge-pass';
    }

    if (modelInfo.status === 'success') {
      const m = modelInfo.model;
      document.getElementById('modelStatusBadge').textContent = `🧠 ${m.model_type} (${m.feature_count} Features)`;
      document.getElementById('healthModelStatus').textContent = 'PASS';
    }
  } catch (err) {
    document.getElementById('apiStatusBadge').innerHTML = '⚠️ API Degraded';
    document.getElementById('healthApiStatus').textContent = 'FAIL';
    document.getElementById('healthApiStatus').className = 'health-badge health-badge-error';
  }
}

function generateCairoSampleSeries(refDate) {
  current168History = [];
  currentTimestamps = [];
  currentReferenceTimeUTC = refDate.toISOString();

  const baseTemp = 28.5;
  const refMs = refDate.getTime();

  for (let i = 168; i >= 1; i--) {
    const t = new Date(refMs - i * 3600 * 1000);
    currentTimestamps.push(t.toISOString());
    const hr = t.getUTCHours();
    const diurnal = Math.cos(((hr - 14) / 24) * 2 * Math.PI) * 5.8;
    const noise = (Math.random() - 0.5) * 0.6;
    const val = parseFloat((baseTemp + diurnal + noise).toFixed(2));
    current168History.push(val);
  }

  const statusText = document.getElementById('historyStatusText');
  if (statusText) {
    statusText.textContent = `✅ 168h Cairo series loaded (Reference: ${refDate.toUTCString().replace('GMT', 'UTC')})`;
  }
  updateActiveUtcDisplay(refDate);
  ForecastChart.render(current168History, currentTimestamps, 168);
}

async function handleAutoFetchOpenMeteo() {
  const lat = parseFloat(document.getElementById('latInput').value);
  const lon = parseFloat(document.getElementById('lonInput').value);
  const btn = document.getElementById('autoFetchBtn');

  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏳ Querying Open-Meteo...';
  }

  try {
    const data = await ApiClient.fetchOpenMeteoHistory(lat, lon);
    const times = data.hourly?.time || [];
    const temps = data.hourly?.temperature_2m || [];

    const nowUtc = new Date();
    const aligned = TemporalValidator.alignHistoricalSlice(times, temps, nowUtc);

    current168History = aligned.history;
    currentTimestamps = aligned.timestamps;
    setReferenceTime(aligned.referenceTime);

    document.getElementById('historyStatusText').textContent =
      `✅ Live Open-Meteo series loaded (Latest t-1h: ${current168History[167]}°C)`;
    document.getElementById('healthWeatherServiceStatus').textContent = 'PASS';
    showToast('Live 7-day weather history verified and loaded!', 'success');
    ForecastChart.render(current168History, currentTimestamps, 168);
  } catch (err) {
    showToast(`Open-Meteo query failed: ${err.message}`, 'error');
    document.getElementById('healthWeatherServiceStatus').textContent = 'WARNING';
    generateCairoSampleSeries(new Date());
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = '⚡ Auto-Fetch Open-Meteo';
    }
  }
}

async function handleGenerateForecast() {
  if (current168History.length !== 168) {
    showToast('Please load 168-hour history first.', 'error');
    return;
  }

  const btn = document.getElementById('generateForecastBtn');
  const btnText = btn?.querySelector('.btn-text');
  const btnSpinner = btn?.querySelector('.btn-spinner');

  if (btn) btn.disabled = true;
  if (btnText) btnText.style.display = 'none';
  if (btnSpinner) btnSpinner.style.display = 'inline';

  const payload = {
    current_time: currentReferenceTimeUTC || new Date().toISOString(),
    latitude: parseFloat(document.getElementById('latInput').value),
    longitude: parseFloat(document.getElementById('lonInput').value),
    temperature_history_168h: current168History
  };

  try {
    TemporalValidator.validatePayload(payload);
    const res = await ApiClient.predictTemperature(payload);

    const pred = res.prediction.predicted_temperature_2m;
    document.getElementById('forecastValue').textContent = pred.toFixed(2);

    const fDate = new Date(res.prediction.forecast_time);
    document.getElementById('targetForecastTimestamp').textContent =
      `Target Horizon: ${fDate.toUTCString().replace('GMT', 'UTC')} (+24h)`;

    // Deltas
    const latestHist = current168History[167]; // t-1h
    const d1 = pred - latestHist;
    document.getElementById('deltaLastHour').textContent = (d1 >= 0 ? '+' : '') + d1.toFixed(2) + '°C';

    const yestHist = current168History[144]; // t-24h
    const d24 = pred - yestHist;
    document.getElementById('deltaYesterday').textContent = (d24 >= 0 ? '+' : '') + d24.toFixed(2) + '°C';

    const mean7d = current168History.reduce((a, b) => a + b, 0) / 168;
    document.getElementById('statMean7d').textContent = mean7d.toFixed(2) + '°C';

    // Exogenous
    if (res.context?.exogenous_weather_used) {
      const exo = res.context.exogenous_weather_used;
      document.getElementById('exoApparent').textContent = `${exo.apparent_temperature.toFixed(1)} °C`;
      document.getElementById('exoPressure').textContent = `${exo.pressure_msl.toFixed(1)} hPa`;
      document.getElementById('exoHumidity').textContent = `${Math.round(exo.relative_humidity_2m)} %`;
    }

    // Diagnostics Table
    if (res.context?.features_engineered) {
      renderFeatureDiagnosticsTable(res.context.features_engineered);
    }

    showToast(`Forecast generated: ${pred.toFixed(2)}°C`, 'success');
    ForecastChart.render(current168History, currentTimestamps, 168, pred, res.prediction.forecast_time);

  } catch (err) {
    showToast(`Forecast failed: ${err.message}`, 'error');
  } finally {
    if (btn) btn.disabled = false;
    if (btnText) btnText.style.display = 'inline';
    if (btnSpinner) btnSpinner.style.display = 'none';
  }
}

function renderFeatureDiagnosticsTable(featuresMap) {
  const tbody = document.getElementById('featureTableBody');
  if (!tbody) return;
  tbody.innerHTML = '';

  let idx = 1;
  for (const [name, val] of Object.entries(featuresMap)) {
    const tr = document.createElement('tr');
    let typeDesc = 'Lag Reading';
    if (name.includes('rolling')) typeDesc = 'Rolling Window';
    else if (name.includes('cos') || name.includes('sin')) typeDesc = 'Cyclical Time';
    else if (['apparent_temperature', 'pressure_msl', 'relative_humidity_2m'].includes(name)) typeDesc = 'Exogenous Weather';

    tr.innerHTML = `
      <td>${idx++}</td>
      <td><code>${name}</code></td>
      <td>${typeof val === 'number' ? val.toFixed(4) : val}</td>
      <td><span class="badge-tag">${typeDesc}</span></td>
    `;
    tbody.appendChild(tr);
  }
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(12px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}
