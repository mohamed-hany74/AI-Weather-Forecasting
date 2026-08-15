// frontend/assets/js/charts.js
// Chart.js rendering controller for 168h history + 24h forecast marker

let chartInstance = null;

const ForecastChart = {
  render(history, timestamps, hoursLimit = 168, predictionVal = null, forecastTimeStr = null) {
    const canvas = document.getElementById('forecastChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const slice = history.slice(-hoursLimit);
    const timeSlice = timestamps.slice(-hoursLimit);

    const labels = [];
    const histData = [];
    const predData = [];

    slice.forEach((temp, i) => {
      let tLabel = `t-${hoursLimit - i}h`;
      if (timeSlice[i]) {
        const d = new Date(timeSlice[i]);
        tLabel = `${d.getUTCMonth() + 1}/${d.getUTCDate()} ${d.getUTCHours()}:00`;
      }
      labels.push(tLabel);
      histData.push(temp);
      predData.push(null);
    });

    if (predictionVal !== null) {
      predData[predData.length - 1] = slice[slice.length - 1]; // connect line
      let fLabel = 'Forecast (t+24h)';
      if (forecastTimeStr) {
        const fd = new Date(forecastTimeStr);
        fLabel = `Forecast (${fd.getUTCMonth() + 1}/${fd.getUTCDate()} ${fd.getUTCHours()}:00 UTC)`;
      }
      labels.push(fLabel);
      histData.push(null);
      predData.push(predictionVal);
    }

    if (chartInstance) {
      chartInstance.destroy();
    }

    const isLight = document.body.classList.contains('light-theme');
    const tickColor = isLight ? '#475569' : '#94a3b8';
    const gridColor = isLight ? 'rgba(100,116,139,0.1)' : 'rgba(148,163,184,0.08)';

    chartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Observed Historical Temperature (°C)',
            data: histData,
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56,189,248,0.08)',
            borderWidth: 2,
            pointRadius: hoursLimit > 48 ? 0 : 3,
            pointHoverRadius: 5,
            fill: true,
            tension: 0.25
          },
          {
            label: 'XGBoost 24-Hour Ahead Forecast',
            data: predData,
            borderColor: '#f97316',
            borderDash: [6, 4],
            borderWidth: 2.5,
            pointRadius: 6,
            pointBackgroundColor: '#f97316',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            pointHoverRadius: 8,
            tension: 0.25
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { intersect: false, mode: 'index' },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: tickColor,
              font: { family: 'Plus Jakarta Sans', size: 12, weight: '600' }
            }
          }
        },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: tickColor, maxTicksLimit: 12, font: { size: 10 } }
          },
          y: {
            grid: { color: gridColor },
            ticks: { color: tickColor, callback: v => `${v}°C`, font: { size: 11 } }
          }
        }
      }
    });
  }
};
