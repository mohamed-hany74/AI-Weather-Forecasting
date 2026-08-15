// frontend/assets/js/validation.js
// Client-side temporal validation rules to prevent future data or malformed payloads

const TemporalValidator = {
  validatePayload(payload) {
    if (!payload.current_time) {
      throw new Error("Missing 'current_time' reference timestamp.");
    }
    if (isNaN(payload.latitude) || payload.latitude < -90 || payload.latitude > 90) {
      throw new Error(`Invalid latitude: ${payload.latitude}`);
    }
    if (isNaN(payload.longitude) || payload.longitude < -180 || payload.longitude > 180) {
      throw new Error(`Invalid longitude: ${payload.longitude}`);
    }
    if (!Array.isArray(payload.temperature_history_168h) || payload.temperature_history_168h.length !== 168) {
      throw new Error(`History must contain exactly 168 readings, got ${payload.temperature_history_168h?.length}`);
    }
    return true;
  },

  // Ensures slice is strictly [t-168h ... t-1h] and no future timestamps enter history
  alignHistoricalSlice(times, temps, referenceUtcDate) {
    const targetHourPrefix = referenceUtcDate.toISOString().slice(0, 13) + ':00';
    let tIdx = times.findIndex(t => t.startsWith(targetHourPrefix.slice(0, 13)));

    if (tIdx === -1 || tIdx < 168) {
      const refMs = referenceUtcDate.getTime();
      tIdx = times.length - 1;
      for (let i = times.length - 1; i >= 0; i--) {
        if (new Date(times[i] + 'Z').getTime() <= refMs) {
          tIdx = i;
          break;
        }
      }
      if (tIdx < 168) {
        throw new Error('Insufficient historical data prior to reference hour.');
      }
    }

    const startIdx = tIdx - 168;
    const historySlice = temps.slice(startIdx, tIdx).map(v => parseFloat(v.toFixed(2)));
    const timestampSlice = times.slice(startIdx, tIdx);

    // Strict temporal boundary assertion:
    const lastHistTime = new Date(timestampSlice[timestampSlice.length - 1] + 'Z').getTime();
    const refTime = new Date(times[tIdx] + 'Z').getTime();
    if (lastHistTime >= refTime) {
      throw new Error(`Temporal leakage: Last historical timestamp (${timestampSlice[timestampSlice.length - 1]}) >= reference t (${times[tIdx]})`);
    }

    return {
      history: historySlice,
      timestamps: timestampSlice,
      referenceTime: new Date(times[tIdx] + 'Z')
    };
  }
};
