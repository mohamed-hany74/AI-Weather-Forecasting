# WeatherCast AI — Frontend

A high-performance, dark-themed AI analytics dashboard for 24-hour weather predictions.

## Architecture

- **`index.html`**: Semantic HTML5 layout with glassmorphism cards and accessibility standards.
- **`assets/css/app.css`**: Vanilla CSS design system with HSL/custom tokens, responsive grid layouts, and micro-animations.
- **`assets/js/api.js`**: REST client for backend communication.
- **`assets/js/validation.js`**: Strict temporal and schema validation preventing future data leakage.
- **`assets/js/charts.js`**: Chart.js integration displaying 168-hour historical trajectory and distinct 24-hour forecast markers.
- **`assets/js/dashboard.js`**: Main UI controller and event coordinator.
