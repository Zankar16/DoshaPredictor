const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const { Pool } = require('pg');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// Serve static frontend files in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../frontend/dist')));
  app.get('*', (req, res) => {
    if (!req.path.startsWith('/api')) {
      res.sendFile(path.join(__dirname, '../frontend/dist/index.html'));
    }
  });
}

app.get('/', (req, res) => {
  res.send('AyurPredict Backend API is running. Please use the Frontend UI on port 5173/5174.');
});

// Logging middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  next();
});

let featureOptions = {};
try {
  featureOptions = require('../feature_options.json');
} catch (e) {
  console.error("CRITICAL: Failed to load feature_options.json:", e.message);
}

app.get('/api/features', (req, res) => {
  if (Object.keys(featureOptions).length === 0) {
    return res.status(500).json({ error: "Feature options not loaded" });
  }
  res.json(featureOptions);
});

app.post('/api/predict', (req, res) => {
  const userInput = req.body;
  console.log("Predicting for:", JSON.stringify(userInput).substring(0, 100) + "...");
  
  const pythonExecutable = 'python'; // or 'python3' depending on environment
  const predictorPath = path.join(__dirname, 'predictor.py');

  const pythonProcess = spawn(pythonExecutable, [
    predictorPath,
    JSON.stringify(userInput)
  ]);

  let result = '';
  let errorResult = '';

  pythonProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    errorResult += data.toString();
  });

  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      console.error(`Python script exited with code ${code}. Error: ${errorResult}`);
      return res.status(500).json({ error: 'Failed to predict', details: errorResult });
    }
    try {
      const prediction = JSON.parse(result);
      if (prediction.error) {
        return res.status(400).json({ error: prediction.error });
      }
      res.json(prediction);
    } catch (e) {
      console.error(`Failed to parse prediction result: ${result}`);
      res.status(500).json({ error: 'Internal server error parsing prediction' });
    }
  });

  pythonProcess.on('error', (err) => {
    console.error("Failed to start Python process:", err);
    res.status(500).json({ error: "Python process failed to start", details: err.message });
  });
});

// Final error handler
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res.status(500).send("Something broke!");
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});

process.on('uncaughtException', (err) => {
  console.error('There was an uncaught error', err);
  process.exit(1);
});
