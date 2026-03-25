# DoshaPredictor - AyurPredict

A full-stack PERN application to classify Ayurvedic doshas (Vata, Pitta, Kapha) based on 20 user input features using K-Means + Random Forest.

## 🚀 Quick Start

### 1. Backend (Node.js + Python)
```bash
cd backend
npm install
node server.js
```
*Requires Python with `numpy` and `scikit-learn` installed.*

### 2. Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

## 🛠️ Tech Stack
- **Frontend**: React, Vite, Lucide React, Axios.
- **Backend**: Node.js, Express, Child Process (Python Bridge).
- **Engine**: Scikit-Learn, NumPy, K-Means, Random Forest.
- **Database**: PostgreSQL (Schema in `/database`).

## 📁 Project Structure
- `/frontend`: React application.
- `/backend`: Express API and Python predictor.
- `/database`: SQL schema.
- `dataset.csv`: Training data.

## ⚖️ Features
- Body Size, Weight, Height, Bone Structure, Complexion, Skin Feel/Texture, Hair Color/Appearance, Face Shape, Eyes/Eyelashes/Blinking, Cheeks, Nose, Teeth/Gums, Lips, Nails, Appetite, Liking Tastes.

---
Created by AyurPredict 2026.
