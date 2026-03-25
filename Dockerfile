# Use Node.js as the base image for the frontend build and backend
FROM node:20-slim AS builder

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && \
    ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy dataset and shared files
COPY dataset.csv .
COPY feature_options.json .

# Setup Backend
WORKDIR /app/backend
COPY backend/package*.json ./
RUN npm install
COPY backend/ ./

# Install Python ML dependencies in the container
RUN pip3 install --no-cache-dir numpy scikit-learn pandas --break-system-packages

# Setup Frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# Build the React app (output will be in /app/frontend/dist)
RUN npm run build

# Final stage: Run the backend and serve the frontend
WORKDIR /app/backend

# Environment variables
ENV PORT=5000
ENV NODE_ENV=production

# Expose the API port
EXPOSE 5000

# Script to start the server (Note: in production, Express could serve the static frontend files)
CMD ["node", "server.js"]
