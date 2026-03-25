-- Database Schema for DoshaPredictor

-- Create table for feature options (for UI population)
CREATE TABLE IF NOT EXISTS feature_options (
    id SERIAL PRIMARY KEY,
    feature_name TEXT NOT NULL,
    options JSONB NOT NULL
);

-- Create table for prediction history
CREATE TABLE IF NOT EXISTS prediction_history (
    id SERIAL PRIMARY KEY,
    user_inputs JSONB NOT NULL,
    predicted_dosha TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Note: The backend currently uses a local JSON file for options to ensure 
-- it works without a pre-populated database, but can be extended to use these tables.
