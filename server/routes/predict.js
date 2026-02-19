/**
 * predict.js — Express Routes for Prediction API
 * =================================================
 * Proxies requests to FastAPI ML service and logs results to MongoDB.
 *
 * Author : Student ML Engineer
 * Project: Stock Price Prediction System
 */

const express = require("express");
const axios = require("axios");
const Prediction = require("../models/Prediction");

const router = express.Router();

// FastAPI ML service URL
const ML_API_URL = process.env.ML_API_URL || "http://localhost:8000";

// Axios instance with default timeout (120s — training is slow)
const mlApi = axios.create({
  baseURL: ML_API_URL,
  timeout: 120_000,
});

// Ticker validation regex: 1-5 uppercase letters
const TICKER_REGEX = /^[A-Z]{1,5}$/;

function validateTicker(ticker) {
  if (!ticker) return "Ticker symbol is required";
  if (!TICKER_REGEX.test(ticker.toUpperCase())) {
    return "Invalid ticker format — must be 1-5 letters (e.g. AAPL, TSLA)";
  }
  return null;
}

/**
 * POST /api/predict
 * Forward prediction request to FastAPI, save result to MongoDB.
 *
 * Body: { ticker: "AAPL", days: 5 }
 */
router.post("/predict", async (req, res) => {
  try {
    const { ticker, days = 5 } = req.body;

    const tickerError = validateTicker(ticker);
    if (tickerError) {
      return res.status(400).json({ error: tickerError });
    }

    // Forward to FastAPI
    const response = await mlApi.post("/predict", {
      ticker: ticker.toUpperCase(),
      days: parseInt(days),
    });

    const result = response.data;

    // Save to MongoDB (non-blocking, don't fail if MongoDB is down)
    try {
      await Prediction.create({
        ticker: result.ticker,
        model_used: result.model_used,
        days_ahead: days,
        predictions: result.predictions,
        metrics: result.metrics,
      });
    } catch (dbErr) {
      console.warn("MongoDB save failed (non-critical):", dbErr.message);
    }

    res.json(result);
  } catch (error) {
    console.error("Prediction error:", error.response?.data || error.message);

    if (error.code === "ECONNABORTED") {
      return res.status(504).json({
        error: "Request timed out. The ML service is taking too long to respond.",
      });
    }

    if (error.response) {
      // Forward FastAPI error
      return res.status(error.response.status).json({
        error: error.response.data?.detail || "ML service error",
      });
    }

    res.status(500).json({
      error: "ML service unavailable. Make sure FastAPI is running on port 8000.",
    });
  }
});

/**
 * POST /api/train
 * Forward training request to FastAPI.
 *
 * Body: { ticker: "AAPL", period: "5y", tune: true }
 */
router.post("/train", async (req, res) => {
  try {
    const { ticker, period = "5y", tune = true } = req.body;

    const tickerError = validateTicker(ticker);
    if (tickerError) {
      return res.status(400).json({ error: tickerError });
    }

    // Training can take a long time — use 5 min timeout
    const response = await mlApi.post("/train", {
      ticker: ticker.toUpperCase(),
      period,
      tune,
    }, { timeout: 300_000 });

    res.json(response.data);
  } catch (error) {
    console.error("Training error:", error.response?.data || error.message);

    if (error.code === "ECONNABORTED") {
      return res.status(504).json({
        error: "Training timed out. Try again with a shorter period or tune=false.",
      });
    }

    if (error.response) {
      return res.status(error.response.status).json({
        error: error.response.data?.detail || "ML service error",
      });
    }

    res.status(500).json({
      error: "ML service unavailable. Make sure FastAPI is running on port 8000.",
    });
  }
});

/**
 * GET /api/history/:ticker
 * Get recent historical data for charting.
 */
router.get("/history/:ticker", async (req, res) => {
  try {
    const { ticker } = req.params;

    const tickerError = validateTicker(ticker);
    if (tickerError) {
      return res.status(400).json({ error: tickerError });
    }

    const days = req.query.days || 90;

    const response = await mlApi.get(
      `/history/${ticker.toUpperCase()}?days=${days}`
    );

    res.json(response.data);
  } catch (error) {
    console.error("History error:", error.response?.data || error.message);

    if (error.response) {
      return res.status(error.response.status).json({
        error: error.response.data?.detail || "ML service error",
      });
    }

    res.status(500).json({ error: "ML service unavailable." });
  }
});

/**
 * GET /api/info/:ticker
 * Get stock metadata (name, sector, price, market cap, etc.).
 */
router.get("/info/:ticker", async (req, res) => {
  try {
    const { ticker } = req.params;

    const tickerError = validateTicker(ticker);
    if (tickerError) {
      return res.status(400).json({ error: tickerError });
    }

    const response = await mlApi.get(
      `/info/${ticker.toUpperCase()}`
    );

    res.json(response.data);
  } catch (error) {
    console.error("Stock info error:", error.response?.data || error.message);

    if (error.response) {
      return res.status(error.response.status).json({
        error: error.response.data?.detail || "ML service error",
      });
    }

    res.status(500).json({ error: "ML service unavailable." });
  }
});

/**
 * GET /api/predictions
 * Retrieve prediction history from MongoDB.
 */
router.get("/predictions", async (req, res) => {
  try {
    const { ticker, limit = 20 } = req.query;
    const filter = ticker ? { ticker: ticker.toUpperCase() } : {};

    const predictions = await Prediction.find(filter)
      .sort({ created_at: -1 })
      .limit(parseInt(limit));

    res.json(predictions);
  } catch (error) {
    console.error("Predictions history error:", error.message);
    res.status(500).json({ error: "Failed to retrieve prediction history." });
  }
});

module.exports = router;

