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

/**
 * POST /api/predict
 * Forward prediction request to FastAPI, save result to MongoDB.
 *
 * Body: { ticker: "AAPL", days: 5 }
 */
router.post("/predict", async (req, res) => {
  try {
    const { ticker, days = 5 } = req.body;

    if (!ticker) {
      return res.status(400).json({ error: "Ticker symbol is required" });
    }

    // Forward to FastAPI
    const response = await axios.post(`${ML_API_URL}/predict`, {
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

    if (!ticker) {
      return res.status(400).json({ error: "Ticker symbol is required" });
    }

    const response = await axios.post(`${ML_API_URL}/train`, {
      ticker: ticker.toUpperCase(),
      period,
      tune,
    });

    res.json(response.data);
  } catch (error) {
    console.error("Training error:", error.response?.data || error.message);

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
    const days = req.query.days || 90;

    const response = await axios.get(
      `${ML_API_URL}/history/${ticker.toUpperCase()}?days=${days}`
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
