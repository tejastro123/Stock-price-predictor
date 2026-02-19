/**
 * Prediction.js — Mongoose Schema for Prediction Logs
 * =====================================================
 * Stores prediction results in MongoDB for history tracking.
 *
 * Author : Student ML Engineer
 * Project: Stock Price Prediction System
 */

const mongoose = require("mongoose");

const PredictionSchema = new mongoose.Schema(
  {
    ticker: {
      type: String,
      required: true,
      uppercase: true,
      trim: true,
      index: true,
    },
    model_used: {
      type: String,
      default: "XGBoost",
    },
    days_ahead: {
      type: Number,
      required: true,
    },
    predictions: [
      {
        date: { type: String, required: true },
        close: { type: Number, required: true },
      },
    ],
    metrics: {
      rmse: Number,
      mae: Number,
      mape: Number,
      directional_accuracy: Number,
    },
    created_at: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true,
  }
);

module.exports = mongoose.model("Prediction", PredictionSchema);
