/**
 * server.js — Express.js Backend Server
 * =======================================
 * Acts as a middleware between the React frontend and the FastAPI ML service.
 * Connects to MongoDB for prediction logging (optional).
 *
 * Author : Student ML Engineer
 * Project: Stock Price Prediction System
 */

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");

const predictRoutes = require("./routes/predict");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const PORT = process.env.PORT || 5000;
const MONGODB_URI =
  process.env.MONGODB_URI || "mongodb://localhost:27017/stock_predictor";

// ---------------------------------------------------------------------------
// Express Setup
// ---------------------------------------------------------------------------
const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Request logger
app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// Routes
app.use("/api", predictRoutes);

// Health check
app.get("/api/health", (_req, res) => {
  res.json({
    status: "healthy",
    service: "stock-predictor-express",
    mongodb: mongoose.connection.readyState === 1 ? "connected" : "disconnected",
  });
});

// ---------------------------------------------------------------------------
// MongoDB Connection (optional — server works without it)
// ---------------------------------------------------------------------------
async function connectDB() {
  try {
    await mongoose.connect(MONGODB_URI);
    console.log(`✅ MongoDB connected: ${MONGODB_URI}`);
  } catch (err) {
    console.warn(
      `⚠️  MongoDB connection failed: ${err.message}\n` +
      "   The server will still work, but prediction history won't be saved.\n" +
      "   To enable MongoDB, install and start MongoDB or update MONGODB_URI."
    );
  }
}

// ---------------------------------------------------------------------------
// Start Server
// ---------------------------------------------------------------------------
connectDB().then(() => {
  app.listen(PORT, () => {
    console.log(`\n🚀 Express server running on http://localhost:${PORT}`);
    console.log(`   API routes: http://localhost:${PORT}/api/predict`);
    console.log(`   Health:     http://localhost:${PORT}/api/health\n`);
  });
});
