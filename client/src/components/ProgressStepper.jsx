/**
 * ProgressStepper.jsx — Multi-Step Progress Indicator
 * =====================================================
 * Shows a visual step-by-step progress indicator during training
 * or prediction, replacing the simple spinner.
 */

import React, { useState, useEffect } from "react";

const TRAINING_STEPS = [
  { label: "Fetching Data", desc: "Downloading historical stock data", duration: 3000 },
  { label: "Engineering Features", desc: "Computing technical indicators", duration: 2000 },
  { label: "Training Models", desc: "Training DT, RF, XGBoost, LightGBM", duration: 8000 },
  { label: "Hyperparameter Tuning", desc: "Optimizing the best model", duration: 12000 },
  { label: "Saving Model", desc: "Persisting model artifacts", duration: 1000 },
];

const PREDICTION_STEPS = [
  { label: "Loading Model", desc: "Loading trained model & scaler", duration: 1000 },
  { label: "Fetching Data", desc: "Getting latest market data", duration: 2000 },
  { label: "Computing Forecast", desc: "Generating recursive predictions", duration: 3000 },
  { label: "Confidence Intervals", desc: "Calculating prediction bounds", duration: 1000 },
];

export default function ProgressStepper({ mode, active }) {
  const [currentStep, setCurrentStep] = useState(0);
  const steps = mode === "training" ? TRAINING_STEPS : PREDICTION_STEPS;

  useEffect(() => {
    if (!active) {
      setCurrentStep(0);
      return;
    }

    setCurrentStep(0);
    let stepIdx = 0;
    let totalElapsed = 0;

    const timers = steps.map((step, idx) => {
      totalElapsed += idx === 0 ? 0 : steps[idx - 1].duration;
      return setTimeout(() => {
        setCurrentStep(idx);
      }, totalElapsed);
    });

    return () => timers.forEach(clearTimeout);
  }, [active, mode]);

  if (!active) return null;

  return (
    <div className="progress-stepper fade-in">
      {steps.map((step, idx) => {
        const isComplete = idx < currentStep;
        const isCurrent = idx === currentStep;
        const isPending = idx > currentStep;

        return (
          <div
            key={step.label}
            className={`ps-step ${isComplete ? "ps-step--done" : ""} ${isCurrent ? "ps-step--active" : ""
              } ${isPending ? "ps-step--pending" : ""}`}
          >
            <div className="ps-step__indicator">
              {isComplete ? (
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path
                    d="M3 7L6 10L11 4"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              ) : (
                <span className="ps-step__number">{idx + 1}</span>
              )}
            </div>
            {idx < steps.length - 1 && <div className="ps-step__line" />}
            <div className="ps-step__content">
              <div className="ps-step__label">{step.label}</div>
              <div className="ps-step__desc">{step.desc}</div>
            </div>
            {isCurrent && (
              <div className="ps-step__spinner">
                <div className="spinner" style={{ width: 14, height: 14 }} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
