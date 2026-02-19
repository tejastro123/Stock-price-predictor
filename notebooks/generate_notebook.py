"""Generate the Stock_Price_Prediction.ipynb notebook programmatically."""
import json, os

def md(source): return {"cell_type":"markdown","metadata":{},"source": source if isinstance(source,list) else [source]}
def code(source): return {"cell_type":"code","metadata":{},"source": source if isinstance(source,list) else [source],"execution_count":None,"outputs":[]}

cells = []

# ===== Section 1: Setup =====
cells.append(md("# 📈 Stock Price Prediction System\n\n**End-to-End ML Pipeline** — From data collection to production-ready predictions.\n\n**Models Used:** Linear Regression · Decision Tree · Random Forest · Logistic Regression (Direction)\n\n**Author:** Final Year Student — ML Engineering & Quant Development\n\n---"))

cells.append(md("## 1. Setup & Imports\nInstall necessary libraries and configure the environment."))
cells.append(code("""# Install dependencies (uncomment if needed)
# !pip install yfinance pandas numpy scikit-learn matplotlib seaborn shap ta statsmodels joblib

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import joblib
import os
from datetime import datetime, timedelta

# Sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             accuracy_score, classification_report)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# Stationarity
from statsmodels.tsa.stattools import adfuller

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Plot style
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12
sns.set_palette('viridis')

# Directories
os.makedirs('../data', exist_ok=True)
os.makedirs('../models', exist_ok=True)

print("✅ All imports successful!")
print(f"   NumPy: {np.__version__}")
print(f"   Pandas: {pd.__version__}")
print(f"   Scikit-learn: {__import__('sklearn').__version__}")"""))

# ===== Section 2: Data Collection =====
cells.append(md("---\n## 2. Data Collection\n\n**Objective:** Fetch historical stock data dynamically using `yfinance`.\n\nFeatures:\n- Dynamic ticker selection\n- Local CSV caching\n- Error handling with retry logic"))
cells.append(code("""# ── Configuration ──
TICKER = 'AAPL'       # Change this to any ticker: GOOGL, MSFT, TSLA, etc.
PERIOD = '5y'          # Data period: 1y, 2y, 5y, max
CACHE_FILE = f'../data/{TICKER}_{PERIOD}.csv'

def fetch_stock_data(ticker, period='5y', use_cache=True):
    \"\"\"Fetch historical OHLCV data with caching and error handling.\"\"\"
    cache_path = f'../data/{ticker}_{period}.csv'

    # Try cache first
    if use_cache and os.path.exists(cache_path):
        print(f"📂 Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, index_col='Date', parse_dates=True)
        return df

    # Download from yfinance with retry
    for attempt in range(3):
        try:
            print(f"📥 Downloading {ticker} data (attempt {attempt+1}/3)...")
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, auto_adjust=False)

            if df.empty:
                raise ValueError(f"No data returned for '{ticker}'")

            df.index.name = 'Date'
            keep = ['Open','High','Low','Close','Volume']
            if 'Adj Close' in df.columns:
                keep.append('Adj Close')
            df = df[[c for c in keep if c in df.columns]]

            df.to_csv(cache_path)
            print(f"✅ Downloaded {len(df)} rows. Cached to {cache_path}")
            return df
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            if attempt == 2:
                raise

# Fetch data
df = fetch_stock_data(TICKER, PERIOD)
print(f"\\n📊 Dataset shape: {df.shape}")
print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
df.head(10)"""))

# ===== Section 3: EDA =====
cells.append(md("---\n## 3. Exploratory Data Analysis (EDA)\n\n### 3.1 Data Overview & Missing Values"))
cells.append(code("""# Basic info
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"\\nShape: {df.shape}")
print(f"\\nData Types:\\n{df.dtypes}")
print(f"\\nMissing Values:\\n{df.isnull().sum()}")
print(f"\\nStatistical Summary:\\n")
df.describe().round(2)"""))

cells.append(md("### 3.2 Price Trend & Volume"))
cells.append(code("""fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

# Price
axes[0].plot(df.index, df['Close'], color='#6366f1', linewidth=1.5, label='Close Price')
axes[0].fill_between(df.index, df['Low'], df['High'], alpha=0.1, color='#6366f1')
axes[0].set_title(f'{TICKER} — Historical Price & Volume', fontsize=16, fontweight='bold')
axes[0].set_ylabel('Price ($)')
axes[0].legend()
axes[0].grid(alpha=0.2)

# Volume
axes[1].bar(df.index, df['Volume'], color='#06b6d4', alpha=0.6, width=2)
axes[1].set_ylabel('Volume')
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.show()"""))

cells.append(md("### 3.3 Moving Averages Visualization"))
cells.append(code("""fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df.index, df['Close'], label='Close', color='white', linewidth=1.5)
for window, color in [(7,'#f59e0b'), (21,'#10b981'), (50,'#ef4444')]:
    sma = df['Close'].rolling(window).mean()
    ax.plot(df.index, sma, label=f'SMA {window}', color=color, linewidth=1, alpha=0.8)

ax.set_title(f'{TICKER} — Moving Averages', fontsize=16, fontweight='bold')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()"""))

cells.append(md("### 3.4 Return Distribution"))
cells.append(code("""daily_returns = df['Close'].pct_change().dropna() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(daily_returns, bins=80, color='#8b5cf6', alpha=0.7, edgecolor='none')
axes[0].axvline(daily_returns.mean(), color='#f59e0b', linestyle='--', label=f'Mean: {daily_returns.mean():.3f}%')
axes[0].set_title('Daily Return Distribution', fontweight='bold')
axes[0].set_xlabel('Daily Return (%)')
axes[0].legend()
axes[0].grid(alpha=0.2)

# Box plot
axes[1].boxplot(daily_returns, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#6366f1', alpha=0.5),
                medianprops=dict(color='#f59e0b'))
axes[1].set_title('Return Box Plot', fontweight='bold')
axes[1].set_ylabel('Daily Return (%)')
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.show()

print(f"Mean return:  {daily_returns.mean():.4f}%")
print(f"Std dev:      {daily_returns.std():.4f}%")
print(f"Skewness:     {daily_returns.skew():.4f}")
print(f"Kurtosis:     {daily_returns.kurtosis():.4f}")"""))

# ===== Section 4: Feature Engineering =====
cells.append(md("---\n## 4. Feature Engineering\n\nCreate technical indicators used by professional traders and quant researchers."))
cells.append(code("""def add_technical_indicators(df):
    \"\"\"Add all technical indicators to the DataFrame.\"\"\"
    data = df.copy()

    # Simple Moving Averages
    for w in [7, 21, 50]:
        data[f'SMA_{w}'] = data['Close'].rolling(w).mean()

    # Exponential Moving Averages
    for w in [12, 26]:
        data[f'EMA_{w}'] = data['Close'].ewm(span=w, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # Bollinger Bands
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    data['BB_Upper'] = sma20 + 2 * std20
    data['BB_Middle'] = sma20
    data['BB_Lower'] = sma20 - 2 * std20
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']

    # Daily Returns
    data['Daily_Return'] = data['Close'].pct_change() * 100

    # Lag Features
    for lag in range(1, 6):
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)

    # Volume features
    data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']

    return data

# Apply feature engineering
df_featured = add_technical_indicators(df)
print(f"Columns after feature engineering: {len(df_featured.columns)}")
print(f"\\nNew features: {[c for c in df_featured.columns if c not in df.columns]}")
df_featured.tail()"""))

cells.append(md("### 4.1 Stationarity Check (ADF Test)"))
cells.append(code("""# Augmented Dickey-Fuller Test for stationarity
result = adfuller(df['Close'].dropna())
print("ADF Statistic (Close Price):", round(result[0], 4))
print("p-value:", round(result[1], 4))
print("Stationary?" , "Yes ✅" if result[1] < 0.05 else "No ❌ (expected for stock prices)")

# Test returns (should be stationary)
returns = df['Close'].pct_change().dropna()
result2 = adfuller(returns)
print(f"\\nADF Statistic (Returns): {round(result2[0], 4)}")
print(f"p-value: {round(result2[1], 6)}")
print("Stationary?", "Yes ✅" if result2[1] < 0.05 else "No ❌")"""))

cells.append(md("### 4.2 Feature Correlation Heatmap"))
cells.append(code("""# Select key features for correlation
key_features = ['Close', 'Volume', 'SMA_7', 'SMA_21', 'RSI', 'MACD', 'BB_Width', 'Daily_Return']
corr_data = df_featured[key_features].dropna()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_data.corr(), dtype=bool))
sns.heatmap(corr_data.corr(), mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, ax=ax, linewidths=0.5,
            square=True, cbar_kws={'shrink': 0.8})
ax.set_title(f'{TICKER} — Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

# ===== Section 5: Train/Test Split =====
cells.append(md("---\n## 5. Data Preparation & Train/Test Split\n\nUsing **time-based split** (80/20) to prevent data leakage."))
cells.append(code("""# Prepare features
df_clean = df_featured.dropna()
exclude_cols = ['Close', 'Adj Close']
feature_cols = [c for c in df_clean.columns if c not in exclude_cols]

X = df_clean[feature_cols]
y = df_clean['Close']

# Time-based split (NO shuffling!)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training set:  {X_train.shape[0]} samples ({X_train.index[0].date()} to {X_train.index[-1].date()})")
print(f"Testing set:   {X_test.shape[0]} samples ({X_test.index[0].date()} to {X_test.index[-1].date()})")
print(f"Features:      {X_train.shape[1]}")

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"\\n✅ Features scaled using MinMaxScaler")"""))

# ===== Section 6: Baseline Models =====
cells.append(md("---\n## 6. Baseline Models\n\n### 6.1 Naive Forecast\nPredicts tomorrow's price = today's price."))
cells.append(code("""# Naive forecast: predict previous day's price
naive_preds = y_test.shift(1).dropna()
naive_actual = y_test.iloc[1:]

naive_rmse = np.sqrt(mean_squared_error(naive_actual, naive_preds))
naive_mae = mean_absolute_error(naive_actual, naive_preds)
print(f"Naive Forecast — RMSE: {naive_rmse:.4f} | MAE: {naive_mae:.4f}")"""))

cells.append(md("### 6.2 Linear Regression"))
cells.append(code("""lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mae = mean_absolute_error(y_test, lr_preds)
print(f"Linear Regression — RMSE: {lr_rmse:.4f} | MAE: {lr_mae:.4f}")"""))

# ===== Section 7: ML Models =====
cells.append(md("---\n## 7. Machine Learning Models\n\n### 7.1 Decision Tree Regressor\n\n**Why?** Captures non-linear patterns without requiring feature scaling assumptions. Simple, interpretable model."))
cells.append(code("""dt_model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_SEED
)
dt_model.fit(X_train_scaled, y_train)
dt_preds = dt_model.predict(X_test_scaled)

dt_rmse = np.sqrt(mean_squared_error(y_test, dt_preds))
dt_mae = mean_absolute_error(y_test, dt_preds)
print(f"Decision Tree — RMSE: {dt_rmse:.4f} | MAE: {dt_mae:.4f}")"""))

cells.append(md("### 7.2 Random Forest Regressor\n\n**Why?** Ensemble of decision trees — reduces overfitting via bagging. Usually the best performer for tabular data."))
cells.append(code("""rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)
print(f"Random Forest — RMSE: {rf_rmse:.4f} | MAE: {rf_mae:.4f}")"""))

cells.append(md("### 7.3 Logistic Regression (Direction Prediction)\n\n**Why?** Classification model to predict if price will go **up or down** the next day. Gives directional accuracy."))
cells.append(code("""# Create binary direction labels: 1 = up, 0 = down
y_train_dir = (y_train.diff().dropna() > 0).astype(int)
y_test_dir = (y_test.diff().dropna() > 0).astype(int)
X_train_dir = X_train_scaled[1:]
X_test_dir = X_test_scaled[1:]

log_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight='balanced')
log_model.fit(X_train_dir, y_train_dir)
log_preds = log_model.predict(X_test_dir)

dir_accuracy = accuracy_score(y_test_dir, log_preds) * 100
print(f"Logistic Regression (Direction) — Accuracy: {dir_accuracy:.2f}%")
print(f"\\nClassification Report:\\n{classification_report(y_test_dir, log_preds, target_names=['Down','Up'])}")"""))

# ===== Section 8: Hyperparameter Tuning =====
cells.append(md("---\n## 8. Hyperparameter Tuning\n\nUsing **TimeSeriesSplit** cross-validation + **RandomizedSearchCV** for Random Forest."))
cells.append(code("""param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5, 0.8],
}

tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=20,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1,
)

print("🔧 Running hyperparameter tuning (this may take 1-2 minutes)...")
search.fit(X_train_scaled, y_train)

print(f"\\n✅ Best Parameters: {search.best_params_}")
print(f"   Best CV RMSE: {-search.best_score_:.4f}")

# Evaluate tuned model
rf_tuned = search.best_estimator_
rf_tuned_preds = rf_tuned.predict(X_test_scaled)
rf_tuned_rmse = np.sqrt(mean_squared_error(y_test, rf_tuned_preds))
rf_tuned_mae = mean_absolute_error(y_test, rf_tuned_preds)
print(f"\\nTuned Random Forest — RMSE: {rf_tuned_rmse:.4f} | MAE: {rf_tuned_mae:.4f}")"""))

# ===== Section 9: Model Evaluation =====
cells.append(md("---\n## 9. Model Evaluation\n\n### 9.1 Performance Comparison"))
cells.append(code("""def compute_all_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_dir == pred_dir) * 100
    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'Dir Acc (%)': dir_acc}

all_models = {
    'Linear Regression': lr_preds,
    'Decision Tree': dt_preds,
    'Random Forest': rf_preds,
    'RF Tuned': rf_tuned_preds,
}

comparison = pd.DataFrame([
    compute_all_metrics(y_test.values, preds, name)
    for name, preds in all_models.items()
])
comparison = comparison.sort_values('RMSE')
print("\\n📊 Model Comparison (sorted by RMSE):\\n")
comparison.round(4)"""))

cells.append(md("### 9.2 Actual vs Predicted Plots"))
cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(16, 10))
colors = ['#6366f1', '#f59e0b', '#10b981', '#ef4444']

for idx, (name, preds) in enumerate(all_models.items()):
    ax = axes[idx // 2][idx % 2]
    ax.plot(y_test.index, y_test.values, label='Actual', color='white', linewidth=1.5)
    ax.plot(y_test.index, preds, label=f'Predicted', color=colors[idx], linewidth=1.5, alpha=0.8)
    ax.set_title(name, fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

plt.suptitle(f'{TICKER} — Actual vs Predicted', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("### 9.3 Residual Analysis"))
cells.append(code("""# Use best model (tuned RF)
best_preds = rf_tuned_preds
residuals = y_test.values - best_preds

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].scatter(best_preds, residuals, alpha=0.4, s=8, color='#8b5cf6')
axes[0].axhline(y=0, color='#f59e0b', linestyle='--')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Residual')
axes[0].set_title('Residuals vs Predicted', fontweight='bold')
axes[0].grid(alpha=0.2)

axes[1].hist(residuals, bins=50, color='#6366f1', alpha=0.7, edgecolor='none')
axes[1].set_title('Residual Distribution', fontweight='bold')
axes[1].grid(alpha=0.2)

axes[2].plot(residuals, color='#06b6d4', linewidth=0.5)
axes[2].axhline(y=0, color='#f59e0b', linestyle='--')
axes[2].set_title('Residuals Over Time', fontweight='bold')
axes[2].grid(alpha=0.2)

plt.tight_layout()
plt.show()

print(f"Mean residual:   {residuals.mean():.4f}")
print(f"Std of residuals: {residuals.std():.4f}")"""))

# ===== Section 10: Model Interpretation =====
cells.append(md("---\n## 10. Model Interpretation\n\n### 10.1 Feature Importance (Random Forest)"))
cells.append(code("""importances = rf_tuned.feature_importances_
feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='#6366f1', alpha=0.8)
ax.set_xlabel('Importance')
ax.set_title(f'{TICKER} — Top 15 Feature Importances (Random Forest)', fontweight='bold')
ax.grid(alpha=0.2, axis='x')
plt.tight_layout()
plt.show()"""))

cells.append(md("### 10.2 SHAP Analysis"))
cells.append(code("""try:
    import shap

    explainer = shap.TreeExplainer(rf_tuned)
    # Use a sample for speed
    sample_size = min(200, len(X_test_scaled))
    X_sample = X_test_scaled[:sample_size]

    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                      plot_type='bar', show=False, max_display=15)
    plt.title(f'{TICKER} — SHAP Feature Importance', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Beeswarm plot
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                      show=False, max_display=15)
    plt.title(f'{TICKER} — SHAP Beeswarm Plot', fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\\n🔍 Key insights from SHAP:")
    print("   • Lag features and moving averages are typically most influential")
    print("   • RSI and MACD provide non-linear signal the model leverages")
except ImportError:
    print("⚠️ SHAP not installed. Run: pip install shap")"""))

# ===== Section 11: Production Pipeline =====
cells.append(md("---\n## 11. Production Pipeline\n\nSave the best model and create a reusable prediction function."))
cells.append(code("""# Save model artifacts
joblib.dump(rf_tuned, f'../models/{TICKER}_model.pkl')
joblib.dump(scaler, f'../models/{TICKER}_scaler.pkl')
joblib.dump({
    'ticker': TICKER,
    'best_model': 'RandomForest_Tuned',
    'feature_names': feature_cols,
    'metrics': compute_all_metrics(y_test.values, rf_tuned_preds, 'RF_Tuned'),
}, f'../models/{TICKER}_meta.pkl')

print(f"✅ Model saved to ../models/{TICKER}_model.pkl")
print(f"✅ Scaler saved to ../models/{TICKER}_scaler.pkl")"""))

cells.append(code("""def predict_stock(ticker, days_ahead=5):
    \"\"\"
    Predict future stock prices for the next N trading days.

    Parameters
    ----------
    ticker : str — Stock ticker symbol
    days_ahead : int — Number of trading days to predict

    Returns
    -------
    dict with predictions, history, and model info
    \"\"\"
    # Load artifacts
    model = joblib.load(f'../models/{ticker}_model.pkl')
    scaler = joblib.load(f'../models/{ticker}_scaler.pkl')
    meta = joblib.load(f'../models/{ticker}_meta.pkl')

    # Fetch latest data
    stock = yf.Ticker(ticker)
    df = stock.history(period='2y', auto_adjust=False)
    df.index.name = 'Date'
    keep = ['Open','High','Low','Close','Volume']
    if 'Adj Close' in df.columns: keep.append('Adj Close')
    df = df[[c for c in keep if c in df.columns]]

    working_df = df.copy()
    last_date = working_df.index[-1]
    predictions = []

    for day in range(1, days_ahead + 1):
        enriched = add_technical_indicators(working_df).dropna()
        feat_cols = meta['feature_names']
        last_row = enriched[feat_cols].iloc[[-1]]
        last_scaled = scaler.transform(last_row)
        pred_price = float(model.predict(last_scaled)[0])

        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        predictions.append({'date': next_date.strftime('%Y-%m-%d'), 'close': round(pred_price, 2)})

        new_row = pd.DataFrame({
            'Open': [pred_price], 'High': [pred_price*1.005],
            'Low': [pred_price*0.995], 'Close': [pred_price],
            'Volume': [working_df['Volume'].tail(5).mean()],
        }, index=[next_date])
        if 'Adj Close' in working_df.columns: new_row['Adj Close'] = pred_price
        working_df = pd.concat([working_df, new_row])
        last_date = next_date

    return {
        'ticker': ticker, 'model': meta['best_model'],
        'predictions': predictions, 'metrics': meta['metrics']
    }

# Test the prediction function
result = predict_stock(TICKER, days_ahead=7)
print(f"\\n📈 {TICKER} — {result['model']} Predictions:")
print("-" * 35)
for p in result['predictions']:
    print(f"   {p['date']}: ${p['close']:.2f}")"""))

# ===== Section 12: Forecast Visualization =====
cells.append(md("### 11.1 Forecast Visualization"))
cells.append(code("""fig, ax = plt.subplots(figsize=(14, 6))

# Plot last 60 days of actual data
recent = df.tail(60)
ax.plot(recent.index, recent['Close'], color='#6366f1', linewidth=2, label='Historical')

# Plot forecast
pred_dates = pd.to_datetime([p['date'] for p in result['predictions']])
pred_values = [p['close'] for p in result['predictions']]

# Connect the lines
bridge_dates = [recent.index[-1]] + list(pred_dates)
bridge_values = [recent['Close'].iloc[-1]] + pred_values

ax.plot(bridge_dates, bridge_values, color='#06b6d4', linewidth=2,
        linestyle='--', marker='o', markersize=6, label='Forecast')
ax.axvline(x=recent.index[-1], color='rgba(255,255,255,0.3)', linestyle=':', label='Forecast Start')

ax.set_title(f'{TICKER} — Price Forecast ({len(pred_values)} days)', fontsize=16, fontweight='bold')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()"""))

# ===== Section 13: Deployment Discussion =====
cells.append(md("""---
## 12. Deployment Design & Discussion

### Architecture
```
React (Vite) → Express.js → FastAPI → sklearn Model
     ↕              ↕
  Browser        MongoDB
```

### API Endpoint
```
POST /predict
{
  "ticker": "AAPL",
  "days": 5
}
```

### Key Considerations

| Aspect | Approach |
|--------|----------|
| **Inference** | Real-time via FastAPI (< 2s per request) |
| **Model Storage** | joblib serialization to disk |
| **Data Freshness** | 6-hour cache with yfinance |
| **Monitoring** | Log predictions + compare to actuals weekly |
| **Retraining** | Monthly on latest data with full pipeline |
| **Drift Detection** | Track RMSE on recent predictions vs actuals |

### Limitations
- Stock markets are inherently unpredictable — no model guarantees accuracy
- External events (earnings, geopolitics) are not captured
- Short forecast horizons (1-7 days) are more reliable than long-term
- Past performance doesn't guarantee future results

### Potential Enhancements
- News sentiment integration (NLP)
- Multi-company training with shared features
- Market index features (S&P 500, VIX)
- Streamlit dashboard for quick visualization
- Docker deployment for reproducibility

---

**⚠️ Disclaimer:** This project is for educational purposes only. Not financial advice."""))

cells.append(md("---\n## ✅ Project Complete!\n\nThis notebook demonstrates a full ML pipeline for stock price prediction, including:\n1. ✅ Data collection with caching\n2. ✅ Comprehensive EDA\n3. ✅ Feature engineering (7 technical indicators + lags)\n4. ✅ Model comparison (4 models)\n5. ✅ Hyperparameter tuning (TimeSeriesSplit + RandomizedSearch)\n6. ✅ Evaluation metrics (RMSE, MAE, MAPE, Directional Accuracy)\n7. ✅ SHAP interpretation\n8. ✅ Production-ready `predict_stock()` function\n9. ✅ Deployment design discussion"))

# Build notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out_path = os.path.join(os.path.dirname(__file__), "Stock_Price_Prediction.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook created: {out_path}")
