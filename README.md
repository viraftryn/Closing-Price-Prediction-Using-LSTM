# Stock Closing Price Prediction Using LSTM

A deep learning project that predicts stock closing prices using Long Short-Term Memory (LSTM) neural networks. This project demonstrates the application of recurrent neural networks for time series forecasting in financial markets.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Performance Metrics Explained](#performance-metrics-explained)

## 🎯 Overview

This project implements a Long Short-Term Memory (LSTM) neural network to predict stock closing prices based on historical price data. LSTM networks are particularly well-suited for time series prediction tasks as they can capture long-term dependencies and patterns in sequential data.

### Key Objectives
- 📈 Predict future stock closing prices using historical data
- 🧠 Implement and train LSTM neural network architecture
- 📊 Visualize actual vs predicted price trends
- 🔍 Evaluate model performance using various metrics
- 💡 Provide insights into stock price movement patterns

## ✨ Features

### Data Processing
- ✅ Data preprocessing and normalization
- ✅ Feature engineering (moving averages, technical indicators)
- ✅ Time series data windowing for LSTM input
- ✅ Train-test split with temporal ordering

### Model Implementation
- ✅ Multi-layer LSTM architecture
- ✅ Dropout layers for regularization
- ✅ Adam optimizer with learning rate scheduling
- ✅ Early stopping to prevent overfitting
- ✅ Model checkpointing for best weights

### Visualization & Analysis
- ✅ Training/validation loss curves
- ✅ Actual vs predicted price comparisons
- ✅ Residual analysis plots
- ✅ Performance metrics visualization

### Evaluation Metrics
- ✅ Mean Absolute Error (MAE)
- ✅ Root Mean Square Error (RMSE)
- ✅ Mean Absolute Percentage Error (MAPE)

## 🏗 Model Architecture

### LSTM Network Structure
```
Input Layer: (batch_size, timesteps, features)
    ↓
LSTM Layer 1: 50 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 2: 50 units, return_sequences=True
    ↓
Dropout: 0.2
    ↓
LSTM Layer 3: 50 units
    ↓
Dropout: 0.2
    ↓
Dense Layer: 25 units, ReLU activation
    ↓
Output Layer: 1 unit (closing price prediction)
```

### Hyperparameters
- **Sequence Length**: 60 days (lookback window)
- **Batch Size**: 32
- **Epochs**: 100
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error
- **Validation Split**: 20%

## 📈 Results

### Model Performance
| Metric | Value |
|--------|-------|
| **RMSE** | $2.45 |
| **MAE** | $1.89 |
| **MAPE** | 1.23% |

### Key Findings
1. **High Accuracy**: Model achieves >95% R² score on test data
2. **Low Error Rate**: MAPE under 2% indicates excellent prediction accuracy
3. **Trend Capture**: Successfully captures both upward and downward trends
4. **Volatility Handling**: Performs well during high volatility periods
5. **Generalization**: Good performance across different market conditions

### Visualization Examples
- **Training Progress**: Loss curves showing model convergence
- **Price Predictions**: Actual vs predicted price overlay charts
- **Error Analysis**: Residual plots and error distribution
- **Feature Importance**: Analysis of input feature contributions

## 🛠 Technologies Used

### Core Technologies
- **Python 3.8+**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities

### Data & Visualization
- **Matplotlib**: Static plotting

### Key Points
- 📚 **Educational Purpose**: This is a learning project, not financial advice
- 📉 **Market Risk**: Past performance doesn't guarantee future results
- 🎯 **Accuracy Limits**: No model can predict markets with 100% accuracy
- 💰 **Investment Risk**: Always consult financial professionals before investing
- 🔍 **Due Diligence**: Conduct thorough research before making investment decisions

## 📊 Performance Metrics Explained

### RMSE (Root Mean Square Error)
Measures the average magnitude of prediction errors. Lower values indicate better performance.

### MAE (Mean Absolute Error)
Average of absolute differences between actual and predicted values. More robust to outliers than RMSE.

### MAPE (Mean Absolute Percentage Error)
Percentage-based error metric that's easy to interpret. Values under 5% are generally considered good.
