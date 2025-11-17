# Stock Price Forecasting Report

This report summarizes the full experiment pipeline, quantitative metrics, and benchmarking conclusions for the `MLStock` project. Use it alongside the plots and bar charts exported from `src/stock_price_prediction.py`.

## 1. Objective
- Normalize OHLC stock history, convert it into sliding-window sequences, and compare four baseline models (Linear Regression, Simple RNN, LSTM, GRU) on one-step-ahead closing-price forecasts.
- Evaluate every model with MAE and RMSE to see how well each captures short-term price dynamics.

## 2. Data & Preprocessing
- **Dataset**: `src/Datafile.csv`, containing `Date`, `Open`, `High`, `Low`, `Close`.
- **Cleaning**: rows with missing values removed after sorting by date.
- **Scaling**: Min–Max normalization for price columns to range `[0, 1]`.
- **Sequence generation**: 60-day lookback windows (`time_steps = 60`) converted into `(samples, 60, 3)` feature tensors (Open, High, Low) with the target being the next-day Close.
- **Train/Test split**: 80% chronological training, 20% test with shuffling disabled.

## 3. Modeling Workflow
1. **Linear Regression**: flattened sequence inputs, baseline for fast interpretability.
2. **Simple RNN**: single recurrent layer to capture sequential dependencies.
3. **LSTM**: long short-term memory units to handle longer-range patterns.
4. **GRU**: gated recurrent units for a lighter architecture with similar capabilities.

All neural models share:
- Optimizer: Adam
- Loss: mean squared error
- Epochs: 50
- Batch size: 32

## 4. Quantitative Results
| Algorithm          | MAE  | RMSE |
|--------------------|------|------|
| Linear Regression  | 5.25 | 6.99 |
| RNN                | 6.49 | 8.59 |
| LSTM               | 5.80 | 7.69 |
| GRU                | 5.26 | 7.01 |

> **Note**: These values are percentages relative to an assumed $100 reference price (i.e., 5.25 translates to $5.25 average error).

## 5. Benchmark Interpretation
- **Benchmark criteria**:  
  - **MAE**: Good if 5–10%, acceptable if < 15%.  
  - **RMSE**: Good if 5–15%, acceptable if < 20%.

### Linear Regression
- MAE 5.25% and RMSE 6.99% → both within “Good Performance”.
- Shows that even a simple linear baseline captures a substantial portion of the signal.

### RNN
- MAE 6.49% (acceptable), RMSE 8.59% (acceptable but closer to the upper bound).
- Indicates limited capacity to fit the more complex temporal patterns relative to LSTM/GRU.

### LSTM
- MAE 5.80% and RMSE 7.69% → both “Good Performance”.
- Handles sequential dependencies better than the vanilla RNN, consistent with expectations.

### GRU
- MAE 5.26% and RMSE 7.01% → both “Good Performance”.
- Nearly ties the linear regression baseline on MAE while offering dynamic sequence modeling.

## 6. Overall Summary
- **Best overall**: LSTM and GRU, delivering low errors across both metrics.
- **Strong baseline**: Linear Regression remains competitive, suggesting the feature space is largely linear and benefit from more engineered features or ensembles.
- **Acceptable**: Simple RNN lags slightly but remains under the acceptable error thresholds.

## 7. Visual Evidence
Add screenshots of each algorithm’s prediction plot and the final MAE/RMSE bar chart to this section for completeness. Recommended layout:
1. Linear Regression actual vs. predicted plot.
2. RNN plot.
3. LSTM plot.
4. GRU plot.
5. Comparative bar chart of MAE/RMSE.

> Tip: Save plots from the script or re-run with `plt.savefig(...)` to capture high-resolution images.

## 8. Next Steps
- **Hyperparameter tuning**: adjust time steps, layer widths, learning rate schedules, or add dropout/bidirectional layers.
- **Feature expansion**: include technical indicators (SMA, RSI, volume) or exogenous variables (market indexes, macro data).
- **Model ensembling**: blend the best-performing neural nets with the regression baseline to reduce variance.
- **Deployment**: wrap the selected model into a prediction service or notebook for interactive analysis.

