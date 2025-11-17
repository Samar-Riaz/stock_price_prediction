# MLStock – Multimodel Stock Price Prediction

This repository trains and compares several stock price forecasters – Linear Regression, Simple RNN, LSTM, and GRU – on a univariate closing-price target derived from historic OHLC data. The workflow normalizes raw prices, builds sliding-window sequences, trains each model, visualizes predictions, and exports error metrics.

## Project Structure
- `src/stock_price_prediction.py` – end-to-end training script.
- `src/Datafile.csv` – sample dataset with `Date`, `Open`, `High`, `Low`, `Close`.
- `src/model_results.xlsx` – populated after running the script; contains MAE/RMSE per model.

## Data Requirements
1. Place a CSV named `Datafile.csv` under `src/`.
2. Required columns (case sensitive): `Date`, `Open`, `High`, `Low`, `Close`.
3. Missing values are dropped; ensure the file has enough rows to build 60-step sequences.

## Getting Started
```bash
cd D:\MLStock
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Experiment
```bash
python src/stock_price_prediction.py
```
The script will:
- Split the normalized sequences into train/test partitions.
- Train Linear Regression, RNN, LSTM, and GRU baselines (50 epochs, batch size 32 for neural nets).
- Display matplotlib comparisons of actual vs. predicted prices for each model.
- Save a metrics table to `src/model_results.xlsx`.

## Customization Tips
- Adjust `time_steps` to change the sequence length window.
- Modify network layers (units, additional layers, dropout) for experimentation.
- Replace `Datafile.csv` with your own dataset, keeping the required columns intact.

## Troubleshooting
- **TensorFlow GPU/CPU warnings**: they are common and usually not fatal; ensure that the installed TensorFlow build matches your hardware.
- **Matplotlib figures not showing**: if running headless, you may need to switch to a non-interactive backend (e.g., add `matplotlib.use("Agg")` before plotting).
- **Excel export errors**: confirm that `openpyxl` is installed and the Excel file is not open in another program.

