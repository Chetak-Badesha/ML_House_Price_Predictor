# ML-Powered Ontario House Price Estimator

**ML Home Valuation Tool for Ontario Real Estate**

## Overview
ML-Powered Ontario House Price Estimator is a lightweight full-stack project that predicts house prices and provides affordability analysis for properties in Ontario. It uses a Random Forest Regressor trained on historical listings and exposes predictions via a simple Flask web interface.

## Features
- Predicts house price from: bedrooms, bathrooms, square footage, year built
- Returns a confidence range derived from ensemble predictions
- Affordability breakdown: down payment, monthly mortgage, taxes, utilities, required income
- CLI training script and a web frontend (Flask) for interactive use
- Server-side input validation and fallback estimator if model file is missing

## Tech Stack
- Python 3.8+
- Flask (web server)
- scikit-learn (RandomForestRegressor)
- pandas, numpy
- joblib (model serialization)
- Vanilla HTML/CSS/JS for frontend

## Quick Start

### 1. Clone repository
```bash
git clone https://github.com/Chetak-Badesha/ML_House_Price_Predictor.git
cd ML_House_Price_Predictor
```

### 2. Create and activate virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare dataset
Place `ontario_houses_sample.csv` in the project root. Columns accepted: `price` (or `Price`), `bedrooms`, `bathrooms`, `sqft`, `year_built` (names are auto-mapped).

### 5. Train the model (first time)
```bash
python simple_house_predictor.py
```
This trains a Random Forest model and saves `house_price_model.pkl`.

### 6. Run the web app
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## Example Request (API)
POST `/predict` with JSON:
```json
{
  "bedrooms": 3,
  "bathrooms": 2,
  "sqft": 1500,
  "yearBuilt": 2010
}
```

Response includes `predicted_price`, `price_low`, `price_high`, `monthly_payment`, `required_income`, etc.

## Model Notes & Performance
- Model: RandomForestRegressor (n_estimators=100, max_depth=20)
- Example training output: `R²` and `MAE` are printed after training. Use MAE to gauge average dollar error.
- Limitations: model trained on provided dataset only; does not use location/neighborhood premium without additional features.

## Project Structure (recommended)
```
ML_House_Price_Predictor/
├── app.py                      # Flask web server
├── simple_house_predictor.py   # Training & CLI tool
├── index.html                  # Frontend (if using file-based)
├── main.js / style.css         # Frontend assets
├── ontario_houses_sample.csv   # Sample dataset
├── house_price_model.pkl       # Saved model (generated after training)
├── requirements.txt
└── README.md
```

## Live Demo
🚀 Live Demo — Coming Soon (deployment planned)

## Contributing
- Fork the repo, create a feature branch, commit, and open a PR.
- Follow PEP8 and include docstrings for new functions.
- Add tests for new model behavior where applicable.

## Lessons Learned
- Data cleaning is more important than model choice.
- Feature selection drastically affects accuracy.
- Even a simple web UI improves usability and project appeal.

## Author
Chetak Badesha  
GitHub: `@Chetak-Badesha`
