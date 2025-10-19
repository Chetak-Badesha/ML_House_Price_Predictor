"""
Ontario House Price Predictor
==============================

A machine learning application that predicts house prices in Ontario based on 
key property features using a Random Forest Regressor trained on historical data.

Features:
    - Price predictions based on bedrooms, bathrooms, square footage, and year built
    - Price range estimates with uncertainty quantification
    - Comprehensive affordability metrics (mortgage, taxes, required income)
    - Interactive command-line interface

Model:
    Random Forest Regressor (100 estimators, max_depth=20)
    Features: Bedrooms, Bathrooms, Square Feet, Year Built
    
Financial Assumptions:
    - 20% down payment
    - 5.5% annual interest rate, 25-year amortization
    - 1% annual property tax
    - 32% Gross Debt Service (GDS) ratio

Author: [Your Name]
Date: 2024
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os


def load_data():
    """
    Load housing data from CSV file with user prompt.
    
    Returns:
        pd.DataFrame: Housing data
        
    Raises:
        FileNotFoundError: If CSV file does not exist
    """
    file_name = input("Enter the CSV file name (default: ontario_houses_sample.csv): ").strip()
    if not file_name:
        file_name = "ontario_houses_sample.csv"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"✓ Loading data from: {file_path}")
    return pd.read_csv(file_path)


def normalize_and_map(df):
    """
    Normalize column names to standardized format.
    
    Args:
        df (pd.DataFrame): Raw dataframe with original column names
        
    Returns:
        pd.DataFrame: Dataframe with standardized column names
    """
    mapping = {
        'price': 'Price',
        'bedrooms': 'Beds',
        'bathrooms': 'Bath',
        'sqft': 'Sq.Ft',
        'neighborhood': 'Place',
        'property_type': 'Property_Type',
        'parking': 'Parking',
        'year_built': 'Year_Built',
        'walk_score': 'Walk_Score'
    }
    
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    print(f"✓ Mapped columns: {[c for c in df.columns if c in mapping.values()]}")
    
    return df


def train_model(df):
    """
    Train Random Forest model on cleaned housing data.
    
    Args:
        df (pd.DataFrame): Housing data with standardized column names
        
    Returns:
        tuple: (trained_model, cleaned_dataframe)
            
    Raises:
        ValueError: If insufficient data remains after cleaning (< 30 rows)
    """
    for col in ["Beds", "Bath", "Sq.Ft", "Price", "Year_Built"]:
        if col not in df.columns:
            df[col] = np.nan

    # Convert to numeric
    df["Beds"] = pd.to_numeric(df["Beds"], errors="coerce")
    df["Bath"] = pd.to_numeric(df["Bath"], errors="coerce")
    df["Sq.Ft"] = pd.to_numeric(df["Sq.Ft"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Year_Built"] = pd.to_numeric(df["Year_Built"], errors="coerce")

    df_clean = df[["Beds", "Bath", "Sq.Ft", "Year_Built", "Price"]].dropna()
    
    # Apply data quality filters
    df_clean = df_clean[
        (df_clean["Beds"] > 0) &
        (df_clean["Bath"] >= 0) &
        (df_clean["Sq.Ft"] > 0) &
        (df_clean["Year_Built"] >= 1800) & (df_clean["Year_Built"] <= 2025)
    ]

    if len(df_clean) < 30:
        raise ValueError(f"Not enough clean rows to train (need >=30). Found: {len(df_clean)}")

    X = df_clean[["Beds", "Bath", "Sq.Ft", "Year_Built"]]
    y = df_clean["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n✓ Model trained.")
    print(f"  Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    print(f"  R²: {r2_score(y_test, y_pred):.3f}  |  MAE: ${mean_absolute_error(y_test, y_pred):,.0f}")

    joblib.dump(model, "house_price_model.pkl")
    print("✓ Model saved -> house_price_model.pkl")
    
    return model, df_clean


def predict_and_afford(model, beds, bath, sqft, year_built):
    """
    Generate price prediction and affordability analysis.
    
    Args:
        model: Trained RandomForestRegressor
        beds (float): Number of bedrooms
        bath (float): Number of bathrooms
        sqft (float): Square footage
        year_built (float): Year property was built
        
    Returns:
        dict: Price prediction and financial breakdown
    """
    inp = np.array([[beds, bath, sqft, year_built]])
    predicted_price = float(model.predict(inp)[0])

    # Calculate uncertainty from ensemble
    tree_preds = np.array([t.predict(inp)[0] for t in model.estimators_])
    std = float(np.std(tree_preds))
    low = max(0, predicted_price - std)
    high = predicted_price + std

    # Financial calculations
    down_payment = predicted_price * 0.20
    mortgage_amount = predicted_price - down_payment
    
    annual_interest = 0.055
    monthly_rate = annual_interest / 12
    n_payments = 25 * 12

    if monthly_rate > 0:
        monthly_payment = mortgage_amount * (monthly_rate * (1 + monthly_rate) ** n_payments) / (
            (1 + monthly_rate) ** n_payments - 1
        )
    else:
        monthly_payment = mortgage_amount / n_payments

    property_tax_monthly = predicted_price * 0.01 / 12
    utilities = 200
    total_monthly = monthly_payment + property_tax_monthly + utilities
    required_income = (total_monthly * 12) / 0.32

    return {
        "predicted_price": predicted_price,
        "range_low": low,
        "range_high": high,
        "down_payment": down_payment,
        "mortgage_amount": mortgage_amount,
        "monthly_payment": monthly_payment,
        "property_tax_monthly": property_tax_monthly,
        "utilities": utilities,
        "total_monthly": total_monthly,
        "required_income": required_income,
    }


def get_numeric_input(prompt, min_val=None, max_val=None):
    """
    Prompt for validated numeric input within specified range.
    
    Args:
        prompt (str): Message to display
        min_val (float, optional): Minimum value
        max_val (float, optional): Maximum value
        
    Returns:
        float: Validated numeric input
    """
    while True:
        val = input(prompt).strip()
        try:
            f = float(val)
            if (min_val is not None and f < min_val) or (max_val is not None and f > max_val):
                print(f"Please enter a value between {min_val} and {max_val}")
                continue
            return f
        except ValueError:
            print("Invalid number. Try again.")


def main():
    """Main application entry point."""
    print("\n" + "=" * 60)
    print("ONTARIO HOUSE PRICE PREDICTOR")
    print("=" * 60)

    try:
        df_raw = load_data()
    except FileNotFoundError as e:
        print("ERROR:", e)
        return

    print("Raw columns found:", list(df_raw.columns))
    df = normalize_and_map(df_raw)
    print("Mapped columns:", [c for c in df.columns if c in ["Price","Beds","Bath","Sq.Ft","Place","Year_Built"]])

    try:
        model, df_clean = train_model(df)
    except Exception as e:
        print("Training failed:", e)
        return

    avg_price = df_clean["Price"].mean()
    print(f"\nDataset: {len(df_clean)} rows  |  Average price: ${avg_price:,.0f}")

    # Interactive prediction loop
    while True:
        ans = input("\nWould you like a price estimate? (y/n): ").strip().lower()
        if ans not in ["y", "yes"]:
            print("\nExiting. Goodbye.")
            break

        beds = get_numeric_input("Bedrooms (e.g. 3): ", min_val=0.5, max_val=10)
        bath = get_numeric_input("Bathrooms (e.g. 2): ", min_val=0.0, max_val=8)
        sqft = get_numeric_input("Square feet (e.g. 1200): ", min_val=50, max_val=20000)
        year_built = get_numeric_input("Year built (e.g. 2005): ", min_val=1900, max_val=2025)

        info = predict_and_afford(model, beds, bath, sqft, year_built)

        print("\n" + "-" * 40)
        print("PREDICTION RESULTS")
        print("-" * 40)
        print(f"  Bedrooms: {beds:.0f}  |  Bathrooms: {bath:g}  |  Sq.Ft: {sqft:,.0f}  |  Year: {year_built:.0f}")
        print(f"\n  Estimated Price: ${info['predicted_price']:,.0f}")
        print(f"  Likely Range: ${info['range_low']:,.0f} - ${info['range_high']:,.0f}")

        print("\n  AFFORDABILITY")
        print(f"    Down payment (20%): ${info['down_payment']:,.0f}")
        print(f"    Mortgage amount: ${info['mortgage_amount']:,.0f}")
        print(f"    Monthly mortgage: ${info['monthly_payment']:,.0f}")
        print(f"    Property tax (monthly): ${info['property_tax_monthly']:,.0f}")
        print(f"    Utilities: ${info['utilities']:,.0f}")
        print(f"    Total monthly: ${info['total_monthly']:,.0f}")
        print(f"\n    Required household income: ${info['required_income']:,.0f}/year")

        if info["predicted_price"] < avg_price * 0.7:
            print("\n  ✓ Below average market price - Good value.")
        elif info["predicted_price"] < avg_price * 1.3:
            print("\n  ✓ Near average market price.")
        else:
            print("\n  ⚠ Above average market price - Premium property.")
        print("-" * 40)


if __name__ == "__main__":
    main()