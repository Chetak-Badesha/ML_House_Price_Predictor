from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load trained model
model = None
try:
    model = joblib.load('house_price_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f" No saved model found: {e}")
    print("Will use fallback calculations")

@app.route('/')
def home():
    """Serve the HTML file directly from main folder"""
    html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indexv2.html')
    if os.path.exists(html_file):
        return send_file(html_file)
    else:
        return "<h1>index.html not found</h1>", 404

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Read values from JSON with defaults
        beds = float(data.get('bedrooms', 3))
        baths = float(data.get('bathrooms', 2))
        sqft = float(data.get('sqft', 1500))
        year_built = float(data.get('yearBuilt', 2010))

        # Clamp values to allowed ranges
        beds = max(1, min(beds, 10))
        baths = max(1, min(baths, 8))
        sqft = max(50, min(sqft, 20000))
        year_built = max(1900, min(year_built, 2025))

        using_fallback = False

        if model is None:
            # Fallback calculation if no model
            base_price = 200000 + (beds * 50000) + (baths * 25000) + (sqft * 250)
            age_factor = max(0.8, min(1.2, 1.2 - (2024 - year_built) * 0.005))
            predicted_price = base_price * age_factor
            std = predicted_price * 0.1
            using_fallback = True
        else:
            # Use trained model
            features = np.array([[beds, baths, sqft, year_built]])
            predicted_price = float(model.predict(features)[0])
            if hasattr(model, 'estimators_'):
                tree_preds = [tree.predict(features)[0] for tree in model.estimators_]
                std = float(np.std(tree_preds))
            else:
                std = predicted_price * 0.1

        # Price range
        price_low = max(0, predicted_price - std)
        price_high = predicted_price + std

        # Affordability calculations
        down_payment = predicted_price * 0.20
        mortgage_amount = predicted_price - down_payment
        monthly_rate = 0.055 / 12
        num_payments = 25 * 12
        monthly_payment = mortgage_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                          ((1 + monthly_rate)**num_payments - 1)
        property_tax_monthly = predicted_price * 0.01 / 12
        utilities = 200
        total_monthly = monthly_payment + property_tax_monthly + utilities
        required_income = (total_monthly * 12) / 0.32

        result = {
            'success': True,
            'using_fallback': using_fallback,
            'predicted_price': round(predicted_price),
            'price_low': round(price_low),
            'price_high': round(price_high),
            'down_payment': round(down_payment),
            'mortgage_amount': round(mortgage_amount),
            'monthly_payment': round(monthly_payment),
            'property_tax_monthly': round(property_tax_monthly),
            'utilities': utilities,
            'total_monthly': round(total_monthly),
            'required_income': round(required_income)
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\nStarting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
