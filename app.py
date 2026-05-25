# ============================================================
#  Breast Cancer Prediction — Flask App 
# ============================================================
#  FIXES APPLIED:
#  1. SSL TLSV1_ALERT_INTERNAL_ERROR  → tlsCAFile=certifi.where()
#  2. Wrong Malignant/Benign label    → 0=Malignant, 1=Benign
#  3. Broken indentation / duplicate  → clean single predict()
#  4. No startup DB check             → ping at startup
#  5. App crash when DB is down       → graceful fallback
# ============================================================

from flask import Flask, request, render_template
from pymongo import MongoClient
import certifi
import ssl
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

# ── Step 1: Load ML model & scaler ──────────────────────────
try:
    model  = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler .pkl', 'rb'))
    print("✅ Model and scaler loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Model file missing: {e}")
    print("   Make sure model.pkl and 'scaler (3).pkl' are in the same folder as app.py")
    raise

# ── Step 2: MongoDB connection with SSL fix ──────────────────
collection = None
try:
    # Create explicit SSL context forcing TLS 1.2+
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ssl_ctx.check_hostname  = True
    ssl_ctx.verify_mode     = ssl.CERT_REQUIRED

    client = MongoClient(
        "mongodb+srv://anjalisharma123:anjalisharma123"
        "@cluster0.jxobkth.mongodb.net/", 
        tls=True,
        tlsCAFile=certifi.where(),       # FIX: use up-to-date CA certs
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000,
        retryWrites=True
    )
    client.admin.command('ping')         # verify connection right away
    print("✅ MongoDB connected successfully!")
    db         = client["cancerDB"]
    collection = db["predictions"]

except Exception as e:
    print(f"⚠️  MongoDB unavailable — predictions will NOT be saved.")
    print(f"   Reason: {e}")
    collection = None                    # app still runs without DB

# ── Routes ───────────────────────────────────────────────────
# Home Page
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


# About Page
@app.route('/about')
def about():
    return render_template('about.html')


# Prediction Page
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


# Dashboard Page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')


# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step A: Read form values
        raw_values = list(request.form.values())

        if not raw_values or all(v.strip() == '' for v in raw_values):
            return render_template(
                'index.html',
                prediction_text="Error: Please fill in all input fields."
            )

        # Step B: Convert to floats
        features = [float(x) for x in raw_values]

        # Step C: Scale and predict
        final_features = np.array(features).reshape(1, -1)
        scaled_data    = scaler.transform(final_features)
        prediction     = model.predict(scaled_data)[0]

        # Step D: Map label correctly
        # Standard sklearn breast cancer: 0 = Malignant, 1 = Benign
        result = "Malignant" if prediction == 0 else "Benign"

        # Step E: Save to MongoDB (non-fatal if DB unavailable)
        if collection is not None:
            try:
                collection.insert_one({
                    "features":   features,
                    "prediction": result,
                    "timestamp":  datetime.now()
                })
                print(f"✅ Saved prediction: {result}")
            except Exception as db_err:
                print(f"⚠️  Could not save to DB: {db_err}")

        # Step F: Return result to user
        return render_template(
            'index.html',
            prediction_text=f"Prediction: {result}"
        )

    except ValueError:
        return render_template(
            'index.html',
            prediction_text="Error: All fields must be valid numbers. Please check your input."
        )
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)


