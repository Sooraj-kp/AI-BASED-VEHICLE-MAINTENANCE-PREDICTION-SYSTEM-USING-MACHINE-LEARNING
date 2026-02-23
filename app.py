"""
AI-Based Vehicle Predictive Maintenance System
Flask Web Application
"""

import os, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ──────────────────────────────────────────────────────────────
# Load models at startup
# ──────────────────────────────────────────────────────────────
engine_model    = joblib.load(os.path.join(MODEL_DIR, "engine_model.pkl"))
engine_scaler   = joblib.load(os.path.join(MODEL_DIR, "engine_scaler.pkl"))
engine_features = joblib.load(os.path.join(MODEL_DIR, "engine_features.pkl"))

cost_model      = joblib.load(os.path.join(MODEL_DIR, "cost_model.pkl"))
item_models     = joblib.load(os.path.join(MODEL_DIR, "item_models.pkl"))
service_features= joblib.load(os.path.join(MODEL_DIR, "service_features.pkl"))

le_brand  = joblib.load(os.path.join(MODEL_DIR, "le_brand.pkl"))
le_model_ = joblib.load(os.path.join(MODEL_DIR, "le_model.pkl"))
le_engine = joblib.load(os.path.join(MODEL_DIR, "le_engine.pkl"))
le_region = joblib.load(os.path.join(MODEL_DIR, "le_region.pkl"))

# Load reference data for dropdowns
df_svc = pd.read_csv(os.path.join(DATA_DIR, "service_records.csv"))

MAINT_ITEMS = ['oil_filter', 'engine_oil', 'washer_plug_drain',
               'dust_and_pollen_filter', 'whell_alignment_and_balancing',
               'air_clean_filter', 'fuel_filter', 'spark_plug',
               'brake_fluid', 'brake_and_clutch_oil', 'transmission_fluid',
               'brake_pads', 'clutch', 'coolant']

ITEM_LABELS = {
    'oil_filter': 'Oil Filter',
    'engine_oil': 'Engine Oil',
    'washer_plug_drain': 'Washer Plug Drain',
    'dust_and_pollen_filter': 'Dust & Pollen Filter',
    'whell_alignment_and_balancing': 'Wheel Alignment & Balancing',
    'air_clean_filter': 'Air Clean Filter',
    'fuel_filter': 'Fuel Filter',
    'spark_plug': 'Spark Plug',
    'brake_fluid': 'Brake Fluid',
    'brake_and_clutch_oil': 'Brake & Clutch Oil',
    'transmission_fluid': 'Transmission Fluid',
    'brake_pads': 'Brake Pads',
    'clutch': 'Clutch',
    'coolant': 'Coolant',
}

GRAPH_ENG = [
    {"file": "eng_model_comparison.png",   "title": "Model Comparison",           "desc": "Train vs Test accuracy across all classifiers"},
    {"file": "eng_roc_curves.png",          "title": "ROC Curves",                 "desc": "AUC-ROC curves for all models"},
    {"file": "eng_confusion_matrix.png",    "title": "Confusion Matrix",           "desc": "Prediction correctness breakdown"},
    {"file": "eng_precision_recall.png",    "title": "Precision-Recall Curves",    "desc": "Precision vs Recall trade-off"},
    {"file": "eng_feature_importance.png",  "title": "Feature Importance",         "desc": "Most influential engine parameters"},
    {"file": "eng_learning_curve.png",      "title": "Learning Curve",             "desc": "Model performance vs training data size"},
    {"file": "eng_cv_scores.png",           "title": "CV Score Distribution",      "desc": "Cross-validation score violin plot"},
    {"file": "eng_distributions.png",       "title": "Data Distribution",          "desc": "Feature distributions by engine condition"},
]

GRAPH_SVC = [
    {"file": "svc_actual_vs_predicted.png", "title": "Actual vs Predicted Cost",   "desc": "Cost regression accuracy scatter"},
    {"file": "svc_residuals.png",            "title": "Residual Analysis",          "desc": "Model error distribution"},
    {"file": "svc_feature_importance.png",   "title": "Feature Importance",         "desc": "Key predictors for service cost"},
    {"file": "svc_item_accuracy.png",        "title": "Maintenance Item Accuracy",  "desc": "Per-item classification accuracy"},
    {"file": "svc_item_frequency.png",       "title": "Item Replacement Frequency", "desc": "How often each item is serviced"},
    {"file": "svc_cost_distribution.png",    "title": "Cost Distribution",          "desc": "Service cost stats and brand comparison"},
    {"file": "svc_mileage_vs_cost.png",      "title": "Mileage vs Cost",            "desc": "Cost vs mileage by vehicle year"},
    {"file": "summary_metrics.png",          "title": "Overall Model Metrics",      "desc": "Summary of all model performance KPIs"},
]

# ──────────────────────────────────────────────────────────────
# Stats for dashboard
# ──────────────────────────────────────────────────────────────
def get_stats():
    df_e = pd.read_csv(os.path.join(DATA_DIR, "engine_data.csv"))
    df_e.columns = df_e.columns.str.strip()
    return {
        "engine_samples": len(df_e),
        "service_samples": len(df_svc),
        "fault_pct": round(df_e['Engine Condition'].mean() * 100, 1),
        "avg_cost": int(df_svc['cost'].mean()),
        "max_cost": int(df_svc['cost'].max()),
        "brands": sorted(df_svc['brand'].unique().tolist()),
        "models": sorted(df_svc['model'].unique().tolist()),
        "engine_types": sorted(df_svc['engine_type'].unique().tolist()),
        "regions": sorted(df_svc['region'].unique().tolist()),
        "years": sorted(df_svc['make_year'].unique().tolist(), reverse=True),
        "mileage_ranges": sorted(df_svc['mileage_range'].unique().tolist()),
    }

# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    stats = get_stats()
    return render_template('index.html', stats=stats)

@app.route('/engine')
def engine_page():
    return render_template('engine.html')

@app.route('/service')
def service_page():
    stats = get_stats()
    return render_template('service.html', stats=stats)

@app.route('/analytics')
def analytics():
    return render_template('analytics.html',
                           graphs_eng=GRAPH_ENG,
                           graphs_svc=GRAPH_SVC)

@app.route('/about')
def about():
    return render_template('about.html')

# ──────────────────────────────────────────────────────────────
# API endpoints
# ──────────────────────────────────────────────────────────────
@app.route('/api/predict-engine', methods=['POST'])
def predict_engine():
    try:
        data   = request.get_json()
        values = {f: float(data.get(f, 0)) for f in engine_features}
        df_in  = pd.DataFrame([values])
        scaled = engine_scaler.transform(df_in)

        condition = int(engine_model.predict(scaled)[0])
        if hasattr(engine_model, 'predict_proba'):
            proba = engine_model.predict_proba(scaled)[0]
            fault_prob = float(proba[1])
        else:
            fault_prob = float(condition)

        # Build alert level
        if fault_prob >= 0.75:
            alert_level = "CRITICAL"
            alert_color = "danger"
            message     = "Immediate maintenance required. Multiple engine parameters are critically out of range."
        elif fault_prob >= 0.50:
            alert_level = "WARNING"
            alert_color = "warning"
            message     = "Engine shows signs of degradation. Schedule maintenance soon."
        elif fault_prob >= 0.30:
            alert_level = "CAUTION"
            alert_color = "caution"
            message     = "Minor anomalies detected. Monitor engine closely."
        else:
            alert_level = "NORMAL"
            alert_color = "success"
            message     = "Engine operating within normal parameters. No immediate action required."

        return jsonify({
            "success": True,
            "condition": condition,
            "fault_probability": round(fault_prob * 100, 2),
            "health_score": round((1 - fault_prob) * 100, 2),
            "alert_level": alert_level,
            "alert_color": alert_color,
            "message": message,
            "label": "Engine Fault Detected" if condition == 1 else "Engine Normal",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/predict-service', methods=['POST'])
def predict_service():
    try:
        data = request.get_json()
        brand    = data['brand']
        model    = data['model']
        eng_type = data['engine_type']
        region   = data['region']
        year     = int(data['make_year'])
        mileage  = int(data['mileage'])
        mil_range= int(data['mileage_range'])

        # Encode categoricals (handle unseen labels)
        def safe_encode(le, val):
            if val in le.classes_:
                return int(le.transform([val])[0])
            return 0

        X_dict = {
            'brand_enc':  safe_encode(le_brand,  brand),
            'model_enc':  safe_encode(le_model_, model),
            'engine_enc': safe_encode(le_engine, eng_type),
            'region_enc': safe_encode(le_region, region),
            'make_year':  year, 'mileage': mileage, 'mileage_range': mil_range
        }
        X = pd.DataFrame([X_dict])
        pred_cost = float(cost_model.predict(X)[0])

        # Predict each maintenance item
        items_needed = []
        items_not_needed = []
        for item in MAINT_ITEMS:
            clf = item_models[item]
            pred = int(clf.predict(X)[0])
            try:
                proba_arr = clf.predict_proba(X)[0]
                proba_val = proba_arr[1] if len(proba_arr) > 1 else float(pred)
            except Exception:
                proba_val = float(pred)
            entry = {
                "key": item,
                "label": ITEM_LABELS[item],
                "needed": bool(pred),
                "probability": round(float(proba_val) * 100, 1)
            }
            if pred == 1:
                items_needed.append(entry)
            else:
                items_not_needed.append(entry)

        urgency_map = {
            0: ("Low", "success"),
            1: ("Medium", "warning"),
            2: ("High", "warning"),
            3: ("Urgent", "danger"),
        }
        n = len(items_needed)
        urgency_key = min(3, n // 3)
        urgency, urgency_color = urgency_map[urgency_key]

        return jsonify({
            "success": True,
            "predicted_cost": round(pred_cost, 0),
            "items_needed": items_needed,
            "items_not_needed": items_not_needed,
            "total_items_needed": len(items_needed),
            "urgency": urgency,
            "urgency_color": urgency_color,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/stats')
def api_stats():
    stats = get_stats()
    return jsonify(stats)


if __name__ == '__main__':
    print("\n🚗  Vehicle Predictive Maintenance System")
    print("    Running on http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
