import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import glob
import os
import json
from fastapi import FastAPI
import uvicorn
import time

# --- 1. INDUSTRIAL DATA INGESTION ---
# This makes the script work on ANY computer once downloaded from GitHub
path = os.path.join(os.getcwd(), "IoTScenarios") 
all_files = glob.glob(os.path.join(path, "**/conn.log.labeled"), recursive=True)

if not all_files:
    print("CRITICAL ERROR: No 'conn.log.labeled' files found!")
    print(f"Current Path: {path}")
    print("Please ensure your CTU-Malware-Capture folders are in this directory.")
    exit()

print(f"Loading {len(all_files)} scenarios for final training...")
li = []
for f in all_files:
    try:
        # Loading 100k rows per scenario for high-fidelity training
        df_temp = pd.read_csv(f, sep='\t', skiprows=8, nrows=100000, low_memory=False, header=None)
        li.append(df_temp)
    except Exception as e:
        print(f"Skipping {f} due to error: {e}")

df = pd.concat(li, axis=0, ignore_index=True)

# Standard Zeek Column Mapping
df.columns = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig', 'local_resp', 'missed_bytes', 'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'label']

# Features: Duration, Bytes, Packets, and Destination Port (23)
cols = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'id.resp_p']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 0 = Normal/Honeypot, 1 = Attack/Malware
df['target'] = df['label'].apply(lambda x: 0 if 'Benign' in str(x) else 1)
X_train, X_test, y_train, y_test = train_test_split(df[cols], df['target'], test_size=0.2, random_state=42)

print("Status: Training Industrial XGBoost Model...")
model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# --- 2. THE XAI: LOGIC VISUALIZATION ---
print("Status: Generating SHAP Explainability Plot...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("xai_logic.png") 
print("Explainability Map saved as 'xai_logic.png'")

# --- 3. THE CLOUD API (REACTION LAYER) ---
app = FastAPI()

@app.get("/predict")
def predict_traffic(port: int, bytes: int, pkts: int, duration: float):
    # Velocity = Bytes per second (Detecting DDoS floods)
    velocity = bytes / duration if duration > 0 else bytes
    
    # Input for the AI Model
    input_data = pd.DataFrame([[duration, bytes, 0, pkts, 0, port]], columns=cols)
    
    prediction = int(model.predict(input_data)[0])
    conf = float(model.predict_proba(input_data)[0].max())
    
    # The JSON "Actionable" Response
    return {
        "system_status": "ACTIVE",
        "threat_decision": "MALICIOUS" if prediction == 1 else "BENIGN",
        "confidence": f"{conf*100:.2f}%",
        "telemetry_data": {
            "destination_port": port,
            "packet_velocity": f"{velocity:.2f} B/s",
            "heartbeat_count": pkts
        },
        "countermeasure": "BLOCK_IP_IMMEDIATELY" if prediction == 1 else "NONE",
        "xai_explanation": "Port 23 activity is a hallmark of Mirai/Botnet" if port == 23 else "Traffic follows benign patterns"
    }

if __name__ == "__main__":
    print("\n[FINALIZED] Cloud Security API is active.")
    print("Submit Heartbeats to: http://127.0.0.1:8000/predict")
    uvicorn.run(app, host="127.0.0.1", port=8000)