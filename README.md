# 📡 Telecom AI Network Dashboard

An AI-powered telecom network monitoring dashboard built using Python and Streamlit.

The project analyzes telecom KPI data and demonstrates anomaly detection, traffic forecasting, QoS prediction, and root cause analysis.

## 🚀 Features

- 📊 Real-time-style telecom KPI monitoring
- 🔍 Anomaly detection using Isolation Forest
- 🧠 LSTM-based anomaly detection
- 📈 Network traffic forecasting
- 🎯 QoS/QoE prediction
- 🔧 Root Cause Analysis (RCA)
- ⚙️ KPI threshold tuning
- 📱 Interactive Streamlit dashboard
- 🏢 Cell-wise network KPI analysis

## 📡 KPIs Analyzed

- RSRP
- SINR
- Latency
- Throughput
- Packet Loss
- Connected Users

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Streamlit
- Plotly
- Matplotlib

## 🤖 Machine Learning

The project includes:

- Isolation Forest for anomaly detection
- LSTM neural networks for sequential anomaly detection and traffic forecasting
- ML classification for QoS/QoE prediction
- Threshold-based KPI labeling and analysis
- Rule-based Root Cause Analysis

## 📂 Project Structure

```text
├── app.py
├── week5amantya.py
├── filedataset.csv
├── requirements.txt
├── .gitignore
├── isolation_forest.pkl
├── lstm_anomaly_model.pt
├── lstm_traffic_forecasting_model.pt
├── qos_prediction_model.pkl
└── supporting datasets and model outputs
