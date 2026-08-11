# ============================================================
# WEEK 5 - TASK 10
# FINAL TELECOM AI DASHBOARD
# ============================================================

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telecom AI Network Dashboard",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# 2. TITLE
# ============================================================

st.title("📡 Telecom AI Network Dashboard")

st.write(
    "AI-powered telecom KPI monitoring, anomaly detection, "
    "traffic forecasting, QoS prediction and root cause analysis."
)


# ============================================================
# 3. FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATA_FILE = os.path.join(
    BASE_DIR,
    "filedataset.csv"
)

ANOMALY_FILE = os.path.join(
    BASE_DIR,
    "detected_anomalies.csv"
)

LSTM_ANOMALY_FILE = os.path.join(
    BASE_DIR,
    "lstm_anomaly_results.csv"
)

FORECAST_24_FILE = os.path.join(
    BASE_DIR,
    "throughput_forecast_24_hours.csv"
)

FORECAST_7_FILE = os.path.join(
    BASE_DIR,
    "throughput_forecast_7_days.csv"
)

RCA_FILE = os.path.join(
    BASE_DIR,
    "rca_report.csv"
)

QOS_MODEL_FILE = os.path.join(
    BASE_DIR,
    "qos_prediction_model.pkl"
)

LABELED_FILE = os.path.join(
    BASE_DIR,
    "labeled_telecom_kpi.csv"
)


# ============================================================
# 4. LOAD MAIN DATASET
# ============================================================

try:

    df = pd.read_csv(
        DATA_FILE
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    df = df.sort_values(
        "Timestamp"
    )

except Exception as e:

    st.error(
        f"Could not load dataset: {e}"
    )

    st.stop()


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard Controls")


# Refresh button

if st.sidebar.button("🔄 Refresh Dashboard"):

    st.rerun()


# Cell selection

cell_options = [
    "All Cells"
] + sorted(
    df["Cell_ID"]
    .dropna()
    .unique()
    .tolist()
)


selected_cell = st.sidebar.selectbox(
    "Select Cell",
    cell_options
)


# KPI selection

kpi_options = [
    "RSRP",
    "SINR",
    "Latency",
    "Throughput",
    "Packet_Loss",
    "Connected_Users"
]


selected_kpi = st.sidebar.selectbox(
    "Select KPI",
    kpi_options
)


# ============================================================
# 6. FILTER DATA
# ============================================================

if selected_cell == "All Cells":

    filtered_df = df.copy()

else:

    filtered_df = df[
        df["Cell_ID"] == selected_cell
    ].copy()


# ============================================================
# 7. TOP KPI CARDS
# ============================================================

st.subheader("📊 Network Overview")


col1, col2, col3, col4, col5 = st.columns(5)


# Average throughput

avg_throughput = filtered_df[
    "Throughput"
].mean()


# Average latency

avg_latency = filtered_df[
    "Latency"
].mean()


# Average RSRP

avg_rsrp = filtered_df[
    "RSRP"
].mean()


# Packet loss

avg_packet_loss = filtered_df[
    "Packet_Loss"
].mean()


# Connected users

avg_users = filtered_df[
    "Connected_Users"
].mean()


col1.metric(
    "📶 Avg Throughput",
    f"{avg_throughput:.2f} Mbps"
)

col2.metric(
    "⏱️ Avg Latency",
    f"{avg_latency:.2f} ms"
)

col3.metric(
    "📡 Avg RSRP",
    f"{avg_rsrp:.2f} dBm"
)

col4.metric(
    "📦 Avg Packet Loss",
    f"{avg_packet_loss:.2f}%"
)

col5.metric(
    "👥 Avg Users",
    f"{avg_users:.0f}"
)


# ============================================================
# 8. LIVE KPI TREND
# ============================================================

st.subheader(
    f"📈 KPI Trend - {selected_kpi}"
)


fig_kpi = px.line(
    filtered_df,
    x="Timestamp",
    y=selected_kpi,
    title=f"{selected_kpi} Over Time"
)


fig_kpi.update_layout(
    xaxis_title="Time",
    yaxis_title=selected_kpi,
    hovermode="x unified"
)


st.plotly_chart(
    fig_kpi,
    use_container_width=True
)


# ============================================================
# 9. KPI TRENDS - ALL KPIs
# ============================================================

st.subheader("📊 All KPI Trends")


kpi_col1, kpi_col2 = st.columns(2)


with kpi_col1:

    fig_rsrp = px.line(
        filtered_df,
        x="Timestamp",
        y="RSRP",
        title="RSRP Trend"
    )

    st.plotly_chart(
        fig_rsrp,
        use_container_width=True
    )


with kpi_col2:

    fig_sinr = px.line(
        filtered_df,
        x="Timestamp",
        y="SINR",
        title="SINR Trend"
    )

    st.plotly_chart(
        fig_sinr,
        use_container_width=True
    )


kpi_col3, kpi_col4 = st.columns(2)


with kpi_col3:

    fig_latency = px.line(
        filtered_df,
        x="Timestamp",
        y="Latency",
        title="Latency Trend"
    )

    st.plotly_chart(
        fig_latency,
        use_container_width=True
    )


with kpi_col4:

    fig_throughput = px.line(
        filtered_df,
        x="Timestamp",
        y="Throughput",
        title="Throughput Trend"
    )

    st.plotly_chart(
        fig_throughput,
        use_container_width=True
    )


# ============================================================
# 10. LOAD ISOLATION FOREST ANOMALIES
# ============================================================

if os.path.exists(ANOMALY_FILE):

    anomalies = pd.read_csv(
        ANOMALY_FILE
    )

    anomalies["Timestamp"] = pd.to_datetime(
        anomalies["Timestamp"],
        errors="coerce"
    )

else:

    anomalies = pd.DataFrame()


# ============================================================
# 11. ANOMALY SECTION
# ============================================================

st.subheader("🚨 Detected Anomalies")


if not anomalies.empty:

    if selected_cell != "All Cells":

        display_anomalies = anomalies[
            anomalies["Cell_ID"] == selected_cell
        ]

    else:

        display_anomalies = anomalies


    anomaly_count = len(
        display_anomalies
    )

    st.metric(
        "Detected Anomalies",
        anomaly_count
    )


    if not display_anomalies.empty:

        st.dataframe(
            display_anomalies[
                [
                    col
                    for col in [
                        "Timestamp",
                        "Cell_ID",
                        "RSRP",
                        "SINR",
                        "Latency",
                        "Throughput",
                        "Packet_Loss",
                        "Connected_Users"
                    ]
                    if col in display_anomalies.columns
                ]
            ],
            use_container_width=True
        )

    else:

        st.success(
            "No anomalies detected for this cell."
        )

else:

    st.info(
        "Isolation Forest anomaly file not found."
    )


# ============================================================
# 12. ANOMALY VISUALIZATION
# ============================================================

if not anomalies.empty:

    st.subheader(
        "🚨 Anomalies on Throughput"
    )


    anomaly_plot_data = filtered_df.copy()


    anomaly_plot = px.line(
        anomaly_plot_data,
        x="Timestamp",
        y="Throughput",
        title="Throughput with Anomaly Locations"
    )


    if selected_cell == "All Cells":

        plot_anomalies = anomalies

    else:

        plot_anomalies = anomalies[
            anomalies["Cell_ID"] == selected_cell
        ]


    if not plot_anomalies.empty:

        anomaly_plot.add_trace(
            go.Scatter(
                x=plot_anomalies["Timestamp"],
                y=plot_anomalies["Throughput"],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    size=10,
                    symbol="x"
                )
            )
        )


    st.plotly_chart(
        anomaly_plot,
        use_container_width=True
    )


# ============================================================
# 13. LSTM ANOMALIES
# ============================================================

st.subheader(
    "🧠 LSTM Anomaly Detection"
)


if os.path.exists(LSTM_ANOMALY_FILE):

    lstm_anomalies = pd.read_csv(
        LSTM_ANOMALY_FILE
    )

    lstm_anomalies["Timestamp"] = pd.to_datetime(
        lstm_anomalies["Timestamp"],
        errors="coerce"
    )


    detected_lstm = lstm_anomalies[
        lstm_anomalies["LSTM_Anomaly"] == True
    ]


    st.metric(
        "LSTM Detected Sequences",
        len(detected_lstm)
    )


    if not detected_lstm.empty:

        st.dataframe(
            detected_lstm,
            use_container_width=True
        )

else:

    st.info(
        "LSTM anomaly result file not found."
    )


# ============================================================
# 14. TRAFFIC FORECAST
# ============================================================

st.subheader(
    "🔮 Network Traffic Forecast"
)


forecast_tab1, forecast_tab2 = st.tabs(
    [
        "Next 24 Hours",
        "Next 7 Days"
    ]
)


# ------------------------------------------------------------
# 24 HOURS
# ------------------------------------------------------------

with forecast_tab1:

    if os.path.exists(
        FORECAST_24_FILE
    ):

        forecast_24 = pd.read_csv(
            FORECAST_24_FILE
        )

        forecast_24["Timestamp"] = pd.to_datetime(
            forecast_24["Timestamp"],
            errors="coerce"
        )


        fig_24 = px.line(
            forecast_24,
            x="Timestamp",
            y="Predicted_Throughput",
            title="Next 24 Hours Throughput Forecast"
        )


        st.plotly_chart(
            fig_24,
            use_container_width=True
        )


        st.dataframe(
            forecast_24,
            use_container_width=True
        )

    else:

        st.warning(
            "24-hour forecast file not found."
        )


# ------------------------------------------------------------
# 7 DAYS
# ------------------------------------------------------------

with forecast_tab2:

    if os.path.exists(
        FORECAST_7_FILE
    ):

        forecast_7 = pd.read_csv(
            FORECAST_7_FILE
        )

        forecast_7["Timestamp"] = pd.to_datetime(
            forecast_7["Timestamp"],
            errors="coerce"
        )


        fig_7 = px.line(
            forecast_7,
            x="Timestamp",
            y="Predicted_Throughput",
            title="Next 7 Days Throughput Forecast"
        )


        st.plotly_chart(
            fig_7,
            use_container_width=True
        )


        st.dataframe(
            forecast_7,
            use_container_width=True
        )

    else:

        st.warning(
            "7-day forecast file not found."
        )


# ============================================================
# 15. QoS / QoE PREDICTION
# ============================================================

st.subheader(
    "📶 QoS / QoE Prediction"
)


if os.path.exists(
    QOS_MODEL_FILE
):

    try:

        qos_model = joblib.load(
            QOS_MODEL_FILE
        )


        qos_features = [
            "RSRP",
            "SINR",
            "Latency",
            "Packet_Loss",
            "Throughput"
        ]


        qos_input = filtered_df[
            qos_features
        ].copy()


        qos_input = qos_input.fillna(
            qos_input.median()
        )


        qos_predictions = (
            qos_model.predict(
                qos_input
            )
        )


        qos_display = filtered_df[
            [
                "Timestamp",
                "Cell_ID"
            ]
        ].copy()


        qos_display[
            "Predicted_QoS"
        ] = qos_predictions


        # Latest prediction

        latest_qos = qos_predictions[-1]


        if latest_qos == "Good":

            st.success(
                f"Current Network Quality: {latest_qos}"
            )

        elif latest_qos == "Fair":

            st.warning(
                f"Current Network Quality: {latest_qos}"
            )

        else:

            st.error(
                f"Current Network Quality: {latest_qos}"
            )


        # Distribution

        qos_counts = pd.Series(
            qos_predictions
        ).value_counts()


        fig_qos = px.bar(
            x=qos_counts.index,
            y=qos_counts.values,
            title="QoS Prediction Distribution",
            labels={
                "x": "Network Quality",
                "y": "Number of Records"
            }
        )


        st.plotly_chart(
            fig_qos,
            use_container_width=True
        )


        st.dataframe(
            qos_display.tail(50),
            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"Could not load QoS model: {e}"
        )

else:

    st.warning(
        "QoS prediction model not found."
    )


# ============================================================
# 16. ROOT CAUSE ANALYSIS
# ============================================================

st.subheader(
    "🔍 Root Cause Analysis"
)


if os.path.exists(
    RCA_FILE
):

    rca = pd.read_csv(
        RCA_FILE
    )


    if selected_cell != "All Cells":

        rca_display = rca[
            rca["Cell_ID"] == selected_cell
        ]

    else:

        rca_display = rca


    if not rca_display.empty:

        st.dataframe(
            rca_display,
            use_container_width=True
        )


        # Root cause frequency

        if "Possible_Root_Cause" in rca_display.columns:

            root_causes = (
                rca_display[
                    "Possible_Root_Cause"
                ]
                .value_counts()
            )


            fig_rca = px.bar(
                x=root_causes.index,
                y=root_causes.values,
                title="Root Cause Frequency",
                labels={
                    "x": "Possible Root Cause",
                    "y": "Number of Occurrences"
                }
            )


            st.plotly_chart(
                fig_rca,
                use_container_width=True
            )

    else:

        st.success(
            "No RCA records for the selected cell."
        )

else:

    st.warning(
        "RCA report not found."
    )


# ============================================================
# 17. CELL-WISE KPI STATUS
# ============================================================

st.subheader(
    "🏢 Cell-wise KPI Status"
)


if os.path.exists(
    LABELED_FILE
):

    labeled_df = pd.read_csv(
        LABELED_FILE
    )


    # Latest record for every cell

    latest_cell_status = (
        labeled_df
        .sort_values("Timestamp")
        .groupby("Cell_ID")
        .tail(1)
    )


    status_columns = [
        col
        for col in [
            "Cell_ID",
            "RSRP_Status",
            "SINR_Status",
            "Latency_Status",
            "Packet_Loss_Status",
            "Network_Status"
        ]
        if col in latest_cell_status.columns
    ]


    st.dataframe(
        latest_cell_status[
            status_columns
        ],
        use_container_width=True
    )


else:

    st.warning(
        "Labeled dataset not found."
    )


# ============================================================
# 18. KPI THRESHOLD ALERTS
# ============================================================

st.subheader(
    "⚠️ KPI Threshold Alerts"
)


# Thresholds from Task 9

RSRP_THRESHOLD = -105
SINR_THRESHOLD = 10
LATENCY_THRESHOLD = 50
PACKET_LOSS_THRESHOLD = 3


latest_record = filtered_df.iloc[-1]


alerts = []


# RSRP

if latest_record["RSRP"] < RSRP_THRESHOLD:

    alerts.append(
        f"🚨 Critical RSRP: "
        f"{latest_record['RSRP']:.2f} dBm"
    )


# SINR

if latest_record["SINR"] < SINR_THRESHOLD:

    alerts.append(
        f"🚨 Critical SINR: "
        f"{latest_record['SINR']:.2f} dB"
    )


# Latency

if latest_record["Latency"] > LATENCY_THRESHOLD:

    alerts.append(
        f"🚨 High Latency: "
        f"{latest_record['Latency']:.2f} ms"
    )


# Packet loss

if latest_record["Packet_Loss"] > PACKET_LOSS_THRESHOLD:

    alerts.append(
        f"🚨 High Packet Loss: "
        f"{latest_record['Packet_Loss']:.2f}%"
    )


if len(alerts) == 0:

    st.success(
        "✅ No critical KPI threshold violations detected."
    )

else:

    for alert in alerts:

        st.error(alert)


# ============================================================
# 19. RAW DATA
# ============================================================

with st.expander(
    "📄 View Raw Telecom Data"
):

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


# ============================================================
# 20. FOOTER
# ============================================================

st.divider()

st.caption(
    "Telecom AI Network Monitoring Dashboard | "
    "Python • Pandas • Scikit-learn • PyTorch • Streamlit • Plotly"
)