import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. LOAD DATASET
# ============================================================


df = pd.read_csv("filename.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. UNDERSTAND THE DATASET
# ============================================================

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 3. CONVERT TIMESTAMP TO DATETIME
# ============================================================

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

print("\nTimestamp data type:")
print(df["Timestamp"].dtype)


# ============================================================
# 4. CHECK INVALID / MISSING TIMESTAMPS
# ============================================================

invalid_timestamps = df["Timestamp"].isna().sum()

print("\nInvalid or missing timestamps:", invalid_timestamps)


# ============================================================
# 5. SORT DATA CHRONOLOGICALLY
# ============================================================

df = df.sort_values("Timestamp").reset_index(drop=True)

print("\nFirst timestamp:", df["Timestamp"].min())
print("Last timestamp:", df["Timestamp"].max())

print("\nData after chronological sorting:")
print(df.head())


# ============================================================
# 6. CHECK DUPLICATE ROWS
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)


# ============================================================
# 7. CHECK MISSING KPI VALUES
# ============================================================

print("\nMissing values in each column:")
print(df.isnull().sum())


# Missing-value report

missing_values = df.isnull().sum()

missing_report = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": (missing_values / len(df)) * 100
})

print("\nMissing Value Report:")
print(missing_report)


# ============================================================
# 8. CHECK 5-MINUTE TIME INTERVALS
# ============================================================

expected_interval = pd.Timedelta(minutes=5)

time_differences = df["Timestamp"].diff()

print("\nMost common time differences:")
print(time_differences.value_counts().head(10))


# ============================================================
# 9. FIND TIMESTAMP GAPS
# ============================================================

timestamp_gaps = df.loc[
    df["Timestamp"].diff() > expected_interval,
    ["Timestamp"]
]

print("\nNumber of timestamp gaps:", len(timestamp_gaps))

print("\nTimestamp gaps:")
print(timestamp_gaps.head(20))


# ============================================================
# 10. CHECK TIMESTAMP GAPS FOR EACH CELL
# ============================================================

df = df.sort_values(
    ["Cell_ID", "Timestamp"]
).reset_index(drop=True)

df["Time_Difference"] = (
    df.groupby("Cell_ID")["Timestamp"].diff()
)

cell_timestamp_gaps = df[
    df["Time_Difference"] > expected_interval
]

print("\nTotal timestamp gaps across cells:",
      len(cell_timestamp_gaps))

print("\nTimestamp gaps by cell:")
print(
    cell_timestamp_gaps[
        ["Cell_ID", "Timestamp", "Time_Difference"]
    ].head(20)
)


# ============================================================
# 11. CREATE DATE / HOUR / DAY FEATURES
# ============================================================

df["Date"] = df["Timestamp"].dt.date
df["Hour"] = df["Timestamp"].dt.hour
df["Day_of_Week"] = df["Timestamp"].dt.day_name()




# ============================================================
# 22. FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("FINAL DATASET SUMMARY")
print("=" * 50)

print("\nDataset shape:")
print(df.shape)

print("\nDate range:")
print(
    df["Timestamp"].min(),
    "to",
    df["Timestamp"].max()
)

print("\nNumber of cells:")
print(df["Cell_ID"].nunique())

print("\nCell IDs:")
print(df["Cell_ID"].unique())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTask 1 completed!")


# ============================================================
# WEEK 5 - TASK 3
# ANOMALY DETECTION USING ISOLATION FOREST
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib


# ============================================================
# 1. SELECT KPI FEATURES
# ============================================================

features = [
    "RSRP",
    "SINR",
    "Latency",
    "Throughput",
    "Packet_Loss",
    "Connected_Users"
]

print("\nFeatures selected for anomaly detection:")
print(features)


# ============================================================
# 2. CREATE FEATURE DATA
# ============================================================

X = df[features].copy()

print("\nFeature data:")
print(X.head())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing values before handling:")
print(X.isnull().sum())


# Fill missing KPI values using median
X = X.fillna(X.median())

print("\nMissing values after handling:")
print(X.isnull().sum())


# ============================================================
# 4. NORMALIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures after normalization:")
print(X_scaled[:5])


# ============================================================
# 5. CREATE ISOLATION FOREST MODEL
# ============================================================

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)


# ============================================================
# 6. TRAIN MODEL AND DETECT ANOMALIES
# ============================================================

predictions = model.fit_predict(X_scaled)


# ============================================================
# 7. ADD RESULTS TO DATAFRAME
# ============================================================

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

df["Anomaly"] = predictions

df["Anomaly_Label"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})


# ============================================================
# 8. COUNT NORMAL AND ANOMALOUS RECORDS
# ============================================================

total_records = len(df)

normal_records = (df["Anomaly"] == 1).sum()

anomaly_records = (df["Anomaly"] == -1).sum()

anomaly_percentage = (
    anomaly_records / total_records
) * 100


print("\n" + "=" * 50)
print("ISOLATION FOREST RESULTS")
print("=" * 50)

print("\nTotal records:", total_records)
print("Normal records:", normal_records)
print("Anomalous records:", anomaly_records)

print(
    "Anomaly percentage:",
    round(anomaly_percentage, 2),
    "%"
)


# ============================================================
# 9. DISPLAY DETECTED ANOMALIES
# ============================================================

anomalies = df[df["Anomaly"] == -1].copy()

print("\nDetected anomalies:")
print(
    anomalies[
        [
            "Timestamp",
            "Cell_ID",
            "RSRP",
            "SINR",
            "Latency",
            "Throughput",
            "Packet_Loss",
            "Connected_Users"
        ]
    ].head(20)
)


# ============================================================
# 10. SAVE LIST OF DETECTED ANOMALIES
# ============================================================

anomalies.to_csv(
    "detected_anomalies.csv",
    index=False
)

print("\nDetected anomalies saved to:")
print("detected_anomalies.csv")


# ============================================================
# 11. VISUALIZE THROUGHPUT ANOMALIES
# ============================================================

normal_data = df[df["Anomaly"] == 1]
anomaly_data = df[df["Anomaly"] == -1]

plt.figure(figsize=(14, 6))

plt.plot(
    normal_data["Timestamp"],
    normal_data["Throughput"],
    label="Normal",
    linewidth=1
)

plt.scatter(
    anomaly_data["Timestamp"],
    anomaly_data["Throughput"],
    label="Anomaly",
    marker="x",
    s=50
)

plt.title("Isolation Forest - Throughput Anomalies")
plt.xlabel("Time")
plt.ylabel("Throughput")

plt.legend()
plt.xticks(rotation=45)



# ============================================================
# 12. VISUALIZE LATENCY ANOMALIES
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    normal_data["Timestamp"],
    normal_data["Latency"],
    label="Normal",
    linewidth=1
)

plt.scatter(
    anomaly_data["Timestamp"],
    anomaly_data["Latency"],
    label="Anomaly",
    marker="x",
    s=50
)

plt.title("Isolation Forest - Latency Anomalies")
plt.xlabel("Time")
plt.ylabel("Latency")

plt.legend()
plt.xticks(rotation=45)




# ============================================================
# 13. VISUALIZE SINR ANOMALIES
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    normal_data["Timestamp"],
    normal_data["SINR"],
    label="Normal",
    linewidth=1
)

plt.scatter(
    anomaly_data["Timestamp"],
    anomaly_data["SINR"],
    label="Anomaly",
    marker="x",
    s=50
)

plt.title("Isolation Forest - SINR Anomalies")
plt.xlabel("Time")
plt.ylabel("SINR (dB)")

plt.legend()
plt.xticks(rotation=45)





# ============================================================
# 14. SAVE THE MODEL
# ============================================================

joblib.dump(
    model,
    "isolation_forest.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("\nSaved model files:")
print("isolation_forest.pkl")
print("scaler.pkl")


# ============================================================
# TASK 3 COMPLETE
# ============================================================

print("\n" + "=" * 50)
print("TASK 3 COMPLETED")
print("=" * 50)


# ============================================================
# WEEK 5 - TASK 4
# LSTM-BASED ANOMALY DETECTION
# ============================================================




# ============================================================
# 1. CHECK PYTORCH AND SELECT DEVICE
# ============================================================

print("\n" + "=" * 60)
print("TASK 4 - LSTM ANOMALY DETECTION")
print("=" * 60)

print("\nPyTorch version:", torch.__version__)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 2. SELECT KPI FEATURES
# ============================================================

features = [
    "RSRP",
    "SINR",
    "Latency",
    "Throughput",
    "Packet_Loss",
    "Connected_Users"
]

print("\nFeatures:")
print(features)


# ============================================================
# 3. PREPARE KPI DATA
# ============================================================

# Use the same features as Task 3
X = df[features].copy()

# Handle missing values
X = X.fillna(X.median())

# Use the scaler created in Task 3
X_scaled = scaler.transform(X)

print("\nScaled data shape:")
print(X_scaled.shape)


# ============================================================
# 4. KEEP ONLY NORMAL DATA FOR TRAINING
# ============================================================

# Task 3 Isolation Forest:
#  1  = Normal
# -1  = Anomaly

normal_mask = df["Anomaly"] == 1

normal_data = X_scaled[normal_mask.values]

print("\nTotal records:", len(X_scaled))
print("Normal records:", len(normal_data))


# ============================================================
# 5. CREATE SLIDING WINDOWS
# ============================================================

sequence_length = 12

print("\nSequence length:", sequence_length)

X_sequences = []
y_sequences = []

for i in range(len(normal_data) - sequence_length):

    sequence = normal_data[
        i:i + sequence_length
    ]

    target = normal_data[
        i + sequence_length
    ]

    X_sequences.append(sequence)
    y_sequences.append(target)


X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print("\nSequence shape:")
print(X_sequences.shape)

print("\nTarget shape:")
print(y_sequences.shape)


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(len(X_sequences) * 0.8)

X_train = X_sequences[:split_index]
y_train = y_sequences[:split_index]

X_val = X_sequences[split_index:]
y_val = y_sequences[split_index:]

print("\nTraining sequences:", len(X_train))
print("Validation sequences:", len(X_val))


# ============================================================
# 7. CONVERT DATA TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.float32
)


print("\nPyTorch tensor shapes:")
print("X_train:", X_train_tensor.shape)
print("y_train:", y_train_tensor.shape)
print("X_val:", X_val_tensor.shape)
print("y_val:", y_val_tensor.shape)


# ============================================================
# 8. CREATE DATALOADERS
# ============================================================

batch_size = 32

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("\nBatch size:", batch_size)


# ============================================================
# 9. BUILD LSTM MODEL
# ============================================================

class LSTMAnomalyDetector(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        # LSTM output
        lstm_output, _ = self.lstm(x)

        # Take the output from the final time step
        last_output = lstm_output[:, -1, :]

        # Convert hidden representation to KPI prediction
        output = self.fc(last_output)

        return output


# ============================================================
# 10. CREATE MODEL
# ============================================================

input_size = len(features)
hidden_size = 64
num_layers = 2
output_size = len(features)

model = LSTMAnomalyDetector(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=output_size
)

model = model.to(device)

print("\nLSTM model:")
print(model)


# ============================================================
# 11. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\nLoss function: MSELoss")
print("Optimizer: Adam")
print("Learning rate: 0.001")


# ============================================================
# 12. TRAINING SETTINGS
# ============================================================

epochs = 50

train_losses = []
val_losses = []


# ============================================================
# 13. TRAIN LSTM MODEL
# ============================================================

print("\nStarting training...\n")

for epoch in range(epochs):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(X_batch)

        # Calculate loss
        loss = criterion(
            predictions,
            y_batch
        )

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    total_val_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            total_val_loss += loss.item()

    average_val_loss = (
        total_val_loss / len(val_loader)
    )


    # Store losses
    train_losses.append(average_train_loss)
    val_losses.append(average_val_loss)


    # Print progress
    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {average_train_loss:.6f} "
            f"Validation Loss: {average_val_loss:.6f}"
        )


# ============================================================
# 14. TRAINING COMPLETE
# ============================================================

print("\nTraining completed!")


# ============================================================
# 15. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, epochs + 1),
    train_losses,
    label="Training Loss"
)

plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")


# ============================================================
# 16. VALIDATION LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, epochs + 1),
    val_losses,
    label="Validation Loss"
)

plt.title("LSTM Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.legend()


# ============================================================
# 17. TRAINING VS VALIDATION LOSS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, epochs + 1),
    train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, epochs + 1),
    val_losses,
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.legend()



# ============================================================
# 18. CALCULATE PREDICTION ERRORS
# ============================================================

model.eval()

all_predictions = []
all_actual = []

with torch.no_grad():

    for X_batch, y_batch in val_loader:

        X_batch = X_batch.to(device)

        predictions = model(X_batch)

        all_predictions.append(
            predictions.cpu().numpy()
        )

        all_actual.append(
            y_batch.numpy()
        )


all_predictions = np.concatenate(
    all_predictions
)

all_actual = np.concatenate(
    all_actual
)


# ============================================================
# 19. CALCULATE MSE FOR EACH SEQUENCE
# ============================================================

prediction_errors = np.mean(
    (all_actual - all_predictions) ** 2,
    axis=1
)

print("\nPrediction error shape:")
print(prediction_errors.shape)

print("\nFirst 10 prediction errors:")
print(prediction_errors[:10])


# ============================================================
# 20. DETERMINE ANOMALY THRESHOLD
# ============================================================

# Use the 95th percentile as the initial threshold

threshold = np.percentile(
    prediction_errors,
    95
)

print("\nAnomaly threshold:")
print(threshold)


# ============================================================
# 21. DETECT ANOMALOUS SEQUENCES
# ============================================================

lstm_anomalies = prediction_errors > threshold

print("\nTotal validation sequences:",
      len(prediction_errors))

print(
    "Detected anomalous sequences:",
    lstm_anomalies.sum()
)

print(
    "Anomaly percentage:",
    round(
        lstm_anomalies.mean() * 100,
        2
    ),
    "%"
)


# ============================================================
# 22. CREATE ANOMALY RESULT DATAFRAME
# ============================================================

validation_start = split_index + sequence_length

# Get timestamps corresponding approximately
# to the prediction points

validation_timestamps = df[
    df["Anomaly"] == 1
]["Timestamp"].iloc[
    validation_start:
    validation_start + len(prediction_errors)
].values

# Make sure lengths match

min_length = min(
    len(validation_timestamps),
    len(prediction_errors)
)

anomaly_results = pd.DataFrame({
    "Timestamp": validation_timestamps[:min_length],
    "Prediction_Error": prediction_errors[:min_length],
    "LSTM_Anomaly": lstm_anomalies[:min_length]
})


# ============================================================
# 23. DISPLAY LSTM ANOMALIES
# ============================================================

detected_lstm_anomalies = anomaly_results[
    anomaly_results["LSTM_Anomaly"] == True
]

print("\n" + "=" * 60)
print("LSTM ANOMALY RESULTS")
print("=" * 60)

print("\nDetected anomalies:")
print(
    detected_lstm_anomalies.head(20)
)


# ============================================================
# 24. SAVE LSTM ANOMALY RESULTS
# ============================================================

anomaly_results.to_csv(
    "lstm_anomaly_results.csv",
    index=False
)

print(
    "\nLSTM anomaly results saved as:"
)
print("lstm_anomaly_results.csv")


# ============================================================
# 25. PLOT PREDICTION ERRORS
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    anomaly_results["Timestamp"],
    anomaly_results["Prediction_Error"],
    label="Prediction Error"
)

plt.axhline(
    y=threshold,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.scatter(
    detected_lstm_anomalies["Timestamp"],
    detected_lstm_anomalies["Prediction_Error"],
    marker="x",
    s=60,
    label="Anomaly"
)

plt.title("LSTM Prediction Error and Anomalies")
plt.xlabel("Time")
plt.ylabel("Prediction Error")

plt.legend()
plt.xticks(rotation=45)




# ============================================================
# 26. SAVE TRAINED LSTM MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "output_size": output_size,
        "sequence_length": sequence_length,
        "features": features,
        "threshold": threshold
    },
    "lstm_anomaly_model.pt"
)

print("\nLSTM model saved as:")
print("lstm_anomaly_model.pt")


# ============================================================
# 27. FINAL TASK 4 SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TASK 4 COMPLETED")
print("=" * 60)

print("\nModel:")
print("LSTM")

print("\nSequence length:")
print(sequence_length)

print("\nHidden units:")
print(hidden_size)

print("\nLSTM layers:")
print(num_layers)

print("\nEpochs:")
print(epochs)

print("\nTraining sequences:")
print(len(X_train))

print("\nValidation sequences:")
print(len(X_val))

print("\nFinal training loss:")
print(train_losses[-1])

print("\nFinal validation loss:")
print(val_losses[-1])

print("\nAnomaly threshold:")
print(threshold)

print("\nDetected anomalous sequences:")
print(lstm_anomalies.sum())

print("\nGenerated files:")
print("- lstm_anomaly_results.csv")
print("- lstm_anomaly_model.pt")

print("\nTask 4 completed successfully!")


# ============================================================
# WEEK 5 - TASK 5
# ANOMALY THRESHOLD TUNING
# ============================================================

print("\n" + "=" * 60)
print("TASK 5 - THRESHOLD TUNING")
print("=" * 60)


# ============================================================
# 1. DEFINE LOW, MEDIUM AND HIGH THRESHOLDS
# ============================================================

# We use prediction error percentiles.
#
# Low threshold    = 90th percentile
# Medium threshold = 95th percentile
# High threshold   = 99th percentile

low_threshold = np.percentile(
    prediction_errors,
    90
)

medium_threshold = np.percentile(
    prediction_errors,
    95
)

high_threshold = np.percentile(
    prediction_errors,
    99
)

print("\nThreshold values:")

print(
    "Low threshold (90th percentile):",
    low_threshold
)

print(
    "Medium threshold (95th percentile):",
    medium_threshold
)

print(
    "High threshold (99th percentile):",
    high_threshold
)


# ============================================================
# 2. DETECT ANOMALIES AT EACH THRESHOLD
# ============================================================

low_anomalies = (
    prediction_errors > low_threshold
)

medium_anomalies = (
    prediction_errors > medium_threshold
)

high_anomalies = (
    prediction_errors > high_threshold
)


# ============================================================
# 3. COUNT DETECTED ANOMALIES
# ============================================================

low_count = low_anomalies.sum()

medium_count = medium_anomalies.sum()

high_count = high_anomalies.sum()

total_sequences = len(prediction_errors)


print("\nNumber of detected anomalies:")

print(
    "Low threshold:",
    low_count
)

print(
    "Medium threshold:",
    medium_count
)

print(
    "High threshold:",
    high_count
)


# ============================================================
# 4. CALCULATE ANOMALY PERCENTAGES
# ============================================================

low_percentage = (
    low_count / total_sequences
) * 100

medium_percentage = (
    medium_count / total_sequences
) * 100

high_percentage = (
    high_count / total_sequences
) * 100


print("\nAnomaly percentages:")

print(
    "Low threshold:",
    round(low_percentage, 2),
    "%"
)

print(
    "Medium threshold:",
    round(medium_percentage, 2),
    "%"
)

print(
    "High threshold:",
    round(high_percentage, 2),
    "%"
)


# ============================================================
# 5. USE TASK 3 ISOLATION FOREST AS REFERENCE
# ============================================================

# NOTE:
# Isolation Forest is NOT ground truth.
#
# We are only using it as a reference detector so that
# we can compare the LSTM threshold results.
#
# True false-positive and missed-anomaly measurements
# require actual labelled anomaly data.

isolation_reference = df["Anomaly"].values


# ============================================================
# 6. ALIGN ISOLATION FOREST RESULTS WITH LSTM RESULTS
# ============================================================

# anomaly_results contains the LSTM validation sequences.
#
# We need the corresponding portion of the Isolation Forest
# results.

reference_normal = (
    df["Anomaly"] == 1
).values

reference_anomalies = (
    df["Anomaly"] == -1
).values


# Use only the length that is available in both datasets

comparison_length = min(
    len(reference_anomalies),
    len(prediction_errors)
)

reference_anomalies = (
    reference_anomalies[
        :comparison_length
    ]
)

low_compare = (
    low_anomalies[
        :comparison_length
    ]
)

medium_compare = (
    medium_anomalies[
        :comparison_length
    ]
)

high_compare = (
    high_anomalies[
        :comparison_length
    ]
)


# ============================================================
# 7. CALCULATE REFERENCE FALSE POSITIVES / MISSED ANOMALIES
# ============================================================

# Again:
# These are NOT true ground-truth metrics.
#
# They measure disagreement with the Isolation Forest
# reference detector.

def calculate_reference_metrics(
    lstm_predictions,
    reference_labels
):

    false_positives = (
        (lstm_predictions == True) &
        (reference_labels == False)
    ).sum()

    missed_anomalies = (
        (lstm_predictions == False) &
        (reference_labels == True)
    ).sum()

    return false_positives, missed_anomalies


low_fp, low_missed = calculate_reference_metrics(
    low_compare,
    reference_anomalies
)

medium_fp, medium_missed = calculate_reference_metrics(
    medium_compare,
    reference_anomalies
)

high_fp, high_missed = calculate_reference_metrics(
    high_compare,
    reference_anomalies
)


# ============================================================
# 8. CREATE THRESHOLD COMPARISON TABLE
# ============================================================

threshold_comparison = pd.DataFrame({

    "Threshold": [
        "Low",
        "Medium",
        "High"
    ],

    "Percentile": [
        "90%",
        "95%",
        "99%"
    ],

    "Threshold Value": [
        low_threshold,
        medium_threshold,
        high_threshold
    ],

    "Detected Anomalies": [
        low_count,
        medium_count,
        high_count
    ],

    "Anomaly Percentage": [
        low_percentage,
        medium_percentage,
        high_percentage
    ],

    "Reference False Positives": [
        low_fp,
        medium_fp,
        high_fp
    ],

    "Reference Missed Anomalies": [
        low_missed,
        medium_missed,
        high_missed
    ]
})


# ============================================================
# 9. DISPLAY COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("THRESHOLD COMPARISON")
print("=" * 60)

print(
    threshold_comparison.to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE COMPARISON TABLE
# ============================================================

threshold_comparison.to_csv(
    "threshold_comparison.csv",
    index=False
)

print(
    "\nThreshold comparison saved as:"
)

print("threshold_comparison.csv")


# ============================================================
# 11. PLOT PREDICTION ERRORS WITH ALL THRESHOLDS
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    anomaly_results["Timestamp"],
    anomaly_results["Prediction_Error"],
    label="Prediction Error"
)

plt.axhline(
    y=low_threshold,
    linestyle="--",
    label="Low Threshold (90%)"
)

plt.axhline(
    y=medium_threshold,
    linestyle="--",
    label="Medium Threshold (95%)"
)

plt.axhline(
    y=high_threshold,
    linestyle="--",
    label="High Threshold (99%)"
)

plt.title(
    "LSTM Prediction Error - Threshold Comparison"
)

plt.xlabel("Time")

plt.ylabel("Prediction Error")

plt.legend()

plt.xticks(rotation=45)


# ============================================================
# 12. PLOT NUMBER OF DETECTED ANOMALIES
# ============================================================

threshold_names = [
    "Low (90%)",
    "Medium (95%)",
    "High (99%)"
]

anomaly_counts = [
    low_count,
    medium_count,
    high_count
]

plt.figure(figsize=(8, 5))

plt.bar(
    threshold_names,
    anomaly_counts
)

plt.title(
    "Number of Detected Anomalies by Threshold"
)

plt.xlabel("Threshold")

plt.ylabel("Number of Anomalies")



# ============================================================
# 13. SELECT RECOMMENDED THRESHOLD
# ============================================================

# For this initial experiment we recommend the
# MEDIUM threshold because:
#
# Low threshold:
# - More sensitive
# - Detects more anomalies
# - Greater chance of false alarms
#
# High threshold:
# - Less sensitive
# - Fewer false alarms
# - Greater chance of missing subtle anomalies
#
# Medium threshold:
# - Balanced starting point

recommended_threshold = medium_threshold

recommended_name = "Medium (95%)"


# ============================================================
# 14. FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 60)
print("RECOMMENDED THRESHOLD")
print("=" * 60)

print(
    "Recommended threshold:",
    recommended_name
)

print(
    "Threshold value:",
    recommended_threshold
)

print(
    "Detected anomalies:",
    medium_count
)

print(
    "Anomaly percentage:",
    round(
        medium_percentage,
        2
    ),
    "%"
)

print("\nJustification:")

print(
    "The medium 95th-percentile threshold is used as the "
    "initial recommended threshold because it provides a "
    "balance between sensitivity and excessive anomaly "
    "detection. The low threshold is more sensitive but can "
    "produce more false alarms, while the high threshold is "
    "more conservative and may miss subtle anomalies."
)


# ============================================================
# 15. SAVE RECOMMENDED THRESHOLD
# ============================================================

threshold_config = pd.DataFrame({
    "Recommended Threshold": [
        recommended_threshold
    ],

    "Percentile": [
        "95%"
    ],

    "Detected Anomalies": [
        medium_count
    ]
})

threshold_config.to_csv(
    "recommended_threshold.csv",
    index=False
)


# ============================================================
# TASK 5 COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TASK 5 COMPLETED")
print("=" * 60)

print("\nGenerated files:")
print("- threshold_comparison.csv")
print("- recommended_threshold.csv")

print("\nRecommended threshold:")
print(recommended_threshold)

# ============================================================
# WEEK 5 - TASK 6
# NETWORK TRAFFIC FORECASTING USING LSTM
# ============================================================


print("\n" + "=" * 60)
print("TASK 6 - NETWORK TRAFFIC FORECASTING")
print("=" * 60)


# ============================================================
# 1. SELECT THROUGHPUT DATA
# ============================================================

# Make sure Timestamp is datetime
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

# Sort chronologically
df = df.sort_values("Timestamp").reset_index(drop=True)

print("\nFirst timestamp:")
print(df["Timestamp"].min())

print("\nLast timestamp:")
print(df["Timestamp"].max())


# ============================================================
# 2. CREATE HOURLY THROUGHPUT DATA
# ============================================================

# Set Timestamp as index temporarily
hourly_df = (
    df.set_index("Timestamp")
    .resample("1h")["Throughput"]
    .mean()
    .dropna()
    .to_frame()
)

print("\nHourly throughput data:")
print(hourly_df.head())

print("\nNumber of hourly observations:")
print(len(hourly_df))


# ============================================================
# 3. PLOT HISTORICAL HOURLY THROUGHPUT
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    hourly_df.index,
    hourly_df["Throughput"]
)

plt.title("Historical Hourly Throughput")
plt.xlabel("Time")
plt.ylabel("Average Throughput")

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 4. NORMALIZE THROUGHPUT
# ============================================================

from sklearn.preprocessing import MinMaxScaler

forecast_scaler = MinMaxScaler()

throughput_scaled = forecast_scaler.fit_transform(
    hourly_df[["Throughput"]]
)


# ============================================================
# 5. CREATE SLIDING WINDOWS
# ============================================================

sequence_length = 24

X_forecast = []
y_forecast = []

for i in range(
    len(throughput_scaled) - sequence_length
):

    sequence = throughput_scaled[
        i:i + sequence_length
    ]

    target = throughput_scaled[
        i + sequence_length
    ]

    X_forecast.append(sequence)
    y_forecast.append(target)


X_forecast = np.array(X_forecast)
y_forecast = np.array(y_forecast)


print("\nForecast input shape:")
print(X_forecast.shape)

print("\nForecast target shape:")
print(y_forecast.shape)


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(
    len(X_forecast) * 0.8
)

X_train_forecast = X_forecast[
    :split_index
]

y_train_forecast = y_forecast[
    :split_index
]

X_val_forecast = X_forecast[
    split_index:
]

y_val_forecast = y_forecast[
    split_index:
]


print("\nTraining sequences:")
print(len(X_train_forecast))

print("\nValidation sequences:")
print(len(X_val_forecast))


# ============================================================
# 7. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train_forecast,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_forecast,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val_forecast,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val_forecast,
    dtype=torch.float32
)


# ============================================================
# 8. CREATE DATALOADERS
# ============================================================

batch_size = 32

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# 9. SELECT DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:")
print(device)


# ============================================================
# 10. BUILD FORECASTING LSTM
# ============================================================

class LSTMForecaster(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )


    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        last_output = (
            lstm_output[:, -1, :]
        )

        output = self.fc(
            last_output
        )

        return output


# ============================================================
# 11. CREATE MODEL
# ============================================================

forecast_model = LSTMForecaster(
    input_size=1,
    hidden_size=64,
    num_layers=2
)

forecast_model = forecast_model.to(device)

print("\nForecasting model:")
print(forecast_model)


# ============================================================
# 12. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    forecast_model.parameters(),
    lr=0.001
)


# ============================================================
# 13. TRAINING SETTINGS
# ============================================================

epochs = 50

forecast_train_losses = []
forecast_val_losses = []


# ============================================================
# 14. TRAIN FORECASTING MODEL
# ============================================================

print("\nStarting forecasting model training...\n")


for epoch in range(epochs):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    forecast_model.train()

    total_train_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        predictions = forecast_model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()


    average_train_loss = (
        total_train_loss /
        len(train_loader)
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    forecast_model.eval()

    total_val_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = forecast_model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            total_val_loss += loss.item()


    average_val_loss = (
        total_val_loss /
        len(val_loader)
    )


    forecast_train_losses.append(
        average_train_loss
    )

    forecast_val_losses.append(
        average_val_loss
    )


    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {average_train_loss:.6f} "
            f"Validation Loss: {average_val_loss:.6f}"
        )


print("\nForecasting model training completed!")


# ============================================================
# 15. TRAINING / VALIDATION LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, epochs + 1),
    forecast_train_losses,
    label="Training Loss"
)

plt.plot(
    range(1, epochs + 1),
    forecast_val_losses,
    label="Validation Loss"
)

plt.title(
    "LSTM Traffic Forecasting - Training vs Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.legend()

plt.show()


# ============================================================
# 16. VALIDATION PREDICTIONS
# ============================================================

forecast_model.eval()

validation_predictions = []

with torch.no_grad():

    for X_batch, y_batch in val_loader:

        X_batch = X_batch.to(device)

        predictions = forecast_model(
            X_batch
        )

        validation_predictions.extend(
            predictions.cpu().numpy().flatten()
        )


validation_predictions = np.array(
    validation_predictions
)


# ============================================================
# 17. CONVERT PREDICTIONS BACK TO ORIGINAL SCALE
# ============================================================

actual_validation = forecast_scaler.inverse_transform(
    y_val_forecast.reshape(-1, 1)
).flatten()

predicted_validation = forecast_scaler.inverse_transform(
    validation_predictions.reshape(-1, 1)
).flatten()


# ============================================================
# 18. CREATE VALIDATION TIMESTAMPS
# ============================================================

validation_timestamps = hourly_df.index[
    split_index + sequence_length:
]

# Match lengths
min_length = min(
    len(validation_timestamps),
    len(actual_validation),
    len(predicted_validation)
)

validation_timestamps = (
    validation_timestamps[:min_length]
)

actual_validation = (
    actual_validation[:min_length]
)

predicted_validation = (
    predicted_validation[:min_length]
)


# ============================================================
# 19. COMPARE ACTUAL VS PREDICTED THROUGHPUT
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    validation_timestamps,
    actual_validation,
    label="Actual Throughput"
)

plt.plot(
    validation_timestamps,
    predicted_validation,
    label="Predicted Throughput"
)

plt.title(
    "Actual vs Predicted Throughput"
)

plt.xlabel("Time")
plt.ylabel("Throughput")

plt.legend()

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 20. FORECAST ACCURACY METRICS
# ============================================================

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

mae = mean_absolute_error(
    actual_validation,
    predicted_validation
)

rmse = np.sqrt(
    mean_squared_error(
        actual_validation,
        predicted_validation
    )
)

# Avoid division by zero for MAPE
non_zero_mask = actual_validation != 0

mape = np.mean(
    np.abs(
        (
            actual_validation[non_zero_mask]
            -
            predicted_validation[non_zero_mask]
        )
        /
        actual_validation[non_zero_mask]
    )
) * 100


print("\n" + "=" * 60)
print("FORECASTING VALIDATION RESULTS")
print("=" * 60)

print("\nMAE:")
print(mae)

print("\nRMSE:")
print(rmse)

print("\nMAPE:")
print(mape, "%")


# ============================================================
# 21. CREATE FUTURE FORECAST FUNCTION
# ============================================================

def forecast_future(
    model,
    last_sequence,
    future_steps,
    scaler,
    device
):

    model.eval()

    sequence = last_sequence.copy()

    future_predictions = []

    with torch.no_grad():

        for _ in range(future_steps):

            input_tensor = torch.tensor(
                sequence,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            prediction = model(
                input_tensor
            )

            prediction_value = (
                prediction.cpu()
                .numpy()
                .flatten()[0]
            )

            future_predictions.append(
                prediction_value
            )

            # Remove oldest observation
            # and add newest prediction

            sequence = np.vstack([
                sequence[1:],
                [[prediction_value]]
            ])


    future_predictions = np.array(
        future_predictions
    )

    future_predictions = scaler.inverse_transform(
        future_predictions.reshape(-1, 1)
    ).flatten()

    return future_predictions


# ============================================================
# 22. PREPARE LAST 24 HOURS
# ============================================================

last_sequence = throughput_scaled[
    -sequence_length:
]


# ============================================================
# 23. FORECAST NEXT 24 HOURS
# ============================================================

forecast_24_hours = forecast_future(
    forecast_model,
    last_sequence,
    24,
    forecast_scaler,
    device
)


future_24_timestamps = pd.date_range(
    start=hourly_df.index[-1]
    + pd.Timedelta(hours=1),
    periods=24,
    freq="1h"
)


print("\nNext 24-hour forecast:")
print(
    pd.DataFrame({
        "Timestamp": future_24_timestamps,
        "Predicted_Throughput": forecast_24_hours
    })
)


# ============================================================
# 24. PLOT NEXT 24 HOURS
# ============================================================

plt.figure(figsize=(14, 6))

# Last 48 historical hours
historical_48 = hourly_df.iloc[-48:]

plt.plot(
    historical_48.index,
    historical_48["Throughput"],
    label="Historical Throughput"
)

plt.plot(
    future_24_timestamps,
    forecast_24_hours,
    label="24-Hour Forecast"
)

plt.title(
    "Next 24 Hours - Throughput Forecast"
)

plt.xlabel("Time")
plt.ylabel("Throughput")

plt.legend()

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 25. FORECAST NEXT 7 DAYS
# ============================================================

future_steps_7_days = 24 * 7

forecast_7_days = forecast_future(
    forecast_model,
    last_sequence,
    future_steps_7_days,
    forecast_scaler,
    device
)


future_7_day_timestamps = pd.date_range(
    start=hourly_df.index[-1]
    + pd.Timedelta(hours=1),
    periods=future_steps_7_days,
    freq="1h"
)


# ============================================================
# 26. DISPLAY 7-DAY FORECAST
# ============================================================

forecast_7_day_df = pd.DataFrame({

    "Timestamp":
        future_7_day_timestamps,

    "Predicted_Throughput":
        forecast_7_days

})

print("\nNext 7-day forecast:")
print(
    forecast_7_day_df.head(20)
)


# ============================================================
# 27. PLOT NEXT 7 DAYS
# ============================================================

plt.figure(figsize=(14, 6))

historical_7_days = hourly_df.iloc[-168:]

plt.plot(
    historical_7_days.index,
    historical_7_days["Throughput"],
    label="Historical Throughput"
)

plt.plot(
    future_7_day_timestamps,
    forecast_7_days,
    label="7-Day Forecast"
)

plt.title(
    "Next 7 Days - Throughput Forecast"
)

plt.xlabel("Time")
plt.ylabel("Throughput")

plt.legend()

plt.xticks(rotation=45)

plt.show()


# ============================================================
# 28. SAVE 24-HOUR FORECAST
# ============================================================

forecast_24_df = pd.DataFrame({

    "Timestamp":
        future_24_timestamps,

    "Predicted_Throughput":
        forecast_24_hours

})

forecast_24_df.to_csv(
    "throughput_forecast_24_hours.csv",
    index=False
)


# ============================================================
# 29. SAVE 7-DAY FORECAST
# ============================================================

forecast_7_day_df.to_csv(
    "throughput_forecast_7_days.csv",
    index=False
)


# ============================================================
# 30. SAVE FORECASTING MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            forecast_model.state_dict(),

        "sequence_length":
            sequence_length,

        "hidden_size":
            64,

        "num_layers":
            2
    },
    "lstm_traffic_forecasting_model.pt"
)


# ============================================================
# 31. FINAL TASK 6 SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TASK 6 COMPLETED")
print("=" * 60)

print("\nModel: LSTM")

print(
    "\nHistorical hourly observations:",
    len(hourly_df)
)

print(
    "\nSequence length:",
    sequence_length,
    "hours"
)

print(
    "\n24-hour forecast points:",
    len(forecast_24_hours)
)

print(
    "\n7-day forecast points:",
    len(forecast_7_days)
)

print("\nValidation MAE:")
print(mae)

print("\nValidation RMSE:")
print(rmse)

print("\nValidation MAPE:")
print(mape, "%")

print("\nGenerated files:")
print("- throughput_forecast_24_hours.csv")
print("- throughput_forecast_7_days.csv")
print("- lstm_traffic_forecasting_model.pt")

print("\nTask 6 completed successfully!")

# ============================================================
# WEEK 5 - TASK 7
# ROOT CAUSE ANALYSIS (RCA)
# ============================================================

print("\n" + "=" * 60)
print("TASK 7 - ROOT CAUSE ANALYSIS")
print("=" * 60)


# ============================================================
# 1. DEFINE RCA THRESHOLDS
# ============================================================

# These are operational thresholds selected for this project.
# They are NOT numerical thresholds provided by the PDF.

RSRP_THRESHOLD = -105
SINR_THRESHOLD = 10
LATENCY_THRESHOLD = 80
PACKET_LOSS_THRESHOLD = 3
THROUGHPUT_THRESHOLD = 40


print("\nRCA thresholds:")

print("RSRP <", RSRP_THRESHOLD)
print("SINR <", SINR_THRESHOLD)
print("Latency >", LATENCY_THRESHOLD)
print("Packet Loss >", PACKET_LOSS_THRESHOLD)
print("Throughput <", THROUGHPUT_THRESHOLD)


# ============================================================
# 2. SELECT ANOMALOUS RECORDS
# ============================================================

# Use Isolation Forest anomalies from Task 3

anomalies = df[
    df["Anomaly"] == -1
].copy()

print("\nTotal anomalous records:")
print(len(anomalies))


# ============================================================
# 3. RCA FUNCTION
# ============================================================

def determine_root_cause(row):

    observations = []
    causes = []
    actions = []


    # --------------------------------------------------------
    # LOW RSRP
    # --------------------------------------------------------

    if row["RSRP"] < RSRP_THRESHOLD:

        observations.append(
            "Low RSRP"
        )

        causes.append(
            "Poor radio coverage"
        )

        actions.append(
            "Optimize antenna tilt"
        )


    # --------------------------------------------------------
    # LOW SINR
    # --------------------------------------------------------

    if row["SINR"] < SINR_THRESHOLD:

        observations.append(
            "Low SINR"
        )

        causes.append(
            "High interference"
        )

        actions.append(
            "Review neighboring cell configuration"
        )


    # --------------------------------------------------------
    # HIGH LATENCY
    # --------------------------------------------------------

    if row["Latency"] > LATENCY_THRESHOLD:

        observations.append(
            "High Latency"
        )

        causes.append(
            "Core network congestion"
        )

        actions.append(
            "Check UPF utilization"
        )


    # --------------------------------------------------------
    # HIGH PACKET LOSS
    # --------------------------------------------------------

    if row["Packet_Loss"] > PACKET_LOSS_THRESHOLD:

        observations.append(
            "High Packet Loss"
        )

        causes.append(
            "Backhaul issues"
        )

        actions.append(
            "Inspect transport network"
        )


    # --------------------------------------------------------
    # LOW THROUGHPUT
    # --------------------------------------------------------

    if row["Throughput"] < THROUGHPUT_THRESHOLD:

        observations.append(
            "Low Throughput"
        )

        causes.append(
            "Heavy traffic load"
        )

        actions.append(
            "Enable load balancing"
        )


    # --------------------------------------------------------
    # NO MATCHING CONDITION
    # --------------------------------------------------------

    if len(observations) == 0:

        observations.append(
            "Unusual KPI combination"
        )

        causes.append(
            "Requires further investigation"
        )

        actions.append(
            "Review all network KPIs"
        )


    return pd.Series({

        "Observation":
            ", ".join(observations),

        "Possible_Root_Cause":
            ", ".join(causes),

        "Recommended_Action":
            ", ".join(actions)

    })


# ============================================================
# 4. APPLY RCA TO ANOMALIES
# ============================================================

rca_results = anomalies.apply(
    determine_root_cause,
    axis=1
)


# ============================================================
# 5. COMBINE RCA WITH ANOMALY DATA
# ============================================================

rca_report = pd.concat(
    [
        anomalies[
            [
                "Timestamp",
                "Cell_ID",
                "RSRP",
                "SINR",
                "Latency",
                "Throughput",
                "Packet_Loss",
                "Connected_Users"
            ]
        ].reset_index(drop=True),

        rca_results.reset_index(drop=True)

    ],
    axis=1
)


# ============================================================
# 6. DISPLAY RCA REPORT
# ============================================================

print("\n" + "=" * 60)
print("RCA REPORT")
print("=" * 60)

print(
    rca_report.head(20).to_string(
        index=False
    )
)


# ============================================================
# 7. COUNT ROOT CAUSES
# ============================================================

print("\n" + "=" * 60)
print("ROOT CAUSE FREQUENCY")
print("=" * 60)

root_cause_counts = (
    rca_report["Possible_Root_Cause"]
    .value_counts()
)

print(root_cause_counts)


# ============================================================
# 8. PLOT ROOT CAUSE FREQUENCY
# ============================================================

plt.figure(figsize=(10, 6))

root_cause_counts.plot(
    kind="bar"
)

plt.title(
    "Root Cause Frequency"
)

plt.xlabel(
    "Possible Root Cause"
)

plt.ylabel(
    "Number of Anomalies"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.show()


# ============================================================
# 9. SAVE RCA REPORT
# ============================================================

rca_report.to_csv(
    "rca_report.csv",
    index=False
)

print(
    "\nRCA report saved as:"
)

print("rca_report.csv")


# ============================================================
# 10. SAVE ROOT CAUSE SUMMARY
# ============================================================

root_cause_summary = (
    rca_report[
        [
            "Observation",
            "Possible_Root_Cause",
            "Recommended_Action"
        ]
    ]
    .value_counts()
    .reset_index(
        name="Count"
    )
)

root_cause_summary.to_csv(
    "root_cause_summary.csv",
    index=False
)

print(
    "\nRoot cause summary saved as:"
)

print("root_cause_summary.csv")


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TASK 7 COMPLETED")
print("=" * 60)

print(
    "\nAnomalies analyzed:",
    len(rca_report)
)

print(
    "\nUnique root-cause combinations:",
    len(root_cause_counts)
)

print("\nGenerated files:")
print("- rca_report.csv")
print("- root_cause_summary.csv")

print("\nTask 7 completed successfully!")

# ============================================================
# WEEK 5 - TASK 8
# QoS / QoE PREDICTION
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib


print("\n" + "=" * 60)
print("TASK 8 - QoS / QoE PREDICTION")
print("=" * 60)


# ============================================================
# 1. SELECT FEATURES
# ============================================================

qos_features = [
    "RSRP",
    "SINR",
    "Latency",
    "Packet_Loss",
    "Throughput"
]

print("\nFeatures used:")
print(qos_features)


# ============================================================
# 2. CREATE COPY OF DATA
# ============================================================

qos_df = df[
    qos_features
].copy()


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before handling:")

print(qos_df.isnull().sum())


qos_df = qos_df.fillna(
    qos_df.median()
)


print("\nMissing values after handling:")

print(qos_df.isnull().sum())


# ============================================================
# 4. CREATE QoS SCORE
# ============================================================

# We create a simple rule-based quality score.
#
# Each KPI gets a score from 0 to 2.
#
# 2 = Good
# 1 = Fair
# 0 = Poor


def calculate_qos_score(row):

    score = 0


    # --------------------------------------------------------
    # RSRP
    # --------------------------------------------------------

    if row["RSRP"] >= -90:
        score += 2

    elif row["RSRP"] >= -105:
        score += 1


    # --------------------------------------------------------
    # SINR
    # --------------------------------------------------------

    if row["SINR"] >= 15:
        score += 2

    elif row["SINR"] >= 10:
        score += 1


    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    if row["Latency"] <= 40:
        score += 2

    elif row["Latency"] <= 80:
        score += 1


    # --------------------------------------------------------
    # PACKET LOSS
    # --------------------------------------------------------

    if row["Packet_Loss"] <= 1:
        score += 2

    elif row["Packet_Loss"] <= 3:
        score += 1


    # --------------------------------------------------------
    # THROUGHPUT
    # --------------------------------------------------------

    if row["Throughput"] >= 50:
        score += 2

    elif row["Throughput"] >= 25:
        score += 1


    return score


qos_df["QoS_Score"] = qos_df.apply(
    calculate_qos_score,
    axis=1
)


# ============================================================
# 5. CONVERT SCORE INTO GOOD / FAIR / POOR
# ============================================================

def classify_qos(score):

    if score >= 8:
        return "Good"

    elif score >= 5:
        return "Fair"

    else:
        return "Poor"


qos_df["QoS"] = qos_df[
    "QoS_Score"
].apply(classify_qos)


print("\nQoS class distribution:")

print(
    qos_df["QoS"].value_counts()
)


# ============================================================
# 6. PREPARE FEATURES AND TARGET
# ============================================================

X = qos_df[
    qos_features
]

y = qos_df[
    "QoS"
]


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# ============================================================
# 8. CREATE RANDOM FOREST CLASSIFIER
# ============================================================

qos_model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    class_weight="balanced"
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

qos_model.fit(
    X_train,
    y_train
)


print("\nQoS model training completed!")


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

y_pred = qos_model.predict(
    X_test
)


# ============================================================
# 11. CALCULATE PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# 12. DISPLAY PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("QoS MODEL PERFORMANCE")
print("=" * 60)

print(
    "\nAccuracy:",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "Good",
        "Fair",
        "Poor"
    ]
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 15. CONFUSION MATRIX VISUALIZATION
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Good",
        "Fair",
        "Poor"
    ]
)

disp.plot()

plt.title(
    "QoS Prediction Confusion Matrix"
)

plt.show()


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": qos_features,

    "Importance":
        qos_model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


print("\nFeature Importance:")

print(
    feature_importance
)


# ============================================================
# 17. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title(
    "QoS Model Feature Importance"
)

plt.xlabel(
    "KPI"
)

plt.ylabel(
    "Importance"
)

plt.xticks(
    rotation=45
)

plt.show()


# ============================================================
# 18. SAVE QoS PREDICTIONS
# ============================================================

qos_predictions = X_test.copy()

qos_predictions["Actual_QoS"] = (
    y_test.values
)

qos_predictions["Predicted_QoS"] = (
    y_pred
)


qos_predictions.to_csv(
    "qos_predictions.csv",
    index=False
)


# ============================================================
# 19. CREATE PERFORMANCE REPORT
# ============================================================

performance_report = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]

})


print("\nPerformance Report:")

print(
    performance_report.to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE PERFORMANCE REPORT
# ============================================================

performance_report.to_csv(
    "qos_performance_report.csv",
    index=False
)


# ============================================================
# 21. SAVE MODEL
# ============================================================

joblib.dump(
    qos_model,
    "qos_prediction_model.pkl"
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TASK 8 COMPLETED")
print("=" * 60)

print("\nQoS Classes:")
print(
    qos_df["QoS"].value_counts()
)

print("\nModel: Random Forest")

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(f1 * 100, 2),
    "%"
)

print("\nGenerated files:")
print("- qos_prediction_model.pkl")
print("- qos_predictions.csv")
print("- qos_performance_report.csv")

print("\nTask 8 completed successfully!")


# ============================================================
# WEEK 5 - TASK 9
# DATA LABELING STRATEGY
# ============================================================

print("\n" + "=" * 60)
print("TASK 9 - DATA LABELING STRATEGY")
print("=" * 60)


# ============================================================
# 1. DEFINE KPI LABELING RULES
# ============================================================

# Rules are taken directly from the assignment.
#
# RSRP:
# Normal   > -90
# Warning  -90 to -105
# Critical < -105
#
# SINR:
# Normal   > 20
# Warning  10 to 20
# Critical < 10
#
# Latency:
# Normal   < 20
# Warning  20 to 50
# Critical > 50
#
# Packet Loss:
# Normal   < 1
# Warning  1 to 3
# Critical > 3


# ============================================================
# 2. RSRP LABELING FUNCTION
# ============================================================

def label_rsrp(value):

    if value > -90:
        return "Normal"

    elif value >= -105:
        return "Warning"

    else:
        return "Critical"


# ============================================================
# 3. SINR LABELING FUNCTION
# ============================================================

def label_sinr(value):

    if value > 20:
        return "Normal"

    elif value >= 10:
        return "Warning"

    else:
        return "Critical"


# ============================================================
# 4. LATENCY LABELING FUNCTION
# ============================================================

def label_latency(value):

    if value < 20:
        return "Normal"

    elif value <= 50:
        return "Warning"

    else:
        return "Critical"


# ============================================================
# 5. PACKET LOSS LABELING FUNCTION
# ============================================================

def label_packet_loss(value):

    if value < 1:
        return "Normal"

    elif value <= 3:
        return "Warning"

    else:
        return "Critical"


# ============================================================
# 6. CREATE KPI-SPECIFIC LABELS
# ============================================================

label_df = df.copy()


label_df["RSRP_Status"] = (
    label_df["RSRP"]
    .apply(label_rsrp)
)


label_df["SINR_Status"] = (
    label_df["SINR"]
    .apply(label_sinr)
)


label_df["Latency_Status"] = (
    label_df["Latency"]
    .apply(label_latency)
)


label_df["Packet_Loss_Status"] = (
    label_df["Packet_Loss"]
    .apply(label_packet_loss)
)


# ============================================================
# 7. CREATE OVERALL KPI STATUS
# ============================================================

def overall_status(row):

    statuses = [
        row["RSRP_Status"],
        row["SINR_Status"],
        row["Latency_Status"],
        row["Packet_Loss_Status"]
    ]


    # If any KPI is Critical
    if "Critical" in statuses:
        return "Critical"


    # Otherwise, if any KPI is Warning
    elif "Warning" in statuses:
        return "Warning"


    # Otherwise everything is Normal
    else:
        return "Normal"


label_df["Network_Status"] = (
    label_df.apply(
        overall_status,
        axis=1
    )
)


# ============================================================
# 8. DISPLAY SAMPLE LABELS
# ============================================================

print("\nSample labeled records:")

print(
    label_df[
        [
            "Timestamp",
            "Cell_ID",
            "RSRP",
            "RSRP_Status",
            "SINR",
            "SINR_Status",
            "Latency",
            "Latency_Status",
            "Packet_Loss",
            "Packet_Loss_Status",
            "Network_Status"
        ]
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# 9. COUNT EACH KPI STATUS
# ============================================================

print("\n" + "=" * 60)
print("KPI STATUS DISTRIBUTION")
print("=" * 60)


print("\nRSRP:")
print(
    label_df["RSRP_Status"].value_counts()
)


print("\nSINR:")
print(
    label_df["SINR_Status"].value_counts()
)


print("\nLatency:")
print(
    label_df["Latency_Status"].value_counts()
)


print("\nPacket Loss:")
print(
    label_df["Packet_Loss_Status"].value_counts()
)


# ============================================================
# 10. OVERALL NETWORK STATUS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("OVERALL NETWORK STATUS")
print("=" * 60)

print(
    label_df["Network_Status"]
    .value_counts()
)


# ============================================================
# 11. PERCENTAGE OF EACH NETWORK STATUS
# ============================================================

status_percentage = (
    label_df["Network_Status"]
    .value_counts(
        normalize=True
    ) * 100
)


print("\nNetwork status percentages:")

print(
    status_percentage
)


# ============================================================
# 12. VISUALIZE OVERALL STATUS
# ============================================================

plt.figure(figsize=(8, 5))

label_df["Network_Status"].value_counts().plot(
    kind="bar"
)

plt.title(
    "Network Status Distribution"
)

plt.xlabel(
    "Network Status"
)

plt.ylabel(
    "Number of Records"
)

plt.xticks(
    rotation=0
)

plt.show()


# ============================================================
# 13. SAVE LABELED DATASET
# ============================================================

label_df.to_csv(
    "labeled_telecom_kpi.csv",
    index=False
)

print(
    "\nLabeled dataset saved as:"
)

print("labeled_telecom_kpi.csv")


# ============================================================
# 14. CREATE LABELING RULE DOCUMENT
# ============================================================

labeling_rules = pd.DataFrame({

    "KPI": [
        "RSRP",
        "SINR",
        "Latency",
        "Packet Loss"
    ],

    "Normal": [
        "> -90 dBm",
        "> 20 dB",
        "< 20 ms",
        "< 1%"
    ],

    "Warning": [
        "-90 to -105 dBm",
        "10 to 20 dB",
        "20 to 50 ms",
        "1 to 3%"
    ],

    "Critical": [
        "< -105 dBm",
        "< 10 dB",
        "> 50 ms",
        "> 3%"
    ]
})


print("\n" + "=" * 60)
print("LABELING RULES")
print("=" * 60)

print(
    labeling_rules.to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE LABELING RULES
# ============================================================

labeling_rules.to_csv(
    "kpi_labeling_rules.csv",
    index=False
)

print(
    "\nLabeling rules saved as:"
)

print("kpi_labeling_rules.csv")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TASK 9 COMPLETED")
print("=" * 60)

print("\nLabels created:")
print("- Normal")
print("- Warning")
print("- Critical")

print("\nGenerated files:")
print("- labeled_telecom_kpi.csv")
print("- kpi_labeling_rules.csv")

print("\nTask 9 completed successfully!")

# ============================================================
# WEEK 5 - TASK 9
# DATA LABELING STRATEGY
# ============================================================

print("\n" + "=" * 60)
print("TASK 9 - DATA LABELING STRATEGY")
print("=" * 60)


# ============================================================
# 1. DEFINE KPI LABELING RULES
# ============================================================

# Rules are taken directly from the assignment.
#
# RSRP:
# Normal   > -90
# Warning  -90 to -105
# Critical < -105
#
# SINR:
# Normal   > 20
# Warning  10 to 20
# Critical < 10
#
# Latency:
# Normal   < 20
# Warning  20 to 50
# Critical > 50
#
# Packet Loss:
# Normal   < 1
# Warning  1 to 3
# Critical > 3


# ============================================================
# 2. RSRP LABELING FUNCTION
# ============================================================

def label_rsrp(value):

    if value > -90:
        return "Normal"

    elif value >= -105:
        return "Warning"

    else:
        return "Critical"


# ============================================================
# 3. SINR LABELING FUNCTION
# ============================================================

def label_sinr(value):

    if value > 20:
        return "Normal"

    elif value >= 10:
        return "Warning"

    else:
        return "Critical"


# ============================================================
# 4. LATENCY LABELING FUNCTION
# ============================================================

def label_latency(value):

    if value < 20:
        return "Normal"

    elif value <= 50:
        return "Warning"

    else:
        return "Critical"


# ============================================================
# 5. PACKET LOSS LABELING FUNCTION
# ============================================================

def label_packet_loss(value):

    if value < 1:
        return "Normal"

    elif value <= 3:
        return "Warning"

    else:
        return "Critical"


# ============================================================
# 6. CREATE KPI-SPECIFIC LABELS
# ============================================================

label_df = df.copy()


label_df["RSRP_Status"] = (
    label_df["RSRP"]
    .apply(label_rsrp)
)


label_df["SINR_Status"] = (
    label_df["SINR"]
    .apply(label_sinr)
)


label_df["Latency_Status"] = (
    label_df["Latency"]
    .apply(label_latency)
)


label_df["Packet_Loss_Status"] = (
    label_df["Packet_Loss"]
    .apply(label_packet_loss)
)


# ============================================================
# 7. CREATE OVERALL KPI STATUS
# ============================================================

def overall_status(row):

    statuses = [
        row["RSRP_Status"],
        row["SINR_Status"],
        row["Latency_Status"],
        row["Packet_Loss_Status"]
    ]


    # If any KPI is Critical
    if "Critical" in statuses:
        return "Critical"


    # Otherwise, if any KPI is Warning
    elif "Warning" in statuses:
        return "Warning"


    # Otherwise everything is Normal
    else:
        return "Normal"


label_df["Network_Status"] = (
    label_df.apply(
        overall_status,
        axis=1
    )
)


# ============================================================
# 8. DISPLAY SAMPLE LABELS
# ============================================================

print("\nSample labeled records:")

print(
    label_df[
        [
            "Timestamp",
            "Cell_ID",
            "RSRP",
            "RSRP_Status",
            "SINR",
            "SINR_Status",
            "Latency",
            "Latency_Status",
            "Packet_Loss",
            "Packet_Loss_Status",
            "Network_Status"
        ]
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# 9. COUNT EACH KPI STATUS
# ============================================================

print("\n" + "=" * 60)
print("KPI STATUS DISTRIBUTION")
print("=" * 60)


print("\nRSRP:")
print(
    label_df["RSRP_Status"].value_counts()
)


print("\nSINR:")
print(
    label_df["SINR_Status"].value_counts()
)


print("\nLatency:")
print(
    label_df["Latency_Status"].value_counts()
)


print("\nPacket Loss:")
print(
    label_df["Packet_Loss_Status"].value_counts()
)


# ============================================================
# 10. OVERALL NETWORK STATUS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("OVERALL NETWORK STATUS")
print("=" * 60)

print(
    label_df["Network_Status"]
    .value_counts()
)


# ============================================================
# 11. PERCENTAGE OF EACH NETWORK STATUS
# ============================================================

status_percentage = (
    label_df["Network_Status"]
    .value_counts(
        normalize=True
    ) * 100
)


print("\nNetwork status percentages:")

print(
    status_percentage
)


# ============================================================
# 12. VISUALIZE OVERALL STATUS
# ============================================================

plt.figure(figsize=(8, 5))

label_df["Network_Status"].value_counts().plot(
    kind="bar"
)

plt.title(
    "Network Status Distribution"
)

plt.xlabel(
    "Network Status"
)

plt.ylabel(
    "Number of Records"
)

plt.xticks(
    rotation=0
)

plt.show()


# ============================================================
# 13. SAVE LABELED DATASET
# ============================================================

label_df.to_csv(
    "labeled_telecom_kpi.csv",
    index=False
)

print(
    "\nLabeled dataset saved as:"
)

print("labeled_telecom_kpi.csv")


# ============================================================
# 14. CREATE LABELING RULE DOCUMENT
# ============================================================

labeling_rules = pd.DataFrame({

    "KPI": [
        "RSRP",
        "SINR",
        "Latency",
        "Packet Loss"
    ],

    "Normal": [
        "> -90 dBm",
        "> 20 dB",
        "< 20 ms",
        "< 1%"
    ],

    "Warning": [
        "-90 to -105 dBm",
        "10 to 20 dB",
        "20 to 50 ms",
        "1 to 3%"
    ],

    "Critical": [
        "< -105 dBm",
        "< 10 dB",
        "> 50 ms",
        "> 3%"
    ]
})


print("\n" + "=" * 60)
print("LABELING RULES")
print("=" * 60)

print(
    labeling_rules.to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE LABELING RULES
# ============================================================

labeling_rules.to_csv(
    "kpi_labeling_rules.csv",
    index=False
)

print(
    "\nLabeling rules saved as:"
)

print("kpi_labeling_rules.csv")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TASK 9 COMPLETED")
print("=" * 60)

print("\nLabels created:")
print("- Normal")
print("- Warning")
print("- Critical")

print("\nGenerated files:")
print("- labeled_telecom_kpi.csv")
print("- kpi_labeling_rules.csv")

print("\nTask 9 completed successfully!")


