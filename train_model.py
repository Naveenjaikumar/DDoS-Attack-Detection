import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==============================
# 1. Load Dataset
# ==============================
data = pd.read_csv("data/cicddos.csv", low_memory=False)

# 🔥 IMPORTANT FIX: clean column names
data.columns = data.columns.str.strip()

print("✅ Columns loaded:")
print(list(data.columns))

# ==============================
# 2. MAIN 14 FEATURES (MATCHING DATASET)
# ==============================
FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std'
]

LABEL_COLUMN = 'Label'

# ==============================
# 3. Verify Columns Exist
# ==============================
missing = [col for col in FEATURES + [LABEL_COLUMN] if col not in data.columns]

if missing:
    print("❌ Missing columns detected:")
    for col in missing:
        print("   -", col)
    raise ValueError("Fix column names in FEATURES list")

# ==============================
# 4. Select Required Columns
# ==============================
data = data[FEATURES + [LABEL_COLUMN]]

# ==============================
# 5. Data Cleaning
# ==============================
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

# Convert labels to binary
data[LABEL_COLUMN] = data[LABEL_COLUMN].apply(
    lambda x: 0 if x.upper() == 'BENIGN' else 1
)

X = data[FEATURES]
y = data[LABEL_COLUMN]

# ==============================
# 6. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 7. Feature Scaling
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 8. Neural Network
# ==============================
model = Sequential([
    Dense(64, activation='relu', input_shape=(14,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# 9. Train Model
# ==============================
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

# ==============================
# 10. Save Model
# ==============================
model.save("ddos_model.h5")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ MODEL TRAINED SUCCESSFULLY")
print("✅ Model saved as 'ddos_model.h5'")
print("✅ Scaler saved as 'scaler.pkl'")