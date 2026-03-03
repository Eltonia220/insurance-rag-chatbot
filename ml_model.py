import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ---- LOAD DATA ----
df = pd.read_csv("Insurance claims data.csv")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ---- PREPARE DATA ----
# Drop policy_id as it's not useful for prediction
df = df.drop(columns=["policy_id"])

# Convert text columns to numbers
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Split into features and target
X = df.drop(columns=["claim_status"])
y = df["claim_status"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---- TRAIN MODEL ----
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- EVALUATE MODEL ----
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- MODEL RESULTS ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---- FEATURE IMPORTANCE ----
print("\nTop 5 Most Important Features:")
importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.nlargest(5))
py