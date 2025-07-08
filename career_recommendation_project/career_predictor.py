import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# 📥 Load dataset
df = pd.read_csv("career_dataset.csv")

# 🛠️ Feature Engineering
df['Academic_Aptitude'] = (df['Numerical'] + df['Logical']) / 2
df['Language_Aptitude'] = df['English']
df['Personality_Level'] = pd.cut(df['Personality'], bins=[0, 3, 7, 10], labels=['low', 'medium', 'high'])

# 🔠 Encode categorical features
categorical_cols = [
    "Stream", "Activity", "Interest1", "Interest2", "Skill1", "Skill2",
    "Subject1", "Subject2", "PreferredEnv", "StudyStyle", "Personality_Level"
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    joblib.dump(le, f"{col.lower()}_encoder.pkl")  # Saves as stream_encoder.pkl, etc.

# 🎯 Encode target
career_encoder = LabelEncoder()
df["Career"] = career_encoder.fit_transform(df["Career"])
joblib.dump(career_encoder, "career_label_encoder.pkl")

# ✅ Define final features
features = [
    "Stream", "Activity", "Interest1", "Interest2", "Skill1", "Skill2",
    "Subject1", "Subject2", "PreferredEnv", "StudyStyle",
    "Numerical", "Logical", "English", "Personality",
    "Academic_Aptitude", "Language_Aptitude", "Personality_Level"
]

X = df[features]
y = df["Career"]

# 💾 Save input column order
joblib.dump(X.columns.tolist(), "model_input_columns.pkl")

# 🔀 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train XGBoost with calibration
xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
model = CalibratedClassifierCV(xgb, method='sigmoid', cv=5)
model.fit(X_train, y_train)

# 📊 Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ XGBoost (Calibrated) Accuracy: {accuracy * 100:.2f}%")

# 💾 Save the trained model
joblib.dump(model, "best_model.pkl")
print("✅ Saved: best_model.pkl")

# 📃 Classification report
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=career_encoder.classes_))
