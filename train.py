import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Loading the data
df = pd.read_csv('health_data.csv')  

# 2. Separate features and target
X = df.copy()
y = X.pop('Disease')  # Disease is the column we want to predict

# 3. Encode categorical variables
label_encoders = {}
for col in ['Gender', 'Smoking', 'FamilyHistory']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

disease_le = LabelEncoder()
y = disease_le.fit_transform(y)

# 4. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# 5. Train a classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7. Save the trained model and label encoders
with open('diabetes_heartdisease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({"label_encoders": label_encoders, "disease_le": disease_le}, f)

print("Training complete. Model and label encoders saved.")