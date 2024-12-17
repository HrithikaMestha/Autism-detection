import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load your dataset
data = pd.read_csv('data.csv')

# Handle missing values (though the dataset description says there are none)
data = data.dropna()
print(data.columns)

# Convert 'No' to 0 and 'Yes' to 1 in the label column
data['Class/ASD Traits'] = data['Class/ASD Traits'].map({'No': 0, 'Yes': 1})

# Separate features and labels
X = data.drop('Class/ASD Traits', axis=1)  # Corrected column name
y = data['Class/ASD Traits']  # Corrected column name

# Encode categorical variables if needed
X = pd.get_dummies(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
