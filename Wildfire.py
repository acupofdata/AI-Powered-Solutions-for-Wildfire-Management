

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('wildfire_dataset.csv')

# Data Cleaning
data['daynight'] = data['daynight'].map({"D": 1, "N": 0})
data['satellite'] = data['satellite'].map({"Terra": 1, "Aqua": 0})

# Feature Engineering
data['year'] = pd.to_datetime(data['acq_date']).dt.year

# Train-Test Split
X = data.drop(columns=['confidence', 'acq_date'])
y = data['confidence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model Building
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Confusion Matrix Plot
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()