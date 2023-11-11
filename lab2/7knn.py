import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("data.csv")

y = df.iloc[:, -1] # y is the last column
X = df.iloc[:, :-1]

#split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.3, random_state=42, shuffle=True)

# Instantiate the K-Nearest Neighbors classifier
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model on the training data
knn_model.fit(X_train, y_train)

# Get predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the performance of the model
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()