import numpy as np
from sklearn.linear_model import LinearRegression

# Input data (2 independent variables)
# Example: [study_hours, sleep_hours]
X = np.array([
    [2, 7],
    [3, 8],
    [4, 6],
    [5, 9],
    [6, 7]
])

# Output data (marks)
y = np.array([50, 60, 65, 80, 85])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict new value
# Example: study_hours=4, sleep_hours=8
prediction = model.predict([[4, 8]])

print("Predicted Marks:", prediction)