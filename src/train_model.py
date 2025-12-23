import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Create dummy data (Weight in lbs -> Height in inches)
# Pattern: Height roughly = weight / 3.5 + 20 (just for demo purposes)
X = np.array([[150], [170], [200], [220], [250]]) # Weights
y = np.array([63, 68, 77, 82, 91])                # Heights

# 2 Train model
model = LinearRegression()
model.fit(X, y)

# Save the model to a file
joblib.dump(model, 'model.joblib')
print("Model trained and saved as model.joblib")