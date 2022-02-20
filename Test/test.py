import pickle
import pandas as pd
from Seoul_Bike_Prediction_Model import seoulbikedemand_prediction

df = pd.read_csv("SeoulBikeData.csv")

model = pickle.load(open("rf_model", "rb"))

# Generate some data for validation
X_test, y = seoulbikedemand_prediction(1000,n_features = 8)

# Test on the model
y_hat = model.predict(X_test)