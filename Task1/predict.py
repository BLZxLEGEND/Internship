import pickle
import numpy as np

# Load the trained model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Enter the feature values:")

MedInc = float(input("Median Income: "))
HouseAge = float(input("House Age: "))
AveRooms = float(input("Average Rooms: "))
AveBedrms = float(input("Average Bedrooms: "))
Population = float(input("Population: "))
AveOccup = float(input("Average Occupancy: "))
Latitude = float(input("Latitude: "))
Longitude = float(input("Longitude: "))

features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                      Population, AveOccup, Latitude, Longitude]])

prediction = model.predict(features)

print(f"\nPredicted House Value: {prediction[0]:.4f}")