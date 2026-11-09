import pickle

with open('titanic_final_model.pkl', 'rb') as file:
    model = pickle.load(file)

print("Model loaded successfully!")
print("Features:", model.feature_names_in_)