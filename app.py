import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template


with open("xgb_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


with open("label_encoders.pkl", "rb") as enc_file:
    label_encoders = pickle.load(enc_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('flight.html', predicted_price=None)

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    departure = form_data['departure']
    destination = form_data['destination']
    airline = form_data['airline']
    flight_class = form_data['flightClass']
    departure_date = form_data['departureDate']

    
    departure_date = pd.to_datetime(departure_date).dayofweek


    try:
        departure_encoded = label_encoders['departure'].transform([departure])[0]
        destination_encoded = label_encoders['destination'].transform([destination])[0]
        airline_encoded = label_encoders['airline'].transform([airline])[0]
        class_encoded = label_encoders['flightClass'].transform([flight_class])[0]
    except KeyError:
        return render_template('flight.html', predicted_price="Invalid input! Check values.")

   
    input_data = np.array([[departure_encoded, destination_encoded, airline_encoded, class_encoded, departure_date]])
    
   
    predicted_price = model.predict(input_data)[0]
    
    return render_template('flight.html', predicted_price=round(predicted_price, 2))

if __name__ == '__main__':
    app.run(debug=True)




 