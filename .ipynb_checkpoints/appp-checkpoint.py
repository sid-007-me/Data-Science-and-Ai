import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))

st.title("🚗 Car Price Prediction ML Model")

cars_data = pd.read_csv('cardetails.csv')

def get_brand_name(car_name):
    return car_name.split(' ')[0].strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand', sorted(cars_data['name'].unique()))
year = st.slider('Car Manufactured Year', 1994, 2026, )
km_driven = st.slider('Kilometers Driven', 0, 300000, )
fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission Type', cars_data['transmission'].unique())
owner = st.selectbox('Owner Type', cars_data['owner'].unique())
mileage = st.slider('Mileage (km/l)', 10, 40, )
engine = st.slider('Engine (CC)', 700, 5000, )
max_power = st.slider('Max Power (BHP)', 20, 300, )
seats = st.slider('Number of Seats', 2, 10, )

if st.button("Predict Price"):

    input_data_model = pd.DataFrame({
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]
    })

    owner_map = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4,
        'Test Drive Car': 5
    }

    fuel_map = {
        'Diesel': 1,
        'Petrol': 2,
        'LPG': 3,
        'CNG': 4
    }

    seller_map = {
        'Individual': 1,
        'Dealer': 2,
        'Trustmark Dealer': 3
    }

    transmission_map = {
        'Manual': 1,
        'Automatic': 2
    }

    brand_map = {
        'Maruti':1,'Skoda':2,'Honda':3,'Hyundai':4,'Toyota':5,
        'Ford':6,'Renault':7,'Mahindra':8,'Tata':9,'Chevrolet':10,
        'Datsun':11,'Jeep':12,'Mercedes-Benz':13,'Mitsubishi':14,
        'Audi':15,'Volkswagen':16,'BMW':17,'Nissan':18,'Lexus':19,
        'Jaguar':20,'Land':21,'MG':22,'Volvo':23,'Daewoo':24,
        'Kia':25,'Fiat':26,'Force':27,'Ambassador':28,'Ashok':29,
        'Isuzu':30,'Opel':31
    }

    input_data_model['owner'] = input_data_model['owner'].map(owner_map)
    input_data_model['fuel'] = input_data_model['fuel'].map(fuel_map)
    input_data_model['seller_type'] = input_data_model['seller_type'].map(seller_map)
    input_data_model['transmission'] = input_data_model['transmission'].map(transmission_map)
    input_data_model['name'] = input_data_model['name'].map(brand_map)

    try:
        prediction = model.predict(input_data_model)[0]

        prediction = max(0, prediction)

        st.success(f"💰 Estimated Car Price: ₹ {prediction:,.0f}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")