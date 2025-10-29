import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import date
# If your model file is a full pipeline, it needs the correct imports
# from sklearn.preprocessing import OneHotEncoder, StandardScaler 
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline 

@st.cache_resource
def load_assets():
    """Loads the model pipeline and feature list from path or disk"""
    try:
        # Load the final pipeline
        pipeline = joblib.load('car_price_Gradient_Boost_pipeline.pkl')

        # Load the original feature names for Streamlit input
        original_features = joblib.load('original_feature_columns.pkl')
        
        return pipeline, original_features
    except FileNotFoundError as e:
        st.error(f"Error loading required files: {e} Please make sure that 'car_price_Gradient_Boost_pipeline.pkl' and 'original_feature_columns.pkl'are in the same directory.")
        return None, None
pipeline, original_features = load_assets()

# Exit if assets failed to load
if pipeline is None:
    st.stop()

# Configuration and Helper Function
categorical_maps = {
    'fuel_type': ['Petrol', 'Diesel', 'CNG', 'LPG'],
    'transmission_type': ['Manual', 'Automatic'],
    'seller_type':['Individual', 'Dealer', 'Trustmark Dealer'],

}
st.set_page_config(
page_title="Car Price Predictor",
page_icon="",
layout="wide"
)
st.title("Car Price Prediction App")
st.markdown("Please fill he car details below to get an estimated selling price.")

# Column layout for cleaner presentation
column1, column2 = st.columns(2)

# Dictionary to hold the all input data
input_data = {}
# --- Define Options ---
brand_options = ['Audi',
'Bentley',
'BMW',
'Datsun',
'Ferrari',
'Force',
'Ford',
'Honda',
'Hyundai',
'Isuzu',
'Jaguar',
'Jeep',
'Kia',
'Land Rover',
'Lexus',
'Mahindra',
'Maruti',
'Maserati',
'Mercedes-AMG',
'Mercedes-Benz',
'MG',
'Mini',
'Nissan',
'Porsche',
'Renault',
'Rolls-Royce',
'Skoda',
'Tata',
'Toyota',
'Volkswagen',
'Volvo']
model_options = {
    'Audi': ['A3', 'A4', 'A6', 'A8', 'Q3', 'Q5', 'Q7'],
    'BMW': ['3', '5', '6', '7', 'X1', 'X4', 'X5', 'Z4'],
    'Maruti': ['Alto','Baleno','Celerio','Ciaz','Dzire LXI','Dzire VXI','Dzire ZXI','Eeco','Ertiga','Ignis','S-Presso','Swift','Swift Dzire','Vitara','Wagon R','XL6'],
    'Hyundai': ['i20', 'Creta', 'Verna', 'Venue', 'Aura', 'Creta', 'Elantra', 'Grand', 'i10', 'Santro', 'Tucson'] ,
    'Ford': ['EcoSport', 'Figo', 'Endeavour', 'Aspire', 'Ecosport', 'Freestyle'],
    'Toyota': ['Innova Crysta', 'Fortuner', 'Yaris', 'Camry', 'Innova', 'Glanza'],
    'Honda': ['City', 'Jazz', 'WR-V', 'Amaze', 'Civic', 'CR', 'CR-V', ],
    'Volkswagen': ['Polo', 'Vento', 'Tiguan'],
    'Mahindra': ['Scorpio', 'XUV300','XUV500', 'Thar', 'Alteras', 'Bolero', 'KUV', 'KUV100', 'Marazzo',],
    'Kia': ['Seltos', 'Sonet', 'Carnival'],
    'Skoda': ['Octavia', 'Rapid', 'Kushaq', 'Superb'],
    'Nissan': ['Magnite', 'Kicks', 'X-Trail'],
    'Jeep': ['Compass', 'Wrangler', ''],
    'Mercedes-Benz': ['C-Class', 'E-Class', 'GLA', 'S-Class', 'GLS', 'GL-Class'],
    'Land Rover': ['Range Rover Evoque', 'Discovery Sport', 'Rover'],
    'MG': ['Hector', 'ZS EV'],
    'Mini': ['Cooper', 'Countryman'],
    'Porsche': ['Cayenne', 'Macan', 'Panamera'],
    'Renault': ['KWID', 'Duster', 'Triber'],
    'Datsun': ['GO', 'RediGO', 'redi-GO'],    
    'Bentley': ['Continental GT', 'Bentayga'],
    'Ferrari': ['488 GTB', 'Portofino', 'GTC4Lusso'],
    'Maserati': ['Ghibli', 'Levante', 'Quattroporte'],
    'Force': ['Gurkha', 'Tornado'],
    'Isuzu': ['D-Max V-Cross', 'D-Max', 'MUX'],
    'Rolls-Royce': ['Phantom', 'Cullinan', 'Ghost'],
    'Mercedes-AMG': ['GT', 'A 45', 'C'],
    'Jaguar': ['F-PACE', 'XE', 'XF'],
    'Lexus': ['ES', 'NX', 'RX'],
    'Tata': ['Altroz','Harrier','Hexa','Nexon','Safari','Tiago','Tigor'],
    'Volvo': ['S90', 'XC', 'XC60', 'XC90']


    
    # Add more brand-model mappings as needed
  
}
with column1:
    input_data['km_driven'] = st.slider("Slide the Kilometers Driven:", min_value=100, max_value=300000, value=50000, step=1000)
    input_data['engine'] = st.slider("Slide the Engine Capacity (in CC):", min_value=500, max_value=5000, value=1500, step=50)
    input_data['max_power'] = st.slider("Slide the Max Power (in bhp):", 20.0, 400.0, 80.0)
    input_data['mileage'] = st.slider("Slide the Mileage (in km/l):", min_value=5.0, max_value=50.0, value=18.0, step=0.1)
    input_data['seats'] = st.slider("Slide the Number of Seats", min_value=2, max_value=12, value=5, step=1)

with column2:
    input_data['fuel_type'] = st.selectbox("Fuel Type", options=categorical_maps['fuel_type'])
    input_data['transmission_type'] = st.selectbox("Transmission Type", options=categorical_maps['transmission_type'])
    input_data['seller_type'] = st.selectbox("Seller Type", options=categorical_maps['seller_type'])
    
    current_year = date.today().year

    year_of_purchase = st.number_input("Year of Manufacturing/Purchase",
                                       min_value=1950,
                                       max_value=current_year,
                                       value=current_year-5)
    input_data['vehicle_age'] = current_year - year_of_purchase
# --- Widgets ---
    input_data['brand'] = st.selectbox("Search and Select Favourite Brand", brand_options)
    input_data['model'] = st.selectbox("Search and select your favourite car model.", model_options.get(input_data['brand'], []))



# vehicle_age = st.slider("Slide the Vehicle Age (in years):", min_value=1, max_value=30, value=5, step=1)

st.markdown("🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹")
st.write(f"Calculated Vehicle Age: **{input_data['vehicle_age']} years**")




# create a button to predict output
predict_clicked = st.button("Get the Car Price Prediction 💰", type='primary')

if predict_clicked:
    try:
        input_df = pd.DataFrame([input_data])

        # Recreation of log features:
        offset = 0

        #if 'vehicle_age_log' in original_features:
            #input_df['vehicle_age_log'] = np.log(input_df['vehicle_age'] + offset + 1)
            #input_df = input_df.drop(columns='vehicle_age', errors='ignore')

        #if 'selling_price_log' in original_features:
            #input_df['selling_price_log'] = np.log(input_df['selling_price'] + offset + 1)
            #input_df = input_df.drop(columns='selling_price_log', errors='ignore')
        #if 'selling_price' in original_features:
            #input_df['selling_price'] = np.log(input_df['selling_price'] + offset + 1)
            #input_df = input_df.drop(columns='selling_price', errors='ignore')
        # Make the prediction using the full pipeline
        with st.spinner('Calculating price....'):
            predicted_price = pipeline.predict(input_df)
        
        # Display the results
        st.success(f"🔹🔹 Estimated Selling Car Price: R {predicted_price[0]:,.2f}")
        st.balloons()

        st.markdown("🔹🔹*** *Disclaimer: This is an estimate based on the trained model and features.*🔹🔹")
    except Exception as e:
        st.error(f"An error occurred during prediction. This is often caused by missing or unexpected input columns. Error: {e}")
        st.error(f"Input DataFrame Columns: {input_df.columns.tolist()}")
        st.error(f"Model Expected Features: {original_features}")
        st.exception(e) # Display the full exception traceback
st.caption(f"© {current_year} [Cedrick Mkhabela]. All rights reserved. | Developed for a Machine Learning Deployment Project. Powered by Streamlit.| @Regenesy School of Technology") 