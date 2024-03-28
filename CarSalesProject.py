import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the prediction model
model = joblib.load('car_price_model.joblib')

# numeric features
numeric_features_array = ["year", "condition", "odometer", "saleyear", "salemonth", "saleday"]
categorical_features_array = ["make", "model", "trim", "state", "color", "interior"]
# Define the list of states
states = ['ca', 'tx', 'pa', 'mn', 'az', 'wi', 'tn', 'md', 'fl', 'ne', 'nj', 
          'oh', 'mi', 'ga', 'va', 'sc', 'in', 'il', 'co', 'ut', 
          'mo', 'nv', 'ma', 'pr', 'nc', 'ny', 'or', 'la', 'wa', 
          'hi', 'qc', 'ab', 'on', 'ok', 'ms', 'nm', 'al', 'ns']

# Convert states to uppercase
states_uppercase = [state.upper() for state in states]

# Define the list of colors
colors = ['white', 'gray', 'black', 'red', 'silver', 'brown', 'beige', 'blue', 'purple', 'burgundy', '—', 'gold', 'yellow', 'green', 'charcoal', 'orange', 'off-white', 'turquoise', 'pink', 'lime']

# Define the list of interiors
interiors = ['black', 'beige', 'tan', 'brown', 'gray', '—', 'burgundy', 'white', 'silver', 'off-white', 'blue', 'red', 'yellow', 'green', 'purple', 'orange', 'gold']

# Load car data from CSV file
@st.cache_data
def load_car_data():
    # Load car data from CSV file
    car_data = pd.read_csv('car_data.csv')
    return car_data

# Function to filter options based on make
def filter_models(make, car_data):
    models = [model for model in car_data['Model'].unique() if car_data[(car_data['Make'] == make) & (car_data['Model'] == model)].shape[0] > 0]
    return models

# Function to filter options based on make and model
def filter_trims(make, model, car_data):
    trims = [trim for trim in car_data['Trim'].unique() if car_data[(car_data['Make'] == make) & (car_data['Model'] == model) & (car_data['Trim'] == trim)].shape[0] > 0]
    return trims


# Define function to make predictions
def make_prediction(input_data):

    #preprocessed_data = model['preprocessor'].transform(input_data)
    # preProcessedData = preProcesssing(input_data)
    # Assume your model takes input_data as input and returns predictions
    # Replace this with actual prediction logic based on your model
    # preprocessed_data = model['preprocessor'].transform(input_data)

    # prediction = model.predict(preprocessed_data)
    # st.write(preprocessed_data)
    # return prediction
    return

# Streamlit app
def main():
    # Load car data
    # car_data = st.cache_data(load_car_data)()
    car_data = load_car_data()

    # st.title('Prediction App')
    st.markdown("<h1 style='text-align: center;'>Car Sales Price Prediction</h1>", unsafe_allow_html=True)

    # Define input widgets
    st.sidebar.header('Input Parameters')

    make = st.sidebar.selectbox('Select Make:', sorted(car_data['Make'].unique()))

    # Filter models based on selected make
    models = filter_models(make, car_data)
    model = st.sidebar.selectbox('Select Model:', models)

    # Filter trims based on selected make and model
    trims = filter_trims(make, model, car_data)
    trim = st.sidebar.selectbox('Select Trim:', trims)

   # Year selection dropdown
    current_year = int(datetime.now().year)
    year = st.sidebar.selectbox('Select Year:', range(2000, current_year + 1), index = current_year - 2000)

    # Define input widget for state selection
    state = st.sidebar.selectbox('Select State:', sorted(states_uppercase))

    # Define input widget for color selection
    color = st.sidebar.selectbox('Select Color:', sorted(colors))

    # Define input widget for color selection
    interior = st.sidebar.selectbox('Select Interior:', interiors)

    # Define input widget for selling month
    selling_month = st.sidebar.number_input('Enter Selling Month (1-12):', min_value=1, max_value=12, value=1)
    selling_day = st.sidebar.number_input('Enter Selling Day (1-31):', min_value=1, max_value=31, value=1)
    selling_year = st.sidebar.number_input('Enter Selling Year:', min_value=2000, max_value=current_year, value=current_year)


    # Define input widget for odometer
    odometer = st.sidebar.number_input('Enter Odometer:', min_value=1)

    # Define input widget for range selection
    condition = st.sidebar.slider('Select condition:', 1, 10, 1)

    # Display prediction
    # st.write('Prediction:', 25000)

    # Button to trigger prediction
    if st.sidebar.button('Predict'):
        # Gather input data
        # input_data = [year, make, model, trim, state, condition, odometer, color, interior, selling_year, selling_month, selling_day]

        input_data = pd.DataFrame({
            'year': [year],
            'make': [make],
            'model': [model],
            'trim': [trim],
            'state': [state.lower()],
            'color': [color],
            'interior': [interior],
            'condition': [condition],
            'saleyear': [selling_year],
            'salemonth': [selling_month],
            'saleday': [selling_day],
            "odometer" :[odometer]
        })
        st.write(input_data)
        # Make prediction
        prediction = make_prediction(input_data)

        # Display prediction
        # st.write('Prediction:', prediction)

if __name__ == "__main__":
    main()
