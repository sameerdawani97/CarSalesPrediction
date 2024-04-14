import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import time
import plotly.graph_objects as go
import main

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Load the price prediction model
model = joblib.load('car_price_model.joblib')
preProcessor = joblib.load('car_price_preprocessor1.joblib')

# Load the insurance prediction model
model_insurance = joblib.load('smaller_insurance_model.joblib')

# numeric features
numeric_features_array = ["year", "condition", "odometer", "saleyear", "salemonth", "saleday"]
categorical_features_array = ["make", "model", "trim", "state", "color", "interior"]
# Define the list of states
states = ['ca', 'tx', 'pa', 'mn', 'az', 'wi', 'tn', 'md', 'fl', 'ne', 'nj', 
          'oh', 'mi', 'ga', 'va', 'sc', 'in', 'il', 'co', 'ut', 
          'mo', 'nv', 'ma', 'pr', 'nc', 'ny', 'or', 'la', 'wa', 
          'hi', 'ok', 'ms', 'nm', 'al']

# Convert states to uppercase
states_uppercase = [state.upper() for state in states]

# Define the list of colors
colors = ['white', 'gray', 'black', 'red', 'silver', 'brown', 'beige', 'blue', 'purple', 'burgundy', '—', 'gold', 'yellow', 'green', 'charcoal', 'orange', 'off-white', 'turquoise', 'pink', 'lime']

# Define the list of interiors
interiors = ['black', 'beige', 'tan', 'brown', 'gray', '—', 'burgundy', 'white', 'silver', 'off-white', 'blue', 'red', 'yellow', 'green', 'purple', 'orange', 'gold']

# Load car data from CSV file
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


# Define function to make prediction
def make_prediction(input_data):
    X_preprocessed = preProcessor.transform(input_data)
    predicted_value = model.predict(X_preprocessed)
    
    return predicted_value

# Define function to make prediction with premium
def make_prediction_insurance(input_data_insurance, carPrice):

    # Get the index of the 'Cylinder_capacity' column
    cylinder_capacity_index = input_data_insurance.columns.get_loc('Cylinder_capacity')

    # Insert the new column after the 'Cylinder_capacity' column
    input_data_insurance.insert(cylinder_capacity_index + 1, 'Value_vehicle', [carPrice])

    # input_data_insurance['Value_vehicle'] = carPrice
    predicted_value = model_insurance.predict(input_data_insurance)
    # print(input_data_insurance)
    return predicted_value
    # return predicted_value

def make_prediction_for_other_states(input_data, selected_state):
    prices = {}
    for state in states_uppercase:
        if state == selected_state:
            continue
        input_data['state'] = state.lower()
        price = make_prediction(input_data)
        prices[state] = price
    return prices

# Streamlit app
def predict():

    default_text = """
        <h3 style='text-align: center;'>Please enter the required information and click the predict button to proceed.</h3>
        """

    # Load car data
    # car_data = st.cache_data(load_car_data)()

    car_data = load_car_data()

    # st.title('Prediction App')
    st.markdown("<h1 style='text-align: center;'>Car Sales Price Prediction</h1>", unsafe_allow_html=True)

    
    prediction_placeholder = st.empty()
    prediction_placeholder.write(default_text, unsafe_allow_html=True)

    # Define input widgets
    st.sidebar.header('Questionnaire for Car Price Prediction')

    make = st.sidebar.selectbox('Select Make:', sorted(car_data['Make'].unique()), help = "Please select the make of car")

    models = filter_models(make, car_data)
    # st.sidebar.markdown("**Select Model:**", unsafe_allow_html=True)
    # model = st.sidebar.selectbox(' ', models, help="Select the model of the car")

    # Filter models based on selected make
    
    models = filter_models(make, car_data)
    model = st.sidebar.selectbox('**Select Model:**', models, help="Select the model of the car")

    # Filter trims based on selected make and model
    trims = filter_trims(make, model, car_data)
    trim = st.sidebar.selectbox('Select Trim:', trims, help="Select the trim of the car")

   # Year selection dropdown
    current_year = int(datetime.now().year)
    year = st.sidebar.selectbox('Select Year:', range(2000, current_year + 1), index = current_year - 2000, help = "year of make")

    # Define input widget for state selection
    state = st.sidebar.selectbox('Select State:', sorted(states_uppercase), help = "state of car bought")

    # Define input widget for color selection
    color = st.sidebar.selectbox('Select Color:', sorted(colors))

    # Define input widget for color selection
    interior = st.sidebar.selectbox('Select Interior:', interiors)

    # Define input widget for selling month
    selling_month = st.sidebar.number_input('Enter Selling Month (1-12):', min_value=1, max_value=12, value=1)
    selling_day = st.sidebar.number_input('Enter Selling Day (1-31):', min_value=1, max_value=31, value=1)
    selling_year = st.sidebar.number_input('Enter Selling Year:', min_value=year, max_value=current_year, value=current_year)


    # Define input widget for odometer
    odometer = st.sidebar.number_input('Enter Odometer:', min_value=1, help="This refers to the distance travelled by car")

    # Define input widget for range selection
    condition = st.sidebar.slider('Select condition:', 1, 50, 1, help="Condition of the car, 50 being the best condition and 1 being the worst condition")

    # insurance price section
    insurance_option = st.sidebar.radio("**Would you like to know the insurance premium cost as well?**", ("Yes", "No"), index = 1)

    # If user selects "Yes", display additional input widgets
    if insurance_option == "Yes":
         # Define input widgets for questionnaire
        st.sidebar.header('Questionnaire for Car Insurance Premium Prediction')

        # start_date = st.sidebar.date_input("Expected Start Date for Insurance", datetime.today())
        insurance_company_years = st.sidebar.number_input("Years Associated with target Insurance Company", min_value=0)

        payment_frequency_option = st.sidebar.radio("Preferred Insurance Payment Frequency", ('Half-yearly', 'Annually'))
        payment_frequency = 1 if payment_frequency_option == 'Half-yearly' else 0

        past_insurance_duration = st.sidebar.number_input("Total Duration of Past Vehicle Insurance Policies (years)", min_value=0)
        past_claims = st.sidebar.number_input("Number of Claims Filed Across All Past Insurance Policies", min_value=0)
        
        # date_of_birth = st.sidebar.date_input("Date of Birth", datetime.today())
        driverAge = st.sidebar.number_input("Driver Age", min_value=16, max_value=96)
        license_issue_year = st.sidebar.number_input("Year of Obtaining Driver's License", min_value=1950, max_value=datetime.today().year)

        # Vehicle Details
        st.sidebar.subheader("Vehicle Details (Can be found on papers)")

        vehicle_registration_year = st.sidebar.number_input("Vehicle Registration Year", min_value=1950, max_value=datetime.today().year)
        vehicle_weight = st.sidebar.number_input("Vehicle Weight (kg)", min_value=0)
        vehicle_power = st.sidebar.number_input("Vehicle Power (horsepower)", min_value=0)
        vehicle_length = st.sidebar.number_input("Vehicle Length (meters)", min_value=0.0)
        engine_capacity = st.sidebar.number_input("Engine Capacity (cc)", min_value=0)


    # Display prediction
    # st.write('Prediction:', 25000)

    # Button to trigger prediction
    if st.sidebar.button('Predict'):

        # Gather input
        input_data = pd.DataFrame({
            'year': [year],
            'make': [make],
            'model': [model],
            'trim': [trim],
            'state': [state.lower()],
            'condition': [condition],
            'odometer': [odometer],
            'color': [color],
            'interior': [interior],
            'saleyear': [selling_year],
            'salemonth': [selling_month],
            'saleday': [selling_day],
        })

        current_date = datetime.today()
        # driverAge = current_date.year - date_of_birth.year - ((current_date.month, current_date.day) < (date_of_birth.month, date_of_birth.day))

        prediction_placeholder.write("<h3 style='text-align: center;'>Please wait while we calculate the estimated price...</h3>", unsafe_allow_html=True)

        # Make prediction
        pricePrediction = make_prediction(input_data)

        premium = 0
        if insurance_option == "Yes":
            vehicleAge = current_year - vehicle_registration_year
            input_data_insurance = pd.DataFrame({
                'Seniority' : [insurance_company_years],
                'Payment' : [payment_frequency],
                'N_claims_history' : [past_claims],
                'R_Claims_history' : [past_claims / past_insurance_duration if past_insurance_duration > 0 else 0.0],
                'Year_matriculation' : [vehicle_registration_year],
                'Power' : [vehicle_power],
                'Cylinder_capacity' : [engine_capacity],
                'Length': [vehicle_length],
                'Weight' : [vehicle_weight],
                'Driving_Experience' : [license_issue_year - current_year],
                'Driver_Age_at_Start_squared' : [driverAge*driverAge],
                'Vehicle_Age_at_Start_squared' : [vehicleAge*vehicleAge],
                'Age_Vehicle_Interaction' : [driverAge*vehicleAge],
            })
            premium = make_prediction_insurance(input_data_insurance, pricePrediction)[0]
        prices = make_prediction_for_other_states(input_data, state)

        time.sleep(2) 
        if premium !=0:
            prediction_placeholder.write(f"<h3 style='text-align: center;'>Estimated price of car in {state} is ${int(pricePrediction[0])}</h3>\n<h3 style='text-align: center;'>Estimated insurance premium of car is ${int(premium)}</h3>\n", unsafe_allow_html=True)
        else:
            prediction_placeholder.write(f"<h3 style='text-align: center;'>Estimated price of car in {state} is ${int(pricePrediction[0])}</h3>\n\n", unsafe_allow_html=True)
        # st.write("<h6 style='text-align: center;'>Following are the car prices in different states:</h6>", unsafe_allow_html=True)
       
       # Display predictions for all other states
        state_prices = []
        for state, price in prices.items():
            # prediction_placeholder.write(f"<h3 style='text-align: center;'>Estimated price of car in {state} is ${price}</h3>", unsafe_allow_html=True)
            state_prices.append({'State': state, 'Price': price})

        # Create a DataFrame from the state prices
        state_prices_df = pd.DataFrame(state_prices)
        
        # Extract state names and prices from the state_prices list
        states_price = [item['State'] for item in state_prices]
        prices = [item['Price'][0] for item in state_prices]

        
        # Define a color palette for the states
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Create the Plotly figure
        fig = go.Figure()

        # Add bar trace to the figure with custom colors
        for i, state in enumerate(states_price):
            fig.add_trace(go.Bar(x=[state], y=[prices[i]], name = state, marker_color=color_palette[i % len(color_palette)]))

        # Update layout for hover information and aesthetics
        fig.update_layout(title='Car Prices Estimation by State',
                        title_x=0.35, 
                        xaxis=dict(title='States',
                                    showgrid=False,  # Hide gridlines
                                    showline=True,   # Show axis line
                                    linecolor='gray',  # Axis line color
                                    linewidth=1),    # Axis line width
                        yaxis=dict(title='Price ($)',
                                    showgrid=True,   # Show gridlines
                                    gridcolor='lightgray',  # Gridline color
                                    gridwidth=0.5),  # Gridline width
                        plot_bgcolor='rgba(255, 255, 255, 0)',  # Transparent plot background
                        hoverlabel=dict(bgcolor='white',  # Hover label background color
                                        font=dict(color='black')),  # Hover label font color
                        margin=dict(l=50, r=50, t=50, b=50))  # Add margin to the plot

        # Display the Plotly figure
        st.plotly_chart(fig)

        #df = pd.DataFrame(car_prices_premium).T.reset_index()

        state_prices_df.columns = ["State", "Price"]

        state_prices_df["Price"] = state_prices_df["Price"]


    
if __name__ == "__main__":
    main.main()