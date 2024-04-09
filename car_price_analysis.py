import streamlit as st
import pandas as pd
import plotly.express as px


# Function to calculate color distribution
def get_color_distribution():

    df = pd.read_csv('car_prices.csv')

    # Calculate color distribution
    color_counts = df['color'].value_counts()

    # Keep the top 12 categories, combine the rest into "Others"
    top_categories = color_counts.head(12)
    other_count = color_counts.sum() - top_categories.sum()
    color_counts = pd.concat([top_categories, pd.Series({'Others': other_count})])

    return color_counts


# Function to calculate car make distribution
def get_make_distribution():

    df = pd.read_csv('car_prices.csv')

    # Calculate car make distribution
    make_counts = df['make'].value_counts()

    # Keep the top 20 categories, combine the rest into "Others"
    top_makes = make_counts.head(20)
    other_count = make_counts.sum() - top_makes.sum()
    make_counts = pd.concat([top_makes, pd.Series({'Others': other_count})])

    return make_counts


# Function to calculate interior distribution
def get_interior_distribution():

    df = pd.read_csv('car_prices.csv')

    # Calculate interior distribution
    interior_counts = df['interior'].value_counts()

    # Keep the top 6 categories, combine the rest into "Others"
    top_interiors = interior_counts.head(6)
    other_count = interior_counts.sum() - top_interiors.sum()
    interior_counts = pd.concat([top_interiors, pd.Series({'Others': other_count})])

    return interior_counts


# Function to calculate body type distribution
def get_body_distribution():

    df = pd.read_csv('car_prices.csv')

    # Convert body types to uppercase to ensure uniformity
    df['body'] = df['body'].str.upper()

    # Calculate body type distribution
    body_counts = df['body'].value_counts()

    return body_counts


# Function to calculate condition distribution
def get_condition_distribution():

    df = pd.read_csv('car_prices.csv')

    # Calculate condition distribution
    condition_counts = df['condition'].value_counts()

    return condition_counts

# Function to calculate odometer distribution
def get_odometer_distribution():

    df = pd.read_csv('car_prices.csv')

    # Calculate odometer distribution
    odometer_counts = df['odometer']

    return odometer_counts


def showAnalysis():
    st.title('Car Data Analysis')

    # Dropdown menu for selecting data analysis
    analysis_type = st.selectbox('Select Analysis', ['Color Distribution', 'Car Make Distribution', 'Interior Distribution', 'Body Type Distribution', 'Condition Distribution', 'Odometer Distribution', 'Selling Price Distribution', 'Feature Importance Analysis'])

    # Perform selected data analysis
    if analysis_type == 'Color Distribution':
        st.subheader('Car Colors Analysis')
        color_distribution_analysis()
    elif analysis_type == 'Car Make Distribution':
        st.subheader('Car Make Analysis')
        make_distribution_analysis()
    elif analysis_type == 'Interior Distribution':
        st.subheader('Car Interior Color Analysis')
        interior_distribution_analysis()
    elif analysis_type == 'Body Type Distribution':
        st.subheader('Car Body Analysis')
        body_distribution_analysis()
    elif analysis_type == 'Condition Distribution':
        st.subheader('Car Condition Analysis')
        condition_distribution_analysis()
    elif analysis_type == 'Odometer Distribution':
        st.subheader('Car Odometer Analysis')
        odometer_distribution_analysis()
    elif analysis_type == 'Selling Price Distribution':
        st.subheader('Selling Price Analysis')
        selling_price_distribution_analysis()
    elif analysis_type == 'Feature Importance Analysis':
        st.subheader('Feature Importance Analysis')
        feature_importance_analysis()

# Function for color distribution analysis
def color_distribution_analysis():
    # Calculate color distribution
    color_counts = get_color_distribution()

    # Get a list of colors for each category
    colors = px.colors.qualitative.Plotly[:len(color_counts)]

    # Display pie chart without title
    fig = px.pie(names=color_counts.index, values=color_counts.values, color_discrete_sequence=colors)
    st.plotly_chart(fig)


# Function for car make distribution analysis
def make_distribution_analysis():
    # Calculate car make distribution
    make_counts = get_make_distribution()

    # Display pie chart without title
    fig = px.pie(names=make_counts.index, values=make_counts.values)
    st.plotly_chart(fig)


# Function for interior distribution analysis
def interior_distribution_analysis():
    # Calculate interior distribution
    interior_counts = get_interior_distribution()

    # Display pie chart without title
    fig = px.pie(names=interior_counts.index, values=interior_counts.values)
    st.plotly_chart(fig)


# Function for body type distribution analysis
def body_distribution_analysis():
    # Calculate body type distribution
    body_counts = get_body_distribution()

    # Display pie chart without title
    fig = px.pie(names=body_counts.index, values=body_counts.values)
    st.plotly_chart(fig)


# Function for condition distribution analysis
def condition_distribution_analysis():
    # Calculate condition distribution
    condition_counts = get_condition_distribution()

    # Display bar chart without title
    fig = px.bar(x=condition_counts.index, y=condition_counts.values, labels={'x':'Condition', 'y':'Frequency'})
    st.plotly_chart(fig)

# Function for odometer distribution analysis
def odometer_distribution_analysis():

    df = pd.read_csv('car_prices.csv')

    # Filter the DataFrame to include only odometer values less than 400,000
    df_filtered = df[df['odometer'] < 400000]

    # Display histogram
    fig = px.histogram(df_filtered, x='odometer', title='Distribution of Odometer Values (Less than 400,000)')
    st.plotly_chart(fig)

# Function to calculate selling price distribution
def get_selling_price_distribution():

    df = pd.read_csv('car_prices.csv')

    # Calculate selling price distribution
    selling_price = df['sellingprice']

    return selling_price

# Function for selling price distribution analysis
def selling_price_distribution_analysis():

    df = pd.read_csv('car_prices.csv')

    # Filter the DataFrame to include only selling prices less than 400,000
    df_filtered = df[df['sellingprice'] < 100000]

    # Display histogram
    fig = px.histogram(df_filtered, x='sellingprice', title='Distribution of Selling Prices (Less than $100,000)')
    st.plotly_chart(fig)


# Function for feature importance analysis
def feature_importance_analysis():
    feature_importances = {
        'color': 65950.23,
        'model': 34593.64,
        'state': 13345.79,
        'trim': 7261.12,
        'year': 4355.97,
        'interior': -5172.95,
        'condition': 455.95,
        'make': -2523.52,
        'odometer': -2388.88,
        'salemonth': -129.05,
        'saleyear': -61.09,
        'saleday': -30.12
    }

    # Create a DataFrame from the dictionary
    feature_df = pd.DataFrame(feature_importances.items(), columns=['Features', 'Importance'])

    # Sort the DataFrame by importance
    feature_df = feature_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    fig = px.bar(feature_df, x='Features', y='Importance')
    st.plotly_chart(fig)