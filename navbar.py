import streamlit as st
from streamlit_option_menu import option_menu
import CarSalesProject
import car_price_analysis
import home


def showNavigation():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options= ["Home", "Car Price Prediction", "Data Analysis"],
            icons=["house", "book", "envelope"],
            menu_icon="cast",
            default_index=0,

        )

    if selected == "Home":
        home.showHome()
        
    if selected == "Car Price Prediction":
        CarSalesProject.predict()
        
    if selected == "Data Analysis":
        car_price_analysis.showAnalysis()
        