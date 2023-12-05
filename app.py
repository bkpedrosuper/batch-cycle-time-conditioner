import streamlit as st
import pandas as pd

st.title('Data Science Challenge - Fabric Conditioner new predictive system and Batch Cycle Time Monitoring')



st.markdown("""
This project solves a dataset containing data from the site of "Amera," a Fabric Conditioner Factory of the "Globins Care Co." corporation in Azeroth. This data was obtained using a Data Engineering tool developed by SMi that uses Machine Learning models to extract useful information from the produced batches based on cloud data. The batch manufacturing process is divided into phases, and the tool can analyze each batch and get specific data from the phases during the production of each batch, for example: Temperature average in phase 2, pressure maximum value in phase 4, weight of the tank in phase 1, etc.
""")

st.markdown("""
## Given Dataset is presented below:
            """)

df = pd.read_csv('FC_Mixer_MO.csv')

st.dataframe(df)