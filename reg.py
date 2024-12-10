import streamlit as st
import pandas as pd
import pickle

st.header("House Price Predictor")
st.write('Enter information about the house you seek to value.')

gr_area = st.number_input('Enter the above ground area of the house.')

lot_area = st.slider('Select the total lot area.', min_value = 0, max_value = 1000)


over_qual = st.selectbox('Select overall quality.', ('Above_Average', 'Average', 'Good', 'Very_Good', 'Excellent', 'Below_Average', 'Fair', 'Poor', 'Very_Excellent', 'Very_Poor'))

sale_cond = st.selectbox("Select sale condition.", ('Normal', 'Partial', 'Family', 'Abnorml', 'Alloca', 'AdjLand'))

X = pd.DataFrame({'Gr_Liv_Area':gr_area, 'Overall_Qual':over_qual,'Sale_Condition':sale_cond, 'Lot_Area':lot_area}, index = [0])

st.dataframe(X)

with open('lr_model.pkl',"rb") as f:
    model = pickle.load(f)

preds = model.predict(X)
st.write(preds)
