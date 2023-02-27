import pandas as pd
import streamlit as st
import joblib
import pandas as pd
import streamlit as st

# Load the saved model
filename = 'Plantwise_model.joblib'
reg_loaded = joblib.load(filename)
# Load the crop_data table into a pandas DataFrame
crop_data = pd.read_csv('new_cropdata.csv')
# One-hot encode the categorical variable
crop_data = pd.get_dummies(crop_data, columns=['Varieties of Crops grown'])
# Define the Streamlit app
st.title('Crop Prediction App')
st.write('Enter the month and click on the Predict button to get the crop varieties predicted by the model for that month.')
# Create input fields for the month
month = st.selectbox('Select a month', crop_data['Months'].unique())
# Make a prediction based on the input month
if st.button('Predict'):
    inputs_df = crop_data[crop_data['Month'] == month].drop(['Varieties of Crops grown_Maize', 'Varieties of Crops grown_Rice'], axis=1)
    prediction = reg_loaded.predict(inputs_df)
    st.write('The predicted crop varieties for', month, 'are:')
    st.write(prediction)
model = joblib.load("Plantwise_model.joblib")

crop_data = pd.read_csv("new_cropdata.csv")

# Create input fields for the month
month = st.selectbox('Select a month', crop_data['Months'].unique())

# Create a button to trigger the prediction
if st.button("Predict"):
    # Preprocess the inputs (e.g., encode categorical features)
    # ...
    # Predict the crop type
    prediction = model.predict([inputs])[0]
    # Display the predicted crop type
    st.write(f"The recommended crop type for {month} is {prediction}.")

if __name__ == "__main__":
    main()






