#!/usr/bin/env python
# coding: utf-8

# In[18]:
# In[19]:
# In[20]:
# In[21]:


import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[22]:


# Load the crop_data table into a pandas DataFrame
crop_data = pd.read_csv('new_cropdata.csv')


# In[23]:


# One-hot encode the categorical variable
crop_data = pd.get_dummies(crop_data, columns=['Varieties of Crops grown'])


# In[24]:


# Split the data into features and labels
X = crop_data.drop(['Varieties of Crops grown_Maize', 'Varieties of Crops grown_Rice'], axis=1)
y = crop_data[['Varieties of Crops grown_Maize', 'Varieties of Crops grown_Rice']]


# In[25]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[26]:


# Train a multi-output regression model on the training set
reg = MultiOutputRegressor(LinearRegression())
reg.fit(X_train, y_train)


# In[27]:


# Evaluate the model on the testing set
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('MAE:', mae)
print('MSE:', mse)
print('R2 Score:', r2)


# In[28]:


# Save the trained model
filename = 'Plantwise_model.sav'
joblib.dump(reg, filename)


# In[29]:


# Load the saved model
reg_loaded = joblib.load(filename)


# In[30]:


# Define the Streamlit app
st.title('Crop Prediction App')


# In[31]:


# Define the Streamlit app
st.title('Crop Prediction App')

st.write('Enter the values for the features and click on the Predict button to get the crop varieties predicted by the model.')


# In[32]:


# Create input fields for the features
inputs = []
for column in X.columns:
    value = st.number_input(f'Enter {column}', min_value=0.0)
    inputs.append(value)


# In[33]:


# Make a prediction based on the input values
if st.button('Predict'):
    inputs_df = pd.DataFrame([inputs], columns=X.columns)
    prediction = reg_loaded.predict(inputs_df)
    st.write('The predicted crop varieties are:')
    st.write(prediction)

