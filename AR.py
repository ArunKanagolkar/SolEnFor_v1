#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import streamlit as st
from statsmodels.tsa.arima.model import ARIMA


# In[3]:


# Load your time series dataset
data = pd.read_csv('Ir_vs_kWhrs_hrs_base.csv')


# In[8]:


# Train an ARIMA model
p = 8  
d = 0  
q = 8  

model = ARIMA(data['kWhrs'], order=(p, d, q))
model_fit = model.fit()


# In[10]:


# Streamlit app
st.title('Modeling Solar Energy Production for Tomorrow')
st.subheader("Accurate Predictions for Renewable Energy")
st.write('## Original Data')
st.line_chart(data,x='Date',y='kWhrs')


# In[12]:


df5 = data.copy()
df5['forecast'] = model_fit.predict(start=0, end=len(df5)-1)
st.write('## Predicted Data')
st.line_chart(df5,x='Date',y='forecast',color='#04f')


# In[14]:


# Forecast
forecast_steps = st.slider('Select number of steps for forecast', 1, 100, 10)
forecast =model_fit.forecast(steps=forecast_steps)
df5=df5.set_index('Date')
forecast_index = pd.date_range(df5.index[-1], periods=forecast_steps + 1, freq='H')[1:]

# Create a DataFrame with forecast and forecast datetime
forecast_df = pd.DataFrame({
    'Forecast Datetime': forecast_index,
    'Forecast kWhrs': forecast
})


# In[16]:


st.write('##  Forecast')
if st.button("Forecast"):
    st.line_chart(forecast_df,x='Forecast Datetime',y='Forecast kWhrs',color='#ffaa0088')


# In[10]:


if st.button("Forecast Days"):
    Days = (forecast_df.set_index('Forecast Datetime').resample('D').sum())
    st.table(Days)


# In[ ]:




