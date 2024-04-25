import streamlit as st
import random

@st.cache_data

def generate_random_value(x): 
  return random.uniform(0, x) 

a = generate_random_value(10) 
b = generate_random_value(20) 

st.write(a) 
st.write(b)