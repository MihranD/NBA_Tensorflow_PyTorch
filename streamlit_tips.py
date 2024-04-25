import streamlit as st
import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data

def generate_random_value(x): 
  return random.uniform(0, x) 

a = generate_random_value(10) 
b = generate_random_value(20) 

st.write(a) 
st.write(b)