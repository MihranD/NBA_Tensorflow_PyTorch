import streamlit as st
import pandas as pd

@st.cache_data
def read_df():
  df = pd.read_csv('NBA Shot Locations 1997 - 2020.csv')
  return df