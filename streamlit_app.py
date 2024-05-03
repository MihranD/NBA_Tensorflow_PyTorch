import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sources.introduction_page import show_intro_page
from sources.preprocessing_page import show_preprocessing_page
from sources.visualisation_page import show_visualisation_page
from sources.preprocessing_for_modelling_page import show_preprocessing_for_modelling_purposes_page
from sources.modelling_page import show_modelling_page
from sources.deep_learning_page import show_deep_learning_page
from sources.conclusion import show_conclusion_page

st.title("NBA player shot analysis")
st.sidebar.title("Table of contents")
pages=["Introduction to the project", 
       "Preprocessing and feature engineering", 
       "Visualizations and Statistics", 
       "Preprocessing for modeling purposes", 
       "Base Models",
       "Deep Learning",
       "Conclusion"]
page=st.sidebar.radio("Go to", pages)

# Context
if page == pages[0] : 
  show_intro_page()

if page == pages[1] : 
  show_preprocessing_page()

if page == pages[2] : 
  show_visualisation_page()

if page == pages[3] : 
  show_preprocessing_for_modelling_purposes_page()

if page == pages[4] : 
  show_modelling_page()

if page == pages[5] : 
  show_deep_learning_page()

if page == pages[6] : 
  show_conclusion_page()
