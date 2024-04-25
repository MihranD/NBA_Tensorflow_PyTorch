import streamlit as st
import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("train.csv")

st.title("Titanic : binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)