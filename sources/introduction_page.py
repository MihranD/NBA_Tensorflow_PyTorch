import streamlit as st

def show_intro_page():
  st.markdown('''
For this project, we utilized the NBA shot dataset spanning the years 1997 to 2020. The dataset contains comprehensive information on shot locations in NBA games, allowing for detailed analysis of shot frequency and efficiency among players during this period.
              ''')

  # Problem Definition
  st.markdown('''
We aim to predict the probability of a shot being made by each player, indicating whether a shot is successful or not. This problem naturally aligns with a binary classification task, where shots are categorized as either made or missed.
            ''')
  
  st.markdown('''
The dataset used in this project is freely available on Kaggle: (https://www.kaggle.com/jonathangmwl/nba-shot-locations).
              ''')