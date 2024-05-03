import streamlit as st

def show_conclusion_page():
  st.write("### Conclusion")
  show_conclusion_text()

  show_experiments()

  show_bibliography()

def show_experiments():
  st.write("#### Model Experiments and Results")
  st.write("Our experiments with different models produced similar results. Our random forests model was able to achieve 63.64% accuracy.")

  show_experiment_1()
  show_experiment_2()

def show_experiment_1():
  if st.checkbox("Experiment 1"):
    st.markdown('''
    It's widely acknowledged that different players exhibit significant variations in their shot-making abilities. Therefore, it would seem logical that the player's name should be a crucial factor. 
                However, even after excluding player names from the dataset, the model still achieved an accuracy of 63.77%. This suggests that the model considered the 'player name' feature to be relatively insignificant.
                ''')
    st.write("*Accuracy on test set after removing 'Player ID_Frequency': 0.6377*")
    st.write("---")

def show_experiment_2():
  if st.checkbox("Experiment 2"):
    st.markdown('''
    The 'Action Type' emerged as the most critical factor for the network, as its removal resulted in a drop in accuracy to 62.07%. This finding underscores the importance of players and coaches focusing on the specific category of shots taken to significantly enhance shot success rates.
                ''')
    st.write("*Accuracy on test set after removing 'Action Type': 0.6207*")
    st.write("---")

def show_bibliography():
  st.write("#### Bibliography")

  st.markdown('''
[1] Made With ML: https://madewithml.com
              ''')
  st.markdown('''
[2] A game theoretic approach to explain the output of any machine learning model: https://github.com/shap/shap
              ''')
  st.markdown('''
[3] NBA Shot Analysis: https://www.kaggle.com/code/nbuhagiar/nba-shot-analysis
              ''')
  st.markdown('''
[4] Evaluating the effectiveness of machine learning models for performance forecasting in basketball: https://link.springer.com/article/10.1007/s10115-024-02092-9
              ''')
  st.markdown('''
The code used throughout this paper can be found here: https://github.com/DataScientest-Studio/feb24_bds_int_nba, https://github.com/MihranD/streamlit
              ''')

def show_conclusion_text():
  st.markdown('''
Achieving an accuracy of around 64% in NBA shot prediction analysis is typical and aligns with similar reports around 65%. This level of accuracy can be influenced by several factors: the complexity of basketball as a sport with numerous variables affecting shot outcomes, potential limitations in data quality or quantity leading to noise or biases, the crucial role of feature engineering and model choice in capturing relevant information, challenges in adapting to temporal dynamics and evolving game trends, the impact of sample size and variability on model robustness, the choice of evaluation metrics beyond accuracy to provide a nuanced view of performance, and the inherent human and random factors in basketball games that statistical models may struggle to fully account for. Overall, these factors collectively contribute to the observed accuracy levels in NBA shot prediction analyses.
              ''')
  st.markdown('''
For further improvement, we suggest exploring more advanced hyperparameters for random forests and deep learning models. Enhanced hyperparameter tuning, especially with access to more powerful computers and a wider range of hyperparameters, could potentially improve our prediction accuracy.
              ''')
  st.markdown('''
Furthermore, the increased data points will allow the network to train for longer and further improve its accuracy.
              ''')