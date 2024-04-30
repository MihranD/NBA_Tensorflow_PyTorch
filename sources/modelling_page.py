import streamlit as st
from sources.utils import read_df

def show_modelling_page():
  st.write("## Modelling")
  
  # Read the dataset into a DataFrame
  df = read_df()

  # Classification of the problem
  classification_of_problem(df)

@st.cache_data
def classification_of_problem(df):
  st.write("### Classification of the problem")

  st.markdown('''
1. Type of Machine Learning Problem:
The machine learning problem in this project is a classification task. Specifically, it involves predicting whether a given NBA shot will be made or missed based on various features related to the shot.

2. Task Related to the Project:
The project is related to NBA shot analysis, which falls under sports analytics. The specific task is to analyze and predict the outcome of shots taken during NBA games, focusing on whether a shot will result in a successful make or a miss.

3. Main Performance Metric:
The main performance metric used to compare different machine learning models in this project is accuracy. Accuracy is chosen as the main metric because it provides a straightforward measure of how often the model's predictions are correct out of all predictions made. In the context of NBA shot analysis, accuracy helps evaluate the overall effectiveness of the model in predicting whether a shot will be made or missed.

4. Additional Performance Metrics:
In addition to accuracy, other quantitative performance metrics are used to evaluate the models' performance comprehensively. These metrics include:

    Precision: It measures the ratio of true positive predictions to the total number of positive predictions. In the context of NBA shot analysis, precision helps assess the model's ability to correctly identify made shots among all shots predicted as made.

    Recall (Sensitivity): It calculates the ratio of true positive predictions to the total number of actual positives. In the NBA shot analysis, recall evaluates the model's capability to capture all made shots, indicating its sensitivity to identifying positive instances (made shots).

    F1 Score: The F1 score is the harmonic mean of precision and recall. It provides a balanced measure of a model's performance, considering both false positives and false negatives. This metric is useful in scenarios where we want to balance precision and recall, such as in the NBA shot analysis task where correctly identifying both made and missed shots is crucial.

    Area Under the ROC Curve (AUC-ROC): This metric evaluates the model's ability to distinguish between classes (made and missed shots) across various threshold values. A higher AUC-ROC value indicates a better-performing model in terms of class separation and predictive power.

    These additional metrics help in gaining a more nuanced understanding of the model's performance beyond just accuracy, taking into account factors like false positives, false negatives, and the balance between precision and recall.

In summary, the NBA shot analysis project involves a classification task where the goal is to predict whether a given shot will be made or missed. The main performance metric used is accuracy, supplemented by other quantitative metrics like precision, recall, F1 score, and AUC-ROC to provide a comprehensive evaluation of the machine learning models employed in the analysis.
              ''')
  st.write("---")