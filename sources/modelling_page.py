import streamlit as st
import pandas as pd
from sources.utils import read_df
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def show_modelling_page():
  st.write("### Modelling")

  # Read the train and test sets from the file 'NBA Shot Locations 1997 - 2020-Report2-train-test.joblib'.
  X_train, X_test, y_train, y_test = load('NBA Shot Locations 1997 - 2020-Report2-train-test.joblib')
    
  choice = ['Logistic Regression', 'Decision Tree', 'Boosting', 'Bagging', 'Random Forest']
  option = st.selectbox('Choice of the model', choice)
  st.write('The chosen model is :', option)

  def prediction(classifier):
    if classifier == 'Logistic Regression':
      clf = load('models/model_best_lr.joblib')
    elif classifier == 'Decision Tree':
      clf = load('models/model_dt.joblib')
    elif classifier == 'Boosting':
      clf = load('models/model_boosting.joblib')
    elif classifier == 'Bagging':
      clf = load('models/model_best_bagging.joblib')
    elif classifier == 'Random Forest':
      clf = load('models/model_best_rf.joblib')
    return clf

  def scores(clf, choice):
    if choice == 'Accuracy':
      return clf.score(X_test, y_test)
    elif choice == 'Confusion matrix':
      return confusion_matrix(y_test, clf.predict(X_test))
    
  clf = prediction(option)
  display = st.radio('What do you want to show?', ('Accuracy', 'Confusion matrix'))
  if display == 'Accuracy':
    st.write(scores(clf, display))
  elif display == 'Confusion matrix':
    st.dataframe(scores(clf, display))
  
  st.write("---")