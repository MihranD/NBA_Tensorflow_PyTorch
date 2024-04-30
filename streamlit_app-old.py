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

df=pd.read_csv("train.csv")

st.title("NBA player shot analysis")
st.sidebar.title("Table of contents")
pages=["Introduction to the project", 
       "Preprocessing and feature engineering", 
       "Visualizations and Statistics", 
       "Preprocessing for modeling purposes", 
       "Modelling"]
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
  st.write("### Modelling")

  df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

  y = df['Survived']
  X_cat = df[['Pclass', 'Sex',  'Embarked']]
  X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

  for col in X_cat.columns:
    X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
  for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())
  
  X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
  X = pd.concat([X_cat_scaled, X_num], axis = 1)

  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
  X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import SVC
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import confusion_matrix
  
  def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()
    elif classifier == 'SVC':
        clf = SVC()
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    import joblib
    joblib.dump(clf, "model")
    return clf
  
  def scores(clf, choice):
    if choice == 'Accuracy':
        return clf.score(X_test, y_test)
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test))
    
  choice = ['Random Forest', 'SVC', 'Logistic Regression']
  option = st.selectbox('Choice of the model', choice)
  st.write('The chosen model is :', option)

  clf = prediction(option)
  display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
  if display == 'Accuracy':
    st.write(scores(clf, display))
  elif display == 'Confusion matrix':
    st.dataframe(scores(clf, display))