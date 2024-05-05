import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from joblib import load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

# Constant names
LOGISTIC_REGRESSION = "Logistic Regression"
DECISION_TREE = "Decision Tree"
BOOSTING = "Boosting"
BAGGING = "Bagging"
RANDOM_FOREST = "Random Forest"

class DataHolder:
  def __init__(self):
    self.X_train = []
    self.X_test = []
    self.y_train = []
    self.y_test = []

  def load_data(self, filename):
    # Read the train and test sets from the file 'NBA Shot Locations 1997 - 2020-Report2-train-test.joblib'.
    self.X_train, self.X_test, self.y_train, self.y_test = load(filename)

# Create an instance of DataHolder
data_holder = DataHolder()

def show_modelling_page():
  st.write("### Modelling")

  data_holder.load_data('NBA Shot Locations 1997 - 2020-Report2-train-test.joblib')

  # Show single model parameters
  show_single_model()

  # Comparison of Accuracies for each Model
  if st.checkbox("Comparison of **Accuracies** for each Model"):
    comparison_of_accurasies()
    st.write("**Random Forest** model showed the highest accuracy score for the training and test data sets.")
  st.write("---")

  # Comparison of ROC Curves for each Model
  st.write("### ROC Curves")
  st.markdown('''
                The **ROC curve** (for Receiver Operating Characteristic) is the ideal tool to summarize the performance of a binary classifier according to all possible thresholds. It avoids the time-consuming task of predicting classes for different thresholds, and evaluating the confusion matrix for each of these thresholds.

                Graphically, the ROC measure is represented as a curve which gives the true positive rate, the sensitivity, as a function of the false positive rate, the antispecificity ( = 1 - specificity). Each classification threshold value will provide a point on the ROC curve, which will go from (0, 0) to (1, 1).

                The closer the curve is to the (0,1) point (top left), the better the predictions. A model with sensitivity and specificity equal to 1 is considered perfect.
                ''')
  if st.checkbox("Comparison of ROC Curves for each Model"):
    comparison_of_ROC_curves()
  st.write("---")

  # Interpretation of results  
  show_SHAP_plots()

def show_SHAP_plots():
  st.write("### Interpretation of results")
  st.markdown('''
We have finished all our models. Let's have a comparative analysis of feature interpretation using **SHAP** (SHapley Additive exPlanations) values across multiple machine learning models.
              ''')

  if st.checkbox("**SHAP** results for ***Boosting*** and ***Random Forest***"):
    # WARNING: This call can take a some hours
    # percentage = 0.00001    # fast run
    percentage = 0.01    # 1% of the full dataset, which is about 40k rows

    # Take 1% of the full dataset
    sample_size = int(len(data_holder.X_train) * percentage)

    # Take a random sample of the data
    random_indices = np.random.choice(len(data_holder.X_train), sample_size, replace=False)
    X_train_short = data_holder.X_train.iloc[random_indices]
    y_train_short = data_holder.y_train.iloc[random_indices]

     # Read models from files that we created before
    model_boosting = load_model(BOOSTING)
    model_rf = load_model(RANDOM_FOREST)

    # Define models and their respective titles
    models = [
        (model_boosting, X_train_short, BOOSTING),
        (model_rf, X_train_short, RANDOM_FOREST)
    ]

    save_dir = "./models/shap_plots"
    # Loop through models and generate SHAP summary plots
    for model, X, title in models:
      plot_filename = f"{save_dir}/{title}_shap_plot.png"

      # Check if the plot image already exists
      if os.path.isfile(plot_filename):
          st.image(plot_filename)
          st.write("")
      else:
        fig = plt.figure(figsize=(10, 7))
        model.fit(X, y_train_short)
        background_summary = shap.kmeans(X, 10)  # Adjust the number of clusters as needed
        explainer = shap.KernelExplainer(model.predict_proba, background_summary)
        shap_values = explainer(X)
        shap.summary_plot(shap_values[:, :, 0], X, show=False)
        plt.title(title)
        plt.savefig(plot_filename)  # Save each plot to a PNG file
        plt.close()  # Close the plot to release resources
        st.pyplot(fig)
    
    # Add comments
    st.markdown('''
The plots reveal that across all models, the most critical features include '**Action Type Frequency**', '**Shot Distance**', '**Y Location**', and '**ShotZoneRange_Less Than 8 ft.**', which is intuitive given their relevance in basketball dynamics. Certain actions significantly increase the probability of a successful shot, while proximity to the basket strongly influences shot outcomes. 

In the Boosting model, '**Y Location**' emerges as the third crucial feature, with a substantial impact when considered, despite being overshadowed by the first two features. '**Action Type Frequency**' and '**Shot Distance**' consistently influence many predictions, albeit with varying degrees, while '**Y Location**' has not so much impact on a lot of predictions, but when it has it has a huge impact.  

In contrast, the **Random Forest** model shows relatively equal importance among all features in predicting shot outcomes.
                ''')

def show_single_model():
  choice = [LOGISTIC_REGRESSION, DECISION_TREE, BOOSTING, BAGGING, RANDOM_FOREST]
  option = st.selectbox('Choice of the model', choice)
  st.write('The chosen model is :')

  clf = load_model(option)
  st.write(clf)

  def scores(clf, choice):
    if choice == 'Accuracy':
      return clf.score(data_holder.X_test, data_holder.y_test)
    elif choice == 'Confusion matrix':  
      return confusion_matrix(data_holder.y_test, clf.predict(data_holder.X_test))
    elif choice == 'Classification report':
      return classification_report(data_holder.y_test, clf.predict(data_holder.X_test))
  
  display = st.radio('What do you want to show?', ('Accuracy', 'Confusion matrix', 'Classification report'))

  if display == 'Accuracy':
    st.write(scores(clf, display))
  elif display == 'Confusion matrix':
    st.dataframe(scores(clf, display))
  elif display == 'Classification report':
    st.text(scores(clf, display))


def comparison_of_accurasies():
  models = models_dict()

  model_lr = models[LOGISTIC_REGRESSION]
  model_dt = models[DECISION_TREE]
  model_boosting = models[BOOSTING]
  model_bagging = models[BAGGING]
  model_rf = models[RANDOM_FOREST]

  accuracy_train_lr = model_lr.score(data_holder.X_train, data_holder.y_train)
  accuracy_test_lr = model_lr.score(data_holder.X_test, data_holder.y_test)

  accuracy_train_dt = model_dt.score(data_holder.X_train, data_holder.y_train)
  accuracy_test_dt = model_dt.score(data_holder.X_test, data_holder.y_test)

  accuracy_train_ac = model_boosting.score(data_holder.X_train, data_holder.y_train)
  accuracy_test_ac = model_boosting.score(data_holder.X_test, data_holder.y_test)

  accuracy_train_bagging = model_bagging.score(data_holder.X_train, data_holder.y_train)
  accuracy_test_bagging = model_bagging.score(data_holder.X_test, data_holder.y_test)

  accuracy_train_rf = model_rf.score(data_holder.X_train, data_holder.y_train)
  accuracy_test_rf = model_rf.score(data_holder.X_test, data_holder.y_test)

  # Sample data (replace with your actual values for each model)
  train_accuracies = [accuracy_train_lr, accuracy_train_dt, accuracy_train_ac, accuracy_train_bagging, accuracy_train_rf]
  test_accuracies = [accuracy_test_lr, accuracy_test_dt, accuracy_test_ac, accuracy_test_bagging, accuracy_test_rf]

  # Define colors for train and test bars
  test_color = '#1f77b4'  # blue
  train_color = '#ff7f0e'  # orange

  # Define the height of each bar
  bar_height = 0.35

  # Define the y positions for each model
  y = np.arange(len(models.keys()))

  # Create a figure with larger size
  fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed

  # Plot the bars for test accuracies
  test_bars = ax.barh(y - bar_height/2, test_accuracies, height=bar_height, color=test_color, label='Test Accuracy')

  # Plot the bars for train accuracies
  train_bars = ax.barh(y + bar_height/2, train_accuracies, height=bar_height, color=train_color, alpha=0.5, label='Train Accuracy')

  # Set the y-axis ticks and labels
  ax.set_yticks(y)
  ax.set_yticklabels(models.keys())  # Use model names as y-axis labels

  # Set the title, labels, and legend
  ax.set_title('Comparison of Model Accuracies')
  ax.set_xlabel('Accuracy')

  # Move the legend to the upper left outside the plot area
  ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

  # Set x-axis limits to ensure all bars are visible
  ax.set_xlim(0, max(max(train_accuracies), max(test_accuracies)) + 0.1)  # Add a buffer for visibility

  # Show grid lines on both axes, set them behind the bars
  ax.grid(True, linestyle='--', axis='x', zorder=0)

  # Annotate the bars with accuracy values
  for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
      ax.text(train_acc + 0.01, i + bar_height/2, f'{train_acc:.5f}', va='center')
      ax.text(test_acc + 0.01, i - bar_height/2, f'{test_acc:.5f}', va='center')

  # Show the plot
  plt.tight_layout()  # Adjust layout to prevent overlapping labels
  st.pyplot(fig)


def comparison_of_ROC_curves():
  models = models_dict()

  # Train and plot ROC curves for each model
  fig = plt.figure(figsize=(10, 7))
  for name, m in models.items():
    y_pred_prob = m.predict_proba(data_holder.X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(data_holder.y_test, y_pred_prob)
    auc_score = roc_auc_score(data_holder.y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

  # Plot ROC curve for random guessing (diagonal line)
  plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Guessing')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend()
  plt.grid(True)
  st.pyplot(fig)

def load_model(classifier):
  if classifier == LOGISTIC_REGRESSION:
    clf = load('models/model_best_lr.joblib')
  elif classifier == DECISION_TREE:
    clf = load('models/model_dt.joblib')
  elif classifier == BOOSTING:
    clf = load('models/model_boosting.joblib')
  elif classifier == BAGGING:
    clf = load('models/model_best_bagging.joblib')
  elif classifier == RANDOM_FOREST:
    clf = load('models/model_best_rf.joblib')
  return clf

@st.cache_data
def models_dict():
  # Initialize models
  model_lr = load_model(LOGISTIC_REGRESSION)
  model_dt = load_model(DECISION_TREE)
  model_boosting = load_model(BOOSTING)
  model_bagging = load_model(BAGGING)
  model_rf = load_model(RANDOM_FOREST)

  models = {
      LOGISTIC_REGRESSION: model_lr,
      DECISION_TREE: model_dt,
      BOOSTING: model_boosting,
      BAGGING: model_bagging,
      RANDOM_FOREST: model_rf
  }
  return models