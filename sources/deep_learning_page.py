import streamlit as st
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import torch
import torchvision.models as models

def show_deep_learning_page():
  st.write("### Deep Learning")

  # Read the train and test sets from the file 'NBA Shot Locations 1997 - 2020-Report2-train-test.joblib'.
  X_train, X_test, y_train, y_test = load('NBA Shot Locations 1997 - 2020-Report2-train-test.joblib')

  # Initialize the StandardScaler
  scaler = StandardScaler()

  columns_to_scale = ['Period',
                      'Minutes Remaining',
                      'Seconds Remaining', 
                      'Shot Distance', 
                      'X Location', 
                      'Y Location']

  # Scale the features for train set and replace the original columns with the scaled features
  X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])

  # Scale the features for test set and replace the original columns with the scaled features
  X_test[columns_to_scale] = scaler.fit_transform(X_test[columns_to_scale])

  # Apply PCA for dimensionality reduction
  pca = PCA(n_components = 20)
  X_train_pca = pca.fit_transform(X_train)
  X_test_pca = pca.transform(X_test)

  # Reshape input data for CNN
  X_train_reshaped = X_train_pca.reshape(-1, 4, 5, 1)  # Adjust based on PCA components
  X_test_reshaped = X_test_pca.reshape(-1, 4, 5, 1)
    
  # Evaluating
  choice = ['CNN (LeNet)', 'PyTorch']
  option = st.selectbox('Choice of the model', choice)
  st.write('The chosen model is :', option)

  def prediction(classifier):
    if classifier == 'CNN (LeNet)':
      # Load the saved model
      clf = load_model('models/model_lenet.keras')
    elif classifier == 'PyTorch':
      # Create an instance of our model
      clf = models.resnet18()
      # Load the saved state dictionary
      state_dict = torch.load('models/model_pytorch.pth')
      # Load the state dictionary into the model
      clf.load_state_dict(state_dict)
    return clf

  def scores(clf, choice):
    if choice == 'Accuracy on Test set':
      return accuracy_cnn(clf, X_test_reshaped, y_test)
      '''
      with torch.no_grad():
          clf.eval()  # Set the model to evaluation mode
          test_outputs = clf(X_test_tensor)  # Get predicted outputs for the test set
          test_outputs = (test_outputs > 0.5).float()  # Convert outputs to binary predictions
          accuracy = (test_outputs == y_test_tensor).float().mean()  # Calculate accuracy
          print(f'Accuracy on test set: {accuracy.item()*100:.2f}%')
          '''
    elif choice == 'Accuracy on Train set':
      return accuracy_cnn(clf, X_train_reshaped, y_train)
    
  clf = prediction(option)
  display = st.radio('What do you want to show?', ('Accuracy on Test set', 'Accuracy on Train set'))
  if display == 'Accuracy on Test set':
    st.write(scores(clf, display))
  elif display == 'Accuracy on Train set':
    st.write(scores(clf, display))
  
  st.write("---")

def accuracy_cnn(clf, X_reshaped, y):
  # Evaluate the model on test data
  evaluation_results = clf.evaluate(X_reshaped, y)

  # Extract test loss and test accuracy from evaluation_results
  test_loss = evaluation_results[0]
  test_accuracy = evaluation_results[1]
  return test_accuracy