import streamlit as st
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torchvision.models as models
import joblib
import os
import sys
from io import StringIO
from torchsummary import summary

# Constant names
MODEL_CNN = "CNN (LeNet Architecture)"
MODEL_PYTORCH = "PyTorch Framework"

MODEL_CNN_FILE_PATH = "models/model_lenet.keras"
MODEL_PYTORCH_FILE_PATH = "models/model_pytorch_state_dict.pth"

# Read the train and test sets from the file 'NBA Shot Locations 1997 - 2020-Report2-train-test.joblib'.
X_train, X_test, y_train, y_test = load('NBA Shot Locations 1997 - 2020-Report2-train-test.joblib')

class ShotPredictor(nn.Module):
  def __init__(self, input_size):
    super(ShotPredictor, self).__init__()
    self.fc1 = nn.Linear(input_size, 128)  # First fully connected layer
    self.fc2 = nn.Linear(128, 64)           # Second fully connected layer
    self.fc3 = nn.Linear(64, 32)            # Third fully connected layer
    self.dropout = nn.Dropout(0.2)          # Dropout layer to prevent overfitting
    self.fc4 = nn.Linear(32, 1)             # Final fully connected layer
    self.sigmoid = nn.Sigmoid()             # Sigmoid activation for binary classification
  
  def forward(self, x):
    x = torch.relu(self.fc1(x))     # ReLU activation for the first layer
    x = self.dropout(x)             # Dropout applied after the first layer
    x = torch.relu(self.fc2(x))     # ReLU activation for the second layer
    x = self.dropout(x)             # Dropout applied after the second layer
    x = torch.relu(self.fc3(x))     # ReLU activation for the third layer
    x = self.dropout(x)             # Dropout applied after the third layer
    x = self.sigmoid(self.fc4(x))   # Sigmoid activation for the final layer
    return x
    
def show_deep_learning_page():
  st.write("### Deep Learning")

  # Evaluating
  ACCURACY_CNN_TEST_FILE_PATH = "models/accuracies/accuracy_cnn_test"
  ACCURACY_CNN_TRAIN_FILE_PATH = "models/accuracies/accuracy_cnn_train"
  ACCURACY_PYTORCH_TEST_FILE_PATH = "models/accuracies/accuracy_pytorch_test"
  ACCURACY_PYTORCH_TRAIN_FILE_PATH = "models/accuracies/accuracy_pytorch_train"

  ACCURACY_ON_TEST = "Accuracy on Test set"
  ACCURACY_ON_TRAIN = "Accuracy on Train set"

  choice = [MODEL_CNN, MODEL_PYTORCH]
  classifier = st.selectbox('Choice of the model', choice)
  st.write('The chosen model is :', classifier)

  accuracy_option = st.radio('What do you want to show?', (ACCURACY_ON_TEST, ACCURACY_ON_TRAIN))  

  if classifier == MODEL_CNN:
    if accuracy_option == ACCURACY_ON_TEST:
      filename = ACCURACY_CNN_TEST_FILE_PATH
    elif accuracy_option == ACCURACY_ON_TRAIN:
      filename = ACCURACY_CNN_TRAIN_FILE_PATH
  elif classifier == MODEL_PYTORCH:
    if accuracy_option == ACCURACY_ON_TEST:
      filename = ACCURACY_PYTORCH_TEST_FILE_PATH
    elif accuracy_option == ACCURACY_ON_TRAIN:
      filename = ACCURACY_PYTORCH_TRAIN_FILE_PATH

  # if accuracy already saved, then use it
  if os.path.exists(filename):
    # Load accuracy from the joblib file
    accuracy = joblib.load(filename)
    if classifier == MODEL_CNN:
      st.write(accuracy)
    elif classifier == MODEL_PYTORCH:
      st.write(accuracy.item())
  else:
    # Accuracy is not saved yet. Let's calculate and save it
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

    # Prepare for (CNN LeNet)
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components = 20)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Reshape input data for CNN
    X_train_reshaped = X_train_pca.reshape(-1, 4, 5, 1)  # Adjust based on PCA components
    X_test_reshaped = X_test_pca.reshape(-1, 4, 5, 1)
    # END of Prepare for (CNN LeNet)

    # Prepare for PyTorch
    # Convert DataFrame to NumPy arrays
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_np)
    X_test_tensor = torch.FloatTensor(X_test_np)
    y_train_tensor = torch.FloatTensor(y_train_np).view(-1, 1)
    y_test_tensor = torch.FloatTensor(y_test_np).view(-1, 1)
    # END of Prepare for PyTorch
    
    def scores(accuracy_option, classifier):
      if classifier == MODEL_CNN:
        if accuracy_option == ACCURACY_ON_TEST:
          return accuracy_cnn(X_test_reshaped, y_test, ACCURACY_CNN_TEST_FILE_PATH)
        elif accuracy_option == ACCURACY_ON_TRAIN:
          return accuracy_cnn(X_train_reshaped, y_train, ACCURACY_CNN_TRAIN_FILE_PATH)
      elif classifier == MODEL_PYTORCH:
        if accuracy_option == ACCURACY_ON_TEST:
          return accuracy_pytorch(X_test_tensor, y_test_tensor, ACCURACY_PYTORCH_TEST_FILE_PATH)
        elif accuracy_option == ACCURACY_ON_TRAIN:
          return accuracy_pytorch(X_train_tensor, y_train_tensor, ACCURACY_PYTORCH_TRAIN_FILE_PATH)
        
    st.write(scores(accuracy_option, classifier))
  # END of else

  if st.checkbox("Show model summary"):
    show_summary(classifier)
  st.write("---")

def accuracy_cnn(X_reshaped, y, joblib_filename):
  # Load the saved model
  clf = load_model(MODEL_CNN_FILE_PATH)

  st.write(clf.summary())

  # Evaluate the model on test data
  evaluation_results = clf.evaluate(X_reshaped, y)

  # Extract test accuracy from evaluation_results
  # test_loss = evaluation_results[0]
  accuracy = evaluation_results[1]
  
  # Save accuracy to joblib file
  joblib.dump(accuracy, joblib_filename)
  # Print accuracy
  #st.write(f'Accuracy on test set: {test_accuracy * 100:.2f}%')
  return accuracy

def accuracy_pytorch(X_tensor, y_tensor, joblib_filename):
  # Load the saved model
  # Create an instance of our model
  input_size = (39,)  # Input size as a tuple (determined by the number of features)
  clf = ShotPredictor(input_size[0])  # Initialize model with the first element of the tuple
  # Load the saved state dictionary
  state_dict = torch.load(MODEL_PYTORCH_FILE_PATH)
  # Load the state dictionary into the model
  clf.load_state_dict(state_dict)

  with torch.no_grad():
    clf.eval()  # Set the model to evaluation mode
    test_outputs = clf(X_tensor)  # Get predicted outputs for the test set
    test_outputs = (test_outputs > 0.5).float()  # Convert outputs to binary predictions
    accuracy = (test_outputs == y_tensor).float().mean()  # Calculate accuracy
    # Save accuracy to joblib file
    joblib.dump(accuracy, joblib_filename)
    # Print accuracy
    #st.write(f'Accuracy on test set: {accuracy.item()*100:.2f}%')
    return accuracy.item()
    
def show_summary(classifier):
  if classifier == MODEL_CNN:
    # Load module from file and compile it
    clf = load_model(MODEL_CNN_FILE_PATH)
    clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', 'mean_absolute_error'])
    
    # Capture the summary output to a string buffer
    buffer = StringIO()
    clf.summary(print_fn=lambda x: buffer.write(x + '\n'))
    summary_str = buffer.getvalue()

    # Display the model summary in Streamlit
    st.text(summary_str)

  elif classifier == MODEL_PYTORCH:
    # Create an instance of our model
    input_size = (39,)  # Input size as a tuple (determined by the number of features)
    clf = ShotPredictor(input_size[0])  # Initialize model with the first element of the tuple
    # Load the saved state dictionary
    state_dict = torch.load(MODEL_PYTORCH_FILE_PATH)
    # Load the state dictionary into the model
    clf.load_state_dict(state_dict)

    # Capture the summary output to a string buffer
    buffer = StringIO()
    # Redirect the stdout to the buffer
    sys.stdout = buffer
    # Print the summary to capture it in the buffer
    summary(clf, input_size)
    # Reset stdout back to normal
    sys.stdout = sys.__stdout__
    summary_str = buffer.getvalue()

    # Display the model summary in Streamlit
    st.text(summary_str)
