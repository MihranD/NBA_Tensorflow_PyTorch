# NBA Data Science
***
This project demonstrates Data Science techniques using a machine learning model to predict NBA shot accuracy. 
The focus is on developing Streamlit app that provides visualizations for model training experiments and results. 
Key elements include feature engineering, hyperparameter tuning, and interpretability through SHAP values, 
allowing for comprehensive model evaluation. 
The project integrates essential data science libraries, such as Pandas, NumPy, and Matplotlib, 
to manage data processing and create visual insights.

![NBA Court](https://github.com/MihranD/nba_ds/blob/main/images/nba_court.png)

## Project Organization
----------------------------------------------------------------------------------------------
    ├── .github             <- Scripts for Github configs
    ├── .gitignore          <- Includes files and folders that we don't want to control
    |
    ├── images              <- Images for use in this project
    │   ├── nba_court.png   <- NBA court image
    │   └── nba-logo.png    <- NBA logo image
    |
    ├── models                          <- Includes trained models and their appropriate parameters: 
    │   │                               <- (accuracies, classification_reports, shap_plots)
    │   ├── accuracies                  <- Accuracy results for CNN and PyTorch models
    │   │   ├── accuracy_cnn_test       <- CNN LeNet test accuracy result
    │   │   ├── accuracy_cnn_train      <- CNN LeNet train accuracy result
    │   │   ├── accuracy_pytorch_test   <- PyTorch test accuracy result
    │   │   └── accuracy_pytorch_train  <- PyTorch train accuracy result
    │   │
    │   ├── classification_reports                  <- Classification reports generated in this project
    │   │   └── classification_report_model_cnn.txt <- CNN LeNet classification report
    │   │   
    │   ├── model_best_bagging.joblib                   <- Trained Bagging model with best hyperparameters
    │   ├── model_best_lr.joblib                        <- Trained Logistic regression model with best hyperparameters
    │   ├── model_best_rf.joblib                        <- Trained Random Forests model with best hyperparameters
    │   ├── model_boosting.joblib                       <- Trained AdaBoosting model
    │   ├── model_dt.joblib                             <- Trained Decision Tree model
    │   ├── model_lenet_training_history_leaky_relu.pkl <- Trained CNN LeNet model's history dictionary
    │   ├── model_lenet.keras                           <- Trained CNN LeNet model
    │   ├── model_pytorch_state_dict.pth                <- Trained PyTorch model's state dictionary
    │   │
    │   └── shap_plots                      <- SHAP images generated in this project
    │       ├── Boosting_shap_plot.png      <- Trained Boosting model's SHAP values'
    │       └── Random Forest_shap_plot.png <- Trained Random Forests model's SHAP values
    |
    ├── sources                                 <- Source code for use in this project. Each file represents each page in streamlit project
    │   ├── introduction_page.py                <- Introduction to the project
    │   ├── preprocessing_page.py               <- Preprocessing and feature engineering
    │   ├── visualisation_page.py               <- Visualizations and Statistics
    │   ├── preprocessing_for_modelling_page.py <- Preprocessing for modeling purposes
    │   ├── modelling_page.py                   <- Base Models
    │   ├── deep_learning_page.py               <- Deep Learning
    │   ├── conclusion.py                       <- Conclusion
    │   └── utils.py                            <- Some util functions
    
    |
    ├── streamlit_app.py    <- Main python file
    |
    ├── NBA player shot analysis - Report.ipynb <- Jupiter Notebook report file
    ├── NBA player shot analysis - Report.pdf   <- PDF report file
    ├── NBA Shot Locations 1997 - 2020.csv      <- contains dataset (IMPORTANT: original file you can download from here: 
    ├──                                         https://www.kaggle.com/datasets/jonathangmwl/nba-shot-locations)
    |
    ├── requirements.txt    <- The required libraries to deploy this project. 
    |                       Generated with `pip freeze > requirements.txt`
    └── README.md   <- The top-level README for developers using this project.

## Using the App

1. Open terminal app

2. Change directory to 'streamlit' folder
 
3. This step is optional: you can replace 'NBA Shot Locations 1997 - 2020.csv' dataset file from here: https://www.kaggle.com/datasets/jonathangmwl/nba-shot-locations

4. run this command:
```bash
streamlit run streamlit_app.py
``` 

5. Select 'Preprocessing for modeling purposes' page for generating joblib file for train_test_split
