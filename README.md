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
    
## Project Introduction

### Business Problem

Our goal is to enhance our competitive edge in the NBA by using advanced statistical analysis and digital tracking tools to evaluate shot frequency and shooting efficiency among top players, enabling data-driven strategies that improve team performance.

### Dataset

NBA shot dataset spanning the years 1997 to 2020. The dataset contains comprehensive information on shot locations in NBA games, allowing for detailed analysis of shot frequency and eﬀiciency among players during this period.
The dataset used in this project is freely available on Kaggle: (https://www.kaggle.com/jonathangmwl/nba-shot-locations).

### Problem Definition

We aim to predict the probability of a shot being made by each player, indicating whether a shot is successful or not. This problem naturally aligns with a `binary classification task`, where shots are categorized as either made or missed.

## Pre-processing and feature engineering

In this step, we prepared the dataset for model training by performing essential data pre-processing and feature engineering tasks. Specifically, we:

* Analyzed Dataset Structure
* Checked for Missing Values and Duplicates
* Explored Value Distribution
* Evaluated Class Balance
* Identified Outliers

These steps provided a strong foundation for building a robust model by ensuring data quality and addressing key feature engineering needs.

## Visualizations and Statistics

To gain deeper insights into the dataset, we performed extensive exploratory data analysis (EDA) through various diagrams and statistical plots. This included visualizations to understand feature distributions, relationships between variables, and trends within the data. Here are a few selected diagrams for illustration:

![Shot distribution by zone](https://github.com/MihranD/nba_ds/blob/main/images/shot-distribution-by-zone.png)

This count plot displays the distribution of both made and missed shots for each shot zone basic. The bars grouped by shot outcome (made or missed), with different colors representing each outcome. This visualization provides a clear comparison between the number of made and missed shots in each shot zone basic.

![Shot accuracy by remaining minutes](https://github.com/MihranD/nba_ds/blob/main/images/shot-accuracy-by-remaining-minutes.png)

The graph highlights how shot accuracy declines as the game progresses, likely due to fatigue and pressure, with a narrowing confidence interval in the final minutes reflecting heightened certainty in critical shot outcomes. This underscores the impact of time management and strategic decisions on game results.

## Preprocessing NBA player shot dataset for modeling purposes

Before we delve into model development, it’s crucial to ensure that our data is adequately prepared. This involves transforming qualitative attributes into quantitative representations, normalizing features, and other necessary preparations to optimize the performance of our models. Particularly we did:
* Transform attributes with high cardinality
* Transform categorical attributes
* Transform quantitative attributes, which have unique id values
* Transform date attribute

After transformation all features into numeric values, we can proceed to create a correlation matrix specifically for the `Shot Made Flag` target variable:

![Correlation heatmap](https://github.com/MihranD/nba_ds/blob/main/images/correlation-heatmap.png)

## Split train and test parts

We randomly divided the matrices into a training set and a test set corresponding to 80% and 20% of the total amount of available data respectively. Added the argument `random_state = 66` for randomness reproducibility.

## NBA player shot analysis with different models

In the NBA shot analysis project, several machine learning algorithms were experimented with to find the most suitable model for predicting shot outcomes. The algorithms tried include:
* Logistic Regression 
* Support Vector Machines (SVM) 
* Random Forests 
* Boosting 
* Bagging 
* Convolutional Neural Networks (CNN)

### Comparison of Hyperparameters for each Model

![Comparison hyperparameters](https://github.com/MihranD/nba_ds/blob/main/images/comparison-hyperparameters.png)

### Comparison of Accuracies for each Model

![Comparison accuracies](https://github.com/MihranD/nba_ds/blob/main/images/comparison-accuracies.png)

`Random Forest` model showed the highest accuracy score for the training and test data sets.

### Comparison of Confusion Matrix for each Model

![Comparison confusion matrixes](https://github.com/MihranD/nba_ds/blob/main/images/comparison-confusion-matrixes.png)

### Comparison of ROC Curves for each Model

![Comparison ROC](https://github.com/MihranD/nba_ds/blob/main/images/comparison-roc.png)

The ROC shows that all models struggle to increase sensitivity without disproportionately increasing specificity. As our goal is to increase recall, this means that these models will further increase false positives significantly. In addition, the AUC shows us that the overall accuracy across all models is similarly.

## NBA player shot analysis with Deep Learning (CNN, LeNet architecture)

The objective is to implement the foundational LeNet5 Convolutional Neural Network (CNN) architecture in Python using Keras, as it forms the basis for many advanced deep learning models.

![CNN LeNet](https://github.com/MihranD/nba_ds/blob/main/images/cnn-lenet.png)

The fact that the CNN model’s accuracy is around 64% suggests that while convolutional neural networks can offer advantages in certain scenarios, they may not always outperform simpler models on every dataset.

## Interpretation of results

Comparative analysis of feature interpretation using SHAP (SHapley Additive exPlanations) values across multiple machine learning models:

![SHAP Random Forests](https://github.com/MihranD/nba_ds/blob/main/images/shap-rf.png)

The plots reveal that the most critical features include `Action Type Frequency`, `Shot Distance` and `Y Location`, which is intuitive given their relevance in basketball dynamics.

## Conclusion

Our main contribution lies in achieving a 64% accuracy in predicting NBA shot success, surpassing existing benchmarks for similar datasets. The accuracy of NBA shot prediction models is influenced by the sport's complexity, data limitations, feature engineering, temporal dynamics, sample size, and evaluation metrics, along with human and random factors in games. Together, these elements impact the model's ability to capture relevant patterns and produce reliable results.

## Using the Streamlit App

1. Open terminal app

2. Change directory to `streamlit` folder
 
3. This step is optional: you can replace 'NBA Shot Locations 1997 - 2020.csv' dataset file from here: https://www.kaggle.com/datasets/jonathangmwl/nba-shot-locations

4. run this command:
```bash
streamlit run streamlit_app.py
``` 

5. Select `Preprocessing for modeling purposes` page for generating joblib file for train_test_split

5. App should work without any problem.

![Streamlit app screenshot](https://github.com/MihranD/nba_ds/blob/main/images/streamlit-app-screenshot.png)
