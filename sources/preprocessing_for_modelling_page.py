import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sources.utils import read_df

def show_preprocessing_for_modelling_purposes_page():
  st.write("### Preprocessing for modeling purposes")
  
  # Read the dataset into a DataFrame
  df = read_df()

  # Presentation of data
  st.markdown('''
In the previous parts we undertook essential preprocessing steps and conducted exploratory data analysis through visualization techniques. Our focus now shifts towards building machine learning models to gain deeper insights and make predictions based on our dataset. However, before we delve into model development, it's crucial to ensure that our data is adequately prepared. This involves transforming qualitative attributes into quantitative representations, normalizing features, and other necessary preparations to optimize the performance of our models.

In this part, we will carry out these preliminary tasks, laying the groundwork for our subsequent modeling endeavors. By converting qualitative attributes to quantitative ones and applying normalization techniques, we aim to create a clean and standardized dataset that is well-suited for machine learning analysis. 

Our ultimate goal is to generate preprocessed and refined training and testing datasets. These datasets will lay the groundwork for our modeling endeavors, facilitating the smooth progression of building and assessing machine learning models in the subsequent phases of our project.
              ''')
  st.write("---")

  # Categorical variable transformation and feature normalization
  st.write("### Categorical variable transformation and feature normalization")
  st.markdown('''
In machine learning and statistical modeling, the quality and format of input data significantly impact the performance and accuracy of predictive models. When working with datasets that contain categorical variables (attributes that represent qualitative characteristics) transforming these variables into numerical representations becomes imperative. Categorical variable transformation enables machine learning algorithms to interpret and analyze these variables effectively, allowing for meaningful insights and accurate predictions.

Furthermore, to ensure fair and optimal treatment of different features during model training, it is essential to normalize numerical features. Feature normalization standardizes the scale of input features, preventing certain variables from dominating the model training process due to their larger magnitude. By scaling features to a common range or distribution, feature normalization promotes stability, convergence, and improved generalization of machine learning models.

In summary, the process of categorical variable transformation and feature normalization is fundamental in preparing datasets for machine learning tasks, ensuring that models can effectively learn from and generalize to new data.
              ''')
  
  # Transform attributes with high cardinality
  df = transform_attributes_with_high_cardinality(df)

  # Transform other categorical attributes
  df = transform_other_categorical_attributes(df)

  # Transform quantitative attributes, which have unique id values
  df = transform_quantitative_attributes(df)

  # Transform date attribute
  df = transform_date_attribute(df)

  # Show Correlation matrix
  show_correlation_matrix(df)

  # Split train and test parts
  split_train_and_test_parts(df)
  save_train_test_set(df)

  # Conclusion for preprocessing
  conclusion_for_preprocessing(df)

# Conclusion for preprocessing
def conclusion_for_preprocessing(df):
  st.write("#### Conclusion for preprocessing")
  st.write("We converted all categorical attributes to quantitative. Our dataset ready for further modeling.")
  if st.checkbox("Show preprocessed dataset"):
    st.dataframe(df.dtypes)
  st.markdown('''
For now, our focus has been on converting categorical attributes into quantitative representations. We'll defer the scaling process to the appropriate modeling stage, where each model can adopt its own scaling approach (Standardization, MinMax Scaling or other type of scaling) based on its requirements and the characteristics of the data. The current emphasis has been on ensuring that our data is prepared in a format conducive to modeling, with further considerations such as feature scaling left for subsequent stages tailored to each specific model.
              ''')
  
# Save train-test sets in a file
def save_train_test_set(df):
  data = df.drop('Shot Made Flag', axis = 1)
  target = df['Shot Made Flag']
  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 66)
  # WARNING: This call can take a few seconds.
  # Save the data
  if st.button("Save data in joblib file"):
    dump((X_train, X_test, y_train, y_test), 'NBA Shot Locations 1997 - 2020-Report2-train-test.joblib')
    st.write("New joblib file generated successfully.")
    # Later model parts can read the data from it with help of: 
    # X_train, X_test, y_train, y_test = load('NBA Shot Locations 1997 - 2020-Report2-train-test.joblib')

# Split train and test parts
def split_train_and_test_parts(df):
  st.write("#### Split train and test parts")
  st.write("Let's create a data DataFrame in which we will store the features and create the target ('Shot Made Flag') data variable.")
  st.markdown('''
In order to test the performance of the classification model, it is necessary to select a part of the data that is dedicated to the evaluation and that is therefore not taken into account in the training of the model.
To do this, the data must be systematically divided into a training set (X_train and y_train) and a test set (X_test and y_test).

Usually, the test set size is between 15% and 30% of the total amount of data. The choice of the distribution depends mainly on the quantity and quality of the available data. We will choose 20%.

Let's randomly divide the matrices into a training set and a test set corresponding to 80% and 20% of the total amount of available data respectively. Add the argument random_state = 66 for randomness reproducibility.
              ''')
  st.write("To build a classification model, we need to train our model on the training set only.")
  st.markdown('''
We intend to store these training and testing sets in a file for later use in modeling tasks. Ensuring consistency, all models will be trained on these identical sets. This approach will provide us with a clearer understanding of each model's performance under the same conditions, aiding in the determination of the most effective model.
              ''')

# Calculate the correlation matrix
@st.cache_data
def correlation_matrix(df):
  return df.corr()

# Get the correlation of features with the target variable
@st.cache_data
def target_correlation(corr_matrix):
  return corr_matrix['Shot Made Flag'].abs().sort_values(ascending=False)

# Show Correlation matrix
def show_correlation_matrix(df):
  st.write("#### Correlation matrix")
  st.write("Now that we have transformed all features into numeric values, we can proceed to create a correlation matrix specifically for the 'Shot Made Flag' target variable.")

  # Calculate the correlation matrix
  corr_matrix = correlation_matrix(df)
  # Get the correlation of features with the target variable
  target_corr = target_correlation(corr_matrix)
  
  if st.checkbox("Top Ten Correlated Features Heatmap"):
    # Get the top ten most correlated features with the target variable
    top_corr_features = target_corr[1:11].index  # Exclude the target itself

    # Select only the top ten most correlated features from the correlation matrix
    corr_matrix_top = corr_matrix.loc[top_corr_features, top_corr_features]

    # Plot the heatmap
    fig = plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    sns.heatmap(corr_matrix_top, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Top Ten Correlated Features Heatmap')
    st.pyplot(fig)

    st.markdown('''
One can see by viewing these map that for an NBA player to increase his chances of making a shot, he should be as close as possible to the basket (ShotZoneRange_Less Than 8 ft., ShotZoneBasic_Restricted Area). 

The high correlation coefficient of 0.99 between 'ShotType_3PT Field Goal' and 'ShotZoneRange_24+ ft.' signifies a strong relationship between attempting three-point shots and shooting from beyond the 24-foot mark on the court. This suggests that NBA players who frequently attempt three-pointers are more likely to do so from long distances, beyond the traditional three-point line. Coaches and analysts may use this insight to strategize offensive plays that leverage players' shooting abilities from long range, optimizing scoring opportunities based on shot selection and player positions on the court.
                ''')
    st.write("---")

  if st.checkbox("Correlation of Features with 'Shot Made Flag'"):
    features = target_corr.index[1:16].tolist()  # Exclude 'Shot Made Flag' and get the first 15 features
    correlations = target_corr.values[1:16].tolist()  # Exclude its correlation value and get the first 15 correlations

    # Plot the bar plot with adjusted spacing and grid lines for the first 15 features
    fig = plt.figure(figsize=(10, len(features) * 0.5))  # Adjust figure height based on number of features
    bars = plt.barh(features, correlations, color='skyblue')
    plt.xlabel('Correlation')
    plt.title('Correlation of Features with "Shot Made Flag" (First 15 Features)')
    plt.gca().invert_yaxis()  # Invert y-axis to display highest correlation at the top
    plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines with dashed style and transparency
    plt.tight_layout()  # Adjust spacing to prevent text overlap
    st.pyplot(fig)

    st.markdown('''
The correlation plot highlights the features most strongly correlated with 'Shot Made Flag'.

'Action Type Frequency' shows a high positive correlation, suggesting that the frequency of specific actions during a shot attempt significantly impacts shot success.

'ShotZoneBasic_Restricted Area' and 'Shot Distance' also exhibit strong positive correlations, indicating that shots taken from restricted areas and shorter distances are more likely to be made.

Furthermore, 'ShotType_2PT Field Goal' and 'ShotType_3PT Field Goal' reflect the influence of shot type on success rates, with both two-point field goals (2PT) and three-point field goals (3PT) having an equal impact on shot success rates.
                ''')
    st.write("---")

# Transform date attribute
def transform_date_attribute(df):
  st.write("#### Transform date attribute")
  df['Game Date'] = pd.to_datetime(df['Game Date'], format='%Y%m%d')
  
  min_date = df['Game Date'].min()
  max_date = df['Game Date'].max()

  st.markdown('''
The approach to handling date features in classification models is generally similar across different types of classifiers, including logistic regression. Date objects cannot be directly used as predictors in classification models, as these models require numerical input features.

Therefore, regardless of the specific classification model we are using, we typically need to perform feature engineering on the date column to extract relevant numerical features.

There are some common approaches to feature engineering with date data in classification models. We will use Extract Components. This approach extracts relevant components from the date, such as year, month, day, day of the week. These components can then be encoded as numerical features.
              ''')
  
  # Extract components
  df['Year'] = df['Game Date'].dt.year
  df['Month'] = df['Game Date'].dt.month
  df['Day'] = df['Game Date'].dt.day
  df['Day_of_Week'] = df['Game Date'].dt.dayofweek  # Monday = 0, Sunday = 6

  if st.checkbox("Display the DataFrame with extracted components."):
    st.write(df[['Game Date', 'Year', 'Month', 'Day', 'Day_of_Week']].head())
    st.write("Monday = 0, Sunday = 6")

  st.write("Finally, we remove 'Game Date' column.")
  df.drop('Game Date', axis = 1, inplace = True)
  return df

# Transform quantitative attributes, which have unique id values
@st.cache_data
def transform_quantitative_attributes(df):
  st.write("#### Transform quantitative attributes, which have unique id values")
  st.write("In our dataset 'Player ID' and 'Player Name' contains exactly the same information, so we remove 'Player Name' column.")
  df.drop(['Player Name'], axis = 1, inplace = True)

  st.markdown('''
Let's investigate 'Game ID', 'Game Event ID', 'Player ID' columns. These contain unique identifiers and lack inherent numerical meaning, it's important to handle them appropriately for machine learning tasks.

Since these unique identifiers are repeated and associated with different instances or records in our dataset, then they can be considered as categorical variables rather than unique identifiers. In this case, we can apply feature encoding techniques to represent these categorical variables numerically.

Frequency encoding can indeed be applied to our unique identifiers.
              ''')
  # We will use our previously created function.
  # Apply frequency encoding to 'Game ID' column
  frequency_encode_column(df, 'Game ID')

  # Apply frequency encoding to 'Game Event ID' column
  frequency_encode_column(df, 'Game Event ID')

  # Apply frequency encoding to 'Player ID' column
  frequency_encode_column(df, 'Player ID')

  # Drop the original columns
  df.drop(['Game ID', 'Game Event ID', 'Player ID'], axis = 1, inplace = True)
  return df

# Transform other categorical attributes
@st.cache_data
def transform_other_categorical_attributes(df):
  st.write("#### Transform other categorical attributes")
  st.write("We have some categorical columns that need to be dichotomized. This involves transforming these categorical variables into 'quantitative' variables that can be interpreted by a machine learning model.")
  st.write("In our dataset 'Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type' columns contain qualitative values. Let's transform them.")
  st.write("We will use One Hot Encoding technique to encode a qualitative variable.")
  
  st.write("Step 1. Perform one-hot encoding for each categorical column")
  shot_type_encoded = pd.get_dummies(df['Shot Type'], prefix = 'ShotType', dtype = int)
  shot_zone_basic_encoded = pd.get_dummies(df['Shot Zone Basic'], prefix = 'ShotZoneBasic', dtype = int)
  shot_zone_area_encoded = pd.get_dummies(df['Shot Zone Area'], prefix = 'ShotZoneArea', dtype = int)
  season_zone_range_encoded = pd.get_dummies(df['Shot Zone Range'], prefix = 'ShotZoneRange', dtype = int)
  season_type_encoded = pd.get_dummies(df['Season Type'], prefix = 'SeasonType', dtype = int)

  st.write("Step 2. Concatenate the one-hot encoded columns with the original DataFrame")
  df = pd.concat([df, shot_type_encoded, shot_zone_basic_encoded, shot_zone_area_encoded, season_zone_range_encoded, season_type_encoded], axis=1)
  st.write("Step 3. Drop the original categorical columns")
  df.drop(['Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type'], axis = 1, inplace = True)
  return df

# Transform attributes with high cardinality
def transform_attributes_with_high_cardinality(df):
  st.write("#### Transform attributes with high cardinality")
  st.write("Convert attributes, which have very large number of unique values.")
  st.write("There are", df['Action Type'].nunique(), "Action Type values.")

  if st.checkbox("Show Action Type values"):
    st.write(df['Action Type'].value_counts())
  
  st.markdown('''
Most machine learning algorithms operate with quantitative variables, hence the need to convert these values into numerical representations. With 70 unique action types, employing one-hot encoding would introduce an excessive number of additional columns, which is impractical. As we lack expertise in NBA terminology to group similar action types for discretization or feature engineering, our best option would be to utilize frequency encoding in this scenario. Instead of creating binary indicator variables for each unique action type, we could encode each action type based on its frequency in the dataset. This approach replaces each action type with the proportion of shots that belong to that category. This can be useful if certain action types occur more frequently than others and we want to capture that information in the model.
              ''')
  
  st.write("Step 1. Calculate the frequency of each unique action type.")
  action_type_frequency = df['Action Type'].value_counts(normalize = True)

  st.write("Step 2. Replace each action type with its frequency in the dataset.")
  df['Action Type Frequency'] = df['Action Type'].map(action_type_frequency)
  df.drop(['Action Type'], axis = 1, inplace = True)

  if st.checkbox("Show Action Type Frequency values"):
    st.write(df['Action Type Frequency'].value_counts())

  st.markdown('''
These values represent the proportions of each category in the dataset, where the sum of all proportions would be equal to 1. In the context of frequency encoding for machine learning, using normalize=True ensures that each category's encoding represents its relative frequency in the dataset, making it suitable for capturing the distribution of categorical variables in a normalized manner.
              ''')
  st.write("---")

  # Team Name
  st.write("With the same reason as for 'Action Type', we can do feature encoding for 'Team Name', 'Home Team' and 'Away Team' columns.")
  st.write("There are", df['Team Name'].nunique(), "Team Name values.")
  if st.checkbox("Show Team Name values"):
    st.write(df['Team Name'].value_counts())

  # Apply frequency encoding to the 'Team Name' column
  frequency_encode_column(df, 'Team Name')

  # Apply frequency encoding to the 'Home Team' column
  frequency_encode_column(df, 'Home Team')

  # Apply frequency encoding to the 'Away Team' column
  frequency_encode_column(df, 'Away Team')

  # Drop the original categorical columns
  df.drop(['Team Name', 'Home Team', 'Away Team'], axis = 1, inplace = True)
  st.write("In our dataset 'Team ID' and 'Team Name' contains exactly the same information, so we can remove 'Team ID' column.")
  return df

# Function to perform frequency encoding
def frequency_encode_column(df, column_name):
  # Step 1. Calculate the frequency of each unique value in the column
  frequency = df[column_name].value_counts(normalize=True)
  
  # Step 2. Replace each value with its frequency in the dataset
  df[column_name + '_Frequency'] = df[column_name].map(frequency)


