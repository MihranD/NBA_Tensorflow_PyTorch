import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
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
  transform_attributes_with_high_cardinality(df)

  # Transform other categorical attributes
  transform_other_categorical_attributes(df)

  # Transform quantitative attributes, which have unique id values
  transform_quantitative_attributes(df)

  # Transform date attribute
  transform_date_attribute(df)

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

# Function to perform frequency encoding
def frequency_encode_column(df, column_name):
  # Step 1. Calculate the frequency of each unique value in the column
  frequency = df[column_name].value_counts(normalize=True)
  
  # Step 2. Replace each value with its frequency in the dataset
  df[column_name + '_Frequency'] = df[column_name].map(frequency)


