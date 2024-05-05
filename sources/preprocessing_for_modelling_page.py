import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from joblib import dump
from sklearn.model_selection import train_test_split
from sources.utils import read_df

def show_preprocessing_for_modelling_purposes_page():
  # Read the dataset into a DataFrame
  df = read_df()
  
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
  if os.path.isfile('NBA Shot Locations 1997 - 2020-Report2-train-test.joblib'):
    st.write("<span style='color:green'>Joblib file already exists.</span>", unsafe_allow_html=True)
  else:
    save_train_test_set(df)

  # Conclusion for preprocessing
  conclusion_for_preprocessing(df)

# Conclusion for preprocessing
def conclusion_for_preprocessing(df):
  st.write("#### Conclusion for preprocessing")
  st.write("We converted all categorical attributes to quantitative. Our dataset ready for further modeling.")
  if st.checkbox("Show preprocessed dataset"):
    st.dataframe(df.head())
    st.dataframe(df.dtypes)
  
# Save train-test sets in a file
def save_train_test_set(df):
  data = df.drop('Shot Made Flag', axis = 1)
  target = df['Shot Made Flag']
  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 66)
  # WARNING: This call can take a few seconds.
  # Save the data
  #if st.button("Save data in joblib file"):
  dump((X_train, X_test, y_train, y_test), 'NBA Shot Locations 1997 - 2020-Report2-train-test.joblib')
  st.write("<span style='color:green'>New joblib file generated successfully.</span>", unsafe_allow_html=True)
  # Later model parts can read the data from it with help of: 
  # X_train, X_test, y_train, y_test = load('NBA Shot Locations 1997 - 2020-Report2-train-test.joblib')

# Split train and test parts
def split_train_and_test_parts(df):
  st.write("#### Split train and test parts")
  st.markdown('''
Let's randomly divide the matrices into a **training set** and a **test set** corresponding to **80%** and **20%** of the total amount of available data respectively. Add the argument **random_state = 66** for randomness reproducibility.
              ''')
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
  st.write("Now that we have transformed all features into numeric values, we can proceed to create a correlation matrix specifically for the '**Shot Made Flag**' target variable.")

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

  if st.checkbox("Correlation of Features with '**Shot Made Flag**'"):
    features = target_corr.index[1:16].tolist()  # Exclude 'Shot Made Flag' and get the first 15 features
    correlations = target_corr.values[1:16].tolist()  # Exclude its correlation value and get the first 15 correlations

    # Plot the bar plot with adjusted spacing and grid lines for the first 15 features
    fig = plt.figure(figsize=(10, len(features) * 0.5))  # Adjust figure height based on number of features
    bars = plt.barh(features, correlations, color='skyblue')
    plt.xlabel('Correlation')
    plt.title('Correlation of Features with "**Shot Made Flag**" (First 15 Features)')
    plt.gca().invert_yaxis()  # Invert y-axis to display highest correlation at the top
    plt.grid(True, linestyle='--', alpha=0.6)  # Add grid lines with dashed style and transparency
    plt.tight_layout()  # Adjust spacing to prevent text overlap
    st.pyplot(fig)

    st.markdown('''
The correlation plot highlights the features most strongly correlated with '**Shot Made Flag**'.

'**Action Type Frequency**' shows a high positive correlation, suggesting that the frequency of specific actions during a shot attempt significantly impacts shot success.

'**ShotZoneBasic_Restricted Area**' and '**Shot Distance**' also exhibit strong positive correlations, indicating that shots taken from restricted areas and shorter distances are more likely to be made.

Furthermore, '**ShotType_2PT Field Goal**' and '**ShotType_3PT Field Goal**' reflect the influence of shot type on success rates, with both two-point field goals (2PT) and three-point field goals (3PT) having an equal impact on shot success rates.
                ''')
    st.write("---")

# Transform date attribute
def transform_date_attribute(df):
  st.write("#### Transform date attribute")
  df['Game Date'] = pd.to_datetime(df['Game Date'], format='%Y%m%d')
  
  min_date = df['Game Date'].min()
  max_date = df['Game Date'].max()

  st.markdown('''
There are some common approaches to feature engineering with date data in classification models. We will use **Extract Components**. This approach extracts relevant components from the date, such as year, month, day, day of the week. These components can then be encoded as numerical features.
              ''')
  
  # Extract components
  df['Year'] = df['Game Date'].dt.year
  df['Month'] = df['Game Date'].dt.month
  df['Day'] = df['Game Date'].dt.day
  df['Day_of_Week'] = df['Game Date'].dt.dayofweek  # Monday = 0, Sunday = 6

  if st.checkbox("Display the DataFrame with extracted components."):
    st.write(df[['Game Date', 'Year', 'Month', 'Day', 'Day_of_Week']].head())
    st.write("Monday = 0, Sunday = 6")

  st.write("Finally, we remove '**Game Date**' column.")
  df.drop('Game Date', axis = 1, inplace = True)
  return df

# Transform quantitative attributes, which have unique id values
@st.cache_data
def transform_quantitative_attributes(df):
  st.write("#### Transform quantitative attributes, which have unique id values")
  st.write("In our dataset '**Player ID**' and '**Player Name**' contains exactly the same information, so we remove '**Player Name**' column.")
  df.drop(['Player Name'], axis = 1, inplace = True)

  st.markdown('''
Let's investigate '**Game ID**', '**Game Event ID**', '**Player ID**' columns. These contain **unique identifiers** and lack inherent numerical meaning, it's important to handle them appropriately for machine learning tasks.

Since these unique identifiers are repeated and associated with different instances or records in our dataset, then they can be considered as **categorical** variables rather than unique identifiers. In this case, we can apply **feature encoding** techniques to represent these categorical variables numerically.

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
  st.write("We have some categorical columns that need to be dichotomized. This involves transforming these **categorical variables into quantitative** variables that can be interpreted by a machine learning model.")
  st.write("In our dataset '**Shot Type**', '**Shot Zone Basic**', '**Shot Zone Area**', '**Shot Zone Range**', '**Season Type**' columns contain **qualitative** values. Let's transform them.")
  st.write("We will use **One Hot Encoding** technique to encode a qualitative variable.")
  
  st.write("*Step 1*. Perform one-hot encoding for each categorical column")
  shot_type_encoded = pd.get_dummies(df['Shot Type'], prefix = 'ShotType', dtype = int)
  shot_zone_basic_encoded = pd.get_dummies(df['Shot Zone Basic'], prefix = 'ShotZoneBasic', dtype = int)
  shot_zone_area_encoded = pd.get_dummies(df['Shot Zone Area'], prefix = 'ShotZoneArea', dtype = int)
  season_zone_range_encoded = pd.get_dummies(df['Shot Zone Range'], prefix = 'ShotZoneRange', dtype = int)
  season_type_encoded = pd.get_dummies(df['Season Type'], prefix = 'SeasonType', dtype = int)

  st.write("*Step 2*. Concatenate the one-hot encoded columns with the original DataFrame")
  df = pd.concat([df, shot_type_encoded, shot_zone_basic_encoded, shot_zone_area_encoded, season_zone_range_encoded, season_type_encoded], axis=1)
  st.write("*Step 3*. Drop the original categorical columns")
  df.drop(['Shot Type', 'Shot Zone Basic', 'Shot Zone Area', 'Shot Zone Range', 'Season Type'], axis = 1, inplace = True)
  return df

# Transform attributes with high cardinality
def transform_attributes_with_high_cardinality(df):
  st.write("#### Transform attributes with high cardinality")
  st.write("Convert attributes, which have very large number of unique values.")
  st.write("There are", df['Action Type'].nunique(), "**'Action Type'** values.")

  if st.checkbox("Show **'Action Type'** values"):
    st.write(df['Action Type'].value_counts())
  
  st.write("*Step 1*. Calculate the frequency of each unique action type.")
  action_type_frequency = df['Action Type'].value_counts(normalize = True)

  st.write("*Step 2*. Replace each action type with its frequency in the dataset.")
  df['Action Type Frequency'] = df['Action Type'].map(action_type_frequency)
  df.drop(['Action Type'], axis = 1, inplace = True)

  if st.checkbox("Show **'Action Type Frequency'** values"):
    st.write(df['Action Type Frequency'].value_counts())

  st.markdown('''
These values represent the proportions of each category in the dataset, where the sum of all proportions would be equal to 1. In the context of **frequency encoding** for machine learning, it ensures that each category's encoding represents its relative frequency in the dataset, making it suitable for capturing the distribution of categorical variables in a normalized manner.
              ''')

  # Team Name
  st.write("With the same reason as for **'Action Type'**, we can do **feature encoding** for '**Team Name**', '**Home Team**' and '**Away Team**' columns.")
  
  # Apply frequency encoding to the 'Team Name' column
  frequency_encode_column(df, 'Team Name')

  # Apply frequency encoding to the 'Home Team' column
  frequency_encode_column(df, 'Home Team')

  # Apply frequency encoding to the 'Away Team' column
  frequency_encode_column(df, 'Away Team')

  # Drop the original categorical columns
  df.drop(['Team Name', 'Home Team', 'Away Team'], axis = 1, inplace = True)

  # In our dataset 'Team ID' and 'Team Name' contains exactly the same information. Let's remove 'Team ID' column.
  df.drop(['Team ID'], axis = 1, inplace = True)
  return df

# Function to perform frequency encoding
def frequency_encode_column(df, column_name):
  # Step 1. Calculate the frequency of each unique value in the column
  frequency = df[column_name].value_counts(normalize=True)
  
  # Step 2. Replace each value with its frequency in the dataset
  df[column_name + '_Frequency'] = df[column_name].map(frequency)


