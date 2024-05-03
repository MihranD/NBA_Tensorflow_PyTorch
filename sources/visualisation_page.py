import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns
from sources.utils import read_df

def show_visualisation_page():
  st.write("Let's generate some visualizations to gain further insights.")
  st.write("---")

  # Read the dataset into a DataFrame
  df = read_df()

  # Shot Distribution by Zone
  shot_distance_by_zone(df)

  # Shot Accuracy by Shot Zone Range
  shot_accuracy_by_shot_zone_range(df)

  # Shot Type Distribution
  shot_type_distribution(df)

  # Shot Type Distribution by Shot Made Flag
  shot_type_distribution_by_shot_made_flag(df)

  # Shooting Efficiency by Period
  shooting_efficiency_by_period(df)

  # Shot Accuracy by Remaining Minutes
  shot_accuracy_by_remaining_minutes(df)

@st.cache_data
def shot_distance_by_zone(df):
  # Shot Distribution by Zone
  fig = plt.figure(figsize = (10, 6))
  sns.countplot(x = 'Shot Zone Basic', data = df, hue = 'Shot Made Flag', palette = ['gold', 'lightseagreen'])
  plt.title('Shot Distribution by Zone')
  plt.xlabel('Shot Zone Basic')
  plt.ylabel('Number of Shots')
  plt.xticks(rotation = 45)
  plt.legend(title = 'Shot Made', loc = 'upper right')
  plt.grid(True)
  st.pyplot(fig)

  st.markdown('''
This count plot displays the distribution of both made and missed shots for each shot zone basic. The bars grouped by shot outcome (made or missed), with different colors representing each outcome. This visualization provides a clear comparison between the number of made and missed shots in each shot zone basic.

The shot accuracy in the **Restricted Area** stands out as notably higher when compared to other areas.
              ''')
  st.markdown('''
This graph shows something interesting about how accurate NBA players are with their shots. Usually, we expect accuracy to drop as the shot gets farther from the basket. But between 5 to 25 feet away, accuracy stays about the same. There could be a few reasons. Players practice free throws a lot, which are taken from 15 feet away, so they're pretty good at shots from nearby distances. Also, defenders guard differently depending on how close the shooter is to the basket, which might make closer shots harder. But between 5 to 25 feet, these factors seem to balance out.
So, even though we might think farther shots are always harder, this graph shows that's not always the case. It gives us a new perspective on how distance affects shooting accuracy.
              ''')
  st.write("---")

@st.cache_data
def shot_type_distribution(df):
  # Shot Type Distribution
  # Calculate the count of each shot type
  shot_type_counts = df['Shot Type'].value_counts()

  # Determine which slice to separate
  explode = [0.1 if label == '3PT Field Goal' else 0 for label in shot_type_counts.index]

  # Create a pie chart
  fig = plt.figure(figsize = (8, 8))
  plt.pie(shot_type_counts, labels = shot_type_counts.index, autopct = '%1.1f%%', startangle = 90, explode = explode)
  plt.title('Shot Type Distribution')
  st.pyplot(fig)

  st.markdown('''
From the visualization, it's evident that over three-quarters of the shots taken are 2-point shots, while the remaining proportion represents 3-point shots. This distribution underscores the prevalence of 2-point field goal attempts in NBA gameplay, reflecting the traditional emphasis on scoring close to the basket.
              ''')
  st.write("---")

@st.cache_data
def shot_type_distribution_by_shot_made_flag(df):
  # Shot Type Distribution by Shot Made Flag
  fig = plt.figure(figsize = (8, 5))
  sns.countplot(data = df, x = 'Shot Type', hue = 'Shot Made Flag', palette = 'pastel')
  plt.title('Shot Type Distribution by Shot Made Flag')
  plt.xlabel('Shot Type')
  plt.ylabel('Number of Shots')
  plt.grid(True)
  st.pyplot(fig)

  st.markdown('''
The plot reveals a substantial discrepancy in the distribution of shot types. The majority of shots taken are '2PT Field Goals', outnumbering '3PT Field Goals' by approximately threefold. Consequently, the differing proportions between these shot types indicate a notable disparity in the frequency of attempts between two-point and three-point shots.

This observation underscores the importance of analyzing shooting efficiency and strategy across different shot types. Further exploration into the factors influencing shot selection and success rates for both two-point and three-point shots can offer valuable insights into player performance and strategic decision-making during NBA games.
              ''')
  st.markdown('''
From the heatmap, we can discern patterns in shot selection and shooting accuracy across different court regions. Areas with a higher density of gold points indicate regions where players have successfully made shots, while areas with a higher density of blue points suggest regions where shots were missed more frequently.

Analyzing the shot heatmap provides valuable insights into player positioning, shooting tendencies, and areas of strength or weakness on the court. Further investigation into spatial patterns can inform strategic decisions related to player positioning, defensive strategies, and offensive playcalling in basketball games.
              ''')
  st.write("---")

@st.cache_data
def shot_accuracy_by_shot_zone_range(df):
  # Shot Accuracy by Shot Zone Range
  # Define the order of Shot Zone Range values
  shot_zone_order = ['Less Than 8 ft.', '8-16 ft.', '16-24 ft.', '24+ ft.', 'Back Court Shot']

  # Plotting countplot with specified order
  fig = plt.figure(figsize = (10, 6))
  sns.countplot(data = df, x = 'Shot Zone Range', hue = 'Shot Made Flag', palette = ['gold', 'lightseagreen'], order = shot_zone_order)
  plt.title('Shot Accuracy by Shot Zone Range')
  plt.xlabel('Shot Zone Range')
  plt.ylabel('Count')
  plt.xticks(rotation = 45)
  plt.legend(title = 'Shot Made Flag', loc = 'upper right')
  plt.grid(True)
  st.pyplot(fig)

  st.markdown('''
The plot highlights a distinct pattern in shot accuracy based on shot zone range. Shots taken in the nearest area (less than 8 ft. from the basket) exhibit notably higher accuracy compared to shots attempted from further distances. This observation suggests that players tend to have a higher likelihood of making shots when positioned closer to the basket.

The decline in shot accuracy with increasing distance from the basket aligns with the inherent difficulty of making shots from greater distances, attributed to factors such as shot angle, defensive pressure, and player skill. Understanding these patterns in shot accuracy across different shot zones is crucial for player positioning, offensive strategies, and shot selection during basketball games.
              ''')
  st.write("---")

@st.cache_data
def shooting_efficiency_by_period(df):
  # Shooting Efficiency by Period
  fig = plt.figure(figsize=(10, 6))
  sns.barplot(x = 'Period', y = 'Shot Made Flag', data = df)
  plt.title('Shooting Efficiency by Period')
  plt.xlabel('Period')
  plt.ylabel('Shot Made Flag (Mean)')
  plt.grid(True)
  st.pyplot(fig)

  st.markdown('''
Periods (1-4): Shooting efficiency remains stable, with consistent mean values and low variability. This suggests reliable shooting performance in the initial stages of the game.

Overtime-Periods (5-6): There is a slight decrease in shooting efficiency, accompanied by a small increase in variability. Shooting performance becomes moderately less consistent during these periods.

Overtime-Periods (7-8): Shooting efficiency experiences a sharper decline, with shorter columns indicating lower mean values and longer error bars indicating higher variability. Shooting performance becomes less consistent and more variable as the game progresses.
              ''')
  st.write("---")

@st.cache_data
def shot_accuracy_by_remaining_minutes(df):
  # Shot Accuracy by Remaining Minutes
  # Create a line plot for remaining minutes
  fig = plt.figure(figsize = (12, 6))
  sns.lineplot(data = df, x = 'Minutes Remaining', y = 'Shot Made Flag', color = 'green')
  plt.title('Shot Accuracy by Remaining Minutes')
  plt.xlabel('Minutes Remaining')
  plt.ylabel('Shot Made Flag (Mean)')
  plt.grid(True)
  st.pyplot(fig)

  st.markdown('''
The graph reveals interesting insights into how shot accuracy evolves over the course of the game. The slight fluctuations in shot accuracy during the early minutes may reflect the teams' strategies, adjustments, or warm-up period. As the game progresses, we observe a gradual decline in shot accuracy, possibly due to increased fatigue, defensive pressure, or higher stakes in critical game moments. The narrowing confidence interval towards the end of the game indicates heightened certainty in shot accuracy estimation during crunch time, where players' decisions and execution become more decisive in determining game outcomes. This visualization underscores the importance of time management and strategic decision-making in basketball games, as they can significantly impact shot outcomes and ultimately, game results.
              ''')
  st.write("---")
