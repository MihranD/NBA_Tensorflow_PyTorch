import streamlit as st

def show_intro_page():
  st.write("### Context")
  st.markdown('''
Incorporating advanced statistical analysis into our operations is crucial for staying competitive in the realm of American sports, particularly in the NBA. As technology continues to evolve, we recognize the importance of leveraging digital tracking tools to gather comprehensive data on player performance. By focusing on shot frequency and shooting efficiency among top NBA players, as identified by ESPN, we aim to gain valuable insights that can inform our strategic decisions and enhance our team's performance on the court. This project aligns with our commitment to utilizing data-driven approaches to drive success in the highly competitive landscape of professional basketball.

From a technical point of view:

This project entails harnessing cutting-edge digital tracking tools to capture real-time data on player movements during NBA games. We will employ sophisticated statistical analysis techniques to process the vast amount of information collected, focusing on shot frequency and shooting efficiency across various game situations and court locations. Additionally, predictive modeling techniques will be utilized to estimate the probability of shot success for each of the 20 esteemed NBA players included in the dataset. By leveraging advanced analytics and machine learning algorithms, we aim to extract actionable insights that can optimize player performance and inform strategic decision-making within our organization.

From an economic point of view:

Investing in this project holds significant economic potential for our organization within the sports industry. By leveraging data analytics to understand player performance and optimize strategic decision-making, we can gain a competitive edge on the court. Improved shot selection and efficiency among our players can lead to increased game success, fan engagement, and ultimately, revenue generation through ticket sales, merchandise, and sponsorships. Additionally, by demonstrating our commitment to data-driven approaches, we enhance our brand reputation and attract top talent, further bolstering our long-term economic viability and success in the market.

From a scientific point of view:

This project represents an opportunity to advance the scientific understanding of basketball performance through rigorous data analysis and predictive modeling. By examining shot data from top NBA players in various game contexts, we aim to uncover underlying patterns and trends that contribute to successful outcomes on the court. Through the application of statistical methods and machine learning algorithms, we can identify key factors influencing shot selection and effectiveness, contributing to the broader body of knowledge in sports analytics. Moreover, the insights gained from this research have the potential to inform coaching strategies, player development programs, and future scientific inquiries into sports performance optimization.
              ''')
  
  # Understanding and manipulation of data
  st.write("## Understanding and manipulation of data")
  # Framework
  st.write("### Framework")
  st.markdown('''
For this project, we utilized the NBA shot dataset spanning the years 1997 to 2020. The dataset contains comprehensive information on shot locations in NBA games, allowing for detailed analysis of shot frequency and efficiency among players during this period.

The dataset used in this project is freely available on Kaggle (https://www.kaggle.com/jonathangmwl/nba-shot-locations), a platform known for hosting various datasets for analysis and machine learning tasks. As such, it is accessible to anyone with an internet connection and an account on the Kaggle platform. The dataset is provided by the user "jonathangmwl" and can be accessed without any restrictions.

The dataset encompasses a considerable amount of data, capturing detailed shot information from NBA games spanning a 23-year period, ranging from 1997 to 2020. The exact size of the dataset, including the number of rows and columns, will be explored further in subsequent topics.
              ''')
  
  # Relevance
  st.write("### Relevance")
  st.markdown('''
The dataset contains a comprehensive array of features that encapsulate various aspects of shot-taking behavior in NBA games. Here's a brief description of the key features included, along with their relevance for the classification problem:

Game-related features: Unique identifiers for each game, including game ID, event ID, and date of the game. While these features provide contextual information about when and where shots were taken, they are not directly relevant to the classification problem. However, they may indirectly influence player performance due to factors such as fatigue or opponent strength.

Player-related features: Unique identifiers for players, as well as their full names. While player identity is crucial for tracking individual performance, it may not directly impact the shot classification task unless specific player tendencies are considered in the modeling process.

Team-related features: Unique identifiers for teams participating in the game, including home and away teams. Similar to game-related features, team information provides contextual background but may not directly influence shot outcomes.

Shot-related features: Details about each shot taken, such as shot type (e.g., jump shot, layup), shot distance, shot zone (basic and area), shot zone range, and X-Y coordinates of shot location. These features are highly relevant for the classification problem as they directly characterize the shot-taking behavior of players. Shot type, distance, and location are likely to significantly influence the likelihood of a shot being successful, making them crucial predictors in the classification model.

Other features: Period of the game, minutes and seconds remaining in the period, season type (regular season or playoffs). While these features provide additional context about the game situation, their direct relevance to shot classification may be limited compared to shot-related features. However, game context, such as time remaining or playoff status, could indirectly influence shot outcomes by affecting player behavior or defensive strategies.

In summary, shot-related features, including shot type, distance, and location, are the most relevant predictors for the classification problem, as they directly characterize the shot-taking behavior of NBA players. Other contextual features, such as game-related and team-related features, may provide additional background information but are not as directly influential in predicting shot outcomes.
            ''')

  # Problem Definition
  st.write("### Problem Definition")
  st.markdown('''
We aim to predict the probability of a shot being made by each player, indicating whether a shot is successful or not. This problem naturally aligns with a binary classification task, where shots are categorized as either made or missed.
            ''')