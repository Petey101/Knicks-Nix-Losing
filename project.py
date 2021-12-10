import numpy as np
import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler

# site
# https://pleong101.wixsite.com/knicksnixlosing

# data
# https://www.kaggle.com/wyattowalsh/basketball
# https://www.kaggle.com/umutalpaydn/nba-20202021-season-player-stats?select=nba2021_advanced.csv
# https://www.kaggle.com/umutalpaydn/nba-20202021-season-player-stats?select=nba2021_per_game.csv
# https://www.basketball-reference.com/leagues/NBA_2021.html

#reference
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html


# connecting to SQL database
db_path = '../project/basketball.sqlite'  #https://www.kaggle.com/wyattowalsh/basketball
connection = sql.connect(db_path) # create connection object to database
print("SQL database connected")
table = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", connection)
# List of tables
#                      name
# 0                  Player
# 1                    Team
# 2         Team_Attributes
# 3            Team_History
# 4       Player_Attributes
# 5          Game_Officials
# 6   Game_Inactive_Players
# 7             Team_Salary
# 8           Player_Salary
# 9                   Draft
# 10          Draft_Combine
# 11          Player_Photos
# 12            Player_Bios
# 13                   Game
# 14                   News
# 15           News_Missing

#Reading the Player_Salary Table into a pandas dataframe
salaries = pd.read_sql("""SELECT * FROM Player_Salary; """, connection)

#Processing the salaries dataframe into Knicks players and non Knicks players for the 2020-2021 season
knicksSalaries = salaries[(salaries['nameTeam'] == 'New York Knicks') & (salaries['slugSeason'] == '2020-21')]
knicksSalaries = knicksSalaries[['namePlayer', 'value']]
nonKnicksSalaries = salaries[(salaries['nameTeam'] != 'New York Knicks') & (salaries['slugSeason'] == '2020-21')]
nonKnicksSalaries = nonKnicksSalaries[['namePlayer', 'value']]

#Reading the 'team_stats.csv' into a dataframe and processing the data
teamStats = pd.read_csv('team_stats.csv') # https://www.basketball-reference.com/leagues/NBA_2021.html
teamStats = teamStats.rename(columns = {'W':'Wins', 'ORtg': 'Offensive Rating', 'DRtg': 'Defensive Rating', '3PAr': '3 Point Attempt Rate', 'eFG%': 'Effective Field Goal %'})
teamStats = teamStats[['Team', 'Wins', 'Offensive Rating', 'Defensive Rating', 'Pace', '3 Point Attempt Rate', 'Effective Field Goal %']]
stats = ['Offensive Rating', 'Defensive Rating', 'Pace', '3 Point Attempt Rate', 'Effective Field Goal %']

#Creating a dataframe of the stats scaled for better visualization
scaledteamStats = teamStats[['Team', 'Wins', 'Offensive Rating', 'Defensive Rating', 'Pace', '3 Point Attempt Rate', 'Effective Field Goal %']]
scaler = MinMaxScaler()
scaledteamStats[['Wins', 'Offensive Rating', 'Defensive Rating', 'Pace', '3 Point Attempt Rate', 'Effective Field Goal %']] = scaler.fit_transform(scaledteamStats[['Wins', 'Offensive Rating', 'Defensive Rating', 'Pace', '3 Point Attempt Rate', 'Effective Field Goal %']])
scaledteamStats.plot(x='Team', kind='bar', stacked=False, title='Relative Statistics of each NBA Team')
plt.show()

#Appending the teams that made the playoffs in the 2020-2021 season into the dataframe
inPlayoffs = [1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
teamStats['inPlayoffs'] = inPlayoffs

#Finding the highest correlation between Wins and the different stats
def findHighestCorr(colName,colLst,df):
    largest_corre = colLst[0]
    largest_correN = df[colName].corr(df[colLst[0]])
    for i in range(len(colLst)):
        if df[colName].corr(df[colLst[i]]) > largest_correN:
            largest_corre = colLst[i]
            largest_correN = df[colName].corr(df[colLst[i]])
    return largest_corre, largest_correN

print(f'Wins have highest r with {findHighestCorr("Wins", stats, teamStats)}.')
#Offensive Rating was the column with the highest correlation

#A model to find the column which best predicts playoff appearance, which we would assume to be Offensive Rating also
def playoffPredict(df, columns, y_col = "inPlayoffs", test_size = 25, random_state = 42):
    col_name = ''
    acc = 0
    for i in range(len(columns)):
        rows = df[[columns[i]]].to_dict('records')
        oneshot = DictVectorizer(sparse = False).fit(rows)
        X = oneshot.transform(rows)
        X_train, X_test, y_train, y_test = train_test_split(X, df[y_col], test_size = test_size, random_state = random_state)
        clf = LogisticRegression(max_iter = 1000)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        if score > acc:
            acc = score
            col_name = columns[i]
    return acc, col_name

acc,col_name = playoffPredict(teamStats, stats, test_size = 15, random_state = 17)
print(f'The highest accuracy, {acc}, was obtained by including {col_name}.')
#Offensive Rating best predicts a team's playoff appearance

#Creating a plot of the different correlations for visualization
correlations = []
for i in range(len(stats)):
    correlations.append(teamStats['Wins'].corr(teamStats[stats[i]]))
correlationDF = pd.DataFrame(correlations, index = stats, columns =['Correlation to # of Wins'])
sns.set_theme(style="dark", palette='flare')
correlationDF.plot(kind='barh', title='Team Statistics Relation to Higher Season Win Rates')
plt.show()

#Creating a regression line visualization
def compute_r_line(xes,yes):
    sd_x = statistics.stdev(xes)
    sd_y = statistics.stdev(yes)
    r = np.corrcoef(xes, yes)
    m = r*sd_y/sd_x
    b = yes[0] - m * xes[0]
    return m[0][1], b[0][1]

s1 = teamStats['Wins'].tolist()
s2 = teamStats['Offensive Rating'].tolist()
m, b = compute_r_line(s1,s2)
xes = np.array([0,72])
yes = m*xes + b
sns.set_theme(style="dark", palette='vlag')
plt.scatter(s1,s2)
plt.plot(xes,yes)
plt.xlabel("Wins")
plt.ylabel("Offensive Rating")
plt.title(f'Regression line with m = {m:{4}.{2}} and y-intercept = {b:{4}.{2}}')
plt.show()

#Reading and cleaning player stats data
advancedStats = pd.read_csv('nba2021_advanced.csv') #https://www.kaggle.com/umutalpaydn/nba-20202021-season-player-stats?select=nba2021_advanced.csv
perGameStats = pd.read_csv('nba2021_per_game.csv') #https://www.kaggle.com/umutalpaydn/nba-20202021-season-player-stats?select=nba2021_per_game.csv
perGameStats = perGameStats.rename(columns={"MP": "MPG", 'Player': 'Player Name', 'Tm' : 'Team'})
teamStats = pd.concat([advancedStats, perGameStats], axis=1)
teamStats = teamStats[['Player Name','Team', 'MPG', 'PER']]
teamStats['PER * MPG'] = teamStats['MPG'] * teamStats['PER']
averagePER = teamStats['PER'].mean()
knickStats = teamStats[teamStats['Team'] == 'NYK']
knickStats = knickStats[['Player Name', 'PER']]
nonKnickStats = teamStats[teamStats['Team'] != 'NYK']
nonKnickStats = nonKnickStats[['Player Name', 'PER']]
teamStats = teamStats[['Team', 'PER * MPG']]

#Aggregate each team's Player efficient rating into a Team efficiency Rating
TER = teamStats.groupby('Team').sum()
TER = TER.rename(columns={'PER * MPG': "Team Efficiency Score"})
sns.set_theme(style="dark", palette='mako')
TER.plot(kind='barh', title='Team Score from player PERs')
plt.show()

#Find the Knicks PER and plot them
knicksPlayersPER = knickStats.set_index('Player Name')
nonKnicksPlayersPER = nonKnickStats.set_index('Player Name').join(nonKnicksSalaries.set_index('namePlayer'))
nonKnicksPlayersPER = nonKnicksPlayersPER[nonKnicksPlayersPER['value'].notnull()]
sns.set_theme(style="dark", palette='magma')
knicksPlayersPER.plot(kind='barh', title='Knicks\' PERs')
plt.show()

#Find the Knicks that are above and below the average league PER
aboveAVGKnicks = knickStats[knickStats['PER'] > averagePER]
belowAVGKnicks = knickStats[knickStats['PER'] < averagePER].sort_values(by='PER')
belowAVGKnicks = belowAVGKnicks.set_index('Player Name').join(knicksSalaries.set_index('namePlayer'))
belowAVGKnicks = belowAVGKnicks[belowAVGKnicks['value'].notnull()]
belowAVGKnicksList = belowAVGKnicks.reset_index().values.tolist()
nonKnicksPlayerList = nonKnicksPlayersPER.reset_index().values.tolist()

#For the Knicks that fall below the league average, find players that have higher PER and lower pay for potential trades
trades = {}
for kIndex, knick in enumerate(belowAVGKnicksList):
    potentialTrades = []
    for nkIndex, player in enumerate(nonKnicksPlayerList):
        if player[0] != knick[0] and player[1] > knick[1] and player[2] < knick[2]:
            potentialTrades.append(player[0])
    trades[knick[0]] = potentialTrades
print(trades.keys())
print(trades[belowAVGKnicksList[0][0]])
