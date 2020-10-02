import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt

#read data from Match.csv, extract columns into pandas dataframe
data = pd.read_csv (r'Match.csv')
matches = pd.DataFrame(data , columns= ['id', 'league_id', 'home_team_goal','away_team_goal', 'home_player_1','home_player_2','home_player_3',\
                                        'home_player_4','home_player_5','home_player_6','home_player_7','home_player_8','home_player_9','home_player_10',\
                                        'home_player_11','away_player_1','away_player_2','away_player_3','away_player_4','away_player_5','away_player_6',\
                                        'away_player_7','away_player_8','away_player_9','away_player_10','away_player_11'])


#drop rows with missing information
matches = matches.dropna()

#convert all values to int
matches = matches.astype(int)

#take only EPL matches (league_id  == 1729)
matches = matches.loc[matches['league_id'] == 1729 ]

#remove all draw results
matches = matches.drop(matches[matches['home_team_goal'] == matches['away_team_goal']].index)

#add new column where 1 = home team win, 0 = away team win
matches['goal_diff'] = (matches['home_team_goal'] > matches['away_team_goal']).astype(int)

#remove columns not used in ML model
matches = matches.drop('home_team_goal', axis = 1)
matches = matches.drop('away_team_goal', axis = 1)
matches = matches.drop('id', axis = 1)
matches = matches.drop('league_id', axis = 1)


#read data from Player_Attributes.csv, extract columns into pandas dataframe
data = pd.read_csv (r'Player_Attributes.csv')   
players = pd.DataFrame(data , columns= ['player_api_id','overall_rating','potential',
                                        'heading_accuracy','free_kick_accuracy'])

#drop rows with missing information
players = players.dropna()

#take only the first instance of each player (because csv has different player ratings for different seaons
players = players.drop_duplicates(subset=['player_api_id'])


#replace all player id in matches with their ratings
for i in range(1,12):
    curr = 'home_player_' + str(i)
    matches[curr] = matches[curr].map(players.set_index('player_api_id')['overall_rating'])

for i in range(1,12):
    curr = 'away_player_' + str(i)
    matches[curr] = matches[curr].map(players.set_index('player_api_id')['overall_rating'])

def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.show()
    
#features_names = ['player_api_id','overall_rating','potential', 'heading_accuracy','free_kick_accuracy','agility']


features_names = ['home_player_1','home_player_2','home_player_3',\
                                        'home_player_4','home_player_5','home_player_6','home_player_7','home_player_8','home_player_9','home_player_10',\
                                        'home_player_11','away_player_1','away_player_2','away_player_3','away_player_4','away_player_5','away_player_6',\
                                        'away_player_7','away_player_8','away_player_9','away_player_10','away_player_11']





print (matches)

#split into X and y
X = matches.drop('goal_diff', axis = 1)
y = matches['goal_diff']

#train-test split
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size = 0.20)

#model used: SVM
model = SVC(kernel = 'linear')
model.fit(X_train, y_train)

#predict on test set
y_pred = model.predict(X_test)

#generate confusion matrix
matrix = confusion_matrix(y_test, y_pred)
accuracy = (matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])
print("accuracy = " + str(accuracy * 100) + "%")
print(classification_report(y_test, y_pred))

f_importances(abs(model.coef_[0]), features_names)



