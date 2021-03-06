#analyse diff algos --> kNN should not perform well
#https://scikit-learn.org/stable/modules/svm.html
#https://intellipaat.com/blog/tutorial/machine-learning-tutorial/svm-algorithm-in-python/#How-Does-Support-Vector-Machine-Work


### X labels: 2 x 11 player ratings ###
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import cross_validate, KFold
from helper import *
#from sklearn import metrics
#disp = metrics.plot_confusion_matrix(classifier, test_features, test_labels)


algo = input(algos_available)
print('\n')
label = input(labels_available)
print('\n')
scorers = input(scorers_available)
print('\nPlease wait...\n')

#suppress warnings
import warnings
warnings.filterwarnings("ignore")

#read data from Match.csv, extract columns into pandas dataframe
data = pd.read_csv (r'Match.csv')
matches = pd.DataFrame(data , columns= ['id', 'league_id', 'home_team_goal','away_team_goal', \
                                        'home_player_1','home_player_2','home_player_3',\
                                        'home_player_4','home_player_5','home_player_6',\
                                        'home_player_7','home_player_8','home_player_9',\
                                        'home_player_10','home_player_11','away_player_1',\
                                        'away_player_2','away_player_3','away_player_4',
                                        'away_player_5','away_player_6','away_player_7',\
                                        'away_player_8','away_player_9','away_player_10',\
                                        'away_player_11'])

#drop rows with missing information
matches = matches.dropna()

#convert all values to int
matches = matches.astype(int)

#take only EPL matches (league_id  == 1729)
matches = matches.loc[matches['league_id'] == 1729 ]

#read data from Player_Attributes.csv, extract columns into pandas dataframe
data = pd.read_csv (r'Player_Attributes.csv')   
players = pd.DataFrame(data , columns= ['player_api_id','overall_rating'])

#drop rows with missing information
players = players.dropna()

#take only the first rating of each player (csv has diff player ratings for diff seasons)
players = players.drop_duplicates(subset=['player_api_id'])

#replace all player id in matches with their ratings
for i in range(1,12):
    curr = 'home_player_' + str(i)
    matches[curr] = matches[curr].map(players.set_index('player_api_id')['overall_rating'])

for i in range(1,12):
    curr = 'away_player_' + str(i)
    matches[curr] = matches[curr].map(players.set_index('player_api_id')['overall_rating'])

matches, label_name = getLabel(label, matches)

scorers = getScorers(scorers)

#remove columns not used in ML model
matches = matches.drop('id', axis = 1)
matches = matches.drop('league_id', axis = 1)
matches = matches.drop('home_team_goal', axis = 1)
matches = matches.drop('away_team_goal', axis = 1)

#debugging print statements
#print(matches)
#with pd.option_context('display.max_rows', 20, 'display.max_columns', None):  print(matches)

    
#MODEL TRAINING

#None: single core CV, -1: all cores CV
multicore = None

#X-y split
X = matches.drop('result', axis = 1)
y = matches['result']

#train model with 10-fold cross validation, using on given algo
kfold = KFold(n_splits=10, shuffle=True, random_state=420)
algos = ('svm', 'knn', 'nb', 'dt', 'rf', 'lr')
#algos = ('svm', 'knn', 'nb', 'dt', 'rf', 'lr', 'p')
if algo == 'all':
    for a in algos:
        model, algo_name = getAlgo(a)
        scores = cross_validate(model, X, y, cv=kfold, scoring = scorers, n_jobs = multicore)
        print(algo_name + ' to predict ' + label_name + ' using player ratings')
        if 'f1_weighted' in scorers:
            print('F1-weighted: ' + str(round(scores['test_f1_weighted'].mean(), 4)) + ' (+/-' + str(round(scores['test_f1_weighted'].std(), 4)) + ')')
        if 'accuracy' in scorers:
            print('Accuracy: ' + str(round(scores['test_accuracy'].mean(), 4)) + ' (+/-' + str(round(scores['test_accuracy'].std(), 4)) + ')')
        if 'roc_auc_ovo_weighted' in scorers:
            print('AUC-ROC: ' + str(round(scores['test_roc_auc_ovo_weighted'].mean(), 4)) + ' (+/-' + str(round(scores['test_roc_auc_ovo_weighted'].std(), 4)) + ')')
        print('\n')
else:
    model, algo_name = getAlgo(algo)
    scores = cross_validate(model, X, y, cv=kfold, scoring = scorers, n_jobs = multicore)
    print(algo_name + ' to predict ' + label_name + ' using player ratings')
    if 'f1_weighted' in scorers:
        print('F1-weighted: ' + str(round(scores['test_f1_weighted'].mean(), 4)) + ' (+/-' + str(round(scores['test_f1_weighted'].std(), 4)) + ')')
    if 'accuracy' in scorers:
        print('Accuracy: ' + str(round(scores['test_accuracy'].mean(), 4)) + ' (+/-' + str(round(scores['test_accuracy'].std(), 4)) + ')')
    if 'roc_auc_ovo_weighted' in scorers:
        print('AUC-ROC: ' + str(round(scores['test_roc_auc_ovo_weighted'].mean(), 4)) + ' (+/-' + str(round(scores['test_roc_auc_ovo_weighted'].std(), 4)) + ')')

#train model with 80-20 train-test split, using on given algo
#X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size = 0.20)
#model, algo_name = getAlgo(a)
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#print(algo_name + ' to predict ' + label_name + ' using player ratings' + '\n')
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
