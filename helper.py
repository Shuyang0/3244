from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

algos_available = """Algorithms available:
    svm - support vector machine
    knn - k-nearest neighbors
    nb - naive bayes
    dt - decision tree
    rf - random forest
    lr - logistic regression
    all - run all algorithms
CHOICE: """
labels_available = """y-Labels available:
    wl - win loss
    wdl - win draw loss
    gd - goal difference
CHOICE: """
scorers_available = """Scoring metrics available:
    1 - f1 weighted
    2 - accuracy
    3 - auc-roc
    all - use all of the above
***If goal difference is the y-label, DO NOT include auc-roc for scoring***
***Seperate choices with a space (e.g. "2 3")***
CHOICE: """

def getLabel(label, matches):
    if label == 'wl':
        #remove all draw results
        matches = matches.drop(matches[matches['home_team_goal'] == matches['away_team_goal']].index)
        #add new column where 1 = home team win, 0 = away team win
        matches['result'] = (matches['home_team_goal'] > matches['away_team_goal']).astype(int)
        return matches, 'win-loss'

    elif label == 'wdl':
        #add new column where 1 = home win, 0 = draw, -1 = home loss
        matches['result'] = 0
        home_win = matches['home_team_goal'] > matches['away_team_goal']
        away_win = matches['home_team_goal'] < matches['away_team_goal']
        matches.loc[home_win, 'result'] = 1
        matches.loc[away_win, 'result'] = -1
        return matches, 'win-draw-loss'

    elif label == 'gd':
        #convert home_team_goal and away_goal into goal_diff
        matches['result'] = (matches['home_team_goal'] - matches['away_team_goal']).astype(int)
        return matches, 'goal-difference'

def getAlgo(algo):
    if algo == "svm":
        return SVC(kernel = 'rbf', probability = True), 'Support Vector Machine'
    elif algo == "knn":
        return neighbors.KNeighborsClassifier(), 'k-Nearest Neighbors'
    elif algo == "nb":
        return GaussianNB(), 'Naive-Bayes'
    elif algo == 'dt':
    	return DecisionTreeClassifier(), 'Decision Tree'
    elif algo == 'rf':
        return RandomForestClassifier(), 'Random Forest'
    elif algo == 'lr':
        return LogisticRegression(multi_class = 'ovr'), 'Logistic Regression'
    elif algo == 'p':
        return Perceptron(tol=1e-3, random_state=0), 'Perceptron'

def getScorers(scorers):
    if scorers == "all":
        return ['f1_weighted', 'accuracy', 'roc_auc_ovo_weighted']
    out = []
    if '1' in scorers:
        out.append('f1_weighted')
    if '2' in scorers:
        out.append('accuracy')
    if '3' in scorers:
        out.append('roc_auc_ovo_weighted')
    return out
    
