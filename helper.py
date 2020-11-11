from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

algos_available = '{svm(support vector machine)/ knn(k-nearest neighbors)/ nb(naive bayes)/ dt(decision tree)/ rf(random forest)/ lr(logistic regression)}'
labels_available = '{wl(win-loss)/ wdl(win-draw-loss)/ gd(goal-diff)}'

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
        return SVC(kernel = 'rbf'), 'Support Vector Machine'
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
