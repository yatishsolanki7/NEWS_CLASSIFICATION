import pickle
import timeit

with open('Data/processed_training_data_temp.pickle','rb') as f:
    [X_train, y_train, X_test, y_test]=pickle.load(f)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Define models to train
classifiers = [
        ("K Nearest Neighbors", KNeighborsClassifier()),
        ("Decision Tree"      , DecisionTreeClassifier()),
        ("Random Forest"      , RandomForestClassifier()),
        ("Logistic Regression", LogisticRegression()),
        ("SGD Classifier"     , SGDClassifier(max_iter = 100)),
        ("Naive Bayes"        , MultinomialNB()),
        ("SVM Linear"         , SVC(kernel = 'linear'))

        ]

for t in classifiers:
    start = timeit.default_timer()
    clf = t[1]
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)*100
    stop = timeit.default_timer()
    print("{} \t Accuracy: {}% \t Time: {} secs".format(t[0], accuracy, stop-start))

#  Voting classifier
from sklearn.ensemble import VotingClassifier
start = timeit.default_timer()
voting_clf = VotingClassifier(estimators = classifiers, voting='hard', weights=None)
voting_clf = voting_clf.fit(X_train,y_train)
accuracy = voting_clf.score(X_test,y_test)*100
stop = timeit.default_timer()
print("Voting classifier \t Accuracy: {}% \t Time: {} secs".format( accuracy, stop-start))
