from sklearn.tree import DecisionTreeClassifier

def train(features, labels):
    # Train DT with the obtained features.
    clf = DecisionTreeClassifier()
    print('Training Decision Tree on extracted train features')
    clf.fit(X=features, y=labels)
    print('Training complete')
    return clf
