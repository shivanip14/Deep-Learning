from sklearn.ensemble import RandomForestClassifier

def train(features, labels):
    # Train RF with the obtained features.
    clf = RandomForestClassifier()
    print('Training Random Forest on extracted train features')
    clf.fit(X=features, y=labels)
    print('Training complete')
    return clf
