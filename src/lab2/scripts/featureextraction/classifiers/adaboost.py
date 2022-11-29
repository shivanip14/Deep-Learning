from sklearn.ensemble import AdaBoostClassifier

def train(features, labels):
    # Train AdaBoost with the obtained features.
    clf = AdaBoostClassifier()
    print('Training AdaBoost on extracted train features')
    clf.fit(X=features, y=labels)
    print('Training complete')
    return clf
