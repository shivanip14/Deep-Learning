from sklearn.naive_bayes import GaussianNB

def train(features, labels):
    # Train Gaussian Naive Bayes with the obtained features.
    clf = GaussianNB()
    print('Training Gaussian Naive Bayes on extracted train features')
    clf.fit(X=features, y=labels)
    print('Training complete')
    return clf
