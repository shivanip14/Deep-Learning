from sklearn.svm import LinearSVC

def train(features, labels):
    # Train SVM with the obtained features.
    clf = LinearSVC()
    print('Training SVM on extracted train features')
    clf.fit(X=features, y=labels)
    print('Training complete')
    return clf
