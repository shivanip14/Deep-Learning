from fne import full_network_embedding

def classify(trained_classifier, fne_features, classifier_name, dataset_category):
    print('Testing [' + classifier_name + '] on extracted ' + dataset_category + ' features')
    predicted_labels = trained_classifier.predict(fne_features)
    print('Done testing [' + classifier_name + '] on extracted features of ' + dataset_category + ' set')

    return predicted_labels