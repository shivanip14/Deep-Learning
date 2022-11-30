import pandas as pd
import pickle
import json
import numpy as np
import sys
from fne import full_network_embedding
from keras import optimizers
from keras.models import Model
import glob
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications import VGG16, Xception, DenseNet121, InceptionV3
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn
from classifierevaluator import classify

exp_no = sys.argv[1]
base_model_exp = sys.argv[2]
classifiers = ['adaboost', 'decisiontree', 'gaussiannb', 'svm', 'randomforest']
if len(sys.argv) > 3:
    classifiers = [sys.argv[3]]
print('Experiment no: [' + str(exp_no) + '], with pre-trained model: [' + str(base_model_exp) + '] on imagenet weights, using classifier(s) [' + str(classifiers) + ']')

try:
    _impobject = __import__('fe_' + str(exp_no), globals(), locals(), ['target_layer_names'], 0)
    target_layer_names = _impobject.target_layer_names
except ImportError:
    print('fe_' + str(exp_no) + ".py not found")

print('Target layers: ' + str(len(target_layer_names)))

batch_size = 32
epochs = 100
base_path = './lab2data/'
base_img_path = base_path + 'data_256/'

mame_dataset = pd.read_csv(base_path + 'MAMe_dataset.csv')

# read images from previously saved pkls
with open(base_path + 'pkls/train_imgs.pkl', 'rb') as f:
    mame_train_imgs = pickle.load(f)
with open(base_path + 'pkls/val_imgs.pkl', 'rb') as f:
    mame_val_imgs = pickle.load(f)
with open(base_path + 'pkls/test_imgs.pkl', 'rb') as f:
    mame_test_imgs = pickle.load(f)

mame_train_imgs = np.array(mame_train_imgs, dtype=np.float32) / 255
mame_val_imgs = np.array(mame_val_imgs, dtype=np.float32) / 255
mame_test_imgs = np.array(mame_test_imgs, dtype=np.float32) / 255

train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

mame_train_labels = mame_dataset[mame_dataset['Subset'] == 'train']['Medium']
mame_val_labels = mame_dataset[mame_dataset['Subset'] == 'val']['Medium']
mame_test_labels = mame_dataset[mame_dataset['Subset'] == 'test']['Medium']

print('Train images shape: ' + str(mame_train_imgs.shape))
print('Train labels shape: ' + str(mame_train_labels.shape))
print('Validation images shape: ' + str(mame_val_imgs.shape))
print('Validation labels shape: ' + str(mame_val_labels.shape))
print('Test images shape: ' + str(mame_test_imgs.shape))
print('Test labels shape: ' + str(mame_test_labels.shape))

if base_model_exp == 'vgg16':
    final_model = VGG16(include_top=False, weights=base_path + '/savedmodels/weights/vgg16-base-no-top.h5', input_shape=(256, 256, 3))
elif base_model_exp == 'xception':
    final_model = Xception(include_top=False, weights=base_path + '/savedmodels/weights/xception-base-no-top.h5', input_shape=(256, 256, 3))
elif base_model_exp == 'densenet121':
    final_model = DenseNet121(include_top=False, weights=base_path + '/savedmodels/weights/densenet121-base-no-top.h5', input_shape=(256, 256, 3))
elif base_model_exp == 'inceptionv3':
    final_model = InceptionV3(include_top=False, weights=base_path + '/savedmodels/weights/inceptionv3-base-no-top.h5', input_shape=(256, 256, 3))
else:
    print('Chosen pre-trained base model [' + str(base_model_exp) + '] not supported yet. Choose one of [vgg16, xception, densenet121, inceptionv3]')
    exit(1)

final_model.compile(optimizer=optimizers.SGD(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
final_model.summary()

# call FNE method on the train set
print('Extracting features of training set')
fne_features, fne_stats_train = full_network_embedding(final_model, mame_train_imgs, batch_size, target_layer_names, None)
print('Done extracting features of training set. Embedding size:', fne_features.shape)

# call FNE method on the test set, using stats from training
print('Extracting features of validation set')
fne_val_features, fne_stats_val = full_network_embedding(final_model, mame_val_imgs, batch_size, target_layer_names, stats=fne_stats_train)
print('Done extracting features of validation set')

print('Extracting features of test set')
fne_test_features, fne_stats_train = full_network_embedding(final_model, mame_test_imgs, batch_size, target_layer_names, stats=fne_stats_train)
print('Done extracting features of test set')

train = []
for classifier in classifiers:
    try:
        _impobject = __import__(str(classifier), globals(), locals(), ['train'], 0)
        train.append(_impobject.train)
    except ImportError:
        print(str(classifier) + ".py not found")

classes = mame_dataset['Medium'].unique()

for itr, classifier in enumerate(classifiers):
    clf = train[itr](fne_features, mame_train_labels)
    predicted_val_labels = classify(clf, fne_val_features, classifier, 'validation')
    print('Validation accuracy for [' + classifier + ']: ' + str(accuracy_score(mame_val_labels, predicted_val_labels)))
    predicted_test_labels = classify(clf, fne_test_features, classifier, 'test')

    conf = confusion_matrix(mame_test_labels, predicted_test_labels)
    conf_df = pd.DataFrame(conf, index=classes, columns=classes)
    plt.figure(figsize=(10, 10))
    conf_fig = sn.heatmap(conf_df, annot=False, square=True, xticklabels=classes, yticklabels=classes)
    conf_fig.get_figure().savefig(base_path + 'savedmodels/fe/conf/' + exp_no + '_' + classifier + '_confusion_matrix.png')

    with open(base_path + 'savedmodels/fe/reports/' + exp_no + '_' + classifier + '_report.txt', 'w') as report_file:
        report_file.write('\nTest accuracy = ' + str(accuracy_score(mame_test_labels, predicted_test_labels)))
        report_file.write('\nCohen-Kappa score = ' + str(cohen_kappa_score(mame_test_labels, predicted_test_labels)))
        report_file.write('\n')
        report_file.write(classification_report(mame_test_labels, predicted_test_labels))
