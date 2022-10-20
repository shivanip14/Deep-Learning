# for reproducibility of results
import os
os.environ['PYTHONHASHSEED']=str(14)
import random
random.seed(14)
from numpy.random import seed
seed(14)
import tensorflow as tf
tf.random.set_seed(14)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import model_from_json
import json
import glob
import sys
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Activation

def gelu(x):
    cdf = 0.5*(1.0 + tf.tanh((np.sqrt(2/np.pi)*(x + 0.044715*tf.pow(x, 3)))))
    return x*cdf

model_to_test = sys.argv[1]

print('Calculating scores for ' + model_to_test)

base_path = './lab1data/'
base_img_path = base_path + 'data_256/'

mame_dataset = pd.read_csv(base_path + 'MAMe_dataset.csv')

# read image lab1data from previously saved pkls
with open(base_path + 'pkls/test_imgs.pkl', 'rb') as f:
    mame_test_imgs =  pickle.load(f)

# normalising the images
mame_test_imgs = np.array(mame_test_imgs, dtype=np.float32) / 255

# Preparing the labels as OHE
le = LabelEncoder()
ohe = OneHotEncoder(handle_unknown='ignore')
test_df = pd.DataFrame()
test_df['Medium'] = le.fit_transform(mame_dataset[mame_dataset['Subset'] == 'test']['Medium'])
mame_test_labels = pd.DataFrame(ohe.fit_transform(test_df[['Medium']]).toarray()).values

print('Test images shape: ' + str(mame_test_imgs.shape))
print('Test labels shape: ' + str(mame_test_labels.shape))

# loading CNN arch
with open(base_path + 'savedmodels/json/' + model_to_test + '.json') as f:
    model_as_json = json.load(f)

if model_to_test == "exp_19":
    loaded_model = model_from_json(json.dumps(model_as_json), custom_objects={'gelu': Activation(gelu)})
else:
    loaded_model = model_from_json(json.dumps(model_as_json))

weights_path = base_path + 'savedmodels/weights/weights-MAMe-' + model_to_test + '_' + '*.hdf5'
for filename in glob.glob(weights_path):
    loaded_model.load_weights(filename)
loaded_model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

# evaluate & save performance
preds_test = loaded_model.evaluate(mame_test_imgs, mame_test_labels)
pred_labels = loaded_model.predict(mame_test_imgs, verbose=1)
pred_labels_bool = np.argmax(pred_labels, axis=1)
test_labels = np.argmax(mame_test_labels, axis = 1)

with open(base_path + 'savedmodels/reports/' + model_to_test + '_report.txt', 'w') as report_file:
    report_file.write('\nTest loss = ' + str(preds_test[0]))
    report_file.write('\nTest accuracy = ' + str(preds_test[1]))
    report_file.write('\n')
    report_file.write(classification_report(test_labels, pred_labels_bool, target_names=mame_dataset['Medium'].unique()))

