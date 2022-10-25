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
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, BatchNormalization

base_path = './lab1data/'
base_img_path = base_path + 'data_256/'

mame_dataset = pd.read_csv(base_path + 'MAMe_dataset.csv')

# Since this is per model, read image lab1data from previously saved pkls
with open(base_path + 'pkls/train_imgs.pkl', 'rb') as f:
    mame_train_imgs =  pickle.load(f)
with open(base_path + 'pkls/val_imgs.pkl', 'rb') as f:
    mame_val_imgs =  pickle.load(f)
with open(base_path + 'pkls/test_imgs.pkl', 'rb') as f:
    mame_test_imgs =  pickle.load(f)





########## Preprocessing lab1data ###########
# Normalising the images
mame_train_imgs = np.array(mame_train_imgs, dtype=np.float32) / 255
mame_val_imgs = np.array(mame_val_imgs, dtype=np.float32) / 255
mame_test_imgs = np.array(mame_test_imgs, dtype=np.float32) / 255



# Preparing the labels as OHE

le = LabelEncoder()
ohe = OneHotEncoder(handle_unknown='ignore')

train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

train_df['Medium'] = le.fit_transform(mame_dataset[mame_dataset['Subset'] == 'train']['Medium'])
val_df['Medium'] = le.fit_transform(mame_dataset[mame_dataset['Subset'] == 'val']['Medium'])
test_df['Medium'] = le.fit_transform(mame_dataset[mame_dataset['Subset'] == 'test']['Medium'])

mame_train_labels = pd.DataFrame(ohe.fit_transform(train_df[['Medium']]).toarray()).values
mame_val_labels = pd.DataFrame(ohe.fit_transform(val_df[['Medium']]).toarray()).values
mame_test_labels = pd.DataFrame(ohe.fit_transform(test_df[['Medium']]).toarray()).values

print('Train images shape: ' + str(mame_train_imgs.shape))
print('Val images shape: ' + str(mame_val_imgs.shape))
print('Test images shape: ' + str(mame_test_imgs.shape))
print('Train labels shape: ' + str(mame_train_labels.shape))
print('Val labels shape: ' + str(mame_val_labels.shape))
print('Test labels shape: ' + str(mame_test_labels.shape))





############## Define CNN arch #######################
input = Input(shape=(256, 256, 3))
model = Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3), padding="same") (input)
model = Conv2D(32, (3, 3), activation='relu', padding="same") (model)
model = BatchNormalization() (model)
model = MaxPool2D(pool_size=(2, 2), strides=(2, 2)) (model)

model = Conv2D(64, (3, 3), activation='relu', padding="same") (model)
model = Conv2D(64, (3, 3), activation='relu', padding="same") (model)
model = BatchNormalization() (model)
model = MaxPool2D(pool_size=(2, 2), strides=(2, 2)) (model)

model = Conv2D(128, (3, 3), activation='relu', padding="same") (model)
model = Conv2D(128, (3, 3), activation='relu', padding="same") (model)
model = BatchNormalization() (model)
model = MaxPool2D(pool_size=(2, 2), strides=(2, 2)) (model)

model = GlobalAveragePooling2D() (model)
model = Dropout(0.6) (model)
model = Flatten() (model)
model = Dense(29, activation='softmax') (model)
finalmodel = Model(inputs=input, outputs=model)
finalmodel.summary()
finalmodel.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = finalmodel.fit(mame_train_imgs, mame_train_labels, batch_size=128, epochs=100, validation_data=(mame_val_imgs, mame_val_labels))

# test & evaluate
preds_test = finalmodel.evaluate(mame_test_imgs, mame_test_labels)
preds_labels = finalmodel.predict(mame_test_imgs)
print ("Test Loss = " + str(preds_test[0]))
print ("Test Accuracy = " + str(preds_test[1]))

# saving model and weights
with open(base_path + 'savedmodels/json/exp_14.json', 'w') as json_file:
        json_file.write(finalmodel.to_json())
weights_file = base_path + 'savedmodels/weights/weights-MAMe-exp_14_'+str(preds_test[1])+'.hdf5'
finalmodel.save_weights(weights_file, overwrite=True)

# saving accuracy and loss plots
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(base_path + 'savedmodels/accuracy/exp_14_accuracy.pdf')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(base_path + 'savedmodels/loss/exp_14_loss.pdf')
