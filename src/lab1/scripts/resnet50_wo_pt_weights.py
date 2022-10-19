import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, LeakyReLU, Input, concatenate, Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.backend import clear_session
from sklearn.metrics import confusion_matrix
import seaborn as sn


base_path = '../lab1data/'
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
clear_session()
base_model = ResNet50(weights=None, include_top=False, input_shape=(256,256,3))
op = base_model.output
op = Dense(29, activation= 'softmax')(op)
finalmodel = Model(inputs = base_model.input, outputs = op)
finalmodel.summary()

finalmodel.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = finalmodel.fit(mame_train_imgs, mame_train_labels, batch_size=64, epochs=100, validation_data=(mame_val_imgs, mame_val_labels))

# save history and trained model first - for some reason, doesn't work on P9
# with open(base_path + 'pkls/histandtrainedmods/resnet50_wo_pt_weights_history.pkl', 'wb') as f:
#     pickle.dump(history, f)
# with open(base_path + 'pkls/histandtrainedmods/resnet50_wo_pt_weights_trainedmodel.pkl', 'wb') as f:
#     pickle.dump(finalmodel, f)

# option to load saved history and trained model directly - comment if training again
# with open(base_path + 'pkls/histandtrainedmods/resnet50_wo_pt_weights_history.pkl', 'rb') as f:
#     history = pickle.load(f)
# with open(base_path + 'pkls/histandtrainedmods/resnet50_wo_pt_weights_trainedmodel.pkl', 'rb') as f:
#     finalmodel = pickle.load(f)

# test & evaluate
preds_test = finalmodel.evaluate(mame_test_imgs, mame_test_labels)
preds_labels = finalmodel.predict(mame_test_imgs)
print ("Test Loss = " + str(preds_test[0]))
print ("Test Accuracy = " + str(preds_test[1]))

# saving model and weights

with open(base_path + 'savedmodels/resnet50_wo_pt_weights.json', 'w') as json_file:
        json_file.write(finalmodel.to_json())
weights_file = "weights-MAMe-resnet50-wo-pt-weights_"+str(preds_test[1])+".hdf5"
finalmodel.save_weights(weights_file, overwrite=True)

# saving accuracy and loss plots
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(base_path + 'savedmodels/mame_resnet50_wo_pt_weights_accuracy.pdf')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(base_path + 'savedmodels/mame_resnet50_wo_pt_weights_loss.pdf')

# plotting & saving the confusion matrix as a heatmap

predictions = np.argmax(preds_labels, axis = 1)
labels = np.argmax(mame_test_labels, axis = 1)

conf = confusion_matrix(labels, predictions)
classes = np.arange(0, 29)

conf_df = pd.DataFrame(conf, index = classes, columns = classes)
plt.figure(figsize = (10, 8))
plt.axis('off')
conf_fig = sn.heatmap(conf_df, annot=True)
conf_fig.get_figure().savefig(base_path + 'savedmodels/mame_resnet50_wo_pt_weights_test_confusion_matrix.pdf', dpi=400)
