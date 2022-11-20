from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
import seaborn as sn

exp_no = sys.argv[1]

try:
    _impobject = __import__('fine_tune_' + str(exp_no), globals(), locals(), ['fine_tune_model'], 0)
    fine_tune_model = _impobject.fine_tune_model
except ImportError:
    print("fine_tune_" + str(exp_no) + ".py not found")

img_width, img_height = 256, 256
batch_size = 128
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

mame_train_imgs = np.array(mame_train_imgs, dtype=np.float32)
mame_val_imgs = np.array(mame_val_imgs, dtype=np.float32)
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

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255)  # ,
# horizontal_flip = True,
# fill_mode = "nearest",
# zoom_range = 0.3,
# width_shift_range = 0.3,
# height_shift_range=0.3,
# rotation_range=30)
val_datagen = ImageDataGenerator(
    rescale=1. / 255)  # ,
# horizontal_flip = True,
# fill_mode = "nearest",
# zoom_range = 0.3,
# width_shift_range = 0.3,
# height_shift_range=0.3,
# rotation_range=30)
train_generator = train_datagen.flow(
    mame_train_imgs,
    mame_train_labels,
    batch_size=batch_size,
    shuffle=False)
val_generator = val_datagen.flow(
    mame_val_imgs,
    mame_val_labels,
    batch_size=batch_size,
    shuffle=False)

# loading best CNN arch from Lab1 - exp_13 renamed and copied as base_model
with open(base_path + 'savedmodels/json/base_model.json') as f:
    model_as_json = json.load(f)

loaded_model = model_from_json(json.dumps(model_as_json))

weights_path = base_path + 'savedmodels/weights/weights-MAMe-base-model.hdf5'
for filename in glob.glob(weights_path):
    loaded_model.load_weights(filename)

print('Pre-trained model:\n')
loaded_model.summary()

final_model = fine_tune_model(loaded_model)

print('Fine-tuned model:\n')
final_model.summary()

# run training on tweaked model
early = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10, verbose=1, mode='auto')
history = final_model.fit(train_generator, steps_per_epoch=(train_generator.n / batch_size), epochs=epochs, validation_data=val_generator, validation_steps=(val_generator.n / batch_size), callbacks=[early])

# accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.title('Training and validation accuracy')
plt.savefig(base_path + 'savedmodels/accuracy/ft_' + exp_no + '.pdf')
plt.close()

# loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.title('Training and validation loss')
plt.savefig(base_path + 'savedmodels/loss/ft_' + exp_no + '.pdf')

# evaluate & save performance
preds_test = final_model.evaluate(mame_test_imgs, mame_test_labels)
pred_labels = final_model.predict(mame_test_imgs, verbose=1)
pred_labels_bool = np.argmax(pred_labels, axis=1)
test_labels = np.argmax(mame_test_labels, axis=1)

conf = confusion_matrix(test_labels, pred_labels_bool)
classes = mame_dataset['Medium'].unique()

conf_df = pd.DataFrame(conf, index=classes, columns=classes)
plt.figure(figsize=(10, 10))
conf_fig = sn.heatmap(conf_df, annot=False, square=True, xticklabels=classes, yticklabels=classes)
conf_fig.get_figure().savefig(base_path + 'savedmodels/conf/' + exp_no + '_confusion_matrix.png')

with open(base_path + 'savedmodels/reports/' + exp_no + '_report.txt', 'w') as report_file:
    report_file.write('\nTest loss = ' + str(preds_test[0]))
    report_file.write('\nTest accuracy = ' + str(preds_test[1]))
    report_file.write('\nCohen-Kappa score = ' + str(cohen_kappa_score(test_labels, pred_labels_bool)))
    report_file.write('\n')
    report_file.write(classification_report(test_labels, pred_labels_bool, target_names=classes))

# saving fine-tuned model and weights
with open(base_path + 'savedmodels/json/' + exp_no + '.json', 'w') as json_file:
        json_file.write(final_model.to_json())
weights_file = base_path + 'savedmodels/weights/weights-MAMe-ft_' + exp_no + '_' + str(preds_test[1]) + '.hdf5'
final_model.save_weights(weights_file, overwrite=True)
