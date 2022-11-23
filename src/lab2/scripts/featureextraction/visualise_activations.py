from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import json
import glob
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt

# model summary
# Model: "functional_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 256, 256, 3)]     0
# _________________________________________________________________
# conv2d (Conv2D)              (None, 256, 256, 32)      896
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 256, 256, 32)      9248
# _________________________________________________________________
# batch_normalization (BatchNo (None, 256, 256, 32)      128
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 128, 128, 32)      0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 128, 128, 64)      18496
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 128, 128, 64)      36928
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 128, 128, 64)      256
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 64, 64, 64)        0
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 64, 64, 128)       73856
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 64, 64, 128)       147584
# _________________________________________________________________
# batch_normalization_2 (Batch (None, 64, 64, 128)       512
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 32, 32, 128)       0
# _________________________________________________________________
# global_average_pooling2d (Gl (None, 128)               0
# _________________________________________________________________
# dropout (Dropout)            (None, 128)               0
# _________________________________________________________________
# flatten (Flatten)            (None, 128)               0
# _________________________________________________________________
# dense (Dense)                (None, 29)                3741
# =================================================================


batch_size = 128
epochs = 100
base_path = './lab2data/'
base_img_path = base_path + 'data_256/'
model_path = base_path + 'savedmodels/json/base_model.json'
weights_path = base_path + 'savedmodels/weights/weights-MAMe-base-model.hdf5'

mame_dataset = pd.read_csv(base_path + 'MAMe_dataset.csv')
mame_labels = pd.read_csv(base_path + 'MAMe_labels.csv', header=None)

# read images from previously saved pkls
with open(base_path + 'pkls/val_imgs.pkl', 'rb') as f:
    mame_val_imgs = pickle.load(f)

mame_val_imgs = np.array(mame_val_imgs, dtype=np.float32) / 255

# Preparing the labels as OHE
le = LabelEncoder()
ohe = OneHotEncoder(handle_unknown='ignore')

val_df = pd.DataFrame()
val_df['Medium'] = le.fit_transform(mame_dataset[mame_dataset['Subset'] == 'val']['Medium'])

mame_val_labels = pd.DataFrame(ohe.fit_transform(val_df[['Medium']]).toarray()).values

print('Val images shape: ' + str(mame_val_imgs.shape))
print('Val labels shape: ' + str(mame_val_labels.shape))

# load model & weights
with open(model_path) as f:
    model_as_json = json.load(f)
loaded_model = model_from_json(json.dumps(model_as_json))

for filename in glob.glob(weights_path):
    loaded_model.load_weights(filename)
loaded_model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# print('\nPre-trained model:\n')
# loaded_model.summary()
# predict for a random sample image from validation set
input_img = np.expand_dims(mame_val_imgs[0], axis=0)
prediction = loaded_model.predict(input_img)
print('Activations: ' + str(prediction))
print('Class: ' + str(mame_labels[mame_labels[0] == np.argmax(prediction)].values[0][1]))

# create model to capture output of activations across first 12 layers (till its flattened)
all_layer_ops = [layer.output for layer in loaded_model.layers[:12]]
activation_model = Model(inputs=loaded_model.input, outputs=all_layer_ops)

# capture activations across all layers
activations = activation_model.predict(input_img)
print(len(activations)) # 12, one for each layer

# visualise captured activations
for layer_no in range(12):
    layer_act = activations[layer_no][0, ...]
    size = layer_act.shape[0]
    channels = layer_act.shape[-1]
    subplot_cols_req = (channels // 8) + 1
    fig, ax = plt.subplots(subplot_cols_req, 8, squeeze=False)
    fig.set_figheight(subplot_cols_req*2)
    fig.set_figwidth(16)
    print('Layer ' + str(layer_no) + ', channels ' + str(channels) + ', created axes object of shape ' + str(ax.shape))
    channel_no = 0
    for i in range(subplot_cols_req):
        for j in range(8):
            ax[i][j].axis('off')
            if channel_no < channels:
                ax[i][j].imshow(layer_act[:, :, channel_no])
                ax[i][j].set_title('Layer' + str(layer_no) + '_Channel' + str(channel_no))
                channel_no += 1
    plt.axis('off')
    plt.savefig('layer' + str(layer_no) + '_activations.png')
    plt.close()
