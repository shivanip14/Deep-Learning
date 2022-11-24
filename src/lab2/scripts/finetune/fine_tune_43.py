from keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Dense
import glob

base_path = './lab2data/'

def fine_tune_model(loaded_model, delay_loading_weights=False, base_model_exp=None):
    for layer in loaded_model.layers:
        layer.trainable = False
    unfreeze_last_n_layers = 1
    for layer in loaded_model.layers[::-1]:
        if unfreeze_last_n_layers > 0:
            layer.trainable = True
            unfreeze_last_n_layers -= 1
        else:
            break

    # adding custom layers
    x = loaded_model.output
    x = Flatten(name='added_flatten_1')(x)
    x = Dense(512, activation='relu', name='added_dense_1')(x)
    op = Dense(29, activation='softmax', name='added_dense_2')(x)

    # creating the final model
    final_model = Model(loaded_model.input, op)

    if delay_loading_weights:
        print('Loading weights from experiment ' + str(base_model_exp))
        weights_path = base_path + 'savedmodels/weights/weights-MAMe-ft_' + base_model_exp + '_' + '*.hdf5'
        for filename in glob.glob(weights_path):
            final_model.load_weights(filename)

    final_model.compile(optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return final_model
