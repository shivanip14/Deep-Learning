from keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Dense
import glob
from keras.losses import CategoricalCrossentropy

base_path = './lab2data/'

def fine_tune_model(loaded_model, delay_loading_weights=False, base_model_exp=None):
    unfreeze_last_n_layers = 1
    for layer in loaded_model.layers[::-1]:
        if unfreeze_last_n_layers > 0:
            if len(layer.trainable_weights) > 0:
                layer.trainable = True
                unfreeze_last_n_layers -= 1
        else:
            layer.trainable = False

    # adding custom layers
    x = loaded_model.output
    x = Flatten(name='added_flatten_1')(x)
    op = Dense(29, activation='softmax', name='added_dense_1')(x)

    # creating the final model
    final_model = Model(loaded_model.input, op)

    if delay_loading_weights:
        print('Loading weights from experiment ' + str(base_model_exp))
        weights_path = base_path + 'savedmodels/weights/weights-MAMe-ft_' + base_model_exp + '_' + '*.hdf5'
        for filename in glob.glob(weights_path):
            final_model.load_weights(filename)

    final_model.compile(optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9), loss=CategoricalCrossentropy(label_smoothing=0.05), metrics=['accuracy'])
    return final_model
