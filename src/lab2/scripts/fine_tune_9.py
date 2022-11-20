from keras import optimizers
from keras.models import Model
import glob

base_path = './lab2data/'

def fine_tune_model(loaded_model, delay_loading_weights=False, base_model_exp=None):
    if delay_loading_weights:
        print('Loading weights from experiment ' + str(base_model_exp))
        weights_path = base_path + 'savedmodels/weights/weights-MAMe-ft_' + base_model_exp + '_' + '*.hdf5'
        for filename in glob.glob(weights_path):
            loaded_model.load_weights(filename)

    # freezing layers which will not be trained
    for layer in loaded_model.layers:
        layer.trainable = True
    for layer in loaded_model.layers[:12]:
        layer.trainable = False

    # creating the final model
    final_model = Model(loaded_model.input, loaded_model.output)

    final_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # or below
    # final_model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return final_model
