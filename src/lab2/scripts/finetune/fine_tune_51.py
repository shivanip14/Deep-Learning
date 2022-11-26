from keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Dense, BatchNormalization

def fine_tune_model(loaded_model, delay_loading_weights=False, base_model_exp=None):
    ##### tweak loaded_model to make a final_model
    # freezing layers which will not be trained
    for layer in loaded_model.layers:
        layer.trainable = False

    # adding custom layers
    x = loaded_model.output
    x = Flatten(name='added_flatten_1')(x)
    x = BatchNormalization()(x)
    op = Dense(29, activation='softmax', name='added_dense_1')(x)

    # creating the final model
    final_model = Model(loaded_model.input, op)
    ##### end tweaking

    final_model.compile(optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return final_model
