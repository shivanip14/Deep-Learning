from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense

def fine_tune_model(loaded_model):
    ##### tweak loaded_model to make a final_model
    # freezing layers which will not be trained
    for layer in loaded_model.layers:
        layer.trainable = False

    # adding custom layers
    x = loaded_model.output
    x = Dense(512, activation='relu', name='added_dense_1')(x)
    x = Dense(512, activation='relu', name='added_dense_2')(x)
    x = Dropout(0.2, name='added_dropout_1')(x)
    op = Dense(29, activation='softmax', name='added_dense_3')(x)

    # creating the final model
    final_model = Model(loaded_model.input, op)
    ##### end tweaking

    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # or below
    # final_model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return final_model
