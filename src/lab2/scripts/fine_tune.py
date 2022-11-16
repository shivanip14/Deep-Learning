from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

img_width, img_height = 256, 256
batch_size = 128
epochs = 100
base_path = './lab2data/'
base_img_path = base_path + 'data_256/'

def fine_tune_model(exp_no, loaded_model, train_generator, val_generator):
    ##### tweak loaded_model to make a final_model - TODO
    # freeze the layers which you don't want to train. Here I am freezing the first 10 layers.
    for layer in loaded_model.layers[:10]:
        layer.trainable = False

    # adding custom Layers
    x = loaded_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(29, activation='softmax')(x)

    # creating the final model
    final_model = Model(loaded_model.input, predictions)
    ##### end tweaking

    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # or below
    # final_model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

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

    # saving fine-tuned model and weights
    with open(base_path + 'savedmodels/json/' + exp_no + '.json', 'w') as json_file:
            json_file.write(final_model.to_json())
    weights_file = base_path + 'savedmodels/weights/weights-MAMe-ft_' + exp_no + '.hdf5'
    final_model.save_weights(weights_file, overwrite=True)

