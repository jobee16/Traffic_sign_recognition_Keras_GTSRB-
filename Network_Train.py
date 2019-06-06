import os
import cv2
import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import StratifiedKFold

IMG_SIZE = 48
batch_size = 32
Numb_class = 43
epochs = 400


#loaddata
def loaddata():
    X = []
    Y = []

    path = glob.glob('/home/jobee16/Downloads/GTSRB/*/*.ppm')
    for img in path:
        #print(img)
        image = cv2.imread(img)
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        image = cv2.resize(draw, (IMG_SIZE, IMG_SIZE))
        c = int(img.split('/', -1)[-2])
        y = np.zeros(43)
        y[c] = 1.
        X.append(image)
        Y.append(y)
    #print(X)
    X = np.array(X, dtype=np.uint8)
    Y = np.array(Y,dtype=np.float32)
    return X, Y


def load_test():
    X_test = []
    Y_test = []
    img_origine=[]
    with open('/home/jobee16/Downloads/GTSRB_Final_Test_GT/GT-final_test.csv', 'r') as csvfile:
        reader_csv = csv.reader(csvfile)
        next(reader_csv)

        for line in reader_csv:
            convx=line[0].split(';', -1)[0]
            conbine_path='/home/scan-projet-7/Downloads/GTSRB_Final_Test_GT/Final_Test/Images/'+convx
            imag=cv2.imread(conbine_path)
            draw = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
            image = cv2.resize(draw, (IMG_SIZE, IMG_SIZE))

            convy=int(line[0].split(';', -1)[-1])
            y = np.zeros(43)
            y[convy] = 1.0

            X_test.append(image)
            Y_test.append(y)
            img_origine.append(conbine_path)

        X_test = np.array(X_test, dtype=np.uint8)
        Y_test = np.array(Y_test, dtype=np.float32)

        print(img_origine)
        #print(np.argmax(Y_test[3]))

        return X_test, Y_test, img_origine


def model():
    input = keras.Input(shape=(IMG_SIZE , IMG_SIZE, 3))
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation=keras.activations.relu)(input)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation=keras.activations.relu)(x)

    # 1er Maxpooling Et Normalisation
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation=keras.activations.relu)(x)

    # 2eme Maxpooling Et Normalisation
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)


    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation=keras.activations.relu)(x)

    #  4e Maxpooling Et Normalisation
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.Conv2D(512, (3, 3), padding='same', activation=keras.activations.relu)(x)


    #  5e Maxpooling passe l'image a 8
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    #flatten
    x = keras.layers.Flatten()(x)


    x = keras.layers.Dense(64, activation=keras.activations.relu)(x)
    # Normalisation
    x = keras.layers.BatchNormalization()(x)

    output = keras.layers.Dense(43, kernel_initializer='uniform', activation=keras.activations.softmax)(x)

    return keras.Model(inputs=input, outputs=output)



def save_keras_model(model, filename):
    save_weights_path='/home/jobee16/Downloads/GTSRB/'+ filename +"_weights.h5"
    save_keras_modelh5_path='/home/jobee16/Downloads/GTSRB/'+ filename+'_h5' +".h5"
    save_model_jason_path='/home/jobee16/Downloads/GTSRB/'+filename+".json"

    # serialize model to JSON
    model_json = model.to_json()
    with open(save_model_jason_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(save_weights_path)
    model.save(save_keras_modelh5_path)


X , Y =loaddata();
X_test, Y_test, image=load_test();

model = model()
model.summary()


sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
adam=keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=sgd, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#history = model.fit(X, Y, batch_size, epochs, verbose=1, validation_split=0.25, callbacks=callbacks_list)
history = model.fit(X, Y, batch_size, epochs, verbose=1, validation_data=(X_test, Y_test), callbacks=callbacks_list)


"""
loss, acc = model.evaluate(X_test, Y_test, verbose=0)validation_data=(X_test, Y_test)
print(loss, acc)

# evaluate the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
"""


# Plot Accuracy & Epoch values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy', fontsize=10)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss', fontsize=10)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.suptitle('Optimizer : SGD\n', fontsize=10)
plt.ylabel('Validation Loss', fontsize=9)
plt.xlabel('Training Loss', fontsize=9)
plt.legend(loc='upper right')
plt.show()

#Save in weights and model
save_keras_model(model, 'model-keras')
