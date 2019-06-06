from keras.models import load_model
import cv2
import numpy as np
import csv
import glob
import keras
from sklearn.metrics import accuracy_score
from keras.models import model_from_json
import matplotlib.pyplot as plt
import operator

from tensorflow.contrib.gan.python.eval import preprocess_image

IMG_SIZE = 48

##################################################################################################################
# Function to load Data test
##################################################################################################################

def load_test():
    X_test = []
    Y_test = []
    img_origine=[]
    with open('/home/scan-projet-7/Downloads/GTSRB_Final_Test_GT/GT-final_test.csv', 'r') as csvfile:
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

        #print(img_origine)
        #print(np.argmax(Y_test[3]))

        return X_test, Y_test, img_origine


##################################################################################################################
# Function to load classes names
##################################################################################################################
# load classes and indices
def load_class():
    classes = {}
    with open('/home/scan-projet-7/Downloads/GTSRB/classes.csv', 'r') as csvfile:
        reader_csv = csv.reader(csvfile)
        #next(reader_csv)

        for line in reader_csv:
            id = str(line[0])
            label = str(line[1])
            classes[id] = label
            #print(str(id) + ':' + str(label))
    return classes


##################################################################################################################
# Function to load the saved model
##################################################################################################################

def load_keras_model(filename):
    # load json and create model
    json_file = open('/home/scan-projet-7/Downloads/GTSRB/'+filename+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('/home/scan-projet-7/PycharmProjects/untitled/model_weights'+".h5")
    #model.load_weights('/home/scan-projet-7/Downloads/GTSRB/model-keras_weights'+".h5")

    return model

filename_to_load='model-keras'
model = load_keras_model(filename_to_load)
#model.summary()

##################################################################################################################
# compile  the loaded model
##################################################################################################################
#sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam=keras.optimizers.Adam(lr=1e-6)
sgd = keras.optimizers.SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
#model.compile(optimizer=sgd, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.compile(optimizer=sgd, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])



X_test, Y_test, imgload=load_test();

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Model performance (%s): %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("Model error rate : %.2f%%" % (100-scores[1]*100))

##################################################################################################################
# Function to predict take one image
##################################################################################################################

# Prediction
#img0 = cv2.imread('/home/scan-projet-7/Desktop/for_test/t.jpg')


def predict_image(image):

    draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(draw, (IMG_SIZE, IMG_SIZE))
    img = np.reshape(img, [1, IMG_SIZE, IMG_SIZE, 3])

    img_class = model.predict(img)
    prediction=img_class[0]

    listPrd=[]
    for i in prediction:
        listPrd.append(i)
        index, value = max(enumerate(listPrd), key=operator.itemgetter(1))
    print(index, value)
    print(prediction)

    # show image plot
    plt.imshow(draw)
    plt.title("Classe: %s, predict: %.2f%%\n" % (load_class().get(str(index)), value*100), fontsize=13)
    plt.show()

##################################################################################################################
# Function to predict by decoding vector prediction
##################################################################################################################

#for x in X_test:
prediction= model.predict(X_test[:len(X_test)])


def prediction_evaluation(vect):
    count=0; nbr=0
    for i in vect:
        nbr=nbr+1
        print("class predicted: %s, precent: %.2f%%" % (np.argmax(i), max(i)*100))
        print("True Class: %s " % (np.argmax(Y_test[nbr-1])))
        if np.argmax(i)==np.argmax(Y_test[nbr-1]):
            count=count+1
    evaluation= (count/X_test.shape[0])*100
    print(nbr)
    print("Preformance: %s%% " % (evaluation))

    return evaluation


##################################################################################################################
# Function to predict
##################################################################################################################
"""
import datetime
cap = cv2.VideoCapture('/home/scan-projet-7/Downloads/video.mp4')

counter = 0
sum_time = 0
while (True):
    ret, draw = cap.read()
    if not ret:
        break
    bgr = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    # preprocess image for network
    #image = preprocess_image(bgr)
    image, = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    scale= cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    #img = np.reshape(image, [1, IMG_SIZE, IMG_SIZE, 3])

    # process image
    start = datetime.time.time()
    _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
    t = datetime.time.time() - start


    predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
    scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

    # correct for image scale
    detections[0, :, :4] /= scale

    # visualize detections
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
        b = detections[0, idx, :4].astype(int)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), 245, 6)
        caption = "%s: %.1f%%" % ('label', score * 100)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.imwrite('/tmp/img%08d.jpg' % counter, draw)
    counter = counter + 1
    sum_time += t

cap.release()
cv2.destroyAllWindows()


path = glob.glob('/home/scan-projet-7/Deskktop/test1/*.ppm')
for img in path:
    print(img)
    image = cv2.imread(img)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image=np.reshape(draw, [1, IMG_SIZE, IMG_SIZE, 3])
    i=predict_image(image)

##############################################################################################
image= cv2.imread('/home/scan-projet-7/Desktop/for_test/test.jpg')
draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = cv2.resize(draw, (IMG_SIZE, IMG_SIZE))
img = np.reshape(img, [1, IMG_SIZE, IMG_SIZE, 3])

img_class = model.predict(img)
prediction = img_class[0]

listPrd = []
for i in prediction:
    listPrd.append(i)
    index, value = max(enumerate(listPrd), key=operator.itemgetter(1))

print(index, value)
print(prediction)

# show image plot
plt.imshow(draw)
plt.title("Classe: %s, predict: %.2f%%\n" % (load_class().get(str(index)), value * 100), fontsize=13)
plt.show()
"""""



prediction_evaluation(prediction)

img0 = cv2.imread('/home/scan-projet-7/Downloads/GTSRB_Final_Test_GT/Final_Test/Images/00012.ppm')
predict_image(img0)
