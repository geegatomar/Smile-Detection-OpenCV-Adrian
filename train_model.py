from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from LeNet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

data = []
labels = []

for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    print("[INFO] processing image ", imagePath)
    #print(imagePath)
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-3]
    #print("label: ", label)
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)
    

data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

classTotals = labels.sum(axis=0)
classWeight = classTotals.max()/classTotals
#print(classTotals)
#print(classTotals.max())
#print(classWeight)

# This shows our dataset is skewed, and this can be fixed by using "stratify" in train_test_split as below

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, stratify=labels, test_size=0.2)

print("[INFO] compiling model...")
model = LeNet.build(height=28, width=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[INFO] training the network...")
H =model.fit(X_train, y_train, validation_data=(X_test, y_test), class_weight=classWeight, batch_size=64, epochs=14)

print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=64)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO serializing network")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 14), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 14), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 14), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 14), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


# notice that after 12 epochs it starts to overfit the model

