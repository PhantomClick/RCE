from keras.datasets import mnist
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Download Mnist Dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

print(len(x_train))


hog = cv2.HOGDescriptor()
hog_train_1 = []
hog_train_2 = []
hog_train_3 = []
hog_train_4 = []
hog_train_5 = []
hog_train_6 = []
hog_train_7 = []
hog_train_8 = []
hog_train_9 = []
hog_train_0 = []

for i in range(len(x_train)):
    img = np.reshape(x_train[i], (28, 28))

    img = cv2.resize(img, (64,128), interpolation=cv2.INTER_CUBIC)

    print(i)

    h = hog.compute(img)

    if y_train[i] == 0:
        hog_train_0.append(h)
    elif y_train[i] == 1:
        hog_train_1.append(h)
    elif y_train[i] == 2:
        hog_train_2.append(h)
    elif y_train[i] == 3:
        hog_train_3.append(h)
    elif y_train[i] == 4:
        hog_train_4.append(h)
    elif y_train[i] == 5:
        hog_train_5.append(h)
    elif y_train[i] == 6:
        hog_train_6.append(h)
    elif y_train[i] == 7:
        hog_train_7.append(h)
    elif y_train[i] == 8:
        hog_train_8.append(h)
    elif y_train[i] == 9:
        hog_train_9.append(h)


print(len(hog_train_0))
print(len(hog_train_1))
print(len(hog_train_2))
print(len(hog_train_3))
print(len(hog_train_4))
print(len(hog_train_5))
print(len(hog_train_6))
print(len(hog_train_7))
print(len(hog_train_8))
print(len(hog_train_9))


# Take mean of each hog features and calculate cosine similarity to
#   check which ones are more close to others.

mean_features = np.zeros(shape=(10,3780))

mean_0 = np.mean(np.array(hog_train_0),axis=0)
mean_features[0,:] = np.reshape(mean_0,3780)

mean_1 = np.mean(np.array(hog_train_1),axis=0)
mean_features[1,:] = np.reshape(mean_1,3780)

mean_2 = np.mean(np.array(hog_train_2),axis=0)
mean_features[2,:] = np.reshape(mean_2,3780)

mean_3 = np.mean(np.array(hog_train_3),axis=0)
mean_features[3,:] = np.reshape(mean_3,3780)

mean_4 = np.mean(np.array(hog_train_4),axis=0)
mean_features[4,:] = np.reshape(mean_4,3780)

mean_5 = np.mean(np.array(hog_train_5),axis=0)
mean_features[5,:] = np.reshape(mean_5,3780)

mean_6 = np.mean(np.array(hog_train_6),axis=0)
mean_features[6,:] = np.reshape(mean_6,3780)

mean_7 = np.mean(np.array(hog_train_7),axis=0)
mean_features[7,:] = np.reshape(mean_7,3780)

mean_8 = np.mean(np.array(hog_train_8),axis=0)
mean_features[8,:] = np.reshape(mean_8,3780)

mean_9 = np.mean(np.array(hog_train_9),axis=0)
mean_features[9,:] = np.reshape(mean_9,3780)

np.zeros(shape=(9,1),dtype=float)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import manhattan_distances

from math import sqrt

diff = np.zeros(shape=(10, 10),dtype=float)

for i in range(0,10):
    for j in range(0,10):
        sim = (manhattan_distances(mean_features[i],mean_features[j]))
        diff[i,j] = sim

for i in range(10):
    print("For: ",i)
    L = np.ndarray.tolist(diff[i,:])
    print("Indicies:",sorted(range(len(L)), key=lambda i: L[i]))
