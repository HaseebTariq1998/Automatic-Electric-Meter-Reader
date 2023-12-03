
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

import sys

np.set_printoptions(threshold=sys.maxsize)

DATADIR = "Digits_Dataset_Preprocessed_Augmented/"
CATEGORIES = ["zero","one","two","three","four","five","six","seven","eight","nine","th"]
dic={"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"th":10}
encoding=[[1,0,0,0,0,0,0,0,0,0,0] ,[0,1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0],
[0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,1]]
'''
for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!
        IMG_SIZE = 130

        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        plt.imshow(new_array, cmap='gray')
        plt.show()
        break

          # we just want one for now so break

'''


IMG_SIZE = 50
training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, dic[category]])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()


import random

random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
Y = []

for features,label in training_data:
    X.append(features)
    Y.append(encoding[label])



X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)



print(training_data[0][1])
print(Y)

import pickle

pickle_out = open("X_multiclass.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_multiclass.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

