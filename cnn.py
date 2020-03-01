
# making a CNN classifier
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


# Initialising the CNN
classifier = Sequential()

# Convolutional layer 1
classifier.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,input_shape=(200,200,3),activation='relu'))
classifier.add(Dropout(0.4))

# Pooling layer 1
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Convolutional layer 2
classifier.add(Conv2D(filters=64,kernel_size=(3,3),strides=1,activation='relu'))
classifier.add(Dropout(0.4))

# Pooling layer 2
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Convolutional layer 3
classifier.add(Conv2D(filters=64,kernel_size=(2,2),strides=1,activation='relu'))
classifier.add(Dropout(0.4))

# Pooling layer 3
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

# compiling model
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# importing dataset
import glob
import cv2
import numpy as np

data = []
data_labels = []

files = glob.glob("C:/Users/Skanda/Documents/Machine learning/IndianCurrencyDetection/currency_dataset/dataset/ten/*.jpg")
for file in files:
    
    image = cv2.imread(file)
    
    image = cv2.resize(image,(200,200))
    
    data.append(image)
    
    data_labels.append(0.0)     # 0 for 10


files = glob.glob("C:/Users/Skanda/Documents/Machine learning/IndianCurrencyDetection/currency_dataset/dataset/twenty/*.jpg")
for file in files:
    
    image = cv2.imread(file)
    
    image = cv2.resize(image,(200,200))
    
    data.append(image)
    
    data_labels.append(1.0)     # 0 for 10



data = np.array(data)
data_labels = np.array(data_labels)



# splitting training and test datasets
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(data,data_labels,test_size=0.2,random_state=2)

X_train = X_train/255.0


# training CNN
classifier.fit(X_train,y_train,epochs=200)


# prediction
X_test = X_test/255.0

y_pred = classifier.predict(X_test)

y_pred[y_pred >= 0.5] = 1.0
y_pred[y_pred < 0.5] = 0.0


# creating confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

# accuracy
accuracy = cm.diagonal().sum()/cm.sum() * 100
