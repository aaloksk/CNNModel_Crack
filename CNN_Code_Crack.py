# -*- coding: utf-8 -*-
"""
Created on Wed May  3 20:46:33 2023

@author: Aalok
"""

#Importing necessary libraries
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


#Working directory to the path with ::
#Özgenel, Çağlar Fırat (2019), “Concrete Crack Images for Classification”, Mendeley Data, V2, doi: 10.17632/5y9wdsg2zt.2
#cracked_folder = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\CNN\\Image\\archive(4)\\1'
#uncracked_folder = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\CNN\\Image\\archive(4)\\0'




# Define a function to load images and labels from a folder and append images with 90,270 and 180 degree rotation
def load_images(folder, label):
    images = []
    for root, dirs, files in os.walk(folder): #Looking into the folder and all of its subfolders for images
        for filename in files:
            img = cv2.imread(os.path.join(root, filename))
            img = cv2.resize(img, (img_size, img_size))
            if img is not None:
                
                # Rotate the image by 90 degrees, 180 degrees, and 270 degrees
                img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                img_180 = cv2.rotate(img, cv2.ROTATE_180)
                img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Append all four images to the list
                images.append((img, label))
                images.append((img_90, label))
                images.append((img_180, label))
                images.append((img_270, label))
            
    return images

# Define a function to load images and labels from a folder and append images with 180 degree rotation
def load_images2(folder, label):
    images = []
    for root, dirs, files in os.walk(folder): #Looking into the folder and all of its subfolders for images
        for filename in files:
            img = cv2.imread(os.path.join(root, filename)) #Reading the image
            img = cv2.resize(img, (img_size, img_size)) #Maintaining the size of image
            if img is not None:
                
                # Rotate the image by  180 degrees, 
                img_180 = cv2.rotate(img, cv2.ROTATE_180)
                
                # Append images to the list
                images.append((img, label))
                images.append((img_180, label))
                
            
    return images


# set the path to the folders containing the images around beaumont
cracked_folder = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\CNN\\Image\\All\\1'
uncracked_folder = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\CNN\\Image\\All\\0'


#Defining image size
img_size = 256

# Load the images and labels from the folders
cracked_images = load_images(cracked_folder, 1) #Prprocess the crracked images and rotate them by 90, 180 and 270 degree
uncracked_images = load_images2(uncracked_folder, 0) #Only rotate by 180 degree for uncracked image #Helps also in matching the number of cracked and uncracked images


# Combine the images and labels and shuffle them
images = cracked_images + uncracked_images


# Split the data into training and testing sets by shuffling the order
train_images, test_images, train_labels, test_labels = train_test_split(
    [image[0] for image in images], [image[1] for image in images],
    test_size=0.2, random_state=42, shuffle=True)

# Convert the data to NumPy arrays
train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Display some of the images from both classes
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.flatten()

#fIVE PLOTS FOR EACH CLASS
for i in range(5):
    axs[i].imshow(train_images[train_labels==0][i]) #Label for 0
    axs[i].set_title('Uncracked Concrete')
    axs[i+5].imshow(train_images[train_labels==1][i]) #label for 1
    axs[i+5].set_title('Cracked Concrete')

plt.show() #Final plot


#Initialize the model
model = Sequential()

#First layer is a convolutional layer 
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(256, 256, 3))) #16 filtyers of size 3X3 and relu AF
model.add(MaxPooling2D(pool_size = (2, 2))) #MaxPool
model.add(Dropout(.3)) #30% dropout

model.add(Conv2D(32, (3, 3), activation = "relu")) #Second layer is a convolutional 
model.add(MaxPooling2D(pool_size = (2, 2))) #MaxPool
model.add(Dropout(.3)) #Dropout

model.add(Conv2D(32, (3, 3), activation = "relu")) #THird layer is a convolutional 
model.add(MaxPooling2D(pool_size = (2, 2))) #MaxPool
model.add(Dropout(.3)) #Dropout

model.add(Flatten()) #Flatten as vector
model.add(Dense(258, activation = "relu")) #Fully connected layer with relu

model.add(Dense(1, activation = "sigmoid")) #Output probability layer with sigmoidal

#Model Compilation 
#Binary cross-entropy loss 
##Adam optimizer
#Evaluation metric - Accuracy
print("Compiling the model...")
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
print("Model successfully compiled!!")



#Fitting the model
print("Fitting the model...")
model.fit(train_images, train_labels, batch_size = 64, epochs = 10)
print("Model successfully fitted!!")


#Testing the model with test data
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


#Plotting the ROC Curve

# Get the predicted probabilities for the test set
y_scores = model.predict(test_images)

# Compute the false positive rate (fpr), true positive rate (tpr), and threshold values
fpr, tpr, thresholds = roc_curve(test_labels, y_scores)

# Compute the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


#Looking at the confusion matrix


# Get predicted labels on the training set
train_pred = model.predict(train_images)

# Get predicted labels on the test set
test_pred = model.predict(test_images)

# Compute the confusion matrix for the training set
train_cm = confusion_matrix(train_labels, train_pred.round())

# Compute the confusion matrix for the test set
test_cm = confusion_matrix(test_labels, test_pred.round())

# Print the confusion matrices
print("Confusion matrix for training set:")
print(train_cm)
print("Confusion matrix for test set:")
print(test_cm)


# Define a function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot the confusion matrix for the training set
plt.figure(figsize=(5,5))
plot_confusion_matrix(train_cm, classes=['Uncracked', 'Cracked'], title='Training Confusion Matrix')
plt.show()

# Plot the confusion matrix for the testing set
plt.figure(figsize=(5,5))
plot_confusion_matrix(test_cm, classes=['Uncracked', 'Cracked'], title='Testing Confusion Matrix')
plt.show()


# Set the path to the folder containing the test images
test_folder = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\CNN\\Image\\All\\2'

# Load the test images and resize them to the same size used for training
test_images = []
for filename in os.listdir(test_folder):
    img = cv2.imread(os.path.join(test_folder,filename))
    img = cv2.resize(img, (img_size, img_size))
    test_images.append(img)

# Convert the test images to a NumPy array and normalize the pixel values
test_images = np.array(test_images) / 255.0

# Use the model to predict the class probabilities of the test images
predictions = model.predict(test_images)

# Print the predicted class probabilities
for i, pred in enumerate(predictions):
    print(f"Image {i+1}: {pred[0]}")
    



# Set the path to the folder containing the test images never seen by the model
test_folder = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\CNN\\Image\\All\\2'

# Load the test images and resize them to the same size used for training
test_images = []
for filename in os.listdir(test_folder):
    img = cv2.imread(os.path.join(test_folder,filename))
    img = cv2.resize(img, (img_size, img_size))
    test_images.append(img)

# Convert the test images to a NumPy array and normalize the pixel values
test_images = np.array(test_images) 

# Use the model to predict the class probabilities of the test images
predictions = model.predict(test_images)

# Show the test images and predicted class probabilities
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
for i, (img, pred) in enumerate(zip(test_images, predictions)):
    axs[i].imshow(img)
    axs[i].axis('off')
    if pred[0] < 0.5:
        axs[i].set_title('No Cracks: {:.2f}'.format(pred[0]))
    else:
        axs[i].set_title('Crack Detected!: {:.2f}'.format(pred[0]), color='red')

plt.show()