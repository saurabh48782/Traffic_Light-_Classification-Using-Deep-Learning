# Importing important libraries
import pickle
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
from matplotlib import pyplot as plt

# Making a Convolutional Neural Network
def CNNetwork():
    
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    # Setting activation function as relu
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(4))
    # Applying softmax activation function
    model.add(Activation('softmax'))
    
    return model


def train_model(Training_data, model):
    # Reading my Training dataset
    x_,y_ = pickle.load( open(Training_data, "rb" ) )
    # Training dataset has 9259 samples
    print(x_.shape,y_.shape)
    
    random_state = 130
    # Dividing my dataset for training and validation using train_test_split method from sklearn library
    # 80% dataset is used for training and remaining 20% for validation purpose
    X_train, x_validation, Y_train, y_validation = train_test_split(x_, y_, train_size = 0.80,
                                                                    test_size = 0.2,
                                                                    random_state = random_state)
    # Preprocessing of data
    X_normalized = np.array(X_train / 255.0 - 0.5 )
    
    # LabelBinarizer() method from sklearn.preprocessing module binarizes our labels using One-vs-All approach
    binarizered_label = LabelBinarizer()
    
    # fit_transfrom() method transforms multi-class labels into binary labels
    # Y_labels is a one-hot-encoded training labels
    Y_train_labels = binarizered_label.fit_transform(Y_train)
    
    # model.summary() prints a string summary of the network.
    model.summary()

    ''' model.compile() method configures the  model for training purpose
    It takes in three parameters:
    - adam is a  type of optimizer which is a stochastic gradient descent method
    - categorical_crossentropy is a loss function which computes the loss between labels and predictions.
    - third parameter is a list of metrics to be evaluated by the model
    '''
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    
    # model.fit() trains the model for a fixed number of epochs (iterations on a dataset).
    history = model.fit(X_normalized, Y_train_labels, nb_epoch=20, validation_split=0.2)
    
    # summarizing history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarizing history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    # Saving my model in 'model.h5' file
    model.save('my_model.h5')
    return history



def test_model(file_path, model):
    
    # Reading my testing dataset
    X_test,Y_test = pickle.load( open(file_path, "rb" ) )
    
    # There are 1030 samples in test dataset
    print(X_test.shape,Y_test.shape)
    
    ## Preprocessing of testing data
    
    X_test_standard = np.array(X_test / 255.0 - 0.5 )
    
    # LabelBinarizer() method from sklearn.preprocessing module binarizes our labels using One-vs-All approach
    binarizered_label = LabelBinarizer()
    # binarizered_label.fit_transform() method fits label binarizer and transforms multi-class labels to binary labels.
    Y_test_labels = binarizered_label.fit_transform(Y_test)

    print("\n\n----------Testing my Model----------------")
    
    # model.evaluate() method returns the loss value & metrics values for the model in testing phase.
    metrics = model.evaluate(X_test_standard, Y_test_labels)
    for i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[i]
        metric_value = metrics[i]
        # Prints metrics like loss and accuraxy
        print('{}: {}'.format(metric_name, metric_value))


def test_sample_img(file_path, model):
    
    # Firstly, resizing the input image to [32,32,3] shape and then feeding it into neural network

    desired_dim=(32,32)
    img = cv2.imread(file_path)
    resized_img = cv2.resize(img, desired_dim, interpolation=cv2.INTER_LINEAR)
    img_ = np.expand_dims(np.array(resized_img), axis=0)
    
    # Predicting the class of the input sample image
    predicted_state = model.predict_classes(img_)

    return predicted_state

# Driver Function
if __name__ == "__main__":
    model = CNNetwork()
    Training_set = "./datasets/Train_Dataset.p"
    Testing_set = "./datasets/Test_Dataset.p"

    # Training my CNN Model
    train_model(Training_set, model)

    # Testing my CNN Model
    test_model(Testing_set, model=load_model('my_model.h5'))

    # Testing a single image (any random traffic light image) after training our dataset
    flag = True
    file_path = './datasets/yellow.jpg'
    states = ['red', 'yellow', 'green', 'off']
    if flag:
        predicted_state = test_sample_img(file_path, model=load_model('my_model.h5'))
        for idx in predicted_state:
            print("Colour inferred from the sample image is: ", states[idx])



