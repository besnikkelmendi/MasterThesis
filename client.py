import argparse
import os


import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from PIL import Image
import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU
from sklearn.model_selection import train_test_split

import flwr as fl

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

INPUT_SHAPE = 224, 224, 3

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    parser.add_argument("--clients", type=int, choices=range(0, 11), required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    # model = tf.keras.applications.EfficientNetB0(
    #     input_shape=(32, 32, 3), weights=None, classes=10
    # )
    
    model = build_model(INPUT_SHAPE)
    # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load a subset of CIFAR-10 to simulate the local data partition
    train_df = pd.read_csv('train-backup.csv')
    test_df = pd.read_csv('test.csv')
    
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition, args.clients, train_df)

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("localhost:8080", client=client)

def build_model(input_shape):
    model = tf.keras.applications.DenseNet121((input_shape), classes=5, weights=None)
    
    # model = squeezenet(input_shape, 5)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def squeezenet(input_shape, n_classes):
  
    def fire(x, fs, fe):
        s = Conv2D(fs, 1, activation='relu')(x)
        e1 = Conv2D(fe, 1, activation='relu')(s)
        e3 = Conv2D(fe, 3, padding='same', activation='relu')(s)
        output = Concatenate()([e1, e3])
        return output
  
  
    input = Input(input_shape)
  
    x = Conv2D(96, 7, strides=2, padding='same', activation='relu')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)
  
    x = fire(x, 16, 64)
    x = fire(x, 16, 64)
    x = fire(x, 32, 128)
    x = MaxPool2D(3, strides=2, padding='same')(x)
  
    x = fire(x, 32, 128)
    x = fire(x, 48, 192)
    x = fire(x, 48, 192)
    x = fire(x, 64, 256)
    x = MaxPool2D(3, strides=2, padding='same')(x)
  
    x = fire(x, 64, 256)
    x = Conv2D(n_classes, 1)(x)
    x = GlobalAvgPool2D()(x)
  
    output = Activation('softmax')(x)
  
    model = Model(input, output)
    return model  

def load_partition(idx: int, clients ,train_df):
    """Load 1/10th of the training and test data to simulate a partition."""
    N = train_df.shape[0]
    N = int(N/clients)
    # x_train = []
    x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)
    
    train_partition = train_df[N * idx : N * (idx+1)] 
    for i, image_id in enumerate(train_partition['id_code']):
        # image_id = train_partition.loc[i,'id_code']
        # print(image_id)
        # img = cv2.imread(f'train_images_resized/{image_id}.png')
        # x_train.append(img)
        x_train[i, :, :, :] = preprocess_image(
            f'train_images_resized/{image_id}.png'
        )
    
    # x_train_cl = np.array(x_train, np.float32) / 255.0
        
    
    
    y = train_partition['diagnosis']
    # y_train = y
    y_train = tf.keras.utils.to_categorical(y, num_classes=5)
    # print(y_train.shape)
    
    X_train, x_val, Y_train, y_val = train_test_split(
        x_train, y_train, 
        test_size=0.15, 
        random_state=42
    )
    
    return (X_train, Y_train),(x_val, y_val)
    # assert idx in range(10)
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # return (
    #     x_train[idx * 5000 : (idx + 1) * 5000],
    #     y_train[idx * 5000 : (idx + 1) * 5000],
    # ), (
    #     x_test[idx * 1000 : (idx + 1) * 1000],
    #     y_test[idx * 1000 : (idx + 1) * 1000],
    # )

def mobilenet(input_shape, n_classes):
  
  def mobilenet_block(x, f, s=1):
    x = DepthwiseConv2D(3, strides=s, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(f, 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x
    
    
  input = Input(input_shape)

  x = Conv2D(32, 3, strides=2, padding='same')(input)
  x = BatchNormalization()(x)
  x = ReLU()(x)

  x = mobilenet_block(x, 64)
  x = mobilenet_block(x, 128, 2)
  x = mobilenet_block(x, 128)

  x = mobilenet_block(x, 256, 2)
  x = mobilenet_block(x, 256)

  x = mobilenet_block(x, 512, 2)
  for _ in range(5):
    x = mobilenet_block(x, 512)

  x = mobilenet_block(x, 1024, 2)
  x = mobilenet_block(x, 1024)
  
  x = GlobalAvgPool2D()(x)
  
  output = Dense(n_classes, activation='softmax')(x)
  
  model = Model(input, output)
  return model

def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    # im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im

if __name__ == "__main__":
    main()
