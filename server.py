from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import keras.backend as K
from PIL import Image
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU
from sklearn.model_selection import train_test_split


INPUT_SHAPE = 224, 224, 3

def build_model(input_shape):
    # model = tf.keras.applications.EfficientNetB3((input_shape), classes=5, weights=None)
    
    model = squeezenet(input_shape, 5)
    # model = mobilenet(input_shape, 5)s
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    # model.set_weights([np.random.uniform(0,1, i.shape) for i in model.get_weights()])
    
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


def densenet(img_shape, n_classes, f=32):
  repetitions = 6, 12, 24, 16
  
  def bn_rl_conv(x, f, k=1, s=1, p='same'):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(f, k, strides=s, padding=p)(x)
    return x
  
  
  def dense_block(tensor, r):
    for _ in range(r):
      x = bn_rl_conv(tensor, 4*f)
      x = bn_rl_conv(x, f, 3)
      tensor = Concatenate()([tensor, x])
    return tensor
  
  
  def transition_block(x):
    x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
    x = AvgPool2D(2, strides=2, padding='same')(x)
    return x
  
  
  input = Input(img_shape)
  
  x = Conv2D(64, 7, strides=2, padding='same')(input)
  x = MaxPool2D(3, strides=2, padding='same')(x)
  
  for r in repetitions:
    d = dense_block(x, r)
    x = transition_block(d)
  
  x = GlobalAvgPool2D()(d)
  
  output = Dense(n_classes, activation='softmax')(x)
  
  model = Model(input, output)
  return model

def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    # im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    # model = tf.keras.applications.EfficientNetB0(
    #     input_shape=(32, 32, 3), weights=None, classes=10
    # )
    # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    model = build_model(INPUT_SHAPE)
    #model = initial_fit(model)
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5, #0.3
        fraction_eval=0.3, #0.2
        min_fit_clients=5,
        min_eval_clients=3,
        min_available_clients=5, #10
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("localhost:8086", config={"num_rounds": 100}, strategy=strategy)

# 
    
def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    train_df = pd.read_csv('train-backup.csv')
    
    (x,y) = load_partition(train_df)
    
    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights): 
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x, y)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 5 if rnd < 2 else 5,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


def load_partition(train_df):
    idx = 1
    """Load 1/10th of the training and test data to simulate a partition."""
    N = train_df.shape[0]
    train_df = train_df[-162:]
    N = train_df.shape[0]
    x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

    # train_partition = train_df[N * idx : N * (idx+1)] 
    
    
    for i, image_id in enumerate(train_df['id_code']):
        # image_id = train_partition.loc[i,'id_code']
        # print(image_id)
        # img = cv2.imread(f'train_images_resized/{image_id}.png')
        # x_train.append(img)
        
        x_train[i, :, :, :] = preprocess_image(
            f'train_images_resized/{image_id}.png'
        )
    
    # x_train_cl = np.array(x_train, np.float32) / 255.0
    
    y = train_df['diagnosis']
    # y_train = y
    y_train = tf.keras.utils.to_categorical(y, num_classes=5)
    # print(y_train.shape)
    
    # X_train, x_val, Y_train, y_val = train_test_split(
    #     x_train, y_train, 
    #     test_size=0.30, 
    #     random_state=42
    # )
    
    return (x_train, y_train)
    

if __name__ == "__main__":
    main()


# def initial_fit(model):
#     train_df = pd.read_csv('train.csv')
    
#     (x,y), (x_val, y_val) = load_partition(train_df)
    
#     model.fit(x, y)
    
#     return model