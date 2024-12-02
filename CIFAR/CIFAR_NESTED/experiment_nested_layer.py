import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization

from datetime import datetime
import numpy as np
import os
import argparse

import custom_components.custom_callbacks as custom_callbacks
import custom_components.custom_layers as custom_layers

import utils.plot_scripts as plot_scripts
import utils.logs as logs


def prepare_model_dir(scenario_dir, model, penalty_rate, learning_rate, run_timestamp):
    log_dir = f'logs/{scenario_dir}/{run_timestamp}_lr_{learning_rate}_pr_{penalty_rate}'
    os.makedirs(log_dir)
    logs.log_model_structure(model, log_dir, filename="model_structure.txt")
    return log_dir


def initialize_callbacks(model, log_dir):
    callbacks = []

    for i, layer in enumerate(model.layers):
        if "custom_conv2d" in layer.name:
            #replace accordingly with new callback
            continue
#            scale_tracking_callback = custom_callbacks.NestedScaleTrackingCallback(layer, log_dir)
#            callbacks.append(scale_tracking_callback)

    accuracy_loss_callback = custom_callbacks.AccuracyLossTrackingCallBack(log_dir)
    callbacks.append(accuracy_loss_callback)
    return callbacks


def initialize_model(penalty_rate, input_shape = (32,32,3), seed=42):
    input_layer = Input(shape=input_shape)

    conv_layer_1 = custom_layers.CustomConv2DLayer(filters=32, kernel_size=(3, 3))
    conv_layer_1_output = conv_layer_1(input_layer)
    conv_layer_1_activation = tf.keras.activations.relu(conv_layer_1_output)
    
    conv_layer_1_activation_batch_normalized = BatchNormalization()(conv_layer_1_activation)
    pooling_layer_1 = MaxPooling2D(pool_size=(2,2))(conv_layer_1_activation_batch_normalized)

    conv_layer_2 = custom_layers.CustomConv2DLayer(filters=64, kernel_size=(3, 3))
    conv_layer_2_output = conv_layer_2(pooling_layer_1)
    conv_layer_2_activation = tf.keras.activations.relu(conv_layer_2_output)

    conv_layer_2_activation_batch_normalized = BatchNormalization()(conv_layer_2_activation)
    pooling_layer_2 = MaxPooling2D(pool_size=(2,2))(conv_layer_2_activation_batch_normalized)

    conv_layer_3 = custom_layers.CustomConv2DLayer(filters=128, kernel_size=(3, 3))
    conv_layer_3_output = conv_layer_3(pooling_layer_2)
    conv_layer_3_activation = tf.keras.activations.relu(conv_layer_3_output)

    conv_layer_3_activation_batch_normalized = BatchNormalization()(conv_layer_3_activation)
    pooling_layer_3 = MaxPooling2D(pool_size=(2,2))(conv_layer_3_activation_batch_normalized)

    flatten_layer = Flatten()(pooling_layer_3)
    
    dense_layer_1 = Dense(1024)
    dense_layer_1_output = dense_layer_1(flatten_layer)
    dense_layer_1_activation = tf.keras.activations.relu(dense_layer_1_output)

    dense_layer_2 = Dense(10)
    dense_layer_2_output = dense_layer_2(dense_layer_1_activation)
    dense_layer_2_activation = tf.keras.activations.softmax(dense_layer_2_output)

    quantized_model = Model(inputs=input_layer, outputs=dense_layer_2_activation)

    return quantized_model    


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )


def train_model(model, epochs, train_data, validation_data, callbacks, batch_size=32):
    (x_train, y_train) = train_data
    model.fit(
        x=x_train, 
        y=y_train,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,  
        batch_size=batch_size
    )


def evaluate_model(model, validation_data):
    (x_test, y_test) = validation_data
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Model Test Accuracy: {accuracy}\nModel Test Loss: {loss}')
    return accuracy, loss


def main():
    parser = argparse.ArgumentParser(description="Train model for differen penalty rates for this scenarion.")
    parser.add_argument(
        "scenario_dir", 
        type=str, 
        help="The scenario directory containing different train configs (e.g., 'with_maxbin_threshold_3')."
    )
    args = parser.parse_args()

    scenario_dir = args.scenario_dir


    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    train_data = (x_train, y_train)
    validation_data = (x_test, y_test)
    input_data_shape = (32,32,3)

    learning_rate = 0.001
    epochs = 20
    batch_size = 64

    penalty_rates = [
        0.0,
    ]
    seed = 42

    penalty_rate = 0.0

    for penalty_rate in penalty_rates:

        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        model = initialize_model(penalty_rate=penalty_rate, input_shape=input_data_shape, seed=seed)
        log_dir = prepare_model_dir(scenario_dir=scenario_dir, model=model, penalty_rate=penalty_rate, learning_rate=learning_rate, run_timestamp=run_timestamp)
        compile_model(model=model, learning_rate=learning_rate)

        callbacks = initialize_callbacks(model=model, log_dir=log_dir)
        train_model(model=model, epochs=epochs, train_data=train_data, validation_data=validation_data, batch_size=batch_size, callbacks=callbacks)

        accuracy, loss = evaluate_model(model=model, validation_data=validation_data)


if __name__ == "__main__":
    main()