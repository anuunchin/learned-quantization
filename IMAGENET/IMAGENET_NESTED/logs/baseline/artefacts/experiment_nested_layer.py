import argparse
import concurrent.futures
import logging
import os
import shutil
from datetime import datetime
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Activation,
    Add,
    GlobalAveragePooling2D
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import custom_components.custom_callbacks as custom_callbacks
import custom_components.custom_layers as custom_layers
from utils import logs
from utils import plot_scripts

seed = 42
tf.random.set_seed(seed)

os.environ["TF_DETERMINISTIC_OPS"] = "1"


print("\nGPUs Available:", tf.config.list_physical_devices("GPU"))
build_info = tf.sysconfig.get_build_info()
print("CUDA Version:", build_info["cuda_version"])
print("cuDNN Version:", build_info["cudnn_version"])


def prepare_model_dir(scenario_dir, model, penalty_rate, learning_rate, run_timestamp):
    log_dir = (
        f"logs/{scenario_dir}/{run_timestamp}_lr_{learning_rate}_pr_{penalty_rate}"
    )
    os.makedirs(log_dir)
    logs.log_model_structure(model, log_dir, filename="model_structure.txt")
    return log_dir


def initialize_callbacks(model, log_dir):
    callbacks = []

    for i, layer in enumerate(model.layers):
        if "custom_conv2d_layer_test" in layer.name:
            scale_tracking_callback = custom_callbacks.NestedScaleTrackingCallback(
                layer, log_dir
            )
            callbacks.append(scale_tracking_callback)

    accuracy_loss_callback = custom_callbacks.AccuracyLossTrackingCallBack(log_dir)
    callbacks.append(accuracy_loss_callback)
    return callbacks


def residual_block(x, filters, kernel_size=(3, 3), stride=(1, 1)):
    """A simple Residual Block"""
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)  

    x = Conv2D(filters, kernel_size, strides=(1, 1), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)  # Add activation here
    x = Dropout(0.2)(x)  # Dropout after second ReLU

    # Add shortcut for residual connection
    if stride != (1, 1):
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding="same", kernel_initializer="he_normal")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Dropout(0.2)(x) 
    x = Activation("relu")(x)
    return x

def build_imagenette_model(penalty_rate, input_shape=(224, 224, 3), seed=42, num_classes=10):
    """A ResNet-like model for Imagenette"""
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", kernel_initializer="random_normal")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = Dropout(0.2)(x) 

    # Residual Blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=(2, 2))
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=(2, 2))
    x = residual_block(x, 256)
    
    x = residual_block(x, 512, stride=(2, 2))
    x = residual_block(x, 512)
    
    # Global Pooling and Dense Layers
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def initialize_model(penalty_rate, input_shape=(32, 32, 3), seed=42, num_classes=10):
    # Input Layer
    input_layer = Input(shape=input_shape)

    # First Block
    conv_layer_1_a = custom_layers.CustomConv2DLayerTest(
        penalty_rate=penalty_rate,
        seed=seed,
        filters=32,
        kernel_size=(3, 3),
        kernel_initializer="random_normal",
    )
    conv_layer_1_a_output = conv_layer_1_a(input_layer)
    conv_layer_1_a_activation = tf.keras.activations.relu(conv_layer_1_a_output)
    batch_norm_1_a = BatchNormalization()(conv_layer_1_a_activation)

    conv_layer_1_b = custom_layers.CustomConv2DLayer(
        penalty_rate=penalty_rate,
        seed=seed,
        filters=32,
        kernel_size=(3, 3),
        kernel_initializer="random_normal",
    )
    conv_layer_1_b_output = conv_layer_1_b(batch_norm_1_a)
    conv_layer_1_b_activation = tf.keras.activations.relu(conv_layer_1_b_output)
    batch_norm_1_b = BatchNormalization()(conv_layer_1_b_activation)

    pool_layer_1 = MaxPooling2D(pool_size=(2, 2))(batch_norm_1_b)
    drop_layer_1 = Dropout(0.2, seed=seed)(pool_layer_1)

    # Second Block
    conv_layer_2_a = custom_layers.CustomConv2DLayer(
        penalty_rate=penalty_rate,
        seed=seed,
        filters=64,
        kernel_size=(3, 3),
        kernel_initializer="random_normal",
    )
    conv_layer_2_a_output = conv_layer_2_a(drop_layer_1)
    conv_layer_2_a_activation = tf.keras.activations.relu(conv_layer_2_a_output)
    batch_norm_2_a = BatchNormalization()(conv_layer_2_a_activation)

    conv_layer_2_b = custom_layers.CustomConv2DLayer(
        penalty_rate=penalty_rate,
        seed=seed,
        filters=64,
        kernel_size=(3, 3),
        kernel_initializer="random_normal",
    )
    conv_layer_2_b_output = conv_layer_2_b(batch_norm_2_a)
    conv_layer_2_b_activation = tf.keras.activations.relu(conv_layer_2_b_output)
    batch_norm_2_b = BatchNormalization()(conv_layer_2_b_activation)

    pool_layer_2 = MaxPooling2D(pool_size=(2, 2))(batch_norm_2_b)
    drop_layer_2 = Dropout(0.3, seed=seed)(pool_layer_2)

    # Third Block
    conv_layer_3_a = custom_layers.CustomConv2DLayer(
        penalty_rate=penalty_rate,
        seed=seed,
        filters=128,
        kernel_size=(3, 3),
        kernel_initializer="random_normal",
    )
    conv_layer_3_a_output = conv_layer_3_a(drop_layer_2)
    conv_layer_3_a_activation = tf.keras.activations.relu(conv_layer_3_a_output)
    batch_norm_3_a = BatchNormalization()(conv_layer_3_a_activation)

    conv_layer_3_b = custom_layers.CustomConv2DLayer(
        penalty_rate=penalty_rate,
        seed=seed,
        filters=128,
        kernel_size=(3, 3),
        kernel_initializer="random_normal",
    )
    conv_layer_3_b_output = conv_layer_3_b(batch_norm_3_a)
    conv_layer_3_b_activation = tf.keras.activations.relu(conv_layer_3_b_output)
    batch_norm_3_b = BatchNormalization()(conv_layer_3_b_activation)

    pool_layer_3 = MaxPooling2D(pool_size=(2, 2))(batch_norm_3_b)
    drop_layer_3 = Dropout(0.4, seed=seed)(pool_layer_3)

    # Flatten and Dense Layers
    flatten_layer = Flatten()(drop_layer_3)
    dense_layer_1 = Dense(
        128, activation="relu", kernel_initializer=RandomNormal(seed=seed)
    )(flatten_layer)
    batch_norm_dense = BatchNormalization()(dense_layer_1)
    drop_layer_dense = Dropout(0.5, seed=seed)(batch_norm_dense)
    output_layer = Dense(
        10, activation="softmax", kernel_initializer=RandomNormal(seed=seed)
    )(drop_layer_dense)

    # Create Model
    quantized_model = Model(inputs=input_layer, outputs=output_layer)

    return quantized_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_model(model, epochs, train_data, validation_data, callbacks, batch_size=32):
    (x_train, y_train) = train_data
    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        batch_size=batch_size,
    )


def evaluate_model(model, validation_data):
    (x_test, y_test) = validation_data
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Model Test Accuracy: {accuracy}\nModel Test Loss: {loss}")
    return accuracy, loss


def plot_results(log_dir_path):
    logging.info(f"Processing {log_dir_path}...")

    plot_scripts.plot_values_logged_on_epoch_end(log_dir_path)
    plot_scripts.plot_accuracy_per_epoch(log_dir_path)
    plot_scripts.plot_total_loss_per_epoch(log_dir_path)


def plot_pareto(log_scenario_dir):
    logging.info(f"Pareto plots...")

    plot_scripts.pareto_unique_values_accuracy(log_scenario_dir)
    plot_scripts.pareto_unique_values_loss(log_scenario_dir)
    plot_scripts.pareto_range_accuracy(log_scenario_dir)
    plot_scripts.pareto_range_loss(log_scenario_dir)

    logging.info(f"Pareto plots done.")


def save_artefacts(log_scenario_dir):
    current_script_dir = os.path.dirname(__file__)

    custom_components_dir = os.path.join(current_script_dir, "custom_components")

    destination_dir = os.path.join(current_script_dir, log_scenario_dir, "artefacts")
    os.makedirs(destination_dir, exist_ok=True)

    python_scripts = [
        os.path.join(custom_components_dir, python_script)
        for python_script in os.listdir(custom_components_dir)
        if ".py" in python_script
    ]
    python_scripts += [__file__]
    for script in python_scripts:
        shutil.copy(script, destination_dir)


def prepare_data():
    def preprocess(image, label):
        image = tf.image.resize(image, (224, 224))  # Resize images to 224x224
        return image, label

    train_data = tfds.load('imagenette', split='train', as_supervised=True)
    validation_data = tfds.load('imagenette', split='validation', as_supervised=True)

    train_data = train_data.map(preprocess)
    validation_data = validation_data.map(preprocess)

    train_data_numpy = tfds.as_numpy(train_data)
    validation_data_numpy = tfds.as_numpy(validation_data)

    x_train, y_train = [], []
    for image, label in train_data_numpy:
        x_train.append(image)
        y_train.append(label)

    x_test, y_test = [], []
    for image, label in validation_data_numpy:
        x_test.append(image)
        y_test.append(label)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    input_data_shape = x_train[0].shape

    return (x_train, y_train), (x_test, y_test), input_data_shape


def main():
    # Parse scenario_dir argument
    parser = argparse.ArgumentParser(
        description="Train model for different penalty rates for this scenario."
    )
    parser.add_argument(
        "scenario_dir",
        type=str,
        help="The scenario directory containing different train configs (e.g., 'with_maxbin_threshold_3').",
    )
    args = parser.parse_args()
    scenario_dir = args.scenario_dir

    # Prepare scenario_dir and temp dir within it - temp dir contains temporary logs that are overwritten for each training
    os.makedirs("logs/" + scenario_dir, exist_ok=True)

    # Setup logger for plotting
    plotting_log_file_path = "logs/" + scenario_dir + "/plotting_process.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(plotting_log_file_path),
        ],
    )

    train_data, validation_data, input_data_shape = prepare_data()

    learning_rate = 0.001
    epochs = 100
    batch_size = 32

    penalty_rates = [
        0.0
        #        1e-12,
        #        1e-12,
#        1e-11,
#        1e-10,
#        1e-9,
#        1e-8,
#        1e-7,
#        1e-6,
#        1e-5,
#        1e-4,
#        1e-3,
#        1e-2,
#        1e-1,
#        1.0,
#        1e1,
#        1e2,
#        1e3,
#        1e4,
#        1e5,
#        1e6,
#        1e7,
#        1e8,
#        1e9,
#        1e10,
#        1e11,
#        1e12,
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for penalty_rate in penalty_rates:
            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model = build_imagenette_model(
                penalty_rate=penalty_rate, input_shape=input_data_shape, seed=seed
            )
            log_dir = prepare_model_dir(
                scenario_dir=scenario_dir,
                model=model,
                penalty_rate=penalty_rate,
                learning_rate=learning_rate,
                run_timestamp=run_timestamp,
            )
            compile_model(model=model, learning_rate=learning_rate)

            callbacks = initialize_callbacks(model=model, log_dir=log_dir)
            train_model(
                model=model,
                epochs=epochs,
                train_data=train_data,
                validation_data=validation_data,
                batch_size=batch_size,
                callbacks=callbacks,
            )

            accuracy, loss = evaluate_model(
                model=model, validation_data=validation_data
            )

            futures.append(executor.submit(plot_results, log_dir))

        concurrent.futures.wait(futures)

    log_scenario_dir = "logs/" + scenario_dir
    plot_pareto(log_scenario_dir)
    save_artefacts(log_scenario_dir)


if __name__ == "__main__":
    main()
