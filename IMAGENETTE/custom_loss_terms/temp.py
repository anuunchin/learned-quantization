import argparse
import concurrent.futures
import logging
import random
import os
import shutil
from datetime import datetime
import numpy as np
from typing import List, Literal, Tuple, Any

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

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import RandomNormal, HeNormal
from tensorflow.keras import regularizers

import custom_components.custom_callbacks as custom_callbacks
import custom_components.custom_layers as custom_layers
from custom_components.custom_loss_functions import SCCE, SCCEDifference, SCCEMaxBin, SCCEInverse
from utils import log_scripts
from utils import plot_scripts


os.environ["TF_DETERMINISTIC_OPS"] = "1"


print("\nGPUs Available:", tf.config.list_physical_devices("GPU"))
build_info = tf.sysconfig.get_build_info()
print("CUDA Version:", build_info["cuda_version"])
print("cuDNN Version:", build_info["cudnn_version"])

def prepare_model_dir(
    scenario_dir: str, 
    model: Model, 
    penalty_threshold: float, 
    learning_rate: float, 
    run_timestamp: str
) -> str:
    log_dir = (
        f"logs/{scenario_dir}/{run_timestamp}_lr_{learning_rate}_pr_{penalty_threshold}"
    )
    os.makedirs(log_dir)
    log_scripts.log_model_structure(model, log_dir)
    return log_dir


def scheduler(epoch, lr):
    if epoch == 40:
        return lr * 0.5
    if epoch == 60:
        return lr * 0.2
    return lr


def initialize_callbacks(model: Model, log_dir: str) -> List[Callback]:
    callbacks = []

    for i, layer in enumerate(model.layers):
        if "custom_conv2d_layer" in layer.name:
            scale_tracking_callback = custom_callbacks.NestedScaleTrackingCallback(
                layer, log_dir
            )
            callbacks.append(scale_tracking_callback)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callbacks.append(lr_callback)

    accuracy_loss_callback = custom_callbacks.AccuracyLossTrackingCallBack(log_dir)
    callbacks.append(accuracy_loss_callback)
    return callbacks

from typing import Tuple
from typing_extensions import Literal
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Input, BatchNormalization, Activation, MaxPooling2D, Dropout,
    GlobalAveragePooling2D, Dense, Add
)
from tensorflow.keras.initializers import RandomNormal

# Assume you have your custom CustomConv2DLayer somewhere above:
# import custom_layers  # or wherever your layer is defined

from typing import Tuple
from typing_extensions import Literal
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Input, BatchNormalization, Activation, MaxPooling2D, Dropout,
    GlobalAveragePooling2D, Dense, Add
)
from tensorflow.keras.initializers import RandomNormal

# Assume you have your custom CustomConv2DLayer somewhere above:
# import custom_layers  # or wherever your layer is defined


def initialize_model_post_training(
    penalty_threshold: float, 
    orientation: Literal["rowwise", "columnwise", "channelwise", "scalar"],
    seed: int = 42, 
    input_shape: Tuple[int, ...] = (224, 224, 3), 
    num_classes: int = 10
) -> Model:
    """
    A ResNet-like model for Imagenette with reproducible random behavior via a seed.
    This version inlines all the residual-block logic into one single function 
    rather than using a separate `residual_block` function.
    """
    trained_model = tf.keras.models.load_model('trained_model.keras')

    trained_weights = []

    for layer in trained_model.layers:
        if isinstance(layer, (Dense, Conv2D, BatchNormalization)):
            weights = layer.get_weights()
            trained_weights.append(weights)

    # Seeded initializer
    initializer = RandomNormal(seed=seed)
    # L2 regularization for the residual convs
    residual_regularizer = regularizers.l2(1e-4)

    inputs = Input(shape=input_shape)

    # Initial Conv Block (no residual yet)
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding="same",
        name="custom_conv2d_layer_64_0",
        regularizer=None,  
        trained_weights=trained_weights.pop(0)
    )(inputs)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x)

    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = Dropout(0.2, seed=seed)(x)

    # -------------------------------------------------------------------------
    # Below, we inline each residual block’s logic:
    # (conv → bn → relu) + (conv → bn) + shortcut + relu
    # -------------------------------------------------------------------------

    # --------------------- BLOCK GROUP: filters = 64 -------------------------
    # ---------------------------- block 1 (idx=1) ----------------------------
    shortcut = x
    # conv1
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_64_1_0",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)
    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x)

    x = Activation("relu")(x)

    # conv2
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_64_1_1",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)
    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x)

    # since stride=(1,1) for this block, no shortcut conv is needed
    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    # ---------------------------- block 2 (idx=2) ----------------------------
    shortcut = x
    # conv1
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_64_2_0",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)
    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x)
    x = Activation("relu")(x)

    # conv2
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_64_2_1",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)
    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x)

    # again stride=(1,1), so no shortcut conv
    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    # --------------------- BLOCK GROUP: filters = 128 ------------------------
    # ---------------------------- block 1 (idx=1) ----------------------------
    shortcut = x
    # conv1
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=128,
        kernel_size=(3, 3),
        strides=(2, 2),   # downsample
        padding="same",
        name="custom_conv2d_layer_128_1_0",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)
    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x)
    x = Activation("relu")(x)

    # conv2
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_128_1_1",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(1))
    x = bn(x)

    # shortcut path to match dimensions if stride != (1,1)
    shortcut = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=128,
        kernel_size=(1, 1),
        strides=(2, 2),
        padding="same",
        name="custom_conv2d_layer_shortcut_128_1",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(shortcut)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *shortcut.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    shortcut = bn(shortcut)

#    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    # ---------------------------- block 2 (idx=2) ----------------------------
    shortcut = x
    # conv1
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_128_2_0",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x)

    x = Activation("relu")(x)

    # conv2
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_128_2_1",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x)

    # stride=(1,1) => no change to the shortcut
    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    # --------------------- BLOCK GROUP: filters = 256 ------------------------
    # ---------------------------- block 1 (idx=1) ----------------------------
    shortcut = x
    # conv1
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=256,
        kernel_size=(3, 3),
        strides=(2, 2),  # downsample
        padding="same",
        name="custom_conv2d_layer_256_1_0",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)
    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x)
    x = Activation("relu")(x)

    # conv2
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_256_1_1",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(1))
    x = bn(x) 
    
    # shortcut path
    shortcut = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=256,
        kernel_size=(1, 1),
        strides=(2, 2),
        padding="same",
        name="custom_conv2d_layer_shortcut_256_1",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(shortcut)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *shortcut.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    shortcut = bn(shortcut) 

    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    # ---------------------------- block 2 (idx=2) ----------------------------
    shortcut = x

    # conv1
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_256_2_0",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x) 

    x = Activation("relu")(x)

    # conv2
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_256_2_1",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)
    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x) 

    # stride=(1,1) => no change to shortcut
    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    # --------------------- BLOCK GROUP: filters = 512 ------------------------
    # ---------------------------- block 1 (idx=1) ----------------------------
    shortcut = x
    # conv1
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=512,
        kernel_size=(3, 3),
        strides=(2, 2),  # downsample
        padding="same",
        name="custom_conv2d_layer_512_1_0",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)

    )(x)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x) 
    x = Activation("relu")(x)

    # conv2
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_512_1_1",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)

    )(x)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(1))
    x = bn(x) 

    # shortcut path
    shortcut = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=512,
        kernel_size=(1, 1),
        strides=(2, 2),
        padding="same",
        name="custom_conv2d_layer_shortcut_512_1",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)

    )(shortcut)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *shortcut.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    shortcut = bn(shortcut) 

    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    # ---------------------------- block 2 (idx=2) ----------------------------
    shortcut = x
    # conv1
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_512_2_0",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x) 

    x = Activation("relu")(x)

    # conv2
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        name="custom_conv2d_layer_512_2_1",
        regularizer=residual_regularizer,
        trained_weights=trained_weights.pop(0)
    )(x)

    bn = BatchNormalization()
    bn.build(input_shape=(None, *x.shape[1:]))
    bn.set_weights(trained_weights.pop(0))
    x = bn(x) 

    # stride=(1,1) => no change to shortcut
    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    # ---------------------------- HEAD (classification) -----------------------
    x = Dropout(0.3, seed=seed)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dropout(0.3, seed=seed)(x)
    #outputs = Dense(num_classes, activation="softmax", kernel_initializer=initializer)(x)

    last_layer = Dense(10, activation="softmax")
    weights = trained_weights.pop(0)
    last_layer.build(input_shape=(None, weights[0].shape[0]))
    last_layer.set_weights(weights)
    outputs = last_layer(x)


    model = Model(inputs=inputs, outputs=outputs)
    return model



def build_imagenette_model(
    penalty_threshold: float, 
    orientation: Literal["rowwise", "columnwise", "channelwise", "scalar"],
    seed: int = 42, 
    input_shape: Tuple[int, ...] = (224, 224, 3), 
    num_classes: int = 10
) -> Model:
    """A ResNet-like model for Imagenette with reproducible random behavior via a seed."""
    initializer = RandomNormal(seed=seed)  # Seeded initializer for Conv2D

    inputs = Input(shape=input_shape)
    
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold = penalty_threshold,
        orientation = orientation,
        initializer=initializer,
        filters=64,
        kernel_size=(7,7),
        strides=(2,2),
        padding="same",
        name="custom_conv2d_layer_64_0",
        regularizer=None,
        trained_weights=None
    )(inputs)

    # Initial Conv Block
#    x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", kernel_initializer=kernel_init)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = Dropout(0.2, seed=seed)(x)

    # Residual Blocks with seed
    x = residual_block(x, 64, orientation=orientation, penalty_threshold=penalty_threshold, idx = 1, seed=seed)
    x = residual_block(x, 64, orientation=orientation, penalty_threshold=penalty_threshold, idx = 2, seed=seed)
    
    x = residual_block(x, 128, stride=(2, 2), orientation=orientation, penalty_threshold=penalty_threshold, idx = 1, seed=seed)
    x = residual_block(x, 128, orientation=orientation, penalty_threshold=penalty_threshold, idx = 2, seed=seed)
    
    x = residual_block(x, 256, stride=(2, 2), orientation=orientation, penalty_threshold=penalty_threshold, idx = 1, seed=seed)
    x = residual_block(x, 256, orientation=orientation, penalty_threshold=penalty_threshold, idx = 2, seed=seed)
    
    x = residual_block(x, 512, stride=(2, 2), orientation=orientation, penalty_threshold=penalty_threshold, idx = 1, seed=seed)
    x = residual_block(x, 512, orientation=orientation, penalty_threshold=penalty_threshold, idx = 2, seed=seed)
    
    # Global Pooling and Dense Layers
    x = Dropout(0.3, seed=seed)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3, seed=seed)(x)
    outputs = Dense(num_classes, activation="softmax", kernel_initializer=initializer)(x)
    model = Model(inputs=inputs, outputs=outputs)

    return model


def initialize_loss_function(model: Model, penalty_rate: float, log_dir: str, loss_func = SCCEMaxBin):
    layers = [layer for i, layer in enumerate(model.layers) if isinstance(layer, custom_layers.CustomConv2DLayer)]

    loss_function = loss_func(
        layers=layers,
        penalty_rate=penalty_rate,
        log_dir = log_dir 
    )
    return loss_function


def compile_model(model: Model, learning_rate: float, loss_function) -> None:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_function.compute_total_loss,
        metrics=["accuracy"],
    )


def train_model(model: Model, epochs: int, train_data: tf.data.Dataset, validation_data: tf.data.Dataset, callbacks: List[Callback], batch_size: int = 32) -> None:
    #(x_train, y_train) = train_data
    model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        batch_size=batch_size,
    )


def evaluate_model(model: Model, validation_data: Tuple[Any, Any]) -> Tuple[float, float]:
    #(x_test, y_test) = validation_data
    loss, accuracy = model.evaluate(validation_data)
    print(f"Model Test Accuracy: {accuracy}\nModel Test Loss: {loss}")
    return accuracy, loss


def plot_results(log_dir_path: str) -> None:
    logging.info(f"Processing {log_dir_path}...")

#    plot_scripts.plot_values_logged_on_epoch_end(log_dir_path)
    plot_scripts.plot_accuracy_per_epoch(log_dir_path)
    plot_scripts.plot_total_loss_per_epoch(log_dir_path)
#    plot_scripts.plot_histograms_on_train_begin(log_dir_path)
    plot_scripts.plot_histograms_on_train_end(log_dir_path)


def plot_pareto(log_scenario_dir: str) -> None:
    logging.info(f"Pareto plots...")

    plot_scripts.pareto_unique_values_accuracy(log_scenario_dir)
    plot_scripts.pareto_unique_values_loss(log_scenario_dir)
    plot_scripts.pareto_range_accuracy(log_scenario_dir)
    plot_scripts.pareto_range_loss(log_scenario_dir)
    plot_scripts.val_acc_over_penalty(log_scenario_dir)
    plot_scripts.val_loss_over_penalty(log_scenario_dir)

    logging.info(f"Pareto plots done.")


def save_artefacts(log_scenario_dir: str) -> None:
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


def augment_image(image, label):
    # Resize *up* to 256, then random crop to 224
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, [224, 224, 3])
    
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Optional slight color jitter
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    
    # Ensure valid pixel range after these augmentations
    image = tf.clip_by_value(image, 0.0, 255.0)
    
    return image, label


def preprocess_for_validation(image, label):
    # Resize to 224, 224 for validation (central crop also possible)
    image = tf.image.resize(image, [224, 224])
    return image, label


def prepare_data(batch_size=64):
    # Load tfds splits
    train_data = tfds.load('imagenette', split='train', as_supervised=True)
    val_data   = tfds.load('imagenette', split='validation', as_supervised=True)

    # Augment training
    train_data = (
        train_data
        .shuffle(10_000, seed=42)
        .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # Validation data
    val_data = (
        val_data
        .map(preprocess_for_validation, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_data, val_data, (224,224,3)


def main():
    parser = argparse.ArgumentParser(
        description="Train model for different penalty rates for this scenario."
    )
    parser.add_argument(
        "scenario_dir",
        type=str,
        help="The scenario directory containing different train results for different penaly rates (e.g., 'post_training_maxbin_rowwise').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        choices=["rowwise", "columnwise", "channelwise", "scalar"], 
        help="How the scalers are to be applied to the weight matrix. Options are rowwise, columnwise, channelwise or scalar.",
    )
    parser.add_argument(
        "--custom_loss",
        type=str,
        choices=["maxbin", "difference", "inverse"], 
        help="Custom loss function term to be applied.",
    )
    parser.add_argument(
        "--post_training",
        type=bool,
        help="If true weights of a trained model will be used.",
    )

    args = parser.parse_args()
    scenario_dir = args.scenario_dir
    seed = args.seed
    orientation = args.orientation
    custom_loss = args.custom_loss
    post_training = args.post_training

    if custom_loss == "maxbin":
        custom_loss = SCCEMaxBin
    elif custom_loss == "difference":
        custom_loss = SCCEDifference
    else:
        custom_loss = SCCEInverse

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    scenario_dir_path = os.path.join("logs", scenario_dir)
    os.makedirs(scenario_dir_path, exist_ok=True)

    # Setup logger for plotting
    plotting_log_file_path = os.path.join(scenario_dir_path, "plotting_process.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(plotting_log_file_path),
        ],
    )

    train_data, validation_data, input_data_shape = prepare_data()

    learning_rate = 0.0001
    epochs = 100
    batch_size = 64
    quantized = True

    penalty_thresholds = [
        #0.0,
        #1e-11,
        #1e-10,
        #1e-9,
        #1e-8,
        #1.0,
    ]        

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for penalty_threshold in penalty_thresholds:
            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if post_training == True:
                model = initialize_model_post_training(
                    penalty_threshold=penalty_threshold, 
                    orientation=orientation,
                    input_shape=input_data_shape, 
                    seed=seed,                    
                )
            else:
                model = initialize_model(
                    penalty_threshold=penalty_threshold, 
                    orientation=orientation,
                    input_shape=input_data_shape, 
                    seed=seed
                )
            log_dir = prepare_model_dir(
                scenario_dir=scenario_dir,
                model=model,
                penalty_threshold=penalty_threshold,
                learning_rate=learning_rate,
                run_timestamp=run_timestamp,
            )
            loss_function = initialize_loss_function(model=model, penalty_rate=penalty_threshold, log_dir=log_dir, loss_func=SCCEDifference)
            compile_model(model=model, learning_rate=learning_rate, loss_function=loss_function)

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

            #model.save("trained_model.keras")

            log_scripts.save_parameters(model=model, log_dir=log_dir, quantized=quantized)

            #futures.append(executor.submit(plot_results, log_dir))

            plot_pareto(scenario_dir_path)
            save_artefacts(scenario_dir_path)

#        concurrent.futures.wait(futures)
    plot_pareto(scenario_dir_path)
    save_artefacts(scenario_dir_path)


if __name__ == "__main__":
    main()
