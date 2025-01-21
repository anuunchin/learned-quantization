import argparse
import concurrent.futures
import logging
import random
import os
from datetime import datetime
import numpy as np
from typing import List, Literal, Tuple, Any

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Conv2D,
    Add,
    GlobalAveragePooling2D,
    Activation
)
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

import custom_components.custom_callbacks as custom_callbacks
import custom_components.custom_layers as custom_layers
import utils.log_scripts as log_scripts
import utils.plot_scripts as plot_scripts

os.environ["TF_DETERMINISTIC_OPS"] = "1"

#print("\nGPUs Available:", tf.config.list_physical_devices("GPU"))
build_info = tf.sysconfig.get_build_info()
#print("CUDA Version:", build_info["cuda_version"])
#print("cuDNN Version:", build_info["cudnn_version"])

eps_float32 = np.finfo(np.float32).eps

baseline_model = "baseline_model.keras"


def prepare_model_dir(
    scenario_dir: str,
    model: Model,
    penalty_threshold: float,
    learning_rate: float,
    run_timestamp: str,
) -> str:
    """
    Prepares a directory for the model logs and structure, 
    organized by scenario, timestamp, learning rate, and penalty threshold.
    """
    log_dir = (
        f"{scenario_dir}/{run_timestamp}_lr_{learning_rate}_pr_{penalty_threshold}"
    )
    os.makedirs(log_dir)
    log_scripts.log_model_structure(model, log_dir)
    return log_dir


def scheduler(
    epoch: int, 
    lr: float
) -> float:
    """
    Used to schedule learning rate decay.
    """
    if epoch == 40:
        return lr * 0.5
    if epoch == 60:
        return lr * 0.2
    return lr


def initialize_callbacks(
    model: Model, 
    log_dir: str
) -> List[Callback]:
    """
    Initializes and returns a list of callbacks 
    for tracking scale values, resulting quantization, accuracy, and loss.
    Additionally schedules learning rate decay.
    """    
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


def initialize_model_with_trained_weights(
    penalty_threshold: float, 
    seed: int, 
    orientation: Literal["rowwise", "columnwise", "channelwise", "scalar"],
    input_shape: Tuple[int, ...] = (32, 32, 3), 
) -> Model:    
    """
    A ResNet-like model for Imagenette with trained weights from the baseline model.
    This version inlines all the residual-block logic into one single function 
    rather than using a separate `residual_block` function.
    The reason is that the order of weights is saved 
    differenly so wee need to account for that to keep things clear.
    """
    trained_model = tf.keras.models.load_model('trained_model.keras')

    trained_weights = []

    for layer in trained_model.layers:
        if isinstance(layer, (Dense, Conv2D, BatchNormalization)):
            weights = layer.get_weights()
            trained_weights.append(weights)

    initializer = RandomNormal(seed=seed)

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

    last_layer = Dense(10, activation="softmax")
    weights = trained_weights.pop(0)
    last_layer.build(input_shape=(None, weights[0].shape[0]))
    last_layer.set_weights(weights)
    outputs = last_layer(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def residual_block(
    x, 
    filters, 
    orientation,
    penalty_threshold,
    idx,
    kernel_size=(3, 3), 
    stride=(1, 1), 
    seed=None,
):
    """
    A “standard” ResNet v1 basic block:
      1. conv → bn → relu
      2. conv → bn
      3. add shortcut
      4. relu
    """
    initializer = RandomNormal(seed=seed)
    regularizer = regularizers.l2(1e-4)

    # Save the input for the shortcut path
    shortcut = x

    # --- Conv 1 ---
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        name=f"custom_conv2d_layer_{filters}_{idx}_0",
        regularizer=regularizer,
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # --- Conv 2 ---
    x = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation=orientation,
        initializer=initializer,
        filters=filters,
        kernel_size=kernel_size,
        strides=(1,1),
        padding="same",
        name=f"custom_conv2d_layer_{filters}_{idx}_1",
        regularizer=regularizer,
    )(x)        

    x = BatchNormalization()(x)

    # If we changed spatial dims via 'stride', apply a 1x1 conv to the shortcut
    if stride != (1, 1):
        shortcut = custom_layers.CustomConv2DLayer(
            seed=seed,
            penalty_threshold=penalty_threshold,
            orientation=orientation,
            initializer=initializer,
            filters=filters,
            kernel_size = (1,1),
            strides=stride,
            padding="same",
            name=f"custom_conv2d_layer_shortcut",
            regularizer=regularizer
        )(shortcut)

        shortcut = BatchNormalization()(shortcut)

    # Add shortcut
    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    return x


def initialize_model(
    penalty_threshold: float, 
    seed: int, 
    orientation: Literal["rowwise", "columnwise", "channelwise", "scalar"],
    input_shape: Tuple[int, ...] = (32, 32, 3), 
    num_classes: int = 10
) -> Model:
    """
    A ResNet-like model for Imagenette with reproducible random behavior via a seed.
    """
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


def compile_model(
    model: Model, 
    learning_rate: float,
) -> None:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_model(
    model: Model, 
    epochs: int, 
    train_data: Tuple[Any, Any], 
    validation_data: Tuple[Any, Any], 
    callbacks: List[Callback], 
    batch_size: int = 32
) -> None:
    model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        batch_size=batch_size,
    )


def evaluate_model(
    model: Model, 
    validation_data: Tuple[Any, Any]
) -> Tuple[float, float]:
    loss, accuracy = model.evaluate(validation_data)
    print(f"Model Test Accuracy: {accuracy}\nModel Test Loss: {loss}")
    return accuracy, loss


def plot_results(log_dir_path: str) -> None:
    """
    Plots various things from the logs created during training.
    The middle three are commented out currently as they take a while to be created.
    We therefore recomment to uncomment these only when necessary.
    """
    logging.info(f"Processing {log_dir_path}...")

#    plot_scripts.plot_values_logged_on_epoch_end(log_dir_path)
#    plot_scripts.plot_accuracy_per_epoch(log_dir_path)
#    plot_scripts.plot_total_loss_per_epoch(log_dir_path)
#    plot_scripts.plot_histograms_on_train_begin(log_dir_path)
#    plot_scripts.plot_histograms_on_train_end(log_dir_path)

    logging.info(f"Training plots done.")


def plot_pareto(log_scenario_dir: str) -> None:
    """
    Plots pareto plots, as well as validation accuracy and loss oevr epochs 
    for different penalty threshold trainings in the same directory.
    """
    logging.info(f"Pareto plots...")

    plot_scripts.pareto_unique_values_accuracy(log_scenario_dir)
    plot_scripts.pareto_unique_values_loss(log_scenario_dir)
    plot_scripts.pareto_range_accuracy(log_scenario_dir)
    plot_scripts.pareto_range_loss(log_scenario_dir)
    plot_scripts.val_acc_over_penalty(log_scenario_dir)
    plot_scripts.val_loss_over_penalty(log_scenario_dir)

    logging.info(f"Pareto plots done.")


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
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        required=True,
        choices=["rowwise", "columnwise", "channelwise", "scalar"], 
        help="How the scalers are to be applied to the weight matrix. Options are rowwise, columnwise, channelwise or scalar.",
    )
    parser.add_argument(
        "--training",
        type=str,
        required=True,
        choices=["post_training", "from_scratch"],
        help="If true weights of a trained model will be used.",
    )

    args = parser.parse_args()
    seed = args.seed
    orientation = args.orientation
    training = args.training
    scenario_dir = f"{orientation}_{training}"

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    scenario_dir_path = os.path.join("logs", scenario_dir, f"seed_{seed}")
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
    epochs = 1
    batch_size = 128

    penalty_thresholds = [
        0.0,
        #1e-11,
        #1e-10,
        #1e-9,
        #1e-8,
    ]

    # Create a thread pool for concurrent execution so that we can plot the previous
    # training logs, while continuing with the next one
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for penalty_threshold in penalty_thresholds:
            
            # Initialize model with or without pre-trained weights
            if training == "post_training":
                model = initialize_model_with_trained_weights(
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
                    seed=seed,
                )

            # Prepare the directory for model logging
            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = prepare_model_dir(
                scenario_dir=scenario_dir_path,
                model=model,
                penalty_threshold=penalty_threshold,
                learning_rate=learning_rate,
                run_timestamp=run_timestamp,
            )

            # Compile the model with the specified learning rate and loss function
            compile_model(model=model, learning_rate=learning_rate)

            # Initialize callbacks for training
            callbacks = initialize_callbacks(model=model, log_dir=log_dir)

            # Train the model and log results
            train_model(
                model=model,
                epochs=epochs,
                train_data=train_data,
                validation_data=validation_data,
                batch_size=batch_size,
                callbacks=callbacks,
            )

            # Evaluate the model on validation data
            accuracy, loss = evaluate_model(
                model=model, validation_data=validation_data
            )

            # Save model parameters and artifacts
            log_scripts.save_compress_parameters(model=model, log_dir=log_dir)
            current_script_dir = os.path.dirname(__file__)
            log_scripts.save_artefacts(scenario_dir_path, current_script_dir)

            # Submit plotting task to executor
            #futures.append(executor.submit(plot_results, log_dir))

            # Generate Pareto plot after each penalty threshold, so we can see it as soon as possible
            plot_pareto(scenario_dir_path)

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    plot_pareto(scenario_dir_path)

if __name__ == "__main__":
    main()
