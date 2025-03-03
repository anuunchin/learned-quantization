import argparse
import concurrent.futures
import logging
import random
import os
from datetime import datetime
import numpy as np
from typing import List, Literal, Tuple, Any

import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Conv2D
)
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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


def initialize_callbacks(
    model: Model, 
    log_dir: str
) -> List[Callback]:
    """
    Initializes and returns a list of callbacks 
    for tracking scale values, resulting quantization, accuracy, and loss.
    """    
    callbacks = []

    # TODO: add the custom dense layer callback
    for i, layer in enumerate(model.layers):
        if "custom_conv2d_layer" in layer.name:
            scale_tracking_callback = custom_callbacks.NestedScaleTrackingCallback(
                layer, log_dir
            )
            callbacks.append(scale_tracking_callback)

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
    Initializes a custom quantized model using trained weights from a baseline model.
    This is essentially similar to the next function.
    The only difference is we provide pretrained weights.
    """
    trained_model = tf.keras.models.load_model('trained_model.keras')

    trained_weights = []

    for layer in trained_model.layers:
        if isinstance(layer, (Dense, Conv2D, BatchNormalization)):
            weights = layer.get_weights()
            trained_weights.append(weights)

    input_layer = Input(shape=input_shape)

    initializer = tf.keras.initializers.RandomNormal(seed=seed)

    # First Block
    conv_layer_1_a = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer = initializer,  # For the post-training scenario this is trivial
        filters=32,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_32_1",
        regularizer=None,
        trained_weights=trained_weights.pop(0)
    )

    conv_layer_1_a_output = conv_layer_1_a(input_layer)
    conv_layer_1_a_activation = tf.keras.activations.relu(conv_layer_1_a_output)
   
    batch_norm_1_a = BatchNormalization()
    batch_norm_1_a.build(input_shape=(None, *conv_layer_1_a_activation.shape[1:]))
    batch_norm_1_a.set_weights(trained_weights.pop(0))
    batch_norm_1_a = batch_norm_1_a(conv_layer_1_a_activation)

    conv_layer_1_b = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer=initializer,  # For the post-training scenario this is trivial
        filters=32,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_32_2",
        regularizer=None,
        trained_weights=trained_weights.pop(0)
    )

    conv_layer_1_b_output = conv_layer_1_b(batch_norm_1_a)
    conv_layer_1_b_activation = tf.keras.activations.relu(conv_layer_1_b_output)
    
    batch_norm_1_b = BatchNormalization()
    batch_norm_1_b.build(input_shape=(None, *conv_layer_1_b_activation.shape[1:]))
    batch_norm_1_b.set_weights(trained_weights.pop(0))
    batch_norm_1_b = batch_norm_1_b(conv_layer_1_b_activation)

    pool_layer_1 = MaxPooling2D(pool_size=(2, 2))(batch_norm_1_b)
    drop_layer_1 = Dropout(0.2, seed=seed)(pool_layer_1)

    # Second Block
    conv_layer_2_a = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer=initializer,  # For the post-training scenario this is trivial
        filters=64,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_64_1",
        regularizer=None,
        trained_weights=trained_weights.pop(0)
    )

    conv_layer_2_a_output = conv_layer_2_a(drop_layer_1)
    conv_layer_2_a_activation = tf.keras.activations.relu(conv_layer_2_a_output)
    
    batch_norm_2_a = BatchNormalization()
    batch_norm_2_a.build(input_shape=(None, *conv_layer_2_a_activation.shape[1:]))
    batch_norm_2_a.set_weights(trained_weights.pop(0))
    batch_norm_2_a = batch_norm_2_a(conv_layer_2_a_activation)

    conv_layer_2_b = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer=initializer,  # For the post-training scenario this is trivial
        filters=64,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_64_2",
        regularizer=None,
        trained_weights=trained_weights.pop(0)
    )

    conv_layer_2_b_output = conv_layer_2_b(batch_norm_2_a)
    conv_layer_2_b_activation = tf.keras.activations.relu(conv_layer_2_b_output)
    
    batch_norm_2_b = BatchNormalization()
    batch_norm_2_b.build(input_shape=(None, *conv_layer_2_b_activation.shape[1:]))
    batch_norm_2_b.set_weights(trained_weights.pop(0))
    batch_norm_2_b = batch_norm_2_b(conv_layer_2_b_activation)

    pool_layer_2 = MaxPooling2D(pool_size=(2, 2))(batch_norm_2_b)
    drop_layer_2 = Dropout(0.3, seed=seed)(pool_layer_2)

    # Third Block
    conv_layer_3_a = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer=initializer,  # For the post-training scenario this is trivial
        filters=128,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_128_1",
        regularizer=None,
        trained_weights=trained_weights.pop(0)
    )

    conv_layer_3_a_output = conv_layer_3_a(drop_layer_2)
    conv_layer_3_a_activation = tf.keras.activations.relu(conv_layer_3_a_output)
    
    batch_norm_3_a = BatchNormalization()
    batch_norm_3_a.build(input_shape=(None, *conv_layer_3_a_activation.shape[1:]))
    batch_norm_3_a.set_weights(trained_weights.pop(0))
    batch_norm_3_a = batch_norm_3_a(conv_layer_3_a_activation)

    conv_layer_3_b = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer=initializer,  # For the post-training scenario this is trivial
        filters=128,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_128_2",
        regularizer=None,
        trained_weights=trained_weights.pop(0)
    )

    conv_layer_3_b_output = conv_layer_3_b(batch_norm_3_a)
    conv_layer_3_b_activation = tf.keras.activations.relu(conv_layer_3_b_output)
    
    batch_norm_3_b = BatchNormalization()
    batch_norm_3_b.build(input_shape=(None, *conv_layer_3_b_activation.shape[1:]))
    batch_norm_3_b.set_weights(trained_weights.pop(0))
    batch_norm_3_b = batch_norm_3_b(conv_layer_3_b_activation)

    pool_layer_3 = MaxPooling2D(pool_size=(2, 2))(batch_norm_3_b)
    drop_layer_3 = Dropout(0.4, seed=seed)(pool_layer_3)

    # Flatten and Dense Layers
    flatten_layer = Flatten()(drop_layer_3)

    dense_layer_1 = Dense(128, activation="relu")
    weights = trained_weights.pop(0)
    dense_layer_1.build(input_shape=(None, weights[0].shape[0]))
    dense_layer_1.set_weights(weights)
    dense_layer_1_output = dense_layer_1(flatten_layer)

    batch_norm_dense = BatchNormalization()
    batch_norm_dense.build(input_shape=(None, *dense_layer_1_output.shape[1:]))
    batch_norm_dense.set_weights(trained_weights.pop(0))
    batch_norm_dense = batch_norm_dense(dense_layer_1_output)

    drop_layer_dense = Dropout(0.5, seed=seed)(batch_norm_dense)

    last_layer = Dense(10, activation="softmax")
    weights = trained_weights.pop(0)
    last_layer.build(input_shape=(None, weights[0].shape[0]))
    last_layer.set_weights(weights)
    output_layer = last_layer(drop_layer_dense)

    quantized_model = Model(inputs=input_layer, outputs=output_layer)

    return quantized_model


def initialize_model(
    penalty_threshold: float, 
    seed: int, 
    orientation: Literal["rowwise", "columnwise", "channelwise", "scalar"],
    input_shape: Tuple[int, ...] = (32, 32, 3), 
    num_classes: int = 10
) -> Model:
    """
    Initializes a custom model with six convolutional layers, configured using 
    a specified penalty threshold, orientation (quantization granularity), and specified random seed.
    """
    initializer = tf.keras.initializers.RandomNormal(seed=seed)

    input_layer = Input(shape=input_shape)

    # First Block
    conv_layer_1_a = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer = initializer,
        filters=32,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_32_1",
        regularizer=None,
        trained_weights=None
    )

    conv_layer_1_a_output = conv_layer_1_a(input_layer)
    conv_layer_1_a_activation = tf.keras.activations.relu(conv_layer_1_a_output)
    batch_norm_1_a = BatchNormalization()(conv_layer_1_a_activation)

    conv_layer_1_b = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer=initializer,
        filters=32,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_32_2",
        regularizer=None,
        trained_weights=None
    )

    conv_layer_1_b_output = conv_layer_1_b(batch_norm_1_a)
    conv_layer_1_b_activation = tf.keras.activations.relu(conv_layer_1_b_output)
    batch_norm_1_b = BatchNormalization()(conv_layer_1_b_activation)

    pool_layer_1 = MaxPooling2D(pool_size=(2, 2))(batch_norm_1_b)
    drop_layer_1 = Dropout(0.2, seed=seed)(pool_layer_1)

    # Second Block
    conv_layer_2_a = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer=initializer,
        filters=64,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_64_1",
        regularizer=None,
        trained_weights=None
    )

    conv_layer_2_a_output = conv_layer_2_a(drop_layer_1)
    conv_layer_2_a_activation = tf.keras.activations.relu(conv_layer_2_a_output)
    batch_norm_2_a = BatchNormalization()(conv_layer_2_a_activation)

    conv_layer_2_b = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer=initializer,
        filters=64,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_64_2",
        regularizer=None,
        trained_weights=None
    )

    conv_layer_2_b_output = conv_layer_2_b(batch_norm_2_a)
    conv_layer_2_b_activation = tf.keras.activations.relu(conv_layer_2_b_output)
    batch_norm_2_b = BatchNormalization()(conv_layer_2_b_activation)

    pool_layer_2 = MaxPooling2D(pool_size=(2, 2))(batch_norm_2_b)
    drop_layer_2 = Dropout(0.3, seed=seed)(pool_layer_2)

    # Third Block
    conv_layer_3_a = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer=initializer,
        filters=128,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_128_1",
        regularizer=None,
        trained_weights=None
    )

    conv_layer_3_a_output = conv_layer_3_a(drop_layer_2)
    conv_layer_3_a_activation = tf.keras.activations.relu(conv_layer_3_a_output)
    batch_norm_3_a = BatchNormalization()(conv_layer_3_a_activation)

    conv_layer_3_b = custom_layers.CustomConv2DLayer(
        seed=seed,
        penalty_threshold=penalty_threshold,
        orientation = orientation,
        initializer=initializer,
        filters=128,
        kernel_size=(3, 3),
        strides=(1,1),
        padding="same",
        name="custom_conv2d_layer_128_2",
        regularizer=None,
        trained_weights=None
    )

    conv_layer_3_b_output = conv_layer_3_b(batch_norm_3_a)
    conv_layer_3_b_activation = tf.keras.activations.relu(conv_layer_3_b_output)
    batch_norm_3_b = BatchNormalization()(conv_layer_3_b_activation)

    pool_layer_3 = MaxPooling2D(pool_size=(2, 2))(batch_norm_3_b)
    drop_layer_3 = Dropout(0.4, seed=seed)(pool_layer_3)

    # Flatten and Dense Layers
    flatten_layer = Flatten()(drop_layer_3)

    dense_layer_1 = Dense(
        128, activation="relu", kernel_initializer=initializer
    )
    dense_layer_1_output = dense_layer_1(flatten_layer)

    batch_norm_dense = BatchNormalization()(dense_layer_1_output)

    drop_layer_dense = Dropout(0.5, seed=seed)(batch_norm_dense)

    output_layer = Dense(
        10, activation="softmax", kernel_initializer=initializer
    )(drop_layer_dense)

    quantized_model = Model(inputs=input_layer, outputs=output_layer)

    return quantized_model


def compile_model(
    model: Model, 
    learning_rate: float
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
    (x_train, y_train) = train_data
    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        batch_size=batch_size,
    )


def evaluate_model(
    model: Model, 
    validation_data: Tuple[Any, Any]
) -> Tuple[float, float]:
    (x_test, y_test) = validation_data
    loss, accuracy = model.evaluate(x_test, y_test)
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


def prepare_data() -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[int, ...]]:

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    input_data_shape = x_train[0].shape

    return (x_train, y_train), (x_test, y_test), input_data_shape


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
    epochs = 100
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

            # Compile the model with the specified learning rate
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
