import argparse
import concurrent.futures
import logging
import os
import shutil
from datetime import datetime
from typing import Any, List, Literal, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (
    Flatten,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

import custom_components.custom_callbacks as custom_callbacks
import custom_components.custom_layers as custom_layers
from custom_components.custom_loss_functions import SCCEDifference, SCCEMaxBin, SCCEInverse
import utils.log_scripts as log_scripts
import utils.plot_scripts as plot_scripts

eps_float32 = np.finfo(np.float32).eps

baseline_model = "baseline_model.keras"


def prepare_model_dir(
    scenario_dir: str,
    model: Model,
    penalty_rate: float,
    learning_rate: float,
    run_timestamp: str,
) -> str:
    """
    Prepares a directory for the model logs and structure, 
    organized by scenario, timestamp, learning rate, and penalty rate.
    """
    log_dir = (
        f"{scenario_dir}/{run_timestamp}_lr_{learning_rate}_pr_{penalty_rate}"
    )
    os.makedirs(log_dir)
    log_scripts.log_model_structure(model, log_dir)
    return log_dir


def initialize_callbacks(
    model: Model, 
    log_dir: str, 
) -> List[Callback]:
    """
    Initializes and returns a list of callbacks 
    for tracking scale values, resulting quantization, accuracy, and loss.
    """
    callbacks = []

    for i, layer in enumerate(model.layers):
        if "custom_dense_layer" in layer.name:
            scale_tracking_callback = custom_callbacks.NestedScaleTrackingCallback(
                layer, log_dir
            )
            callbacks.append(scale_tracking_callback)

    accuracy_loss_callback = custom_callbacks.AccuracyLossTrackingCallBack(log_dir)
    callbacks.append(accuracy_loss_callback)
    return callbacks

def initialize_model_with_trained_weights(
    penalty_rate: float,
    seed: int,
    orientation: Literal["rowwise", "columnwise", "scalar"],
    input_shape: tuple = (28, 28, 1),
) -> Model:
    """
    Initializes a custom quantized model using trained weights from a baseline model.
    This is essentially similar to the next function.
    The only difference is we provide pretrained weights.
    """
    trained_model = tf.keras.models.load_model(baseline_model)
    trained_weights_1 = trained_model.get_layer("dense").get_weights()
    trained_weights_2 = trained_model.get_layer("dense_1").get_weights()

    initializer = tf.keras.initializers.RandomNormal(seed=seed)

    input_layer = Input(shape=input_shape)
    flatten_layer = Flatten()(input_layer)

    dense_layer = custom_layers.CustomDenseLayer(
        seed=seed,
        units=128,
        penalty_rate=penalty_rate,
        orientation=orientation,
        initializer=initializer, # For the post-training scenario this is trivial
        name="custom_dense_layer_1",
        regularizer=None,
        trained_weights=trained_weights_1,
    )

    dense_layer_output = dense_layer(flatten_layer)
    dense_layer_activation = tf.keras.activations.relu(dense_layer_output)

    dense_layer_2 = custom_layers.CustomDenseLayer(
        seed=seed,
        units=10,
        penalty_rate=penalty_rate,
        orientation=orientation,
        initializer=initializer, # For the post-training scenario this is trivial
        name="custom_dense_layer_2",
        regularizer=None,
        trained_weights=trained_weights_2,
    )

    dense_layer_2_output = dense_layer_2(dense_layer_activation)
    dense_layer_2_activation = tf.keras.activations.softmax(dense_layer_2_output)
    quantized_model = Model(inputs=input_layer, outputs=dense_layer_2_activation)
    return quantized_model


def initialize_model(
    penalty_rate: float,
    seed: int,
    orientation: Literal["rowwise", "columnwise", "scalar"],
    input_shape: tuple = (28, 28, 1),
) -> Model:
    """
    Initializes a custom model with two quantized dense layers, configured using 
    a specified penalty rate, orientation (quantization granularity), and specified random seed.
    """
    initializer = tf.keras.initializers.RandomNormal(seed=seed)

    input_layer = Input(shape=input_shape)
    flatten_layer = Flatten()(input_layer)

    dense_layer = custom_layers.CustomDenseLayer(
        seed=seed,
        units=128, 
        penalty_rate=penalty_rate, 
        orientation=orientation,
        initializer=initializer,
        name="custom_dense_layer_1",
        regularizer=None,
    )

    dense_layer_output = dense_layer(flatten_layer)
    dense_layer_activation = tf.keras.activations.relu(dense_layer_output)

    dense_layer_2 = custom_layers.CustomDenseLayer(
        seed=seed,
        units=10, 
        penalty_rate=penalty_rate, 
        orientation=orientation,
        initializer=initializer,
        name="custom_dense_layer_2",
        regularizer=None,
    )
    
    dense_layer_2_output = dense_layer_2(dense_layer_activation)
    dense_layer_2_activation = tf.keras.activations.softmax(dense_layer_2_output)
    quantized_model = Model(inputs=input_layer, outputs=dense_layer_2_activation)
    return quantized_model


def initialize_loss_function(
    model: Model, 
    penalty_rate: float, 
    log_dir: str, 
    loss_func = SCCEMaxBin
):
    """
    Initializes a loss function instance for the given model by extracting applicable layers 
    (e.g., custom dense layers) and configuring it with the specified penalty rate and log directory.
    """
    layers = [layer for i, layer in enumerate(model.layers) if "custom_dense_layer" in layer.name]

    loss_function = loss_func(
        layers=layers,
        penalty_rate=penalty_rate,
        log_dir = log_dir 
    )
    return loss_function


def compile_model(
    model: Model, 
    learning_rate: float, 
    loss_function
) -> None:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_function.compute_total_loss,
        metrics=["accuracy"],
    )


def train_model(
    model: Model,
    epochs: int,
    train_data: tuple,
    validation_data: tuple,
    callbacks: List[Callback],
    batch_size: int = 32,
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

    #plot_scripts.plot_values_logged_on_epoch_end(log_dir_path)
    #plot_scripts.plot_accuracy_per_epoch(log_dir_path)
    #plot_scripts.plot_total_loss_per_epoch(log_dir_path)
    #plot_scripts.plot_histograms_on_train_begin(log_dir_path)
    #plot_scripts.plot_histograms_on_train_end(log_dir_path)

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

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
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
        "--custom_loss",
        type=str,
        required=True,
        choices=["maxbin", "difference", "inverse"], 
        help="Custom loss function term to be applied.",
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
    custom_loss = args.custom_loss
    scenario_dir = f"{custom_loss}_{orientation}_{training}"

    if custom_loss == "maxbin":
        custom_loss = SCCEMaxBin
    elif custom_loss == "difference":
        custom_loss = SCCEDifference
    else:
        custom_loss = SCCEInverse

    tf.random.set_seed(seed)

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
    batch_size = 32

    penalty_rates = [
                0.0,
            #    1e-12,
            #    1e-11,
            #    1e-10,
            #    1e-9,
            #    1e-8,
            #    1e-7,
            #    1e-6,
            #    1e-5,
            #    1e-4,
    ]

    # Create a thread pool for concurrent execution so that we can plot the previous
    # training logs, while continuing with the next one
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for penalty_rate in penalty_rates:

            # Initialize model with or without pre-trained weights
            if training == "post_training":
                model = initialize_model_with_trained_weights(
                    penalty_rate=penalty_rate,
                    orientation=orientation,
                    input_shape=input_data_shape,
                    seed=seed,
                )
            else:
                model = initialize_model(
                    penalty_rate=penalty_rate,
                    orientation=orientation,
                    input_shape=input_data_shape,
                    seed=seed,
                )

            # Prepare the directory for model logging
            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = prepare_model_dir(
                scenario_dir=scenario_dir_path,
                model=model,
                penalty_rate=penalty_rate,
                learning_rate=learning_rate,
                run_timestamp=run_timestamp,
            )

            # Initialize the loss function instance 
            loss_function = initialize_loss_function(model=model, penalty_rate=penalty_rate, log_dir=log_dir, loss_func=custom_loss)
            
            # Compile the model with the specified learning rate
            compile_model(model=model, learning_rate=learning_rate, loss_function=loss_function)

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
#            futures.append(executor.submit(plot_results, log_dir))

            # Generate Pareto plot after each penalty threshold, so we can see it as soon as possible
            plot_pareto(scenario_dir_path)

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    plot_pareto(scenario_dir_path)

if __name__ == "__main__":
    main()
