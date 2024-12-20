import argparse
import concurrent.futures
import logging
import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, 
    Flatten, 
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import custom_components.custom_callbacks as custom_callbacks
import custom_components.custom_layers as custom_layers
import utils.logs as logs
import utils.plot_scripts as plot_scripts

seed = 42
tf.random.set_seed(seed)


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
        if "custom_dense_layer" in layer.name:
            scale_tracking_callback = custom_callbacks.NestedScaleTrackingCallback(
                layer, log_dir
            )
            callbacks.append(scale_tracking_callback)

    accuracy_loss_callback = custom_callbacks.AccuracyLossTrackingCallBack(log_dir)
    callbacks.append(accuracy_loss_callback)
    return callbacks


def initialize_model(penalty_rate, input_shape=(28, 28, 1), seed=42):
    input_layer = Input(shape=input_shape)
    flatten_layer = Flatten()(input_layer)
    dense_layer = custom_layers.CustomDenseLayer(
        units=128, penalty_rate=penalty_rate, seed=seed
    )
    dense_layer_output = dense_layer(flatten_layer)
    dense_layer_activation = tf.keras.activations.relu(dense_layer_output)
    dense_layer_2 = Dense(10)
    dense_layer_2_output = dense_layer_2(dense_layer_activation)
    dense_layer_2_activation = tf.keras.activations.softmax(dense_layer_2_output)
    quantized_model = Model(inputs=input_layer, outputs=dense_layer_2_activation)
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

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
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

    learning_rate = 0.0003
    epochs = 5
    batch_size = 32

    penalty_rates = [
        #        1e-60
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
        1e-2,
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
        #        1e12
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for penalty_rate in penalty_rates:
            run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model = initialize_model(
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
