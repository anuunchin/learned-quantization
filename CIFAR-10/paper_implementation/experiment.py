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
from tensorflow.keras.callbacks import (
    Callback,
    ReduceLROnPlateau,
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import (
    Adam,
    SGD,
)

import custom_components.custom_callbacks as custom_callbacks
import custom_components.custom_layers as custom_layers
import utils.log_scripts as log_scripts
import utils.plot_scripts as plot_scripts

import numpy as np
from tensorflow import keras
from resnet import resnet18

from configs import SEED

os.environ["TF_DETERMINISTIC_OPS"] = "1"

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
    """    
    callbacks = []

    for i, layer in enumerate(model.layers):
        if "custom_conv2d_layer" in layer.name:
            scale_tracking_callback = custom_callbacks.NestedScaleTrackingCallback(
                layer, log_dir
            )
            callbacks.append(scale_tracking_callback)

    lr_callback = ReduceLROnPlateau(
        monitor="val_accuracy",  
        factor=0.1,
        patience=3,
        threshold=0.001,
        mode="max",
        verbose=1
    )
    callbacks.append(lr_callback)

    accuracy_loss_callback = custom_callbacks.AccuracyLossTrackingCallBack(log_dir)
    callbacks.append(accuracy_loss_callback)
    return callbacks


def initialize_resnet18(
    penalty_threshold: float,
    seed: int,
    orientation: Literal["rowwise", "columnwise", "channelwise", "scalar"],
    input_shape: Tuple[int, ...] = (32, 32, 3),
    num_classes: int = 10,
) -> Model:
    """
    Initializes the resnet18 model, configured using 
    a specified penalty threshold, orientation (quantization granularity), and specified random seed.
    """
    use_float64 = True
    training=True

    inputs = Input(shape=input_shape)
    outputs = resnet18(inputs, num_classes=10)
    model = Model(inputs, outputs, name="renset18_cifar")
    return model


def compile_model(
    model: Model, 
    learning_rate: float,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    nesterov: bool = True,
) -> None:
    model.compile(
        optimizer=SGD(learning_rate=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov),
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
        help="Random seed for reproducability",
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
    orientation=args.orientation
    training = args.training
    scenario_dir = f"{orientation}_{training}"

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

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

    learning_rate = 0.1
    epochs = 100
    batch_size = 128

    penalty_thresholds = [
        0.0,
    ]

    # for model fine-tuning https://wandb.ai/suvadeep/pytorch/reports/Finetuning-of-ResNet-18-on-CIFAR-10-Dataset--VmlldzoxMDE2NjQ1

    # Create a thread pool for concurrent execution so that we can plot the previous
    # training logs, while continuing with the next one
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for penalty_threshold in penalty_thresholds:

            # Initialize model with or without pre-trained weights
            if training == "post_training":
                raise NotImplementedError("Post training scenario not implemented. Please run from scratch.")
            else:
                model = initialize_resnet18(
                    penalty_threshold=penalty_threshold,
                    seed=seed,
                    orientation=orientation,
                    input_shape=input_data_shape,
                    num_classes=10,
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
            #log_scripts.save_compress_parameters(model=model, log_dir=log_dir)
            #current_script_dir = os.path.dirname(__file__)
            #log_scripts.save_artefacts(scenario_dir_path, current_script_dir)

            # Submit plotting task to executor
            #futures.append(executor.submit(plot_results, log_dir))

            # Generate Pareto plot after each penalty threshold, so we can see it as soon as possible
            #plot_pareto(scenario_dir_path)

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()