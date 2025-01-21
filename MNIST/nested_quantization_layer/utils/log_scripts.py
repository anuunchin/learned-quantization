import os
import shutil
import zipfile
from typing import Optional

import numpy as np
from tensorflow.keras.models import Model


def log_model_structure(
    model: Model, 
    folder_name: str, 
    filename: Optional[str] = "model_structure.log"
):
    """
    Logs the structure of the given model, including each layer's input and output shapes.
    If a layer has scalers, their shapes are also logged.
    """
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, filename)

    with open(file_path, "w") as f:
        f.write("\n" + "-" * 80 + "\n")
        f.write("MODEL STRUCTURE\n")

        for i, layer in enumerate(model.layers):
            f.write(f"\nLAYER {i}: {layer}\n")
            f.write(f"  - Input Shape: {layer.input}\n")
            f.write(f"  - Output Shape: {layer.output}\n")

            if hasattr(layer, "b"):
                f.write(f"  - Bias with shape: {layer.b.shape}\n")

            if hasattr(layer, "nested_q_b_layer"):
                f.write(
                    f"  - Bias scaler with shape: {layer.nested_q_b_layer.scale.shape}\n"
                )

            if hasattr(layer, "w"):
                f.write(f"  - Weight matrix with shape: {layer.W.shape}\n")

            if hasattr(layer, "W"):
                f.write(f"  - Weight matrix with shape: {layer.W.shape}\n")

            if hasattr(layer, "nested_q_w_layer"):
                f.write(
                    f"  - Weight matrix scaler with shape: {layer.nested_q_w_layer.scale.shape}\n"
                )

            if hasattr(layer, "kernel"):
                f.write(f"  - Kernel with shape: {layer.kernel.shape}\n")

            if hasattr(layer, "nested_q_k_layer"):
                f.write(
                    f"  - Kernel scaler with shape: {layer.nested_q_k_layer.scale.shape}\n"
                )

        f.write("-" * 80 + "\n")


def save_compress_parameters(model: Model, log_dir: str) -> None:
    """
    Saves model weights, quantizes if applicable, 
    compresses them, and logs file sizes.
    """
    weights_path = os.path.join(log_dir, "weights.npy")

    weights = {}
    for layer in model.layers:
        if "custom_dense_layer" in layer.name:
            weights[layer.name + "/W"] = np.floor(
                layer.W.numpy() / layer.nested_q_w_layer.scale.numpy()
            ).astype(np.int8)
            weights[layer.name + "/b"] = np.floor(
                layer.b.numpy() / layer.nested_q_b_layer.scale.numpy()
            ).astype(np.int8)
        elif "dense" in layer.name:
            weights[layer.name + "/W"] = layer.kernel.numpy()
            weights[layer.name + "/b"] = layer.bias.numpy()
    np.save(weights_path, weights)

    # Compress quantized weights
    zip_file_path = os.path.join(log_dir, "weights.zip")
    with zipfile.ZipFile(zip_file_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(weights_path, arcname="weights.npy")

    # Get sizes
    size = os.path.getsize(weights_path) / (1024 * 1024)
    zip_size = os.path.getsize(zip_file_path) / (1024 * 1024)

    size_log_path = os.path.join(log_dir, "file_sizes.log")
    with open(size_log_path, "w") as log_file:
        log_file.write(f"Weights size: {size:.4f} MB\n")
        log_file.write(f"Compressed weights size: {zip_size:.4f} MB\n")


def save_artefacts(log_scenario_dir: str, current_script_dir: str) -> None:
    """
    Saves Python scripts, including custom components and the current script, 
    as artefacts in the specified scenario directory.
    """
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
