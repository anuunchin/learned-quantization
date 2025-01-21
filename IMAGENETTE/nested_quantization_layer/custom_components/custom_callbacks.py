import os

import numpy as np
import tensorflow as tf

eps_float32 = np.finfo(np.float32).eps


class NestedScaleTrackingCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer, log_dir):
        super(NestedScaleTrackingCallback, self).__init__()
        self.layer = layer
        self.qk_layer = layer.nested_q_k_layer
        self.qb_layer = layer.nested_q_b_layer
        self.k_scale = layer.nested_q_k_layer.scale
        self.b_scale = layer.nested_q_b_layer.scale

        kernel_name = layer.kernel.name.split("/")[-1].split(":")[0]
        kernel_scale_name = self.k_scale.name.split("/")[-1].split(":")[0]
        bias_name = layer.b.name.split("/")[-1].split(":")[0]
        bias_scale_name = self.b_scale.name.split("/")[-1].split(":")[0]

        os.makedirs(f"{log_dir}/on_epoch_end", exist_ok=True)
        self.log_file_path_kernel = f"{log_dir}/on_epoch_end/{kernel_name}_{layer.kernel.shape}.log"
        self.log_file_path_biases = f"{log_dir}/on_epoch_end/{bias_name}_{layer.b.shape}.log"
        
        self.log_file_path_k_scale = f"{log_dir}/on_epoch_end/{kernel_name}_{layer.kernel.shape}_{kernel_scale_name}_{self.k_scale.shape}.log"
        self.log_file_path_b_scale = f"{log_dir}/on_epoch_end/{bias_name}_{layer.b.shape}_{bias_scale_name}_{self.b_scale.shape}.log"

        self.log_file_path_qk_epoch = f"{log_dir}/on_epoch_end/Number_of_unique_{kernel_name}_{layer.kernel.shape}.log"
        self.log_file_path_qb_epoch = f"{log_dir}/on_epoch_end/Number_of_unique_{bias_name}_{layer.b.shape}.log"
        
        self.log_file_path_qk_epoch_max = f"{log_dir}/on_epoch_end/Max_{kernel_name}_{layer.kernel.shape}.log"
        self.log_file_path_qb_epoch_max = f"{log_dir}/on_epoch_end/Max_{bias_name}_{layer.b.shape}.log"

        os.makedirs(f"{log_dir}/on_train_end", exist_ok=True)
        self.log_file_path_qk = f"{log_dir}/on_train_end/Quantized_{kernel_name}_{layer.kernel.shape}_{kernel_scale_name}_{self.k_scale.shape}.log"
        self.log_file_path_qb = f"{log_dir}/on_train_end/Quantized_{bias_name}_{layer.b.shape}_{bias_scale_name}_{self.b_scale.shape}.log"
        self.log_file_path_qk_unique = f"{log_dir}/on_train_end/Unique_quantized_{kernel_name}_{layer.kernel.shape}_{kernel_scale_name}_{self.k_scale.shape}.log"
        self.log_file_path_qb_unique = f"{log_dir}/on_train_end/Unique_quantized_{bias_name}_{layer.b.shape}_{bias_scale_name}_{self.b_scale.shape}.log"

        os.makedirs(f"{log_dir}/on_train_begin", exist_ok=True)
        self.log_file_path_qk_initial = f"{log_dir}/on_train_begin/Initial_Quantized_{kernel_name}_{layer.kernel.shape}__{kernel_scale_name}__{self.k_scale.shape}.log"
        self.log_file_path_qb_initial = f"{log_dir}/on_train_begin/Initial_Quantized_{bias_name}_{layer.b.shape}_{bias_scale_name}_{self.b_scale.shape}.log"
        self.log_file_path_qk_initial_unique = f"{log_dir}/on_train_begin/Unique_initial_quantized_{kernel_name}_{layer.kernel.shape}_{kernel_scale_name}_{self.k_scale.shape}.log"
        self.log_file_path_qb_initial_unique = f"{log_dir}/on_train_begin/Unique_initial_quantized_{bias_name}_{layer.b.shape}_{bias_scale_name}_{self.b_scale.shape}.log"

    def on_epoch_end(self, epoch, logs=None):
        with open("./logs/temp/signs.log", "a") as file:
            file.write(f"Epoch {epoch}\n")

        # Track values in kernel at the end of each epoch
        kernel = self.layer.kernel.numpy().flatten()

        with open(self.log_file_path_kernel, "a") as file:
            file.write(f"Epoch {epoch}\n")
            for value in kernel:
                file.write(f"{value}\n")

        # Track values in bias vector at the end of each epoch
        biases = self.layer.b.numpy().flatten()

        with open(self.log_file_path_biases, "a") as file:
            file.write(f"Epoch {epoch}\n")
            for value in biases:
                file.write(f"{value}\n")

        # Track values in kernel scales at the end of each epoch
        kernel_scales = self.k_scale.numpy().flatten()

        with open(self.log_file_path_k_scale, "a") as file:
            file.write(f"Epoch {epoch}\n")
            for value in kernel_scales:
                file.write(f"{value}\n")

        # Track values in bias scales at the end of each epoch
        bias_scales = self.b_scale.numpy().flatten()

        with open(self.log_file_path_b_scale, "a") as file:
            file.write(f"Epoch {epoch}\n")
            for value in bias_scales:
                file.write(f"{value}\n")

        # Track unique quantized values in kernel at the end of each epoch
        inputs_quantized_nonrounded = self.layer.kernel / self.k_scale

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_rounded_flattened = inputs_quantized_rounded.numpy().flatten()

        unique_kernel, counts = np.unique(inputs_quantized_rounded_flattened, return_counts=True)

        unique_values_with_counts = list(zip(unique_kernel, counts))

        with open(self.log_file_path_qk_epoch, "a") as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{len(unique_values_with_counts)}\n")

        # Track maximum absolute unique quantized value in weight matrix at the end of each epoch
        max_abs_value = tf.reduce_max(tf.abs(inputs_quantized_rounded), axis=1)
        max_abs_value = max_abs_value.numpy().flatten()

        with open(self.log_file_path_qk_epoch_max, "a") as file:
            file.write(f"Epoch {epoch}\n")
            for val in max_abs_value:
                file.write(f"{val}\n")

        inputs_quantized_nonrounded = self.layer.b / self.b_scale

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_rounded_flattened = inputs_quantized_rounded.numpy().flatten()

        unique_b, counts = np.unique(inputs_quantized_rounded_flattened, return_counts=True)

        unique_values_with_counts = list(zip(unique_b, counts))

        # Track unique quantized values in bias at the end of each epoch
        with open(self.log_file_path_qb_epoch, "a") as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{len(unique_values_with_counts)}\n")

        # Track maximum absolute unique quantized value in bias at the end of each epoch
        max_abs_value = tf.reduce_max(tf.abs(inputs_quantized_rounded))
        max_abs_value = max_abs_value.numpy().flatten()

        with open(self.log_file_path_qb_epoch_max, "a") as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{max_abs_value[0]}\n")

    def on_train_end(self, logs=None):
        # Track quantized values in kernel at the end of training
        inputs_quantized_nonrounded = self.layer.kernel / self.k_scale

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_rounded_flattened = inputs_quantized_rounded.numpy().flatten()

        with open(self.log_file_path_qk, "a") as file:
            for value in inputs_quantized_rounded_flattened:
                file.write(f"{value}\n")

        # Track inique quantized values in kernel at the end of training
        unique_kernel, counts = np.unique(inputs_quantized_rounded_flattened, return_counts=True)
        unique_values_with_counts = list(zip(unique_kernel, counts))

        with open(self.log_file_path_qk_unique, "a") as file:
            for value, count in unique_values_with_counts:
                file.write(f"{value}, {count}\n")

        # Track quantized values in bias at the end of training
        inputs_quantized_nonrounded = self.layer.b / self.b_scale

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_rounded_flattened = inputs_quantized_rounded.numpy().flatten()

        with open(self.log_file_path_qb, "a") as file:
            for value in inputs_quantized_rounded_flattened:
                file.write(f"{value}\n")

        # Track inique quantized values in bias at the end of training
        unique_b, counts = np.unique(inputs_quantized_rounded_flattened, return_counts=True)
        unique_values_with_counts = list(zip(unique_b, counts))

        with open(self.log_file_path_qb_unique, "a") as file:
            for value, count in unique_values_with_counts:
                file.write(f"{value}, {count}\n")


    def on_train_begin(self, logs=None):
        # Track quantized values in kernel at the start of training
        inputs_quantized_nonrounded = self.layer.kernel / self.k_scale

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_rounded_flattened = inputs_quantized_rounded.numpy().flatten()

        with open(self.log_file_path_qk_initial, "a") as file:
            for value in inputs_quantized_rounded_flattened:
                file.write(f"{value}\n")

        # Track inique quantized values in kernel at the start of training
        unique_kernel, counts = np.unique(inputs_quantized_rounded_flattened, return_counts=True)
        unique_values_with_counts = list(zip(unique_kernel, counts))

        with open(self.log_file_path_qk_initial_unique, "a") as file:
            for value, count in unique_values_with_counts:
                file.write(f"{value}, {count}\n")

        # Track quantized values in bias at the start of training
        inputs_quantized_nonrounded = self.layer.b / self.b_scale

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_rounded_flattened = inputs_quantized_rounded.numpy().flatten()

        with open(self.log_file_path_qb_initial, "a") as file:
            for value in inputs_quantized_rounded_flattened:
                file.write(f"{value}\n")

        # Track inique quantized values in bias at the start of training
        unique_b, counts = np.unique(inputs_quantized_rounded_flattened, return_counts=True)
        unique_values_with_counts = list(zip(unique_b, counts))

        with open(self.log_file_path_qb_initial_unique, "a") as file:
            for value, count in unique_values_with_counts:
                file.write(f"{value}, {count}\n")


class AccuracyLossTrackingCallBack(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, accuracy_file="accuracy.log", loss_file="loss.log"):
        super(AccuracyLossTrackingCallBack, self).__init__()
        os.makedirs(f"{log_dir}/accuracy", exist_ok=True)
        self.accuracy_log_file_path = f"{log_dir}/accuracy/train_{accuracy_file}"
        self.val_accuracy_log_file_path = f"{log_dir}/accuracy/val_{accuracy_file}"

        os.makedirs(f"{log_dir}/loss", exist_ok=True)
        self.loss_log_file_path = f"{log_dir}/loss/train_{loss_file}"
        self.val_loss_log_file_path = f"{log_dir}/loss/val_{loss_file}"

    def on_epoch_end(self, epoch, logs=None):
        with open(self.val_accuracy_log_file_path, "a") as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{logs['val_accuracy']}\n")

        with open(self.val_loss_log_file_path, "a") as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{logs['val_loss']}\n")

        with open(self.accuracy_log_file_path, "a") as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{logs['accuracy']}\n")

        with open(self.loss_log_file_path, "a") as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{logs['loss']}\n")
