import tensorflow as tf
import numpy as np
import os


eps_float32 = np.finfo(np.float32).eps


class QuantizedDataTrackingCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer, log_dir):
        super(QuantizedDataTrackingCallback, self).__init__()
        self.layer = layer
        self.log_file_path_weights = f"{log_dir}/{layer.__class__.__name__}_scale_{layer.scale.shape}.log"

    def on_epoch_end(self, epoch, logs=None):
        weights = self.layer.scale.numpy().flatten()

        with open(self.log_file_path_weights, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in weights:
                file.write(f"{value}\n")


class NestedScaleTrackingCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer, log_dir):
        super(NestedScaleTrackingCallback, self).__init__()
        self.layer = layer
        self.qw_layer = layer.nested_q_w_layer
        self.qb_layer = layer.nested_q_b_layer
        self.w_scale = layer.nested_q_w_layer.scale
        self.b_scale = layer.nested_q_b_layer.scale
        
        os.makedirs(f"{log_dir}/on_epoch_end", exist_ok=True)
        self.log_file_path_weights = f"{log_dir}/on_epoch_end/{layer.W.name}_{layer.W.shape}.log"
        self.log_file_path_biases = f"{log_dir}/on_epoch_end/{layer.b.name}_{layer.b.shape}.log"

        self.log_file_path_w_scale = f"{log_dir}/on_epoch_end/{layer.W.name}_{layer.W.shape}_{self.w_scale.name}_{self.w_scale.shape}.log"
        self.log_file_path_b_scale = f"{log_dir}/on_epoch_end/{layer.b.name}_{layer.b.shape}_{self.b_scale.name}_{self.b_scale.shape}.log"

        self.log_file_path_qw_epoch = f"{log_dir}/on_epoch_end/Number_of_unique_{layer.W.name}_{layer.W.shape}.log"
        self.log_file_path_qb_epoch = f"{log_dir}/on_epoch_end/Number_of_unique_{layer.b.name}_{layer.b.shape}.log"

        self.log_file_path_qw_epoch_max = f"{log_dir}/on_epoch_end/Max_{layer.W.name}_{layer.W.shape}.log"
        self.log_file_path_qb_epoch_max = f"{log_dir}/on_epoch_end/Max_{layer.b.name}_{layer.b.shape}.log"

        os.makedirs(f"{log_dir}/on_train_end", exist_ok=True)
        self.log_file_path_wq = f"{log_dir}/on_train_end/Quantized_{layer.W.name}_{layer.W.shape}_{self.w_scale.name}_{self.w_scale.shape}.log"
        self.log_file_path_bq = f"{log_dir}/on_train_end/Quantized_{layer.b.name}_{layer.b.shape}_{self.b_scale.name}_{self.b_scale.shape}.log"
        self.log_file_path_wq_unique = f"{log_dir}/on_train_end/Unique_quantized_{layer.W.name}_{layer.W.shape}_{self.w_scale.name}_{self.w_scale.shape}.log"
        self.log_file_path_bq_unique = f"{log_dir}/on_train_end/Unique_quantized_{layer.b.name}_{layer.b.shape}_{self.b_scale.name}_{self.b_scale.shape}.log"

        os.makedirs(f"{log_dir}/on_train_begin", exist_ok=True)
        self.log_file_path_wq_initial = f"{log_dir}/on_train_begin/Initial_Quantized_{layer.W.name}_{layer.W.shape}_{self.w_scale.name}_{self.w_scale.shape}.log"
        self.log_file_path_bq_initial = f"{log_dir}/on_train_begin/Initial_Quantized_{layer.b.name}_{layer.b.shape}_{self.b_scale.name}_{self.b_scale.shape}.log"
        self.log_file_path_wq_initial_unique = f"{log_dir}/on_train_begin/Unique_initial_quantized_{layer.W.name}_{layer.W.shape}_{self.w_scale.name}_{self.w_scale.shape}.log"
        self.log_file_path_bq_initial_unique = f"{log_dir}/on_train_begin/Unique_initial_quantized_{layer.b.name}_{layer.b.shape}_{self.b_scale.name}_{self.b_scale.shape}.log"
        
    def on_epoch_end(self, epoch, logs=None):
        with open("./logs/test/signs.log", 'a') as file:
            file.write(f"Epoch {epoch}\n")

        # Track values in weight matrix at the end of each epoch
        weights = self.layer.W.numpy().flatten()

        with open(self.log_file_path_weights, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in weights:
                file.write(f"{value}\n")

        # Track values in bias vector at the end of each epoch
        biases = self.layer.b.numpy().flatten()

        with open(self.log_file_path_biases, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in biases:
                file.write(f"{value}\n")

        # Track values in weight scales at the end of each epoch
        weight_scales = self.w_scale.numpy().flatten()

        with open(self.log_file_path_w_scale, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in weight_scales:
                file.write(f"{value}\n")

        # Track values in bias scales at the end of each epoch
        bias_scales = self.b_scale.numpy().flatten()

        with open(self.log_file_path_b_scale, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in bias_scales:
                file.write(f"{value}\n")


        inputs_quantized_nonrounded = self.layer.W / self.w_scale  

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_scaled_back = inputs_quantized_rounded.numpy().flatten()

        unique_weights, counts = np.unique(inputs_quantized_scaled_back, return_counts=True)

        unique_values_with_counts = list(zip(unique_weights, counts))

        with open(self.log_file_path_qw_epoch, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{len(unique_values_with_counts)}\n")

        max_abs_value = tf.reduce_max(tf.abs(inputs_quantized_rounded), axis=1)
        max_abs_value = max_abs_value.numpy().flatten()

        with open(self.log_file_path_qw_epoch_max, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for val in max_abs_value:
                file.write(f"{val}\n")


        inputs_quantized_nonrounded = self.layer.b / self.b_scale  

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_scaled_back = inputs_quantized_rounded.numpy().flatten()

        unique_weights, counts = np.unique(inputs_quantized_scaled_back, return_counts=True)

        unique_values_with_counts = list(zip(unique_weights, counts))

        with open(self.log_file_path_qb_epoch, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{len(unique_values_with_counts)}\n")

        max_abs_value = tf.reduce_max(tf.abs(inputs_quantized_rounded))
        max_abs_value = max_abs_value.numpy().flatten()

        with open(self.log_file_path_qb_epoch_max, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{max_abs_value[0]}\n")


    def on_train_end(self,  logs=None):
        # Track floored quantized values in weight matrix at the end of training
        inputs_quantized_nonrounded = self.layer.W / self.w_scale  

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_scaled_back = inputs_quantized_rounded.numpy().flatten()

        with open(self.log_file_path_wq, 'a') as file:
            for value in inputs_quantized_scaled_back:
                file.write(f"{value}\n")

        unique_weights, counts = np.unique(inputs_quantized_scaled_back, return_counts=True)
        unique_values_with_counts = list(zip(unique_weights, counts))

        # Track unique floored quantized values in weight matrix at the end of training
        with open(self.log_file_path_wq_unique, 'a') as file:
            for value, count in unique_values_with_counts:
                file.write(f"{value}, {count}\n")

        # Track floored quantized values in bias vector at the end of training
        inputs_quantized_nonrounded = self.layer.b / self.b_scale  

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_scaled_back = inputs_quantized_rounded.numpy().flatten()

        with open(self.log_file_path_bq, 'a') as file:
            for value in inputs_quantized_scaled_back:
                file.write(f"{value}\n")

        unique_weights, counts = np.unique(inputs_quantized_scaled_back, return_counts=True)
        unique_values_with_counts = list(zip(unique_weights, counts))

        # Track unique floored quantized values in bias vector at the end of training
        with open(self.log_file_path_bq_unique, 'a') as file:
            for value, count in unique_values_with_counts:
                file.write(f"{value}, {count}\n")

    def on_train_begin(self,  logs=None):
        # Track floored quantized values in weight matrix at the start of training
        inputs_quantized_nonrounded = self.layer.W / self.w_scale  

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_scaled_back = inputs_quantized_rounded.numpy().flatten()

        with open(self.log_file_path_wq_initial, 'a') as file:
            for value in inputs_quantized_scaled_back:
                file.write(f"{value}\n")

        unique_weights, counts = np.unique(inputs_quantized_scaled_back, return_counts=True)
        unique_values_with_counts = list(zip(unique_weights, counts))

        # Track unique floored quantized values in weight matrix at the start of training
        with open(self.log_file_path_wq_initial_unique, 'a') as file:
            for value, count in unique_values_with_counts:
                file.write(f"{value}, {count}\n")

        # Track floored quantized values in bias vector at the start of training
        inputs_quantized_nonrounded = self.layer.b / self.b_scale 

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_scaled_back = inputs_quantized_rounded.numpy().flatten()

        with open(self.log_file_path_bq_initial, 'a') as file:
            for value in inputs_quantized_scaled_back:
                file.write(f"{value}\n")

        unique_weights, counts = np.unique(inputs_quantized_scaled_back, return_counts=True)
        unique_values_with_counts = list(zip(unique_weights, counts))

        # Track unique floored quantized values in bias vector at the start of training
        with open(self.log_file_path_bq_initial_unique, 'a') as file:
            for value, count in unique_values_with_counts:
                file.write(f"{value}, {count}\n")


class AccuracyLossTrackingCallBack(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, accuracy_file = "accuracy.log", loss_file = "loss.log"):
        super(AccuracyLossTrackingCallBack, self).__init__()
        os.makedirs(f"{log_dir}/accuracy", exist_ok=True)
        self.accuracy_log_file_path = f"{log_dir}/accuracy/train_{accuracy_file}"
        self.val_accuracy_log_file_path = f"{log_dir}/accuracy/val_{accuracy_file}"

        os.makedirs(f"{log_dir}/loss", exist_ok=True)
        self.loss_log_file_path = f"{log_dir}/loss/train_{loss_file}"
        self.val_loss_log_file_path = f"{log_dir}/loss/val_{loss_file}"

    def on_epoch_end(self, epoch, logs=None):
        with open(self.val_accuracy_log_file_path, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{logs['val_accuracy']}\n")

        with open(self.val_loss_log_file_path, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{logs['val_loss']}\n")

        with open(self.accuracy_log_file_path, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{logs['accuracy']}\n")

        with open(self.loss_log_file_path, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            file.write(f"{logs['loss']}\n")


