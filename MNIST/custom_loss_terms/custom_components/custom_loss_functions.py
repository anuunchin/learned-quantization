import logging
import os
from typing import Tuple, Literal

import numpy as np
import tensorflow as tf

eps_float32 = np.finfo(np.float32).eps

log_dir = ""


def setup_logger(new_log_dir, logs):
    """
    Sets up a logger to save logs for specified files in the given directory.
    """
    global log_dir
    log_dir = new_log_dir
    os.makedirs(log_dir, exist_ok=True)

    if logs != []:
        for log in logs:
            total_loss_logger = tf.get_logger()
            total_loss_handler = logging.FileHandler(f"{new_log_dir}/{log}", mode="a")
            total_loss_handler.setFormatter(logging.Formatter("%(message)s"))
            total_loss_logger.addHandler(total_loss_handler)
            total_loss_logger.setLevel(logging.INFO)

            with open(f"{new_log_dir}/{log}", "w"):
                pass


class SCCEMaxBin:
    def __init__(self, layers, penalty_rate, log_dir, l2_lambda=0.01):
        self.layers = layers
        self.penalty_rate = tf.Variable(penalty_rate, trainable=False, dtype=tf.float32)
        self.custom_loss_dir = os.path.join(log_dir, "custom_losses")

        setup_logger(
            new_log_dir=self.custom_loss_dir,
            logs=[
                "total_loss.log",                
                "scce_loss.log",
                "maxbin_loss.log"]
        )

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy and 
        a penalty based on the number of bins calculated from the max weights divided by the quantization factor.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred
        )

        maxbin_penalty = self.compute_maxbin_penalty()

        total_loss = cross_entropy_loss + self.penalty_rate * maxbin_penalty

        tf.print(
            tf.reduce_mean(total_loss),
            output_stream=f"file://{self.custom_loss_dir}/total_loss.log",
        )
        tf.print(
            tf.reduce_mean(cross_entropy_loss),
            output_stream=f"file://{self.custom_loss_dir}/scce_loss.log",
        )
        tf.print(
            self.penalty_rate * maxbin_penalty,
            output_stream=f"file://{self.custom_loss_dir}/maxbin_loss.log",
        )

        return total_loss

    def compute_maxbin_penalty(self):
        """
        Computes the penalty based on the number of bins calculated from the max weights divided by the quantization factor.
        Effectively punishes large number of bins.
        """
        total_penalty = 0

        normalizer = 0

        for layer in self.layers:
            W = layer.W
            W_scale = layer.nested_q_w_layer.scale
            b = layer.b
            b_scale = layer.nested_q_b_layer.scale

            W_axes_to_reduce = [i for i in range(len(W_scale.shape)) if W_scale.shape[i] == 1 and len(W_scale.shape) > 1]
            if W_axes_to_reduce != []:
                W_maxbin = tf.reduce_max(tf.abs(W) / W_scale, axis=W_axes_to_reduce)
            else:
                W_maxbin = tf.reduce_max(tf.abs(W) / W_scale)

            b_axes_to_reduce = [i for i in range(len(b_scale.shape)) if b_scale.shape[i] == 1 and len(b_scale.shape) > 1]
            if b_axes_to_reduce != []:
                b_maxbin = tf.reduce_max(tf.abs(b) / b_scale, axis=b_axes_to_reduce)
            else:
                b_maxbin = tf.reduce_max(tf.abs(b) / b_scale)
            
            W_dim = 1.0
            for dim in W.shape:
                W_dim *= dim

            b_dim = 1.0
            for dim in b.shape:
                b_dim *= dim

            layer_penalty = tf.reduce_mean(W_maxbin) * W_dim + tf.reduce_mean(b_maxbin) * b_dim

            total_penalty += layer_penalty

            normalizer += W_dim + b_dim

        return total_penalty / normalizer


class SCCEDifference:
    def __init__(self, layers, penalty_rate, log_dir, l2_lambda=0.01):
        self.layers = layers
        self.penalty_rate = tf.Variable(penalty_rate, trainable=False, dtype=tf.float32)
        self.custom_loss_dir = os.path.join(log_dir, "custom_losses")

        setup_logger(
            new_log_dir=self.custom_loss_dir,
            logs=[
                "total_loss.log",                
                "scce_loss.log",
                "difference_loss.log",
                "maxbin_loss.log"]
        )

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy and a penalty based on the difference
        between the original and near quantized-scaled weights and biases.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred
        )

        difference_penalty = self.compute_difference_penalty()

        total_loss = cross_entropy_loss + self.penalty_rate *  difference_penalty

        tf.print(
            tf.reduce_mean(total_loss),
            output_stream=f"file://{self.custom_loss_dir}/total_loss.log",
        )
        tf.print(
            tf.reduce_mean(cross_entropy_loss),
            output_stream=f"file://{self.custom_loss_dir}/scce_loss.log",
        )
        tf.print(
            self.penalty_rate * difference_penalty,
            output_stream=f"file://{self.custom_loss_dir}/difference_loss.log",
        )
        return total_loss

    def compute_difference_penalty(self):
        """
        Computes the penalty based on the difference between the original and near quantized-scaled weights and biases.
        """
        total_penalty = 0

        normalizer = 0

        for layer in self.layers:
            W = layer.W
            W_scale = layer.nested_q_w_layer.scale
            b = layer.b
            b_scale = layer.nested_q_b_layer.scale

            W_quantized = W / W_scale
            b_quantized = b / b_scale

            W_diff_penalty = tf.reduce_mean(tf.abs(W - W_quantized))
            b_diff_penalty = tf.reduce_mean(tf.abs(b - b_quantized))

            W_dim = 1.0
            for dim in W.shape:
                W_dim *= dim

            b_dim = 1.0
            for dim in b.shape:
                b_dim *= dim

            layer_penalty = W_diff_penalty * W_dim + b_diff_penalty * b_dim

            total_penalty += layer_penalty

            normalizer += W_dim + b_dim

        return total_penalty / normalizer


class SCCEInverse:
    def __init__(self, layers, penalty_rate, log_dir, l2_lambda=0.01):
        self.layers = layers
        self.penalty_rate = tf.Variable(penalty_rate, trainable=False, dtype=tf.float32)
        self.custom_loss_dir = os.path.join(log_dir, "custom_losses")

        setup_logger(
            new_log_dir=self.custom_loss_dir,
            logs=[
                "total_loss.log",                
                "scce_loss.log",
                "inverse_loss.log"]
        )

    def compute_total_loss(self, y_true, y_pred):
        """
        Computes a combined loss that includes sparse categorical cross-entropy 
        and the inverse of the average of scaling factor values.
        """
        cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred
        )

        inverse_penalty = self.compute_inverse_penalty()

        total_loss = cross_entropy_loss + self.penalty_rate * inverse_penalty

        tf.print(
            tf.reduce_mean(total_loss),
            output_stream=f"file://{self.custom_loss_dir}/total_loss.log",
        )
        tf.print(
            tf.reduce_mean(cross_entropy_loss),
            output_stream=f"file://{self.custom_loss_dir}/scce_loss.log",
        )
        tf.print(
            self.penalty_rate * inverse_penalty,
            output_stream=f"file://{self.custom_loss_dir}/inverse_loss.log",
        )

        return total_loss

    def compute_inverse_penalty(self):
        """
        Computes the inverse of the average of scaling factor .
        Effectively punishes small scale factor values.
        """
        total_penalty = 0

        normalizer = 0

        for layer in self.layers:
            W = layer.W
            W_scale = layer.nested_q_w_layer.scale
            b = layer.b
            b_scale = layer.nested_q_b_layer.scale

            W_scale_non_zero = tf.where(W_scale == 0.0, eps_float32, W_scale)
            b_scale_non_zero = tf.where(b_scale == 0.0, eps_float32, b_scale)

            W_scale_inverse = tf.reduce_mean(1.0 / W_scale_non_zero)
            b_scale_inverse = tf.reduce_mean(1.0 / b_scale_non_zero)

            W_dim = 1.0
            for dim in W.shape:
                W_dim *= dim

            b_dim = 1.0
            for dim in b.shape:
                b_dim *= dim

            layer_penalty = W_scale_inverse * W_dim + b_scale_inverse * b_dim

            total_penalty += layer_penalty

            normalizer += W_dim + b_dim

        return total_penalty / normalizer
