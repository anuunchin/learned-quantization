import logging
import os
from typing import Tuple, Literal, Type, Optional
from tensorflow.keras.initializers import Initializer, RandomNormal
from tensorflow.keras.regularizers import Regularizer

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
            total_loss_handler = logging.FileHandler(f'{new_log_dir}/{log}', mode='a')
            total_loss_handler.setFormatter(logging.Formatter('%(message)s'))
            total_loss_logger.addHandler(total_loss_handler)
            total_loss_logger.setLevel(logging.INFO)

            with open(f'{new_log_dir}/{log}', 'w'):
                pass


class MinValueConstraint(tf.keras.constraints.Constraint):
    """
    Ensures the scale factor values stay above a defined minimum value.
    """
    def __init__(self, min_value):
        self.min_value = min_value

    def __call__(self, w):
        return tf.maximum(w, self.min_value)

    def get_config(self):
        return {"min_value": self.min_value}


@tf.custom_gradient
def my_custom_gradient(parameter, scale, penalty_threshold):
    """
    Custom gradient function that penalizes large ratios of gradients 
    to parameters based on a penalty threshold.
    """
    penalty_threshold = tf.stop_gradient(penalty_threshold)
    inputs_quantized_nonrounded = (
        parameter / scale
    )  
    inputs_quantized_rounded = tf.stop_gradient(tf.floor(inputs_quantized_nonrounded))
    inputs_quantized_scaled_back = inputs_quantized_rounded * scale

    def custom_grad(dy, variables=None):        
        # TODO: add threshold based scale gradient calculation
        return dy, tf.zeros_like(scale), None

    return inputs_quantized_scaled_back, custom_grad


class CustomQuantizedScaleLayer(tf.keras.layers.Layer):
    """
    Nested layer that applies a custom quantized scaling operation with a gradient-based penalty threshold mechanism.
    """
    def __init__(self, penalty_threshold, initializer, orientation):
        super(CustomQuantizedScaleLayer, self).__init__()
        self.initializer = initializer
        self.orientation = orientation
        self.penalty_threshold = penalty_threshold

        setup_logger(
            "logs/temp",
            [
                "call_values.log",
                "scale_grads.log",
                "scale_grads_abs.log",
                "maxvalue.log",
                "counts_in_bias.log",
                "neg_grads.log",
                "pos_grads.log",
                "zero_grads.log",
            ],
        )

    def build(self, input_shape):

        if self.orientation == "rowwise":
            scaler_shape = tuple(
                input_shape[i] if i == 0 else 1 for i in range(len(input_shape))
            )
            self.scale = self.add_weight(
                name="Rowwise-scaler",
                shape=scaler_shape,
                initializer=tf.keras.initializers.Constant(eps_float32 * 100),
                trainable=True,
                constraint=MinValueConstraint(eps_float32 * 100),
            )

        elif self.orientation == "columnwise":
            scaler_shape = tuple(
                input_shape[i] if i == 1 else 1 for i in range(len(input_shape))
            )
            self.scale = self.add_weight(
                name="Columnwise-scaler",
                shape=scaler_shape,
                initializer=tf.keras.initializers.Constant(eps_float32 * 100),
                trainable=True,
                constraint=MinValueConstraint(eps_float32 * 100),
            )

        elif self.orientation == "channelwise":
            scaler_shape = tuple(
                input_shape[i] if i == 2 else 1 for i in range(len(input_shape))
            )
            self.scale = self.add_weight(
                name="Columnwise-scaler",
                shape=scaler_shape,
                initializer=tf.keras.initializers.Constant(eps_float32 * 100),
                trainable=True,
                constraint=MinValueConstraint(eps_float32 * 100),
            )

        elif self.orientation == "scalar":
            self.scale = self.add_weight(
                name="Scalar-scaler",
                shape=(1,),
                initializer=tf.keras.initializers.Constant(eps_float32 * 100),
                trainable=True,
                constraint=MinValueConstraint(eps_float32 * 100),
            )

        else:
            raise ValueError(
                f"Invalid scaler application: {self.orientation}. Expected rowwise, columnwise or scalar."
            )

    def call(self, inputs):
        return my_custom_gradient(inputs, self.scale, self.penalty_threshold)


class CustomDenseLayer(tf.keras.layers.Layer):
    """
    Standard dense layer with nested quantization layers for weights and bias.
    """
    def __init__(
        self,
        seed: int, 
        units: int, 
        penalty_threshold: float, 
        orientation: Literal["rowwise", "columnwise", "scalar"],
        initializer: Type[Initializer],
        name: str,
        regularizer: Optional[Type[Regularizer]],
        trained_weights=None,
        **kwargs
    ):
        super(CustomDenseLayer, self).__init__()
        self.seed = seed

        self.nested_q_w_layer = CustomQuantizedScaleLayer(
            penalty_threshold=penalty_threshold, initializer=None, orientation=orientation
        )
        self.nested_q_b_layer = CustomQuantizedScaleLayer(
            penalty_threshold=penalty_threshold, initializer=None, orientation="scalar" 
        )

        self.units = units

        self.initializer = initializer

        self.regularizer = regularizer

        self.trained_weights = trained_weights

    def build(self, input_shape):

        # If weights are provided, set them, otherwise use the initializer
        if self.trained_weights:
            weight_initializer = tf.constant_initializer(self.trained_weights[0])
            bias_initializer = tf.constant_initializer(self.trained_weights[1])
        else:
            weight_initializer = self.initializer
            bias_initializer = self.initializer

        self.W = self.add_weight(
            name="Weights",
            shape=(input_shape[-1], self.units),
            initializer=weight_initializer,
            regularizer=self.regularizer,
            trainable=True,
        )

        self.b = self.add_weight(
            name="Bias",
            shape=(self.units,),
            initializer=bias_initializer,
            regularizer=self.regularizer,
            trainable=True,
        )

    def call(self, inputs):

        qw = self.nested_q_w_layer(self.W)
        qb = self.nested_q_b_layer(self.b)

        return tf.add(tf.matmul(inputs, qw), qb)


class CustomConv2DLayer(tf.keras.layers.Layer):
    """
    Standard convolutional layer with nested quantization layers for kernel and bias.
    """
    def __init__(
        self,
        seed: int,
        penalty_threshold: float,
        orientation: Literal["rowwise", "columnwise", "channelwise", "scalar"],
        initializer: Type[Initializer],
        filters: int,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int],
        padding: str,
        name: str,
        regularizer: Optional[Type[Regularizer]],
        trained_weights=None,
        **kwargs,
    ):
        super(CustomConv2DLayer, self).__init__(**kwargs)
        self.seed = seed

        self.nested_q_k_layer = CustomQuantizedScaleLayer(
            penalty_threshold=penalty_threshold, initializer=None, orientation=orientation
        )
        self.nested_q_b_layer = CustomQuantizedScaleLayer(
            penalty_threshold=penalty_threshold, initializer=None, orientation="scalar"
        )

        self.initializer = initializer

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()  

        self.regularizer = regularizer

        self.trained_weights = trained_weights

    def build(self, input_shape):

        # If weights are provided, set them, otherwise use the initializer
        if self.trained_weights:
            kernel_initializer = tf.constant_initializer(self.trained_weights[0])
            bias_initializer = tf.constant_initializer(self.trained_weights[1])
        else:
            kernel_initializer = self.initializer
            bias_initializer = self.initializer
            
        kernel_shape = (*self.kernel_size, input_shape[-1], self.filters)

        self.kernel = self.add_weight(
            name=f"Kernel_{self.name}",
            shape=kernel_shape,
            initializer=kernel_initializer,
            regularizer=self.regularizer,
            trainable=True,
        )
        self.b = self.add_weight(
            name=f"Bias_{self.name}",
            shape=(self.filters,),
            initializer=bias_initializer,
            regularizer=self.regularizer,
            trainable=True,
        )

    def call(self, inputs):
        # Perform convolution
        qk = self.nested_q_k_layer(self.kernel)
        qb = self.nested_q_b_layer(self.b)

        conv_output = tf.nn.conv2d(
            inputs,
            filters=qk,
            strides=self.strides,
            padding=self.padding,
        )

        return tf.add(conv_output, qb)


class CustomConv2DLayerNoBias(tf.keras.layers.Layer):
    """
    Standard convolutional layer with nested quantization layers for kernel.
    """
    def __init__(
        self,
        seed: int,
        penalty_threshold: float,
        orientation: Literal["rowwise", "columnwise", "channelwise", "scalar"],
        initializer: Type[Initializer],
        filters: int,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int],
        padding: str,
        name: str,
        regularizer: Optional[Type[Regularizer]],
        trained_weights=None,
        **kwargs,
    ):
        super(CustomConv2DLayerNoBias, self).__init__(**kwargs)
        self.seed = seed

        self.nested_q_k_layer = CustomQuantizedScaleLayer(
            penalty_threshold=penalty_threshold, initializer=None, orientation=orientation
        )

        self.initializer = initializer

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()  

        self.regularizer = regularizer

        self.trained_weights = trained_weights

    def build(self, input_shape):

        # If weights are provided, set them, otherwise use the initializer
        if self.trained_weights:
            kernel_initializer = tf.constant_initializer(self.trained_weights[0])
        else:
            kernel_initializer = self.initializer
            
        kernel_shape = (*self.kernel_size, input_shape[-1], self.filters)

        self.kernel = self.add_weight(
            name=f"Kernel_{self.name}",
            shape=kernel_shape,
            initializer=kernel_initializer,
            regularizer=self.regularizer,
            trainable=True,
        )

    def call(self, inputs):
        # Perform convolution
        qk = self.nested_q_k_layer(self.kernel)

        conv_output = tf.nn.conv2d(
            inputs,
            filters=qk,
            strides=self.strides,
            padding=self.padding,
        )

        return conv_output
