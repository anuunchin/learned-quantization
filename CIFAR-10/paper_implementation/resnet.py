# Adapted from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py
import math

from tensorflow.keras.initializers import (
    VarianceScaling,
    RandomUniform,
)
from tensorflow.keras.layers import (
    ZeroPadding2D, 
    Conv2D,
    BatchNormalization,
    ReLU,
    MaxPool2D,
    Dense, 
    GlobalAveragePooling2D,
    Add,
)
from configs import (
    SEED,
    PENALTY_THRESHOLD,
    ORIENTATION,
)
from custom_components.custom_layers import (
    CustomConv2DLayerNoBias,
    CustomDenseLayer,
)

kaiming_normal = VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal', seed=SEED)

def conv3x3(x, out_planes, stride=1, name=None):
    x = ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return CustomConv2DLayerNoBias(
        seed=SEED,
        penalty_threshold=PENALTY_THRESHOLD,
        orientation=ORIENTATION,
        initializer=kaiming_normal,
        filters=out_planes,
        kernel_size=(3, 3),
        strides=stride,
        padding="valid",
        name=name,
        regularizer=None,
        trained_weights=None,
    )(x)
    #uncomment to use the default Conv2D
    #return Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = Add(name=f'{name}.add')([identity, out])
    out = ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            CustomConv2DLayerNoBias(
                seed=SEED,
                penalty_threshold=PENALTY_THRESHOLD,
                orientation=ORIENTATION,
                initializer=kaiming_normal,
                filters=planes,
                kernel_size=(1,1),
                strides=stride,
                padding="valid",
                name=f'{name}.0.downsample.0',
                regularizer=None,
                trained_weights=None,
            ),
            #uncomment to use the default Conv2D
            #Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, num_classes=1000):
    x = ZeroPadding2D(padding=3, name='conv1_pad')(x)
    x = CustomConv2DLayerNoBias(
        seed=SEED,
        penalty_threshold=PENALTY_THRESHOLD,
        orientation=ORIENTATION,
        initializer=kaiming_normal,
        filters=64,
        kernel_size=(7, 7),
        strides=(2,2),
        padding="valid",
        name="conv1",
        regularizer=None,
        trained_weights=None,

    )(x)
    #uncomment to use the default Conv2D
    #x = Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = ReLU(name='relu1')(x)
    x = ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = GlobalAveragePooling2D(name='avgpool')(x)
    initializer = RandomUniform(minval=-1.0 / math.sqrt(512), maxval=1.0 / math.sqrt(512), seed=SEED)
    x = CustomDenseLayer(
        seed=SEED,
        units=num_classes,
        penalty_threshold=PENALTY_THRESHOLD,
        orientation=ORIENTATION,
        initializer=initializer,
        name="fc",
        regularizer=None,
    )(x)
    #uncomment to use the default Dense
    #x = Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)
    return x

def resnet18(x, **kwargs):
    return resnet(x, [2, 2, 2, 2], **kwargs)
