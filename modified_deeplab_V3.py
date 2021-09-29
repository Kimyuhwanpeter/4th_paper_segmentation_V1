# -*- coding:utf-8 -*-
# https://github.com/rishizek/tensorflow-deeplab-v3/blob/master/deeplab_model.py
# https://kuklife.tistory.com/121
# https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_utils.py
# https://github.com/google-research/tf-slim/blob/master/tf_slim/layers/layers.py
# Random crop??
import tensorflow as tf

l2 = tf.keras.regularizers.l2(5e-4)

def block(inputs, filters=256, output_strides=16):

    input_size = tf.shape(inputs)[1:3]
    dilated_rate = [6, 12, 18]
    if output_strides == 8:
        dilated_rate = [2*rate for rate in dilated_rate]

    conv_1x1 = tf.keras.layers.DepthwiseConv2D(kernel_size=1, use_bias=True,
                                               depthwise_regularizer=l2)(inputs)
    conv_1x1 = tf.keras.layers.BatchNormalization()(conv_1x1)
    conv_1x1 = tf.keras.layers.ReLU()(conv_1x1)

    conv_3x3_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same",
                                        use_bias=True, dilation_rate=dilated_rate[0],
                                        kernel_regularizer=l2)(inputs)   # change depthwith with dilated??
    conv_3x3_1 = tf.keras.layers.BatchNormalization()(conv_3x3_1)
    conv_3x3_1 = tf.keras.layers.ReLU()(conv_3x3_1)

    conv_3x3_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same",
                                        use_bias=True, dilation_rate=dilated_rate[1],
                                        kernel_regularizer=l2)(inputs)   # change depthwith with dilated??
    conv_3x3_2 = tf.keras.layers.BatchNormalization()(conv_3x3_2)
    conv_3x3_2 = tf.keras.layers.ReLU()(conv_3x3_2)

    conv_3x3_3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same",
                                        use_bias=True, dilation_rate=dilated_rate[2],
                                        kernel_regularizer=l2)(inputs)   # change depthwith with dilated??
    conv_3x3_3 = tf.keras.layers.BatchNormalization()(conv_3x3_3)
    conv_3x3_3 = tf.keras.layers.ReLU()(conv_3x3_3)

    image_features = tf.reduce_mean(inputs, [1,2], keepdims=True)
    image_features = tf.keras.layers.Conv2D(filters=256, kernel_size=1,
                                            use_bias=True, kernel_regularizer=l2)(image_features)
    image_features = tf.keras.layers.BatchNormalization()(image_features)
    image_features = tf.keras.layers.ReLU()(image_features)
    image_features = tf.image.resize(image_features, input_size)

    h = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_features], -1)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=1, use_bias=True,
                                        depthwise_regularizer=l2)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    return h

def Deep_edge_network(input_shape=(513, 513, 3), num_classes=124):

    model = tf.keras.applications.ResNet50V2(input_shape=(513, 513, 3), include_top=False)

    output = model.output
    decode_outputs = model.get_layer("conv2_block1_1_conv").output
    encode_outputs = block(output)

    low_level_features = tf.keras.layers.Conv2D(filters=48, kernel_size=1, use_bias=True,
                                                kernel_regularizer=l2)(decode_outputs)
    low_level_features_size = tf.shape(low_level_features)[1:3]

    h = tf.image.resize(encode_outputs, low_level_features_size)
    h = tf.concat([h, low_level_features], -1)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same",
                               use_bias=True, kernel_regularizer=l2)(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same",
                               use_bias=True, kernel_regularizer=l2)(h)
    h = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1, padding="valid",
                               use_bias=True, kernel_regularizer=l2)(h)
    h = tf.image.resize(h, [input_shape[0], input_shape[1]])

    return tf.keras.Model(inputs=model.input, outputs=h)


