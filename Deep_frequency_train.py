# -*- coding:utf-8 -*-
from modified_deeplab_V3 import *
from cal_measurement import Measurement
from random import shuffle

import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 256,
                           
                           "label_path": "D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/labels/",
                           
                           "image_path": "D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/images/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 300,

                           "total_classes": 12,

                           "ignore_label": 11,

                           "batch_size": 4,

                           "train": True})


total_step = len(os.listdir(FLAGS.image_path)) // FLAGS.batch_size
warmup_step = int(total_step * 0.6)
power = 1.

lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate = FLAGS.lr,
    decay_steps = total_step - warmup_step,
    end_learning_rate = FLAGS.min_lr,
    power = power
)
lr_schedule = LearningRateScheduler(FLAGS.lr, warmup_step, lr_scheduler)

optim = tf.keras.optimizers.Adam(FLAGS.lr)

# pretrained 모델을 사용해보자
# https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
# https://github.com/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow2.0
# https://www.youtube.com/watch?v=kgkyu7LpBaM
def tr_func(image_list, label_list):

    flip = tf.random.uniform(shape=[1,], minval=0, maxval=2, dtype=tf.int32)[0]

    img = tf.io.read_file(image_list)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    #img = tf.image.random_brightness(img, max_delta=50.)
    #img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    #img = tf.image.random_hue(img, max_delta=0.2)
    #img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    #img = tf.clip_by_value(img, 0, 255)
    #img = tf.case([
    #    (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
    #], default=lambda: img)
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #lab = tf.case([
    #    (tf.greater(flip, 0), lambda: tf.image.flip_left_right(lab))
    #], default=lambda: lab)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, images, labels, ignore_count):

    with tf.GradientTape() as tape:

        batch_labels = tf.reshape(labels, [-1])
        indices = tf.squeeze(tf.where(tf.not_equal(batch_labels, FLAGS.ignore_label)),1)
        # indices 이 부분을 확인해봐야 할것같음
        batch_labels = tf.cast(tf.gather(batch_labels, indices), tf.int32)

        logits = run_model(model, images, True)
        raw_logits = tf.reshape(logits, [-1, FLAGS.total_classes-1])  # -1을 해주어야 하는건가!?
        predict = tf.gather(raw_logits, indices)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(batch_labels, predict)
        #loss = tf.reduce_sum(loss, [1,2]) / ignore_count
        #loss = tf.reduce_mean(loss)
        

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def main():
    tf.keras.backend.clear_session()
    #model = get_model(FLAGS.img_size, FLAGS.total_classes-1)
    model = DeepLabV3Plus(FLAGS.img_size, FLAGS.img_size, 34)
    #model = Deep_edge_network(input_shape=(FLAGS.img_size,FLAGS.img_size,3), num_classes=FLAGS.total_classes-1)
    model.summary()
    out = model.get_layer("activation_decoder_2_upsample").output
    out = tf.keras.layers.Conv2D(FLAGS.total_classes-1, (1,1), name="output_layer")(out)
    model = tf.keras.Model(inputs=model.input, outputs=out)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        #elif isinstance(layer, tf.keras.layers.Conv2D):
        #    layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    
    if FLAGS.train:
        count = 0

        image_dataset = os.listdir(FLAGS.image_path)
        label_dataset = os.listdir(FLAGS.label_path)

        train_img_dataset = []
        train_lab_dataset = []
        for i in range(len(image_dataset)):
            for j in range(len(label_dataset)):
                if image_dataset[i] == label_dataset[j]:
                    train_img_dataset.append(FLAGS.image_path + image_dataset[i])
                    train_lab_dataset.append(FLAGS.label_path + label_dataset[j])

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(train_ge)
            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                #batch_labels = tf.where(batch_labels == FLAGS.ignore_label, 255, batch_labels) # void를 255로 할시
                ignore_count = tf.reshape(batch_labels, [FLAGS.batch_size,FLAGS.img_size*FLAGS.img_size,])
                ignore_count_buf = []
                for b in range(FLAGS.batch_size):
                    ignore_c = np.bincount(ignore_count[b].numpy(), minlength=FLAGS.total_classes)
                    ignore_count_buf.append(np.sum(ignore_c) - ignore_c[-1])

                ignore_count_buf = tf.convert_to_tensor(ignore_count_buf, dtype=tf.float32)

                loss = cal_loss(model, batch_images, batch_labels, ignore_count_buf)
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))


                count += 1

            tr_iter = iter(train_ge)
            miou = 0.
            for i in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    predict = run_model(model, batch_image, False) # type을 batch label과 같은 type으로 맞춰주어야함
                    predict = tf.nn.softmax(predict[0], -1)
                    predict = tf.argmax(predict, -1, output_type=tf.int32)
                    predict = predict.numpy()

                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    ignore_label_axis = np.where(batch_label==FLAGS.ignore_label)   # 출력은 x,y axis로 나옴!
                    predict[ignore_label_axis] = FLAGS.ignore_label # 이거 내일 테스트해봐야함!! 기억해!!

                    miou_ = Measurement(predict=predict, 
                                       label=batch_label, 
                                       shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                       total_classes=FLAGS.total_classes).MIOU()

                    #miou_ = tf.keras.metrics.MeanIoU(FLAGS.total_classes)
                    #miou_.update_state(batch_label, predict)
                    #miou += miou_.result().numpy()
                    miou += miou_
            
            print("Epoch: {}, miou = {}".format(epoch, miou / len(train_img_dataset)))


if __name__ == "__main__":
    main()
