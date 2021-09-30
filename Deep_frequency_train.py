# -*- coding:utf-8 -*-
from modified_deeplab_V3 import *
from cal_measurement import Measurement
from random import random, shuffle

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 513,
                           
                           "label_path": "D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/labels/",
                           
                           "image_path": "D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/images/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "lr": 1e-6,
                           
                           "epochs": 200,

                           "total_classes": 12,

                           "ignore_label": 11,

                           "batch_size": 2,

                           "train": True})

optim = tf.keras.optimizers.Adam(FLAGS.lr)
# pretrained 모델을 사용해보자
# https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

def tr_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

#@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, images, labels):

    with tf.GradientTape() as tape:
        logits = run_model(model, images, True)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def main():
    model = Deep_edge_network(input_shape=(FLAGS.img_size,FLAGS.img_size,3), num_classes=FLAGS.total_classes-1)
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
            #A = list(zip(train_img_dataset, train_lab_dataset))
            #shuffle(A)
            #train_img_dataset, train_lab_dataset = zip(*A)
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
                batch_labels = tf.where(batch_labels == FLAGS.ignore_label, 255, batch_labels)
                batch_labels = tf.one_hot(batch_labels, FLAGS.total_classes - 1)
                batch_labels = tf.squeeze(batch_labels, -2)
                loss = cal_loss(model, batch_images, batch_labels)
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
                    predict = tf.cast(predict, tf.uint8)
                    predict = predict.numpy()

                    batch_label = batch_labels[j].numpy()
                    b = np.bincount(np.reshape(batch_label, [FLAGS.img_size*FLAGS.img_size,]))
                    #print(len(b))
                    miou_ = Measurement(predict=predict, 
                                       label=batch_label, 
                                       shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                       total_classes=FLAGS.total_classes).MIOU()
                    miou += miou_

            print("Epoch: {}, miou = {}".format(epoch, miou / len(train_img_dataset)))




if __name__ == "__main__":
    main()
