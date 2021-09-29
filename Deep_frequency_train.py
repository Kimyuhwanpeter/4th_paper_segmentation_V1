# -*- coding:utf-8 -*-
from modified_deeplab_V3 import *
from cal_measurement import MIOU
from random import random, shuffle

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 513,
                           
                           "label_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/MinneApple/detection/train/masks/",
                           
                           "image_path": "D:/[1]DB/[5]4th_paper_DB/Fruit/MinneApple/detection/train/images/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "lr": 0.0001,
                           
                           "epochs": 50,

                           "total_classes": 124,

                           "ignore_label": 0,

                           "batch_size": 2,

                           "train": True})

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def tr_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

#@tf.function    이 부분도 해결해야함!! 기억해!!!!!!!!!!!
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
                batch_labels = tf.where(batch_labels == FLAGS.ignore_label, 255, batch_labels)
                batch_labels = tf.one_hot(batch_labels, FLAGS.total_classes - 1)
                batch_labels = tf.squeeze(batch_labels, -2)
                loss = cal_loss(model, batch_images, batch_labels)
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))


                count += 1
            #tr_iter = iter(train_ge)
            #miou = 0.
            #for i in range(tr_idx): # 내일 miou 측정하는것까지 완성시키기! 기억해!!!!!!!!!!!!
            #    batch_images, batch_labels = next(tr_iter)
            #    for j in range(FLAGS.batch_size):
            #        predict = run_model(model, batch_images[j], False)

            #        miou += MIOU(predict, batch_labels[j], )



if __name__ == "__main__":
    main()
