# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os

class Measurement:
    def __init__(self, predict, label, shape, total_classes=96):
        self.predict = predict
        self.label = label
        self.total_classes = total_classes
        self.shape = shape

    def MIOU(self):

        self.predict = np.reshape(self.predict, self.shape)
        self.label = np.reshape(self.label, self.shape)

        predict_count = np.bincount(self.predict, minlength=self.total_classes)
        predict_count = np.delete(predict_count, -1) # delete last label (void class)
        label_count = np.bincount(self.label, minlength=self.total_classes)
        label_count = np.delete(label_count, -1)  # delete last label (void class)

        label_ = self.label[self.label != 11]   # 11은 void 클래스
        skip_label_index = np.where(self.label==11) # ==> (array([], dtype),) ==> 그렇기 때문에 앞에 [0]을 해주어야 array([], dtype)이 됨
        predict_ = np.delete(self.predict, [ skip_label_index[0] ])
        #######################################################################################################################
        temp = (self.total_classes - 1) * np.array(label_, dtype="int") + np.array(predict_, dtype="int")  # Get category metrics
    
        temp_count = np.bincount(temp, minlength=(self.total_classes - 1)*(self.total_classes - 1))
        cm = np.reshape(temp_count, [self.total_classes-1, self.total_classes-1])
        cm = np.diag(cm)
        #cm = np.delete(cm, -1)   # delete last label (void class)
    
        U = label_count + predict_count - cm

        miou = cm / U
        miou = np.nanmean(miou)

        return miou


#import matplotlib.pyplot as plt

#if __name__ == "__main__":

    
#    path = os.listdir("D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/labels")

#    b_buf = []
#    for i in range(len(path)):
#        img = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/labels/"+ path[i])
#        img = tf.image.decode_png(img, 1)
#        img = tf.image.resize(img, [513, 513], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#        img = tf.image.convert_image_dtype(img, tf.uint8)
#        img = tf.squeeze(img, -1)
#        #plt.imshow(img, cmap="gray")
#        #plt.show()
#        img = img.numpy()
#        a = np.reshape(img, [513*513, ])
#        print(np.max(a))
#        img = np.array(img, dtype=np.int32) # void클래스가 정말 12 인지 확인해봐야함
#        #img = np.where(img == 0, 255, img)

#        b = np.bincount(np.reshape(img, [img.shape[0]*img.shape[1],]))
#        b_buf.append(len(b))
#        total_classes = len(b)  # 현재 124가 가장 많은 클래스수

#        #miou = MIOU(predict=img, label=img, total_classes=total_classes, shape=[img.shape[0]*img.shape[1],])
#        miou_ = Measurement(predict=img,
#                            label=img, 
#                            shape=[513*513, ], 
#                            total_classes=12).MIOU()
#        print(miou_)

#    print(np.max(b_buf))
