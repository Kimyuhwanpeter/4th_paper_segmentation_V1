# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os

total_classes = 96  # 0 라벨이 void 처리된 것

def MIOU(predict, label, total_classes, shape=[720*1280,]):

    predict_ = np.reshape(predict, shape)
    label_ = np.reshape(label, shape)

    predict_count = np.bincount(predict_, minlength=total_classes)
    predict_count = np.delete(predict_count, 0) # delete 0 label (void class)
    label_count = np.bincount(label_, minlength=total_classes)
    label_count = np.delete(label_count, 0)  # delete 0 label (void class)

    temp = total_classes * label_ + predict_  # Get category metrics
    
    temp_count = np.bincount(temp, minlength=total_classes*total_classes)
    cm = np.reshape(temp_count, [total_classes, total_classes])
    cm = np.diag(cm)
    cm = np.delete(cm, 0)   # delete 0 label (void class)
    
    U = label_count + predict_count - cm

    miou = cm / U
    miou = np.nanmean(miou)

    return miou


#import matplotlib.pyplot as plt

#if __name__ == "__main__":

    
#    path = os.listdir("D:/[1]DB/[5]4th_paper_DB/Fruit/MinneApple/detection/train/masks")

#    b_buf = []
#    for i in range(len(path)):
#        img = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/Fruit/MinneApple/detection/train/masks/"+ path[i])
#        img = tf.image.decode_png(img, 1)
#        img = tf.image.resize(img, [513, 513], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#        img = tf.image.convert_image_dtype(img, tf.uint8)
#        img = tf.squeeze(img, -1)
#        #plt.imshow(img, cmap="gray")
#        #plt.show()
#        img = img.numpy()
#        img = np.array(img, dtype=np.int32)
#        #img = np.where(img == 0, 255, img)

#        b = np.bincount(np.reshape(img, [img.shape[0]*img.shape[1],]))
#        b_buf.append(len(b))
#        total_classes = len(b)  # 현재 124가 가장 많은 클래스수

#        miou = MIOU(predict=img, label=img, total_classes=total_classes, shape=[img.shape[0]*img.shape[1],])
#        print(miou)

#    print(np.max(b_buf))
