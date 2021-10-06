# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os

# 안녕하세요!! 지난번에 이어서 좋은 글 잘 읽었습니다. mIoU 계산시에, true label에서는 등장하지 않고 predict label에서는 등장하는 label의 경우는 iou를 0으로 두고 계산을 해주나요?
# 안녕하세요.네 맞습니다. 간단히 교집합이 없으니 0이 되는 원리 입니다. true label에는 없으나 predict에만 존재한다는 것 자체가 성능이 나쁜것이니 0에 수렴해야 하는 논리와 동일합니다.

# 저 위의 논리대라면, 나누
# 아!!! predict한 이미지에 void부분을 11로 만들고! miou를 구할 때, void클래스 빈도수는 제외하고 진행하면 되지 않을까? 기억해!!!!! 지금생각났음!!!!!!!!!!!!!
# 왜냐면! 어차피 predict와 label 위치에 같은 값이 동일하게 있다면 confusion metrice 할때 중앙성분만있음! 그렇기에 cm을 구한 뒤 대각성분을 추출한 뒤 맨 뒤에있는 void라벨을 제거하면 됨! 오키!! 내일 다시한번더 생각천천히해봐!!!기억해 꼭해!!!!!!!!!!!

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
        label_count = np.bincount(self.label, minlength=self.total_classes)
         
        temp = self.total_classes * np.array(self.label, dtype="int") + np.array(self.predict, dtype="int")  # Get category metrics
    
        temp_count = np.bincount(temp, minlength=self.total_classes*self.total_classes)
        cm = np.reshape(temp_count, [self.total_classes, self.total_classes])
        cm = np.diag(cm)
    
        U = label_count + predict_count - cm
        U = np.delete(U, -1)    # delete last label (void class)
        cm_ = np.delete(cm, -1)

        miou = cm_ / U
        miou = np.nanmean(miou)

        return miou


class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)

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
