import random
from time import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from math import sqrt
import FaBo9Axis_MPU9250
import sys
import RPi.GPIO as GPIO
import requests
import json
import smbus
import time


tf.reset_default_graph() # 再実行時にグラフをクリア

CKPTDIR = "/home/pi/DeepFishing/ckptdir-sqrt-test" # ディレクトリはあらかじめ作っておく
ax=0
ay=0
az=0
gx=0
gy=0
gz=0
mx=0
my=0
mz=0
sess = tf.InteractiveSession()

#-----AI code----------------
def get_sensor(datas):
        a = mpu9250.readAccel()
        g = mpu9250.readGyro()
        m = mpu9250.readMagnet()
        value = (a['x']-ax,  a['y']-ay, a['z']-az, g['x']-gx, g['y']-gy, g['z']-gz,m['x']-mx,m['y']-my,m['z']-mz)
        data = (value[0],value[1],value[2],value[3],value[4],value[5],value[6],value[7],value[8])
        datas.append(data)

def isHit(param):
    Hit = 0;
#         print datas[k][0]
    if(do_test(param)):
        Hit = 1

    if(Hit):
        return True
    else:
        return False

def create_sqrt_data(dataset):
  tmp = []
  sqrt_data = []
  for i in dataset:
      tmp.append(([sqrt(i[0]*i[0]+i[1]*i[1]+i[2]*i[2])]))
  sqrt_data = np.array(tmp)
  return sqrt_data

NUM_CLASSES = 2 #  4クラス分類
NUM_STEPS = 1100 #  学習回数
LEN_SEQ = 10 # 系列長
SIZE_INPUT = 1 # 入力データ数
NUM_DATA = 890  # データ数
NUM_TEST = 100 # テスト用のデータ数
SIZE_BATCH = 3 # バッチサイズ
NUM_NODE = 1024  # ノード数
LEARNING_RATE = 0.01  # 学習率

x = tf.placeholder(tf.float32, [None, LEN_SEQ, SIZE_INPUT])
t = tf.placeholder(tf.int32, [None])
t_on_hot = tf.one_hot(t, depth=NUM_CLASSES, dtype=tf.float32)
x_transpose = tf.transpose(x, [1, 0, 2])
x_reshape = tf.reshape(x_transpose, [-1, SIZE_INPUT])
x_split = tf.split(x_reshape, LEN_SEQ, 0)
lstm_cell = rnn.BasicLSTMCell(NUM_NODE, forget_bias=1.0)
outputs, states = rnn.static_rnn(lstm_cell, x_split, dtype=tf.float32)
w = tf.Variable(tf.random_normal([NUM_NODE, NUM_CLASSES]))
b = tf.Variable(tf.random_normal([NUM_CLASSES]))
logits = tf.matmul(outputs[-1], w) + b
pred = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=t_on_hot, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_step = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(t_on_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
fishing_dic = {0: 'failed', 1: 'hit'}

saver = tf.train.Saver()

frommodel = False

ckpt = tf.train.get_checkpoint_state(CKPTDIR)
if ckpt:
    # checkpointファイルから最後に保存したモデルへのパスを取得する
    last_model = ckpt.model_checkpoint_path
    print(("load {0}".format(last_model)))
    # 学習済みモデルを読み込む
    saver.restore(sess, last_model)
    frommodel = True
else:
    print("initialization")
    # 初期化処理
    init_op = tf.global_variables_initializer()
    sess.run(init_op)


def get_labels(dataset):
    """ラベル(正解データ)を1ofKベクトルに変換する"""
    raw_labels = [item[12] for item in dataset]
    labels = []
    for l in raw_labels:
        if l == 1:
            labels.append([1])
        elif l == 0:
            labels.append([0])
    return np.array(labels)

def get_data(dataset):
    """データセットをnparrayに変換する"""
    raw_data = [list(item)[3:6] for item in dataset]
    return np.array(raw_data)

def conv_shape(seq):
    target = [np.array(seq).reshape(-1,1)]
    return np.array(target)
def do_test(param):
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
    seq = param.reshape(1,10,1)
#     print("start : "+str(start))
#     print(conv_shape(seq))
    answer = sess.run(pred, feed_dict={x: seq})
    #print ans
#     print("end : "+str(start - time.time()))
    answer = np.array(answer)
    #print ans
    #print np.argmax(ans)
    #print("sequence: ", seq)
    #print("answer: ", fishing_dic[np.argmax(ans)])
    if(np.argmax(answer)):
        return True
    else:
        return False
#-----Finalizer--------------
def fin():

    bus.close() #Analog close
    sys.exit() #Exit

#============================
#Analog IO init
bus = smbus.SMBus(1)

mpu9250 = FaBo9Axis_MPU9250.MPU9250()
sensor = []
flag = True

def calibration():
    ax = []
    ay = []
    az = []
    gx = []
    gy = []
    gz = []
    mx = []
    my = []
    mz = []
    i=0
    for i in range(0,15):
        if(i >= 5 and i <15):
            a = mpu9250.readAccel()
            g = mpu9250.readGyro()
            m = mpu9250.readMagnet()
            ax.append(a['x'])
            ay.append(a['y'])
            az.append(a['z'])
            gx.append(g['x'])
            gy.append(g['y'])
            gz.append(g['z'])
            mx.append(m['x'])
            my.append(m['y'])
            mz.append(m['z'])
        time.sleep(0.1)

    return np.mean(ax),np.mean(ay),np.mean(az),np.mean(gx),np.mean(gy),np.mean(gz),np.mean(mx),np.mean(my),np.mean(mz)

ax,ay,az,gx,gy,gz,mx,my,mz = calibration()
#Start fishing!
try:
    while True:
        get_sensor(sensor)
        time.sleep(0.1)

        if(len(sensor) == 10):
            data = get_data(sensor)
            data_sqrt = create_sqrt_data(data)
            #params = data[:,0:3]
            #print params
            ans = isHit(data_sqrt)
#             ans = False

            if(ans):
                print "hit"
                flag = False
#                 servo.ChangeDutyCycle(7)
                #forward(0x3A)
                #count_rotation(ROTATE_COUNT)
#                 stop()
#                 brake()
#                 while True:
#                     # ボタン押下判定
#                     if( GPIO.input(BUTTONPIN)):

#                         servo.ChangeDutyCycle(10)
#                         time.sleep(0.2)
#                         #down()
#                         break;
            else:
                print "failed"
                ans = False

            sensor = []
except KeyboardInterrupt:
    sess.close()
    fin()
