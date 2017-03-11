# coding:utf-8
import random
from time import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from math import sqrt
import matplotlib.pyplot as plt
import sys
import os

%matplotlib inline

# 再実行時にグラフをクリア
tf.reset_default_graph()
# チェックポイントを作成するディレクトリの指定
CKPTDIR = "./ckptdir-test"
if(!os.path.exists(CKPTDIR)):
    print"Please make directory "+CKPTDIR
    sys.exit()

# ロードは最初しないのでFalse
LOAD_MODEL = False

# 不正解のデータが入ったcsvを読み込む
negative_dataset = np.genfromtxt("./negative.csv", delimiter=',', dtype=["S32", int, "S32",float, float, float, float, float, float, float, float, float, int])
# 正解のデータが入ったcsvを読み込む
positive_dataset = np.genfromtxt("./positive.csv", delimiter=',', dtype=["S32", int, "S32",float, float, float, float, float, float, float, float, float, int])

NUM_CLASSES = 2 #  2クラス分類
NUM_STEPS = 500 #  学習回数
LEN_SEQ = 10 # 系列長
SIZE_INPUT = 3 # 入力データ数
NUM_DATA = 2009  # データ数
NUM_TEST = 200 # テスト用のデータ数
SIZE_BATCH = 100 # バッチサイズ
NUM_NODE = 1024  # ノード数
LEARNING_RATE = 0.01  # 学習率

def get_data(dataset):
    #データセットをnparrayに変換する
    #csvに入っている3~6番目の値だけ取得する
    raw_data = [list(item)[3:6] for item in dataset]
    raw_data = np.array(raw_data)
    return raw_data

def set_matrix(dataset,length):
  tmp = []
  ret = []
  #(length,10,3)の配列に整形する
  for i in range(0,length):
    tmp = []
    for j in range(0,10):
      tmp.append(dataset[i*10+j])
    ret.append(np.array(tmp))
  return np.array(ret)

def create_label(num,length):
  #(length,)でnumの値が入った配列を作成する
  label = []
  for i in range(0,length):
    label.append(num)

  return np.array(label)

#正解の加速度の値が入った配列
positive_data = get_data(positive_dataset)
print (positive_data)[:3]
#整形した配列
positive_main_data = set_matrix(positive_data,1400)
print positive_main_data[:3]
#正解ラベルの配列
positive_label = create_label(1,1400)

negative_data = get_data(negative_dataset)
negative_main_data = set_matrix(negative_data,609)
negative_label = create_label(0,609)

#正解と不正解の配列を合わせる
x_data = np.r_[positive_main_data, negative_main_data]
y_data = np.r_[positive_label, negative_label]
print x_data.shape
print y_data.shape

#配列をシャッフルする
index_list = np.arange(0, 2009)
np.random.shuffle(index_list)
x_shuffle_data = x_data[index_list]
y_shuffle_data = y_data[index_list]

#学習とテストで分ける
x_train_data = x_shuffle_data[NUM_TEST:]
y_train_label = y_shuffle_data[NUM_TEST:]
x_test_data = x_shuffle_data[:NUM_TEST]
y_test_label = y_shuffle_data[:NUM_TEST]
print x_train_data.shape
print y_train_label.shape
print x_test_data.shape
print y_test_label.shape

#入力データの代入先
x = tf.placeholder(tf.float32, [None, LEN_SEQ, SIZE_INPUT])
#ラベルの代入先
t = tf.placeholder(tf.int32, [None])
#出力層での値が、分類したいクラスの数と同じ次元のベクトルとなる配列を返す
t_on_hot = tf.one_hot(t, depth=NUM_CLASSES, dtype=tf.float32)

# NUM_STEPSとSIZE_BATCHを転置する
x_transpose = tf.transpose(x, [1, 0, 2])
#(NUM_STEPS*SIZE_BATCH,SIZE_INPUT)にreshapeする
x_reshape = tf.reshape(x_transpose, [-1, SIZE_INPUT])
x_split = tf.split(x_reshape, LEN_SEQ, 0)
lstm_cell = rnn.BasicLSTMCell(NUM_NODE, forget_bias=1.0)
outputs, states = rnn.static_rnn(lstm_cell, x_split, dtype=tf.float32)
#重み
w = tf.Variable(tf.random_normal([NUM_NODE, NUM_CLASSES]))
#バイアス
b = tf.Variable(tf.random_normal([NUM_CLASSES]))
logits = tf.matmul(outputs[-1], w) + b
#どのクラス(正解か不正解)に分類されるのが尤もらしいかを表す
pred = tf.nn.softmax(logits)

#誤差関数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=t_on_hot, logits=logits)
loss = tf.reduce_mean(cross_entropy)
#使用するトレーニングアルゴリズムと最小化
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_step = optimizer.minimize(loss)

#予想したラベルが教師のラベルと一致しているかを表す
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(t_on_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
loss_train = []
acc_train = []
loss_test = []
acc_test = []


saver = tf.train.Saver()
sess = tf.InteractiveSession()
ckpt = tf.train.get_checkpoint_state(CKPTDIR)
if ckpt:
    # checkpointファイルから最後に保存したモデルへのパスを取得する
    last_model = ckpt.model_checkpoint_path
    print(("load {0}".format(last_model)))
    # 学習済みモデルを読み込む
    saver.restore(sess, last_model)
    LOAD_MODEL = True
else:
  #チェックポイントを作成する
  print("initialization")
  sess.run(tf.global_variables_initializer())
  start = time()
  i = 0
  for _ in range(NUM_STEPS):
      cycle = int((NUM_DATA-NUM_TEST)/SIZE_BATCH)
      begin = int(SIZE_BATCH * (i % cycle))
      end = int(begin + SIZE_BATCH)
      batch_x, batch_t = x_train_data[begin:end], y_train_label[begin:end]
      i += 1
      sess.run(train_step, feed_dict={x: batch_x, t: batch_t})
      if i % 10 == 0:
          loss_, acc_ = sess.run([loss, accuracy], feed_dict={x: batch_x, t: batch_t})
          loss_train.append(loss_)
          acc_train.append(acc_)
          loss_test_, acc_test_ = sess.run([loss, accuracy], feed_dict={x: x_test_data, t: y_test_label})
          loss_test.append(loss_test_)
          acc_test.append(acc_test_)
          print("[%i STEPS] %f sec" % (i, (time() - start)))
          print("[TRAIN] loss : %f, accuracy : %f" %(loss_, acc_))
          print("[TEST loss : %f, accuracy : %f" %(loss_test_, acc_test_))

  #チェックポイントを保存する
  saver.save(sess, CKPTDIR+"/model")
