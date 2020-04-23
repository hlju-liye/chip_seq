#coding:utf-8
import numpy as np

from sklearn.model_selection import StratifiedKFold

from Network import *

from util_func import *

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, average_precision_score

pos_file = 'GM12878_POLR2A.fasta'
neg_file = 'neg_collection.fasta'

#读取fasta文件，并将read序列放入列表中
pos_seqs = read_fasta(pos_file)

neg_seqs = read_fasta(neg_file)


pos_seqs_count = len(pos_seqs)


#将read序列转换成one hot编码
pos_matrix = to_one_hot(pos_seqs)
neg_matrix = to_one_hot(neg_seqs)

#转换成numpy数组
np_pos_matrix = np.array(pos_matrix)
#获取one hot编码矩阵个数，高度已经宽度
matrix_count  = np_pos_matrix.shape[0]
matrix_height = np_pos_matrix.shape[1]
matrix_width  = np_pos_matrix.shape[2]

#print np_pos_matrix.shape

#np_3dim_pos_matrix = np_pos_matrix.reshape(matrix_count, matrix_height, matrix_width)

np_neg_matrix_collection = np.array(neg_matrix)

index_ = np.random.randint(0, np_neg_matrix_collection.shape[0], pos_seqs_count)
np_neg_matrix = np_neg_matrix_collection[index_]

matrix_count  = np_neg_matrix.shape[0]
matrix_height = np_neg_matrix.shape[1]
matrix_width  = np_neg_matrix.shape[2]

print np_neg_matrix.shape

#np_3dim_neg_matrix = np_neg_matrix.reshape(matrix_count, matrix_height, matrix_width)


#生成标签的numpy数组(正=1，负=0)
pos_labels = np.ones(len(pos_seqs), dtype = int)
#neg_labels = np.zeros(len(neg_seqs), dtype = int)
neg_labels = np.zeros(matrix_count, dtype = int)

#将阳性和阴性的矩阵数组进行合并，方向为个数方向
matrix = np.concatenate((np_pos_matrix, np_neg_matrix), axis = 0)
labels = np.concatenate((pos_labels, neg_labels), axis = 0)

print labels[0:100]

x1, y1 = data_shuffle(matrix, labels)

print y1[0:100]

datax = x1.reshape(-1, 4, 300, 1)

datay = y1.reshape(labels.shape[0], 1)

count_all = datax.shape[0]
count_train = int(count_all * 0.95)

datax_train = datax[0:count_train]
datay_train = datay[0:count_train]

datax_test = datax[(count_train + 1) : (count_all - 1)]
datay_test = datay[(count_train + 1) : (count_all - 1)]


network = Network()
network.train(datax_train, datay_train, 0.2)
loss, accu = network.evaluate(datax_test, datay_test)

print('accu is ' + str(accu))
