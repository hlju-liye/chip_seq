#coding:utf-8
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc

def read_fasta(file_name):

	seqs = []
	file = open(file_name)

	for line in file.readlines():
		if line.startswith('>'):
			continue
		else:
			seq = line.strip('\n')
	
			result1 = 'N' in seq
			result2 = 'n' in seq
			if result1 == False and result2 == False:
				seqs.append(seq)
	return seqs

def to_one_hot(seqs):
	base_dict = {
	'a' : 0, 'c' : 1, 'g' : 2, 't' : 3, 
	'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3 
	}

	one_hot_4_seqs = []
	for seq in seqs:
		one_hot_matrix = np.zeros([4,len(seq)], dtype = int)
		index = 0
		for seq_base in seq:
			one_hot_matrix[base_dict[seq_base], index] = 1
			index = index + 1

		one_hot_4_seqs.append(one_hot_matrix)

	return one_hot_4_seqs

def data_shuffle(x_data, y_data):

	indices = [i for i in range(len(x_data))]
	#print len(x_data)
	np.random.shuffle(indices)

	return x_data[indices], y_data[indices]



def plot_roc_curve(label_test, label_predict):

	fpr, tpr, thresholds = roc_curve(label_test, label_predict)
	auc_value = auc(fpr, tpr)
	print("auc = " + str(auc_value))
	plt.figure(1)
	plt.title('ROC Curve')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_value))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	#plt.legend(loc='best')

	plt.show()

	return auc_value

def plot_prc_curve(label_test, label_predict):

	precision, recall, thresholds = precision_recall_curve(label_test, label_predict)
	auc_value = auc(precision, recall)

	plt.figure(2)
	plt.title('PRC Curve')
	plt.plot([0, 1], [0, 1], 'k--')
	#plt.plot(recall, precision, label='Keras (area = {:.3f})'.format(auc_value))
	plt.xlabel('Recall')
	plt.ylabel('Precision')

	plt.show()

	return auc_value


#def motif_discover():
