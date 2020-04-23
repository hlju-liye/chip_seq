#coding:utf8

import numpy as np

def to_one_hot(pos_seqs, neg_seqs):
	base_dict = {'A' : 0, 'C' : 1, 'G' : 2, 'T' : 3}

	one_hot_4_pos_seqs = []
	for seq in pos_seqs:
		one_hot_matrix = np.array(4, seq.length())
		index = 0
		for seq_base in seq:
			one_hot_matrix[base_dict[seq_base], index] = 1
			index++

	return one_hot_4_pos_seqs

