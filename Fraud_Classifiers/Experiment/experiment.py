import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pickle
from collections import defaultdict
from helper import reject_outliers, features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from ml_alg import group_graph, save_degree_dic, create_feature_ar, save_data, classify_motifs, run_motifs, preprocess


class Fraud_detector(object):
	"""Fraud_detector is a class for processing the temporal transaction data and
	classifying fraudulent users from normal users
	"""
	
	def __init__(self,preprocess_label, root_path, filename, create_group, feature_file):

		"""
		:param preprocess   : a bool denoting whether preprocessing is needed or not
		:param path 	    : path to the data filename
		:param filename     : name of the datafile
		:param create_group : a bool denoting whether local networks need to be constructed
		:param datafile     : name of the file where the features will be saved
		"""
		super(Fraud_detector, self).__init__()
		datatype_list = ['normal','fraud']

		
		#Preprocess the data if necessary
		if preprocess_label:
			preprocess(filename)

		#create local network from the whole network
		filename = filename[:-4]+'_processed.txt'
		if create_group:	
			group_graph(filename, root_path)


		#Add the input data file and the path to output to the object
		self.root_path = root_path
		self.datatype_list = datatype_list
		self.features = features
		self.feature_file = root_path+feature_file

	def create_degree_dic(self):
		"""A function to calculate the temporal and static degrees of the local network"""

		for datatype in self.datatype_list:
			output_path = self.root_path+datatype+'/'
			#It call the save_degree_dic function from ml_alg module
			save_degree_dic(output_path)

	def count_motif(self,num_node,num_edge,dc,dw):
		"""A function to count the motifs in the local network

		:param num_node     	 : number of nodes in a motif
		:param num_edge     	 : number of edges in a motif
		:param dc    			 : time difference between two consecutive event in a motif
		:param dc    			 : time difference between first and last event in a motif
		:param amount_constraint : a bool denoting whether amount constraint is needed or not
		:param ac 				 : amount difference between two consecutive events in a motif
		"""


		for datatype in self.datatype_list:
			input_path_for_algorithm = self.root_path+datatype+'/'
			#It calls the run_motif function from the ml_alg module
			run_motifs(input_path_for_algorithm,dc,dw,num_node,num_edge,'NO',0,0)

	def feature_creation(self, measure):
		"""A function to crete the features from the motif counts"""

		output_filename = self.feature_file
		dic = create_feature_ar(self.root_path,measure)
		#saves the transformed data
		save_data(dic, output_filename)

		
		

	def run_classifier(self,num_rep,feature_to_keep,classifier_to_use):
		"""A function to classify the fraudulent users from the created features
		Logistic Regression, SVM are Random Forest are used for classifying"""

		feature_for_classifier = [self.features.index(x) for x in feature_to_keep]
		print("feature_for_classifier",feature_for_classifier)
		print("feature_to_keep",feature_to_keep)
		print("self.features",self.features)
		print(len(self.features))

		roc_lr, roc_rf, roc_svm, f1_lr, f1_svm, f1_rf, roc_xgb, f1_xgb = 0,0,0,0,0,0,0,0
		a_p_lr, a_p_svm, a_p_rf, a_p_xgb, a_r_lr, a_r_svm, a_r_rf, a_r_xgb = 0,0,0,0,0,0,0,0 
		for rep in range(num_rep):
			cur_roc_lr, cur_roc_svm, cur_roc_rf, cur_f1_lr, cur_f1_svm, cur_f1_rf, cur_roc_xgb, cur_f1_xgb, p_lr, p_svm, p_rf, p_xgb, r_lr, r_svm, r_rf, r_xgb,  datasize = classify_motifs(self.feature_file, feature_for_classifier, classifier_to_use)
			roc_lr += cur_roc_lr
			roc_rf += cur_roc_rf
			roc_svm += cur_roc_svm
			f1_lr += cur_f1_lr
			f1_svm += cur_f1_svm
			f1_rf += cur_f1_rf
			roc_xgb += cur_roc_xgb
			f1_xgb += cur_f1_xgb
			a_p_lr += p_lr
			a_p_svm += p_svm
			a_p_rf += p_rf
			a_p_xgb += p_xgb
			a_r_lr += r_lr
			a_r_svm += r_svm
			a_r_rf += r_rf
			a_r_xgb += r_xgb 
			if rep %10 == 0:
				if rep == 0:
					print("Size of Normal Data: ", datasize)
					print("Size of Fraud Data: ", datasize)

				print("iteration Number :",rep)

		if 'lr' in classifier_to_use:
			print(roc_lr/float(num_rep))
			print(f1_lr/float(num_rep))
			print(a_p_lr/float(num_rep))
			print(a_r_lr/float(num_rep))
		if 'svm' in classifier_to_use:
			print(roc_svm/float(num_rep))
			print(f1_svm/float(num_rep))
			print(a_p_svm/float(num_rep))
			print(a_r_svm/float(num_rep))
		if 'rf' in classifier_to_use:
			print(roc_rf/float(num_rep))
			print(f1_rf/float(num_rep))
			print(a_p_rf/float(num_rep))
			print(a_r_rf/float(num_rep))
		if 'xgb' in classifier_to_use:
			print(roc_xgb/float(num_rep))
			print(f1_xgb/float(num_rep))
			print(a_p_xgb/float(num_rep))
			print(a_r_xgb/float(num_rep))











