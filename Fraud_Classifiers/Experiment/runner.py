#Code to Run an instance of Fraud_detector
import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml

import pickle
from collections import defaultdict
from experiment import Fraud_detector



def main():

	parser = argparse.ArgumentParser('Detect Fraud in Financial data')
	parser.add_argument('--conf', dest='config_filepath', help='Full path to the configuration file')
	parser.add_argument('--preprocess', dest='preprocess_bool', action='store_true', help = 'Whether to preprocess data or not')
	parser.set_defaults(preprocess_bool=False)


	args = parser.parse_args()
	conf = yaml.load(open(args.config_filepath), Loader=yaml.FullLoader)

	detector = Fraud_detector(args.preprocess_bool,conf['root_path'],conf['filename'],args.preprocess_bool,conf["feature_file"])
	if args.preprocess_bool:
		detector.create_degree_dic()

		print("\n\nPreprocessing Done\n\n")

	#Add motif counting parameters
	if conf["motif_count_params"]["needed"]:
		num_node = conf["motif_count_params"]["num_node"]
		num_edge = conf["motif_count_params"]["num_edge"] 
		dc = conf["motif_count_params"]["dc"] 
		dw = conf["motif_count_params"]["dw"]
		detector.count_motif(num_node,num_edge,dc,dw)

	#Add feature creation parameters
	if conf["feature_creation_params"]["needed"]:
		measure = conf["feature_creation_params"]["measure"]
		detector.feature_creation(measure)

	#Run classification
	if conf["run_classifier"]["needed"]:
		num_rep = conf["run_classifier"]["num_rep"]
		feature_to_keep = conf["run_classifier"]["feature_to_keep"]
		classifier_to_use = conf["run_classifier"]["classifier_to_use"]
		detector.run_classifier(num_rep,feature_to_keep,classifier_to_use)

if __name__=='__main__':
	main()







