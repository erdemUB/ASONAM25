import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pickle
from collections import defaultdict
from helper import reject_outliers, load_pickle, id_rename, features, create_motif_dic, convert_times
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics,svm
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import gc
import networkx as nx
from sklearn.utils import resample
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def preprocess(filename):
	"""A function to preprocess the data to add timestamps, sender and receiver label"""

	df = pd.read_csv("../Data/"+filename)
	df = df.dropna(axis=0, how='all')
	df = df.dropna(axis=1, how='all')
	df.head()
	# print(df.columns)
	# print(df['Label'].unique())
	# print(df.shape, df['Label'].isna())

	senders, buyers = set(list(df['Sender_Id'])), set(list(df['Bene_Id']))
	all_users = senders.union(buyers)
	print("Number of senders :", len(senders))
	print("Number of buyers :", len(buyers))
	print("Number of all users :", len(all_users))


	fraud_users = set(list(df[df['Label']==1]['Sender_Id']))
	print("Number of fraud users :",len(fraud_users))
	normal_users = set(list(df[df['Label']==0]['Sender_Id']))
	print("Number of normal users :",len(normal_users))

	def generate_label(userid):
		return userid in fraud_users

	df['tr_Sender'] = df['Sender_Id'].apply(id_rename)
	df['tr_Bene'] = df['Bene_Id'].apply(id_rename)
	df['tr_label'] = df['Label'].apply(int)
	df['timestamps'] =df['Time_step'].apply(convert_times)
	df['sender_label'] = df['Sender_Id'].apply(generate_label)
	df['bene_label'] = df['Bene_Id'].apply(generate_label)
	df['USD'] = df['USD_amount'].apply(float).apply(int)

	prefix = filename[:-4]
	prefix = prefix+"_processed.txt"

	fmt = '%d','%d','%d','%d','%d','%d'
	np.savetxt("../Data/"+prefix, df[['tr_Sender','tr_Bene','timestamps','USD','sender_label','bene_label','tr_label']].values,fmt="%d")
	# df.to_csv(path+prefix)

	



def group_graph(filename,output_path):
	"""a function to create the local graphs for each users from a transaction data with temporal information"""

	#Loading the data and creating a dataframe
	arr = np.loadtxt("../Data/"+filename,dtype=int)
	df = pd.DataFrame(data = arr, columns = ['seller_id','buyer_id','timestamps','amount','seller_label','buyer_label','fraud_item'])
	
	#Set of all users in the transaction network
	all_users = set(df[['seller_id','buyer_id']].to_numpy().astype(int).reshape(-1))

	#Create set of fraud users from their labels
	fraud_sellers = set(df[df['seller_label']==1]['seller_id'].to_numpy().astype(int))
	fraud_buyers = set(df[df['buyer_label']==1]['buyer_id'].to_numpy().astype(int))
	all_frauds = fraud_sellers.union(fraud_buyers)

	#Create set of non-fraud users from their labels
	all_non_frauds = all_users.difference(all_frauds)


	datatype_list = ["normal",'fraud']

	# Iterating over normal and fraud users seperately to create local graph
	for datatype_ind,seed_ids in enumerate([all_non_frauds, all_frauds]):

		datatype = datatype_list[datatype_ind]

		#A dictionary whose keys are users and its corresponding value is a list of neighbors of type datatype(normal or fraud)
		nbr_seed_dic = defaultdict(set)
		print(datatype+" Data Processing ")
		print("Finding Neighbors ...")
		
		#Iterating over the users and adding neighbors
		for ind,seed in enumerate(seed_ids):
			#neighbors of the current users
			cur_ohi = set(df[(df['seller_id'] == seed) | (df['buyer_id'] == seed)][['seller_id','buyer_id']].to_numpy().astype(int).flatten())
			for user in cur_ohi:
				nbr_seed_dic[user].add(seed)
			
		print("Creating Local Network ...")

		#A dictionary whose keys are the users and value is a list containing connections with neighbors and the connections among neighbors
		final_op = defaultdict(list)
		for ind,line in enumerate(arr):
			
			seller_id, buyer_id = line[0], line[1]
			seed_intersections = nbr_seed_dic[seller_id].intersection(nbr_seed_dic[buyer_id])
			
			for seed in seed_intersections:
				final_op[seed].append(line)
			
		#Writing the local network in text file
		ind = 0

		#Seed_map is a dictionary to save the users value w.r.t the indices they are processed
		seed_map = {}
		print("Writing in files ...\n")
		fraud_graph_size = 0
		for key in final_op:
			cur_ar = np.array(final_op[key])
			# print(cur_ar.shape)
			to_save = cur_ar[cur_ar[:,2].argsort()]
			seed_map[ind] = key
			outfile = output_path +datatype+'/'+datatype+'_seed_'+str(ind)+'.txt'
			np.savetxt(outfile,np.array(to_save),fmt='%i')
			ind += 1

			if datatype=='fraud':
				fraud_graph_size += len(to_save)
		pickle.dump(seed_map,open(output_path+datatype+'/'+'seed_map.pkl','wb'))
		print("!!!!! Fraud SIZE:", fraud_graph_size)
	print("Done")



def save_degree_dic(path):
	""" A function to compute Kodate et al features """

	seed_map = load_pickle(path+'seed_map.pkl')
	dic = {}

	#a string denoting the type of the user (normal or fraud)
	datatype = path.split("/")[-1]

	#Iterate over all the users' local network
	for ind,infile in enumerate(os.listdir(path)):

		if infile.endswith(".txt"):
			key = int(infile[:-4].split("_")[-1])
			arr = np.loadtxt(path+infile)

			#Calculate only for the users with enough data
			if len(arr.shape) > 1:

				#get the seeds from the seed map
				cur_seed = seed_map[key]

				#Calculate the temporal and static out degrees and in degrees of the seed user in the local network
				od_temp, id_temp = arr[arr[:,0] == cur_seed].shape[0], arr[arr[:,1] == cur_seed].shape[0]
				od_stat, id_stat = len(set(arr[arr[:,0] == cur_seed][:,1])), len(set(arr[arr[:,1] == cur_seed][:,0]))
				
				#Calculate the temporal and static out degrees and in degrees of the non-seed user in the local network
				ns_out_degree = (arr.shape[0] - od_temp)/float(od_stat+0.001)

				sp = od_stat / (id_stat + od_stat)
				wsp = od_temp / (od_temp + id_temp)
				k = od_stat
				s = od_temp
				
				key = infile[:-4].split("_")[-1]
				dic[key] = [k, s, wsp, sp, ns_out_degree]
				# if ind%10 == 0:
				# 	print(ind)


	pickle.dump(dic,open(path+'degree_dic.pkl','wb'))



				

def run_motifs(path,dc,dw,num_node,num_edge,amount_constraint,ac,aw):
	"""A function to run the motif counting algorithm"""

	#Iterate over the set of users
	for ind,infile in enumerate(os.listdir(path)):
		if infile.endswith(".txt"):
			arr = np.loadtxt(path+infile)
			#Run the algortihm only if there is enough data for the corresponding data	
			if len(arr.shape) > 1 and arr.shape[0] > 5:

				#Create the command line input for th algorithm
				args = [path+infile,str(dc),str(dw),str(num_node),str(num_edge),'5 NO NO',amount_constraint, str(ac),str(aw)]
				kargs_list  = [' '.join(args)]

				if num_node == 3 or num_edge == 3:
					args = [path+infile,str(dc),str(dw),str(3),str(2),'5 NO NO',amount_constraint, str(ac),str(aw)]
					kargs_list.append(' '.join(args))
				if num_node == 3 and num_edge == 3:
					args = [path+infile,str(dc),str(dw),str(3),str(3),'5 NO NO',amount_constraint, str(ac),str(aw)]
					kargs_list.append(' '.join(args))

				#Run all possible values of motif sizes we are interested in
				for kargs in kargs_list:
					
					command = '../Motif_counting/TMC '+kargs
					os.system(command)
				


def compute_measure_0(val,num):
	"""A function to compute feature value from the raw count"""

	if val[0] or val[1]:
		if num:
			return val[1]/(num*(num-1)/2.0 + 0.00000001)#float(val[0]+val[1])
		else:
			return 0
	else:
		return 0

def compute_measure_1(val,num):
	"""A function to compute feature value from the raw count"""

	if val[0] or val[1]:
		return val[1]/(val[1] + val[0] + 0.00000001)#/(float(num)))#float(val[0]+val[1])
	else:
		return 0

def compute_measure_2(val,deg1,deg2,num_ns):
	"""A function to compute feature value from the raw count"""

	if val[0] or val[1]:
		num = val[1]/(deg1*(deg1-1)/2.0+ 0.00001)
		den = num + val[0]/(num_ns*deg2*(deg2-1)/2.0 + 0.00001)
		return num/den#float(val[0]+val[1])
	else:
		return 0



def create_feature_ar(root_path,measure = 0):

	print("CREATE FEATURES:",features)
	num_features = len(features)
	data_dic = {}

	normal_dic, fraud_dic = pickle.load(open(root_path+'normal/degree_dic.pkl','rb')), pickle.load(open(root_path+'fraud/degree_dic.pkl','rb'))
	for ind,infile in enumerate(os.listdir(root_path)):
		original_filename = infile.split('.txt')[0]
		#print(original_filename)
		

		if infile.endswith(".txt") and "time" not in infile and "jpmc" not in infile:

			datatype,_,key = original_filename.split("_")

			if datatype == 'normal':
				cur_dic = normal_dic
			else:
				cur_dic = fraud_dic

			if key in cur_dic:
			

				motif_filename = root_path+infile
				motif_dic = create_motif_dic(motif_filename)


				if datatype not in data_dic:
					data_dic[datatype] = {}
				if key not in data_dic[datatype]:
					data_dic[datatype][key] = [0 for i in range(num_features)]

				if measure == 0:

					
					num = cur_dic[key][0] #+ normal_dic[key][3]				
					

					for mot in motif_dic:
						if mot in features:
							mot_ind = features.index(mot)
							data_dic[datatype][key][mot_ind] = compute_measure_0(motif_dic[mot],num)

				if measure == 1:

					
					num =  cur_dic[key][2]

					for mot in motif_dic:
						if mot in features:
							mot_ind = features.index(mot)
							data_dic[datatype][key][mot_ind] = compute_measure_1(motif_dic[mot],num)
							

				elif measure == 2:

					
					deg1, deg2, num_ns = cur_dic[key][0], cur_dic[key][4], cur_dic[key][2]

					for mot in motif_dic:
						if mot in features:
							mot_ind = features.index(mot)
							data_dic[datatype][key][mot_ind] = compute_measure_2(motif_dic[mot],deg1, deg2, num_ns)


				
				if datatype == 'normal':
					data_dic[datatype][key][0] = (normal_dic[key][0]+normal_dic[key][1])/float(normal_dic[key][2]+normal_dic[key][3]+0.001)
					data_dic[datatype][key][1] = (normal_dic[key][2])/float(normal_dic[key][2]+normal_dic[key][3]+0.001)
					data_dic[datatype][key][2] = (normal_dic[key][0])/float(normal_dic[key][0]+normal_dic[key][1]+0.001)
					data_dic[datatype][key][3] = (normal_dic[key][0]+normal_dic[key][1])
					data_dic[datatype][key][4] = (normal_dic[key][2]+normal_dic[key][3])

				else:
					data_dic[datatype][key][0] = (fraud_dic[key][0]+fraud_dic[key][1])/float(fraud_dic[key][2]+fraud_dic[key][3]+0.001)
					data_dic[datatype][key][1] = (fraud_dic[key][2])/float(fraud_dic[key][2]+fraud_dic[key][3]+0.001)
					data_dic[datatype][key][2] = (fraud_dic[key][0])/float(fraud_dic[key][0]+fraud_dic[key][1]+0.001)
					data_dic[datatype][key][3] = (fraud_dic[key][0]+fraud_dic[key][1])
					data_dic[datatype][key][4] = (fraud_dic[key][2]+fraud_dic[key][3])


				
				# if ind%10 == 0:
				# 	print(ind)



	final_dic = defaultdict(list)
	for keys in data_dic:
		for ind in data_dic[keys]:
			final_dic[keys].append(data_dic[keys][ind])

	for keys in data_dic:
		final_dic[keys] = np.array(final_dic[keys])

	return final_dic

def save_data(dic,op_filename):

	normal_size = np.array(dic['normal']).shape[0]
	fraud_size = np.array(dic['fraud']).shape[0]
	
	whole_data = np.concatenate((dic['normal'],dic['fraud']),axis=0) 
	whole_label = np.concatenate((np.zeros(normal_size),np.ones(fraud_size)))

	pickle.dump((whole_data,whole_label), open(op_filename,'wb'))



def classify_motifs(inp_filename, feature_to_select, classifier_to_use):
	
	whole_data,whole_label = load_pickle(inp_filename)
	print(whole_data.shape, len(feature_to_select))
	whole_data = whole_data[:,feature_to_select]

	# normal_size = int(len(whole_label) - sum(whole_label))
	# fraud_size = int(sum(whole_label))
	#print(normal_size,fraud_size)
	# data_size = min(normal_size, fraud_size)#+10
	# ind = [normal_size + x for x in np.random.choice(fraud_size,data_size,replace=False)]


	# normal_data = whole_data[:data_size,feature_to_select]
	# fraud_data = whole_data[ind,:]
	# fraud_data = fraud_data[:,feature_to_select]

	# whole_data = np.concatenate((normal_data, fraud_data),axis=0) 
	# whole_label = np.concatenate((np.zeros(data_size),np.ones(data_size)))
	whole_data = np.nan_to_num(whole_data)
	whole_data = whole_data.astype(np.float)

	X_train, X_test, y_train, y_test = train_test_split(whole_data, whole_label, test_size=0.25, random_state=12345)
	#resampling
	# normal_data = X_train[y_train==0]
	# fraud_data = X_train[y_train==1]
	# print(normal_data.shape, fraud_data.shape)
	# fraud_data = resample(fraud_data, n_samples=500)
	# normal_data = resample(normal_data, replace=False, n_samples=1000)
	# print(normal_data.shape, fraud_data.shape)
	# fraud_label = np.full(500,1)
	# normal_label = np.zeros(1000)
	# X_train = np.concatenate([normal_data, fraud_data])
	# y_train = np.concatenate([normal_label, fraud_label])
	# X_train, y_train = shuffle(X_train, y_train)

	roc_lr, roc_svm, roc_rf, roc_xgb = 0, 0, 0, 0

	#options for classifier
	if 'lr' in classifier_to_use:
	

		logisticRegr = LogisticRegression(solver='liblinear')
		# logisticRegr = LogisticRegression(solver='lbfgs')
		logisticRegr.fit(X_train, y_train)
		y_predict_lr = logisticRegr.predict(X_test)

		roc_lr = roc_auc_score(y_test,y_predict_lr)
		f1_lr = f1_score(y_test,y_predict_lr)
		p_lr = precision_score(y_test,y_predict_lr)
		r_lr = recall_score(y_test,y_predict_lr)

	if 'svm' in classifier_to_use:
	
		clf = svm.SVC(gamma='scale')
		clf.fit(X_train, y_train)
		y_predict_svm = clf.predict(X_test)
		
		roc_svm = roc_auc_score(y_test,y_predict_svm)
		f1_svm = f1_score(y_test,y_predict_svm)
		p_svm = precision_score(y_test,y_predict_svm)
		r_svm = recall_score(y_test,y_predict_svm)

	if 'rf' in classifier_to_use:

		rf = RandomForestClassifier(n_estimators=100)
		rf.fit(X_train, y_train)
		y_predict_rf = rf.predict(X_test)

		roc_rf = roc_auc_score(y_test,y_predict_rf)
		f1_rf = f1_score(y_test,y_predict_rf)
		p_rf = precision_score(y_test,y_predict_rf)
		r_rf = recall_score(y_test,y_predict_rf)
	
	if 'xgb' in classifier_to_use:
		xgb = XGBClassifier()
		xgb.fit(X_train, y_train)
		y_predict_xgb = xgb.predict(X_test)

		roc_xgb = roc_auc_score(y_test,y_predict_xgb)
		f1_xgb = f1_score(y_test,y_predict_xgb)
		p_xgb = precision_score(y_test,y_predict_xgb)
		r_xgb = recall_score(y_test,y_predict_xgb)

		# ROC
		preds_proba = xgb.predict_proba(X_test)[:,1]
		fpr, tpr, _ = metrics.roc_curve(y_test, preds_proba)
		fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,7))
		ax.plot(fpr,tpr)
		ax.set_title("ROC Curve", fontsize=25)
		ax.set_ylabel("TPR", fontsize=15)
		ax.set_xlabel("FPR", fontsize=15)
		fig.savefig("ROC", bbox_inches='tight')

		# # # Importance
		# # Features = ['0101','0110','0102','0120','0121','0112']
		# # Features = ['0101', '0110', '0102', '0120', '0121', '0112', '010102', '010120', '010112', '010121', '011002', '011020', '011012', '011021', '010201', '010210', '010202', '010220', '010212', '010221', '012001', '012010', '012002', '012020', '012012', '012021', '011201', '011210', '011202', '011220', '011212', '011221', '012101', '012110', '012102', '012120', '012112', '012121']
		# Features = ['s_by_k','sp','wsp','s','k']
		# importances = xgb.feature_importances_
		# weights = pd.Series(importances,index=Features)
		# print(weights)
		# # plot importance
		# # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,7))
		# # ax = weights.sort_values()[-5:].plot(kind = 'barh')
		# # # ax.barh(weights.sort_values()[-5:])
		# # ax.set_title("Feature Importance", fontsize=25)
		# # fig.savefig("Features", bbox_inches='tight')

	return roc_lr, roc_svm, roc_rf, f1_lr, f1_svm, f1_rf, roc_xgb, f1_xgb, p_lr, p_svm, p_rf, p_xgb, r_lr, r_svm, r_rf, r_xgb, len(whole_data)












