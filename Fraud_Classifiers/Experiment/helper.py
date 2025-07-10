import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from collections import defaultdict
import pickle
from collections import Counter


ep = ['0101','0110','0102','0120','0121','0112']
ev = ['01','10','02','20','12','21']
ev_pair = []
for a in ['S','M','L']:
	for x in ev:
		for y in ev:
			if x ==  '01':
				if y== '01' or y == '10':
					continue
			elif x ==  '10':
				if y== '01' or y == '10':
					continue
			cur = a+'01'+x+y
			ev_pair.append(cur)


# 's_by_k','sp','wsp','s','k' are the static features
features = ['s_by_k','sp','wsp','s','k']+ep+ev_pair






def reject_outliers(data, m=2):
    #return data[abs(data - np.mean(data)) < m * np.std(data)]
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d/mdev if mdev else 0.
	return data[s<m]



def load_pickle(pickle_file):
	"""a function to load pickle files"""

	try:
		with open(pickle_file, 'rb') as f:
			pickle_data = pickle.load(f)
	except UnicodeDecodeError as e:
		with open(pickle_file, 'rb') as f:
			pickle_data = pickle.load(f, encoding='latin1')
	except Exception as e:
		print('Unable to load data ', pickle_file, ':', e)
		raise
	return pickle_data




def id_rename(name):
    # 0 = CLIENT
    # 1 = COMPANY
    # 2 = ELSE
	if name != name:
		return 9999
        
	else:
		splitted = name.split("-")
		if len(splitted) >2:
			return int('3'+splitted[-1])
		else:
			if splitted[0] == 'CLIENT':
				return int('1'+splitted[-1])
			else:
				return int('2'+splitted[-1])

def seller_seed(motif,vert):
	ans = False
	if len(motif) > 4:
		s1, s2 = int(motif[1]), int(motif[3])
	else:
		s1, s2 = int(motif[0]), int(motif[2])
	if vert[s1] == '1' or vert [s2] == '1':
		ans = True
	else:	
		if len(motif) == 6:
			s3 = int(motif[4])
			if vert[s3]: 
				ans = True

	return ans
		


def convert_times(times):
	dt_obj = dt.datetime.strptime(times, '%Y-%m-%d %H:%M:%S')
	return dt.datetime.timestamp(dt_obj)	 



def create_motif_dic(filename):
	M = np.loadtxt(filename, dtype=str, usecols=(0,1))
	op_dic = {}
	if len(M)>4:
		M = M[4:]
		for x in M:
			motif, count = x[0], int(x[1])
			broad_motif, vert, edge = motif.split("|")
			if broad_motif not in op_dic:
				op_dic[broad_motif] = [0,0]


			if seller_seed(broad_motif,vert):
				op_dic[broad_motif][1] += count
			else:
				op_dic[broad_motif][0] += count


	return op_dic



def save_len_dic(path):

	dic = {}
	datatype = path.split("/")[-1]
	for ind,infile in enumerate(os.listdir(path)):
		if infile.endswith(".txt"):
			arr = np.loadtxt(path+infile)
			if len(arr.shape) > 1:
				sellers, buyers = set(arr[:,0]), set(arr[:,1])
				all_users = sellers.union(buyers)
				key = infile[:-4].split("_")[-1]
				dic[key] = len(all_users) - 1
				if ind%10 == 0:
					print(ind)


	pickle.dump(dic,open(path+'len_dic.pkl','wb'))



def create_valid_list(path):
	iet_mean_list, iet_median_list = [], []
	for ind,infile in enumerate(os.listdir(path)):
		if infile.endswith(".txt"):
			# print(ind,infile)
			arr = np.loadtxt(path+infile)
			count = 0
			
			if len(arr.shape) > 1 and arr.shape[0] > 20:

				times = arr[:,2]
				iet = np.array(sorted(reject_outliers(np.ediff1d(times,5.98))))
				iet_mean_list.append(np.mean(iet))
				iet_median_list.append(np.median(iet))

				if ind % 10 == 0:
					print(ind)
	# print(iet_mean_list)
	# print(iet_median_list)

	print("Max of iet mean :",max(reject_outliers(np.array(iet_mean_list),5.98)))
	#print("Max of iet median :",max(reject_outliers(np.array(iet_median_list),2)))
