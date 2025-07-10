import networkx as nx
import pandas as pd
import numpy as np
import os

import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
from stellargraph.data import BiasedRandomWalk
from node2vec import Node2Vec

from tensorflow.keras import layers, optimizers, losses, metrics, Model

import sklearn
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb

from collections import defaultdict

import sys
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

CSV_PATH = sys.argv[1] # csv of input graph
ML_METHOD = sys.argv[2]
# ML_METHOD = 'LR' or 'SVM' or 'RF' or 'XGB'
# ML_METHOD = 'LR'

df = pd.read_csv(CSV_PATH)

is_node_fraud = defaultdict(int)
edges = []
for index, row in df.iterrows():
    u = str(row['Sender_Id'])
    v = str(row['Bene_Id'])

    if pd.isna(u) or pd.isna(v):
        continue
    
    edges.append([u, v])
    is_node_fraud[u] = max(is_node_fraud[u], int(row['Label']))

df_edges = pd.DataFrame({
    'source': [i[0] for i in edges],
    'target': [i[1] for i in edges]
})

all_nodes = list(set(list(df_edges['source']) + list(df_edges['target'])))

gr = nx.from_edgelist(edges)

data_preprocess_execution_time = time.time() - start_time

node2vec = Node2Vec(gr, temp_folder='temp_folder_for_embedding'.format(ML_METHOD))
model = node2vec.fit()

node_embeddings = (
    model.wv.vectors
)

model.save('embedding_path_out')

node_ids = model.wv.index_to_key
node_targets = [is_node_fraud[idx] for idx in node_ids]

node_embeddings_execution_time = time.time() - (start_time + data_preprocess_execution_time)

X = node_embeddings
y = np.array(node_targets)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.75, test_size=0.25)


def get_clf(method_name):
    if method_name == 'LR':
        return LogisticRegression(solver='liblinear', class_weight='balanced', verbose=1, max_iter=100000)

    if method_name == 'SVM':
        return svm.SVC(class_weight='balanced', verbose=1)

    if method_name == 'RF':
        return RandomForestClassifier(class_weight='balanced', verbose=1)

    if method_name == 'XGB':
        return xgb.XGBClassifier(objective="binary:logistic", verbosity=1)

    return None


clf = get_clf(ML_METHOD)
clf.fit(X_train, y_train)

train_execution_time = time.time() - (start_time + data_preprocess_execution_time + node_embeddings_execution_time)

y_pred = clf.predict(X_test)

print('\nconfusion matrix -> ')
print(sklearn.metrics.confusion_matrix(y_test, y_pred))

print('\nclassification report -> ')
print(sklearn.metrics.classification_report(y_test, y_pred, digits=3))

print('\nPrecision -> ', sklearn.metrics.precision_score(y_test, y_pred))
print('Recall -> ', sklearn.metrics.recall_score(y_test, y_pred))
print('AUC ROC -> ', sklearn.metrics.roc_auc_score(y_test, y_pred))
print('F1 Score -> ', sklearn.metrics.f1_score(y_test, y_pred))

model_evaluation_execution_time = time.time() - (start_time + data_preprocess_execution_time + node_embeddings_execution_time + train_execution_time)

total_execution_time = time.time() - start_time

print('\ndata_preprocess_execution_time ->', data_preprocess_execution_time)
print('gnn_configuration_execution_time ->', node_embeddings_execution_time)
print('train_execution_time ->', train_execution_time)
print('model_evaluation_execution_time ->', model_evaluation_execution_time)
print('total_execution_time ->', total_execution_time)

