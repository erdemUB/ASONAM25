import networkx as nx
import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE

from tensorflow.keras import layers, optimizers, losses, metrics, Model

import sklearn
from sklearn import preprocessing, feature_extraction, model_selection

from collections import defaultdict

import sys
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

CSV_PATH = sys.argv[1] # CSV_PATH of the input file
df = pd.read_csv(CSV_PATH)

is_node_fraud = defaultdict(int)
edges = []
for index, row in df.iterrows():
    u = row['Sender_Id']
    v = row['Bene_Id']

    if pd.isna(u) or pd.isna(v):
        continue

    edges.append([u, v])
    is_node_fraud[u] = max(is_node_fraud[u], int(row['Label']))

df_edges = pd.DataFrame({
    'source': [i[0] for i in edges],
    'target': [i[1] for i in edges]
})

all_nodes = list(set(list(df_edges['source']) + list(df_edges['target'])))

idx = 0
identity_features = [[] for _ in range(len(all_nodes))]
for u in all_nodes:
    for i in range(idx):
        identity_features[idx].append(0)
    identity_features[idx].append(1)
    for i in range(idx + 1, len(all_nodes)):
        identity_features[idx].append(0)
    idx = idx + 1

print('identity features -> ', len(identity_features), len(identity_features[0]))

df_nodes = pd.DataFrame(identity_features, index = all_nodes)

data_preprocess_execution_time = time.time() - start_time

gr = sg.StellarGraph(nodes=df_nodes, edges=df_edges)
# gr = sg.StellarGraph(edges=df_edges)
print(gr.info())

gr_nodes = gr.nodes()
node_subjects = pd.Series(data=[is_node_fraud[i] for i in gr_nodes], index=gr_nodes)

train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=0.75, test_size=0.25, stratify=node_subjects
)

target_encoding = preprocessing.LabelBinarizer()
train_targets = target_encoding.fit_transform(train_subjects)
test_targets = target_encoding.transform(test_subjects)

batch_size = int(sys.argv[2])
num_samples = [25, 10]
generator = GraphSAGENodeGenerator(gr, batch_size, num_samples)

train_gen = generator.flow(train_subjects.index, train_targets, shuffle=True)
test_gen = generator.flow(test_subjects.index, test_targets)

graphsage_model = GraphSAGE(
    layer_sizes=[32, 32], generator=generator, bias=True, dropout=0.2
)

x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss=losses.binary_crossentropy,
    metrics=['acc'],
    # Let us rely on the sklearn.metrics here since tf might have different naming conventions upon different versions. 
    # Therefore, sklearn.metrics is more robust IMHO.
)

gnn_configuration_execution_time = time.time() - (start_time + data_preprocess_execution_time)

history = model.fit(
    train_gen, epochs=int(sys.argv[3]), validation_data=test_gen, verbose=2, shuffle=False
)

train_exection_time = time.time() - (start_time + data_preprocess_execution_time + gnn_configuration_execution_time)

y_pred = [1 if i >= 0.5 else 0 for i in model.predict(test_gen)]
y_test = test_targets

print('\nconfusion matrix -> ')
print(sklearn.metrics.confusion_matrix(y_test, y_pred))

print('\nclassification report -> ')
print(sklearn.metrics.classification_report(y_test, y_pred, digits=3))

print('\nPrecision -> ', sklearn.metrics.precision_score(y_test, y_pred))
print('Recall -> ', sklearn.metrics.recall_score(y_test, y_pred))
print('AUC ROC -> ', sklearn.metrics.roc_auc_score(y_test, y_pred))
print('F1 Score -> ', sklearn.metrics.f1_score(y_test, y_pred))

model_evaluation_execution_time = time.time() - (start_time + data_preprocess_execution_time + gnn_configuration_execution_time + train_exection_time)

total_execution_time = time.time() - start_time

print('\ndata_preprocess_execution_time ->', data_preprocess_execution_time)
print('gnn_configuration_execution_time ->', gnn_configuration_execution_time)
print('train_exection_time ->', train_exection_time)
print('model_evaluation_execution_time ->', model_evaluation_execution_time)
print('total_execution_time ->', total_execution_time)

