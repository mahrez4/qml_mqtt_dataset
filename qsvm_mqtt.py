import matplotlib
matplotlib.use("Agg")   # "Agg" = raster graphics, suitable for PNG
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import seaborn as sns
import argparse
import time
from pathlib import Path
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics


p = argparse.ArgumentParser(description="Example parameterized job")
p.add_argument("--backend", required=False, default="CPU", type=str, help="xdd")
p.add_argument("--encoding", required=False, default="angle", type=str, help="xdd") 
p.add_argument("--fraction", required=False, default=0.001, type=float, help ="xdd")
args = p.parse_args()

if args.backend == "GPU":
    backend_type = "lightning.gpu"
elif args.backend == "CPU":
    backend_type = "default.qubit"

data_frac = args.fraction

#feature to extract
featurenamefile = 'Feature_name.dat'
with open(featurenamefile) as file:
    feature_name = [line.rstrip() for line in file]
file.close()
ft_dict = {key: i for i, key in enumerate(feature_name)}
n_feature = len(feature_name)


filename='captured_dataset_07-04-2023_1226.csv'
df = pd.read_csv(filename,na_values='',low_memory=False,names=feature_name, header=None)
df.head(5)

df['label'] = 0
df.loc[df["ip.src"] == '10.45.0.3', "label"] = 1
df.head()

drop_cols = ['ip.protocol','ip.src', 'ip.dst', 'mqtt.clientid', 'mqtt.willtopic', 'mqtt.willmsg','mqtt.username','mqtt.passwd', 'mqtt.topic','mqtt.msgid','mqtt.msg'] 
df.drop(drop_cols,axis=1,inplace=True)
df.fillna(0,inplace=True)
df.head(5)

df.loc[0,'frame.time_delta'] = 0
df.sort_values('frame.time_relative',inplace=True)
df.head(5)

df.loc[df["mqtt.protoname"] == 'MQTT', "mqtt.protoname"] = 1
df.head()

df['mqtt.protoname'] = df['mqtt.protoname'].apply(pd.to_numeric) 
bin_col = ['mqtt.dupflag','mqtt.retain','mqtt.protoname','mqtt.conflag.uname','mqtt.conflag.passwd','mqtt.conflag.willretain','mqtt.conflag.willqos','mqtt.conflag.willflag','mqtt.conflag.cleansess','mqtt.conflag.reserved','mqtt.conact.flags.sp']

df['label'] = df['label'].astype(int)

train, test = train_test_split(df, test_size=0.2)
dataX = df.drop([ 'label'], axis=1)
dataY = df['label']

train_X = train.drop(['label'], axis=1)
train_Y = train['label']

test_X = test.drop(['label'], axis=1)
test_Y = test['label']

type_class = train['label'].unique()

# Extract feature name
test_cols = list(test_X.columns)
featurenamefile = 'Feature_use_training.dat'
f = open(featurenamefile,'w')
for name in test_cols:
    f.write(name+'\n')
f.close()


print("SVC start")
svm = SVC(kernel='linear')
st=time.time()
svm.fit(train_X, train_Y)
et=time.time()
y_pred_svm = svm.predict(test_X)
accuracy_svm = np.mean(y_pred_svm == test_Y) * 100
print(f"Classical Kernel SVM Accuracy: {accuracy_svm:.2f}%")
print(f"Total Time Taken by Classical Kernel:{et-st}")


#Train
lsvc_model = LinearSVC(C=0.1, dual=False, max_iter=10000)
lsvc_model.fit(train_X, train_Y)

lsvc_preds = lsvc_model.predict(train_X)
lsvc_accuracy = metrics.accuracy_score(train_Y, lsvc_preds)
print('train acc:',lsvc_accuracy)

#Test
lsvc_test_preds = lsvc_model.predict(test_X)
lsvc_test_accuracy = metrics.accuracy_score(test_Y, lsvc_test_preds)
print('test acc:',lsvc_test_accuracy)

#Evaluate
print(metrics.classification_report(test_Y,lsvc_test_preds))

conf_matrix = metrics.confusion_matrix(test_Y, lsvc_test_preds, labels=type_class)
metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=type_class).plot(xticks_rotation=90)

# precision_score, recall_score, fbeta_score, count = metrics.precision_recall_fscore_support(test_Y,lsvc_test_preds,labels=type_class)
# print(precision_score, recall_score, diskfbeta_score)

from sklearn.metrics import roc_curve,RocCurveDisplay
roc_display = RocCurveDisplay.from_estimator(lsvc_model,test_X,test_Y)

import joblib

modelfile = 'IDSF_model'
filetype = '.joblib'

# linearSVC
filename = modelfile + '_LSVC' + filetype
joblib.dump(lsvc_model,filename)

### Temp cell, Getting optimal PCA Components
pca = PCA()


features = list(df.columns)
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['label']].values

X_scaled = StandardScaler().fit_transform(x)
pca.fit(X_scaled)

plt.figure()  # start a fresh figure
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig("qsvm_cumulative_var.png")       

threshold = 0.91
# For 90% explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance >= threshold) + 1

print(f"Number of components that explain {threshold*100}% variance: {n_components}")

#function to reduce features using PCA and reduce the dataset size
def shrink_dataset_and_pca(df,num_components,fraction):
    # Reduce features using PCA (to keep the quantum circuit manageable)
    pca = PCA(n_components=num_components)
    
    features = list(df.columns)
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:,['label']].values
    
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    
    principalComponents = pca.fit_transform(x)
    
    new_comps = []
    for j in range(num_components):
        st = "comp"+str(j)
        new_comps.append(st)
    
    principalDf = pd.DataFrame(data = principalComponents, columns = new_comps)
    num_samples = int(fraction * len(principalComponents))
    
    # First split: take a stratified subset
    X_subset, _, y_subset, _ = train_test_split(
        principalComponents, y,
        train_size=num_samples,   # directly pick fraction
        stratify=y,               # preserve label distribution
        random_state=42
    )
    
    # Second split: make train/test from the subset, also stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset,
        test_size=0.2,
        stratify=y_subset,
        random_state=42
    )

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = shrink_dataset_and_pca(df,8,0.002)
print("Size of Train_set:", X_train.shape)
print("Size of Test_set:", X_test.shape)

import pennylane as qml
import multiprocessing as mp
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

NUM_QUBITS = 11

# Define feature map options
def angle_feature_map(x, wires):
    """Angle Embedding feature map with additional rotations"""
    x = np.pi * (x - min(x)) / (max(x) - min(x) + 1e-6)
    qml.AngleEmbedding(x, wires=wires, rotation='Z')
    for i in range(NUM_QUBITS):
        qml.RY(x[i], wires=i)

def amplitude_feature_map(x, wires):
    """Amplitude Embedding feature map with padding and normalization"""
    x = x / np.sqrt(np.sum(np.abs(x)**2) + 1e-6)
    qml.AmplitudeEmbedding(x, wires=wires, pad_with=0.0, normalize=True)
    for i in range(NUM_QUBITS):
        qml.RZ(weights[i], wires=i)

FEATURE_MAPS = {
    'angle': angle_feature_map,
    'amplitude': amplitude_feature_map
}


def make_device():
    """Create a single GPU/CPU device"""
    return qml.device(backend_type, wires=NUM_QUBITS)

def quantum_kernel_element(x1, x2, dev, weights, feature_map_type='angle'):
    feature_map = FEATURE_MAPS[feature_map_type]
    
    @qml.qnode(dev)
    def circuit(weights):
        feature_map(x1, wires=range(NUM_QUBITS))
        qml.adjoint(feature_map)(x2, wires=range(NUM_QUBITS))
        return qml.probs(wires=range(NUM_QUBITS))
    
    return circuit(weights)[0].item()

def compute_kernel_matrix(X1, X2, weights, feature_map_type='angle'):
        """Compute kernel matrix using a single GPU"""
        dev = make_device()
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = quantum_kernel_element(X1[i], X2[j], dev, weights, feature_map_type)
        
        return K

#!/usr/bin/env python3
# file: worker.py
X_train, X_test, y_train, y_test = shrink_dataset_and_pca(df,NUM_QUBITS,data_frac)

feature_map_type = args.encoding
weights = np.random.uniform(low=0, high=2*np.pi, size=(NUM_QUBITS,))

print("Computing training kernel matrix...")
K_train = compute_kernel_matrix(X_train, X_train, weights, feature_map_type)
print("Computing test kernel matrix...")
K_test = compute_kernel_matrix(X_test, X_train, weights, feature_map_type)

# clf = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
clf = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
clf.fit(K_train, y_train)

train_acc = accuracy_score(y_train, clf.predict(K_train))
test_acc = accuracy_score(y_test, clf.predict(K_test))
    
print(f"Training accuracy with {feature_map_type} feature map: {train_acc:.4f}")
print(f"Test accuracy with {feature_map_type} feature map: {test_acc:.4f}")

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, K_train, y_train, cv=5)
print(f"Cross-validation scores: {scores.mean():.4f} ± {scores.std():.4f}")

clf_classical = SVC(kernel='rbf', C=1.0, class_weight='balanced')
clf_classical.fit(X_train, y_train)
print(f"Classical RBF Test Accuracy: {clf_classical.score(X_test, y_test):.4f}")