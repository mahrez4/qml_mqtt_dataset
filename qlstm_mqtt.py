import matplotlib
matplotlib.use("Agg")   # "Agg" = raster graphics, suitable for PNG
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import seaborn as sns
import argparse
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


p = argparse.ArgumentParser(description="Example parameterized job")
p.add_argument("--backend", required=False, default="CPU", type=str, help="xdd")
p.add_argument("--layers", required=False, default=1, type=int, help="Number of Layers (depth)")
args = p.parse_args()

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

print('size row,col:',len(df.index),len(df.columns))

colnames = list(df.columns)
print(colnames)

print(df.dtypes)

for i in range(len(colnames)):
    print(i,' ',colnames[i],' : ')
    vc = df[colnames[i]].value_counts()
    # vc = vc.reindex(sorted(vc.keys()))
    print(vc)

print(len(df))

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)
dataX = df.drop([ 'label'], axis=1)
dataY = df['label']

train_X = train.drop(['label'], axis=1)
train_Y = train['label']

test_X = test.drop(['label'], axis=1)
test_Y = test['label']

type_class = train['label'].unique()
print(type_class)

# Extract feature name
test_cols = list(test_X.columns)

featurenamefile = 'Feature_use_training.dat'
f = open(featurenamefile,'w')
for name in test_cols:
    f.write(name+'\n')
f.close()

pd.DataFrame({
  'train': train['label'].value_counts(),
  'test': test['label'].value_counts()},
  ).plot.bar(rot=0)

NUM_QUBITS = 11
# Reduce features using PCA (to keep the quantum circuit manageable)
pca = PCA(n_components=NUM_QUBITS)

features = list(df.columns)
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['label']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

principalComponents = pca.fit_transform(x)

new_comps = []
for j in range(NUM_QUBITS):
    st = "comp"+str(j)
    new_comps.append(st)

principalDf = pd.DataFrame(data = principalComponents , columns = new_comps)


finalDf = pd.concat([principalDf, df[['label']]], axis = 1)
finalDf.head(5)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SEQ_LEN = 20      # try 10/20/40
STRIDE  = 10      # hop, try 5/10/20
FEATURES = [f"comp{i}" for i in range(NUM_QUBITS)]

pca_X = finalDf[FEATURES].values.astype(np.float32)  # (N, D)
y_row = finalDf["label"].values.astype(np.int64)     # (N,)

def make_sequences(X, y, T=SEQ_LEN, stride=STRIDE, label_rule="any"):
    xs, ys = [], []
    i = 0
    N = len(X)
    while i + T <= N:
        wX = X[i:i+T]
        wy = y[i:i+T]
        seq_label = int((wy == 1).any()) if label_rule=="any" else int(wy[-1])
        xs.append(wX); ys.append(seq_label)
        i += stride
    return np.stack(xs), np.array(ys)

seq_X, seq_y = make_sequences(pca_X, y_row, T=SEQ_LEN, stride=STRIDE, label_rule="any")
print("Sequences:", seq_X.shape, "Labels:", seq_y.shape)  # (M, T, D), (M,)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

Xtr, Xval, ytr, yval = train_test_split(seq_X, seq_y, test_size=0.2, stratify=seq_y, random_state=42)
train_set = SeqDataset(Xtr, ytr)
val_set   = SeqDataset(Xval, yval)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

def make_qgate(n_qubits, depth):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode(inputs, weights):   # name MUST be 'inputs'
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    weight_shapes = {"weights": (depth, n_qubits, 3)}
    return qml.qnn.TorchLayer(qnode, weight_shapes)

class QLSTMCell(nn.Module):
    """
    Quantum-for-gates LSTM cell:
      z_g ≈ Head_g( QNode_g( in_proj([x_t, h_{t-1}]) ) ) + bias_g
      i,f,o = sigmoid(z_i,z_f,z_o),  c̃ = tanh(z_c)
      c_t = f*c_{t-1} + i*c̃,  h_t = o*tanh(c_t)
    """
    def __init__(self, input_size, hidden_size, n_qubits=8, depth=1):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.n_qubits    = n_qubits

        self.in_proj = nn.Linear(input_size + hidden_size, n_qubits, bias=False)

        self.q_i = make_qgate(n_qubits, depth)
        self.q_f = make_qgate(n_qubits, depth)
        self.q_o = make_qgate(n_qubits, depth)
        self.q_c = make_qgate(n_qubits, depth)

        self.head_i = nn.Linear(n_qubits, hidden_size, bias=True)
        self.head_f = nn.Linear(n_qubits, hidden_size, bias=True)
        self.head_o = nn.Linear(n_qubits, hidden_size, bias=True)
        self.head_c = nn.Linear(n_qubits, hidden_size, bias=True)

    def forward(self, x_t, state):
        h_prev, c_prev = state
        qin = self.in_proj(torch.cat([x_t, h_prev], dim=-1))  # (B, n_qubits)

        z_i = self.head_i(self.q_i(qin))
        z_f = self.head_f(self.q_f(qin))
        z_o = self.head_o(self.q_o(qin))
        z_c = self.head_c(self.q_c(qin))

        i_t = torch.sigmoid(z_i)
        f_t = torch.sigmoid(z_f)
        o_t = torch.sigmoid(z_o)
        c_hat = torch.tanh(z_c)

        c_t = f_t * c_prev + i_t * c_hat
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits=8, depth=1):
        super().__init__()
        self.cell = QLSTMCell(input_size, hidden_size, n_qubits, depth)
        self.hidden_size = hidden_size

    def forward(self, x, state=None):   # x: (B, T, D)
        B, T, D = x.shape
        if state is None:
            h = x.new_zeros(B, self.hidden_size)
            c = x.new_zeros(B, self.hidden_size)
        else:
            h, c = state
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1), (h, c)

class QLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_qubits=8, depth=1, num_classes=2):
        super().__init__()
        self.rnn  = QLSTM(input_size, hidden_size, n_qubits, depth)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):                  # x: (B, T, D)
        _, (hT, cT) = self.rnn(x)          # take final hidden state
        logits = self.head(hT)
        return logits

from tqdm import tqdm
import time

device = torch.device("cuda" if (torch.cuda.is_available() and args.backend == "GPU") else "cpu")

model = QLSTMClassifier(
    input_size=NUM_QUBITS,   
    hidden_size=64,          
    n_qubits=NUM_QUBITS,
    depth=1,             
    num_classes=2
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

EPOCHS = 5
log_every = 50   # print a short line every N batches


start = time.time()
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    model.train()
    tot, correct, loss_sum = 0, 0, 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}")
    for step, (xb, yb) in pbar:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # running stats
        loss_sum += loss.item() * xb.size(0)
        pred = logits.argmax(1)
        tot += xb.size(0)
        correct += (pred == yb).sum().item()

        if (step + 1) % log_every == 0:
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "running_acc": f"{(correct/tot):.3f}"})

    train_loss = loss_sum / tot
    train_acc  = correct / tot

    # validation with its own bar
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc="Validating", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(1)
            tot += xb.size(0)
            correct += (pred == yb).sum().item()
    val_loss = loss_sum / tot
    val_acc  = correct / tot
    t1 = time.time()

    print(f"Epoch {epoch:02d} | train {train_loss:.4f}/{train_acc:.3f} "
          f"| val {val_loss:.4f}/{val_acc:.3f} | took {(t1-t0):.1f}s")


end = time.time()
elapsed = end - start
print(f"Total training time: {elapsed:.2f} seconds")

# --- Final metrics on the validation set ---
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

model.eval()
all_logits = []
all_probs  = []
all_preds  = []
all_true   = []

with torch.no_grad():
    for xb, yb in DataLoader(val_set, batch_size=256, shuffle=False):
        xb = xb.to(device)
        logits = model(xb)                       # (B, 2)
        probs  = torch.softmax(logits, dim=1)    # (B, 2)
        preds  = logits.argmax(1).cpu().numpy()  # predicted class ids

        all_logits.append(logits.cpu())
        all_probs.append(probs[:, 1].cpu().numpy())  # prob of class 1
        all_preds.append(preds)
        all_true.append(yb.numpy())

import numpy as np
y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_preds)
y_prob = np.concatenate(all_probs)

# Binary metrics (positive class = 1)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", pos_label=1, zero_division=0
)
auc = roc_auc_score(y_true, y_prob)

print("\n=== Final Validation Metrics ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC AUC:   {auc:.4f}")

# Per-class + macro/weighted summary
print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
tn, fp, fn, tp = cm.ravel()
print("\nConfusion matrix (rows=true, cols=pred):")
print(cm)
print(f"\nTN={tn}  FP={fp}  FN={fn}  TP={tp}")
