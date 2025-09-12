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
p.add_argument("--fraction", required=False, default=0.001, type=float, help ="xdd")
args = p.parse_args()

data_frac = args.fraction
#feature to extract
featurenamefile = '/kaggle/input/feature-name/Feature_name.dat'
with open(featurenamefile) as file:
    feature_name = [line.rstrip() for line in file]
file.close()
ft_dict = {key: i for i, key in enumerate(feature_name)}
n_feature = len(feature_name)

filename='/kaggle/input/ids-linh/captured_dataset_07-04-2023_1226.csv'
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

print(finalDf.shape)

class TabularDataFrameDataset(Dataset):
    def __init__(self, df, feature_cols, label_col):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[label_col].values, dtype=torch.long)  # or float for regression

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Desired fraction of total dataset
frac = data_frac

# Sample from each label group proportionally
subset_df = finalDf.groupby("label", group_keys=False).apply(
    lambda x: x.sample(frac=frac, random_state=42)
)

print("Original distribution:")
print(finalDf["label"].value_counts(normalize=True))

print("\nSubset distribution:")
print(subset_df["label"].value_counts(normalize=True))

feature_cols = ["comp0","comp1","comp2","comp3","comp4","comp5","comp6","comp7","comp8","comp9","comp10"]
label_col = "label"
dataset = TabularDataFrameDataset(subset_df, feature_cols, label_col)


print(dataset.X.size())
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

#for i in range(len(dataset)):
#    data, label = dataset[i]
#    if label == 1:
#        print(f"Found at index {i}")
#        print("Data:", data)
#        print("Label:", label)
#        break

# ======================================================
# Quantum Node
# ======================================================
def create_qnode(n_qubits, depth):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def qnode(inputs, initial_layer_weights, weights):
        # Keep your data embedding
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # SimplifiedTwoDesign ansatz (trainable)
        qml.SimplifiedTwoDesign(
            initial_layer_weights=initial_layer_weights,  # (n_qubits,)
            weights=weights,                               # (depth, n_qubits-1, 2)
            wires=range(n_qubits)
        )
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return qnode

# ======================================================
# Quantum Classifier (no MIL)
# ======================================================

class QNNClassifier(nn.Module):
    def __init__(self, n_qubits=NUM_QUBITS, num_classes=2, depth=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.qnode = create_qnode(n_qubits, depth)

        # IMPORTANT: two separate parameter tensors for SimplifiedTwoDesign
        weight_shapes = {
            "initial_layer_weights": (n_qubits,),
            "weights": (depth, n_qubits - 1, 2),
        }

        self.q_layer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self.classifier = nn.Linear(n_qubits, num_classes)

    def forward(self, x, return_features: bool = False):
        x = x[:, :self.n_qubits]
        q_out = self.q_layer(x)        # (B, n_qubits)
        logits = self.classifier(q_out)
        return (logits, q_out) if return_features else logits


# ======================================================
# Contrastive Loss
# ======================================================
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        return F.cross_entropy(sim, labels)


# ======================================================
# Training + Validation
# ======================================================
def train_joint(model, dataloader, optimizer, criterion_cls, criterion_cont, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for features, labels in tqdm(dataloader, desc="Training"):
        features, labels = features.to(device), labels.to(device)
        noisy = features + 0.3 * torch.randn_like(features)

        logits_clean, z_clean = model(features, return_features=True)
        logits_noisy, z_noisy = model(noisy,   return_features=True)

        loss_cls  = criterion_cls(logits_clean, labels)
        loss_cont = criterion_cont(z_clean, z_noisy)
        loss = loss_cls + 0.1 * loss_cont

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits_clean, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1, all_preds, all_labels

def validate(model, dataloader, criterion_cls, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion_cls(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, accuracy, f1, all_preds, all_labels


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader   = DataLoader(test_set,   batch_size=16)

device = torch.device("cuda" if (torch.cuda.is_available() and args.backend == "GPU") else "cpu")

DEPTH = 1

model = QNNClassifier(n_qubits=NUM_QUBITS, num_classes=2, depth=DEPTH).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(8):
    train_loss, train_acc, train_f1, _, _ = train_joint(
        model, train_loader, optimizer, criterion_cls, criterion_cont, device
    )
    val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion_cls, device)

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    roc_auc_score
)

def evaluate_full_dataset(model, dataset, device, batch_size=64, class_names=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_logits = []
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)                    # shape: (B, C)
            probs  = F.softmax(logits, dim=1)    # for log_loss / AUC
            preds  = torch.argmax(logits, dim=1)

            all_logits.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    y_true = np.array(all_labels)
    y_prob = np.vstack(all_logits)              # (N, C)
    y_pred = np.array(all_preds)

    n_classes = y_prob.shape[1]
    is_binary = n_classes == 2

    # --- Core metrics ---
    metrics = {}
    metrics["accuracy"]            = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"]   = balanced_accuracy_score(y_true, y_pred)
    metrics["precision_macro"]     = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision_weighted"]  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["recall_macro"]        = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_weighted"]     = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["f1_macro"]            = f1_score(y_true, y_pred, average="macro")
    metrics["f1_weighted"]         = f1_score(y_true, y_pred, average="weighted")
    metrics["cohen_kappa"]         = cohen_kappa_score(y_true, y_pred)
    metrics["mcc"]                 = matthews_corrcoef(y_true, y_pred)

    # Log loss (needs valid probabilities; small epsilon stabilization)
    eps = 1e-12
    metrics["log_loss"] = log_loss(y_true, np.clip(y_prob, eps, 1 - eps), labels=np.arange(n_classes))

    # ROC-AUC (binary: positive class prob; multiclass: OVO macro)
    try:
        if is_binary:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics["roc_auc_ovo_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro")
    except Exception:
        # In case a class is missing in y_true, AUC can fail
        pass

    # --- Per-class report & confusion matrix (numbers only; no plotting) ---
    report = classification_report(
        y_true, y_pred,
        target_names=class_names if class_names is not None and len(class_names) == n_classes else None,
        digits=4,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    # --- Optional: distributions (helps spot imbalance) ---
    class_counts_true = np.bincount(y_true, minlength=n_classes)
    class_counts_pred = np.bincount(y_pred, minlength=n_classes)

    # Print neatly
    print("=== Summary Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>20s}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k:>20s}: {v}")

    print("\n=== Per-class Report ===")
    print(report)

    print("=== Confusion Matrix (rows=true, cols=pred) ===")
    print(cm)

    print("\n=== Class Distributions ===")
    print("True counts:     ", class_counts_true)
    print("Predicted counts:", class_counts_pred)

    return metrics, report, cm, class_counts_true, class_counts_pred

# ---- Usage ----
# After training:
metrics, report, cm, true_counts, pred_counts = evaluate_full_dataset(model, dataset, device, batch_size=64, class_names=["class0","class1"])
