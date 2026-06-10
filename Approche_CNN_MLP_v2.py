#!/usr/bin/env python
# coding: utf-8

# # Data Challenge Approche 1 avec Training 

# #### import  TEST
# 



# In[3]:


import os, random, json, io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches, matplotlib.patches as mpatches
import matplotlib.image as mpimg
import requests
from statsmodels.stats.proportion import proportion_confint
from dotenv import load_dotenv
load_dotenv(override=True)
from torch.utils.data import DataLoader


# In[4]:


import transformers
print(torch.__version__)          # doit être >= 2.3.0
print(transformers.__version__)   # doit être 4.49.0

print(f"PyTorch version   : {torch.__version__}")
print(f"CUDA disponible   : {torch.cuda.is_available()}")
print(f"CUDA version      : {torch.version.cuda}")
print(f"Nombre de GPU     : {torch.cuda.device_count()}")
print(f"GPU détecté       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun'}")
print(f"Device actif      : {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")


# #### Chargement Data

# In[5]:


df_train = pd.read_csv("occlusion_datasets/train.csv", delimiter=',')
df_test = pd.read_csv("occlusion_datasets/test_students.csv", delimiter=',')

image_dir = "crops/Crop_224_5fp_100K"


# In[6]:


df_train = df_train.dropna()
df_test = df_test.dropna()


# In[8]:


from sklearn.model_selection import train_test_split

# On garde le vrai test officiel séparé
df_full = df_train.copy()

# 70% train / 30% temporaire
df_train, df_temp = train_test_split(
    df_full,
    test_size=0.30,
    random_state=42,
    shuffle=True,
    stratify=df_full["gender"]
)

# 15% validation / 15% test interne
df_val, df_internal_test = train_test_split(
    df_temp,
    test_size=0.50,
    random_state=42,
    shuffle=True,
    stratify=df_temp["gender"]
)

print(f"Train interne : {len(df_train)}")
print(f"Validation interne : {len(df_val)}")
print(f"Test interne : {len(df_internal_test)}")

# In[9]:


IMAGE_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class FaceOcclusionDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, training=True):
        self.training  = training
        self.image_dir = image_dir
        self.df        = df.reset_index(drop=True)
        self.transform = IMAGE_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row      = self.df.loc[index]
        filename = row['filename']
        img      = Image.open(
                       os.path.join(self.image_dir, filename)
                   ).convert("RGB")
        X = self.transform(img)

        if self.training:
            y      = np.float32(row['FaceOcclusion'])
            gender = row['gender']
            return X, y, gender, filename
        else:
            return X, filename




# In[10]:


training_set   = FaceOcclusionDataset(df_train, image_dir, training=True)
validation_set = FaceOcclusionDataset(df_val,   image_dir, training=True)
test_set       = FaceOcclusionDataset(df_test,  image_dir, training=False)
internal_test_set = FaceOcclusionDataset(df_internal_test, image_dir, training=True)

params_train = {'batch_size': 32, 'shuffle': True,  'num_workers': 0, 'pin_memory': True}
params_val   = {'batch_size': 32, 'shuffle': False, 'num_workers': 0, 'pin_memory': True}

training_generator   = DataLoader(training_set,   **params_train)
validation_generator = DataLoader(validation_set, **params_val)
test_generator       = DataLoader(test_set,        **params_val)
internal_test_generator = DataLoader(internal_test_set, **params_val)

print(
    f"Train : {len(training_set)} | "
    f"Val : {len(validation_set)} | "
    f"Test interne : {len(internal_test_set)} | "
    f"Test officiel : {len(test_set)}"
)


# #### Le Modele chargement et architecture

# In[11]:


from torchvision.models import resnet50, ResNet50_Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

os.makedirs("runs/resnet50_v2", exist_ok=True)

# Modèle ResNet50 pré-entraîné
model_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Remplacement de la tête de classification par une tête de régression
in_features = model_resnet.fc.in_features

model_resnet.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

model_resnet = model_resnet.to(device)


# In[12]:


class WeightedMSELoss(nn.Module):
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        weights = 1/30 + target
        loss = weights * (pred - target) ** 2

        return loss.sum() / weights.sum()

criterion = WeightedMSELoss()


# In[13]:


optimizer = torch.optim.AdamW(
    model_resnet.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

N_EPOCHS = 20


# In[14]:


best_val_loss = float("inf")

for epoch in range(N_EPOCHS):
    model_resnet.train()
    train_loss = 0.0
    train_mae = 0.0

    for X, y, gender, filenames in tqdm(training_generator, desc=f"Epoch {epoch+1}/{N_EPOCHS} - train"):
        X = X.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()

        pred = model_resnet(X).view(-1)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mae += torch.abs(pred.detach() - y).mean().item()

    train_loss /= len(training_generator)
    train_mae /= len(training_generator)

    model_resnet.eval()
    val_loss = 0.0
    val_mae = 0.0

    with torch.no_grad():
        for X, y, gender, filenames in tqdm(validation_generator, desc=f"Epoch {epoch+1}/{N_EPOCHS} - val"):
            X = X.to(device)
            y = y.to(device).float()

            pred = model_resnet(X).view(-1)
            loss = criterion(pred, y)

            val_loss += loss.item()
            val_mae += torch.abs(pred - y).mean().item()

    val_loss /= len(validation_generator)
    val_mae /= len(validation_generator)

    print(
        f"Epoch {epoch+1}/{N_EPOCHS} | "
        f"train_loss={train_loss:.5f} train_mae={train_mae:.5f} | "
        f"val_loss={val_loss:.5f} val_mae={val_mae:.5f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_resnet.state_dict(), "runs/resnet50_v2/resnet50_best.pth")
        print("✓ meilleur modèle sauvegardé")


# In[ ]:


# ============================================================
# ÉVALUATION OFFICIELLE SUR VALIDATION
# ============================================================

import numpy as np
import torch

# Recharger le meilleur modèle sauvegardé
model_resnet.load_state_dict(torch.load("runs/resnet50_v2/resnet50_best.pth", map_location=device))
model_resnet.eval()

all_preds = []
all_targets = []
all_genders = []

with torch.no_grad():
    for X, y, gender, filenames in tqdm(validation_generator, desc="Evaluation officielle"):
        X = X.to(device)
        y = y.to(device).float()

        pred = model_resnet(X).view(-1)
        pred = torch.clamp(pred, 0, 1)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        all_genders.extend(gender)

preds = np.array(all_preds)
targets = np.array(all_targets)
genders = np.array(all_genders)

def weighted_error(preds, targets):
    weights = 1/30 + targets
    return np.sum(weights * (preds - targets) ** 2) / np.sum(weights)

err_by_gender = {}

for g in np.unique(genders):
    mask = genders == g
    err_by_gender[g] = weighted_error(preds[mask], targets[mask])

err_values = list(err_by_gender.values())

official_score = np.mean(err_values) + abs(err_values[0] - err_values[1])

mae = np.mean(np.abs(preds - targets))
rmse = np.sqrt(np.mean((preds - targets) ** 2))

print("MAE validation :", mae)
print("RMSE validation :", rmse)
print("Err par genre :", err_by_gender)
print("Score officiel validation :", official_score)

# ============================================================
# ÉVALUATION SUR TEST INTERNE
# ============================================================

model_resnet.load_state_dict(
    torch.load("runs/resnet50_v2/resnet50_best.pth", map_location=device)
)
model_resnet.eval()

test_preds = []
test_targets = []
test_genders = []

with torch.no_grad():
    for X, y, gender, filenames in tqdm(internal_test_generator, desc="Evaluation test interne"):
        X = X.to(device)
        y = y.to(device).float()

        pred = model_resnet(X).view(-1)
        pred = torch.clamp(pred, 0, 1)

        test_preds.extend(pred.cpu().numpy())
        test_targets.extend(y.cpu().numpy())
        test_genders.extend(gender)

test_preds = np.array(test_preds)
test_targets = np.array(test_targets)
test_genders = np.array(test_genders)

test_mae = np.mean(np.abs(test_preds - test_targets))
test_mse = np.mean((test_preds - test_targets) ** 2)
test_rmse = np.sqrt(test_mse)

test_err_by_gender = {}

for g in np.unique(test_genders):
    mask = test_genders == g
    test_err_by_gender[g] = weighted_error(test_preds[mask], test_targets[mask])

test_err_values = list(test_err_by_gender.values())
test_official_score = np.mean(test_err_values) + abs(test_err_values[0] - test_err_values[1])

print("MAE test interne :", test_mae)
print("MSE test interne :", test_mse)
print("RMSE test interne :", test_rmse)
print("Err test interne par genre :", test_err_by_gender)
print("Score officiel test interne :", test_official_score)