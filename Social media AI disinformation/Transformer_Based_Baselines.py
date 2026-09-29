
import os

import pandas as pd
import numpy as np
import os,  random, tempfile

from sklearn.metrics import precision_score, recall_score, f1_score,   confusion_matrix, roc_auc_score
from sklearn.utils import resample
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit,KFold
from sympy import python
from sympy.abc import lamda
from transformers import DebertaForSequenceClassification, TrainerCallback, TrainerControl, TrainerState, \
    RobertaForSequenceClassification, RobertaTokenizer, TrainingArguments, Trainer, AutoTokenizer, \
    AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import classification_report, accuracy_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import torch
import torch.nn as nn
from transformers import XLNetForSequenceClassification, XLNetTokenizer, BertTokenizer, BertForSequenceClassification
import nltk
import shutil
import warnings
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
from pathlib import Path
from data_sequence import post_level_data

warnings.filterwarnings("ignore")

# nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

class SBERT_CNN_Model(nn.Module):
    def __init__(self,
                 transformer_model_name='sentence-transformers/all-mpnet-base-v2',
                 conv_filters=128,
                 conv_kernel_size=3,
                 num_classes=2,
                 dropout=0.5,
                 loss=None):

        super(SBERT_CNN_Model, self).__init__()

        # SBERT backbone (Transformer only, no classification head)
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

        hidden_size = self.transformer.config.hidden_size

        # CNN layers (same as your original design)
        self.conv1 = nn.Conv1d(in_channels=hidden_size,
                               out_channels=conv_filters * 2,
                               kernel_size=conv_kernel_size,
                               padding=1)

        self.conv2 = nn.Conv1d(in_channels=conv_filters * 2,
                               out_channels=conv_filters,
                               kernel_size=conv_kernel_size,
                               padding=1)

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(conv_filters, num_classes)

        # Loss function
        if loss == "FocalLoss":
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif loss == "FedFocalLoss":
            self.criterion = FedFocalLoss(alpha=0.75, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):

        # SBERT backbone → token embeddings
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # CNN layers
        x = hidden_states.permute(0, 2, 1)  # (batch, hidden_size, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Global max pooling
        x = torch.max(x, dim=2).values  # (batch, conv_filters)

        # Dropout + classifier
        x = self.dropout(x)
        logits = self.fc_out(x)

        # Loss
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

class FedFocalLoss(nn.Module):
    """
    Fed-Focal Loss for Federated Learning with class imbalance.
    Combines focal loss with client-level weighting.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        :param alpha: Weighting factor for minority classes
        :param gamma: Focusing parameter for hard examples
        :param reduction: 'mean' or 'sum'
        """
        super(FedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, client_weight=1.0):
        """
        :param logits: Model predictions (batch_size x num_classes)
        :param targets: Ground truth labels (batch_size)
        :param client_weight: Weight for client contribution (scalar)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * ce_loss

        # Apply client-level weighting
        loss = loss * client_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        alpha: Weighting factor for class imbalance
        gamma: Focusing parameter to down-weight easy examples
        reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: raw model outputs (batch_size, num_classes)
        targets: ground truth labels (batch_size)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # standard CE per sample
        pt = torch.exp(-ce_loss)  # predicted probability for true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MSDMMode_old(nn.Module):
    def __init__(self, transformer_model_name='microsoft/deberta-base',
                 conv_filters=128, conv_kernel_size=3, num_classes=2, dropout=0.5, loss = None): # microsoft/deberta-base
        super(MSDMModel, self).__init__()

        # Transformer backbone
        self.transformer = DebertaForSequenceClassification.from_pretrained(
            transformer_model_name,
            num_labels=num_classes
        )

        hidden_size = self.transformer.config.hidden_size

        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=conv_filters*2, kernel_size=conv_kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=conv_filters*2, out_channels=conv_filters, kernel_size=conv_kernel_size, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(conv_filters, num_classes)

        if loss == "FocalLoss":
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)

        elif loss == "FedFocalLoss":
            self.criterion = FedFocalLoss(alpha=0.75, gamma=2.0)

        else:
            self.criterion = nn.CrossEntropyLoss()


    def forward(self, input_ids, attention_mask, labels=None):
        #  Transformer backbone (DeBERTa / CodeBERT etc.)
        out = self.transformer.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = out.last_hidden_state  # (batch, seq_len, hidden_size)

        #  CNN layers (2-layer CNN)
        x = hidden_states.permute(0, 2, 1)  # (batch, hidden_size, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        #  Global max pooling over sequence length
        x = torch.max(x, dim=2).values  # (batch, num_filters)

        #  Dropout + classifier
        x = self.dropout(x)
        logits = self.fc_out(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

class MSDMModel(nn.Module):
    def __init__(self,
                 transformer_model_name='microsoft/deberta-v3-base',
                 conv_filters=128,
                 conv_kernel_size=3,
                 num_classes=2,
                 dropout=0.5,
                 loss=None,
                 ignore_mismatched_sizes=True):
        super(MSDMModel, self).__init__()

        # Select the correct DeBERTa class based on model name
        if "v3" in transformer_model_name.lower():
            DebertaClass = DebertaV2ForSequenceClassification
        else:
            DebertaClass = DebertaForSequenceClassification

        # Transformer backbone
        self.transformer = DebertaClass.from_pretrained(
            transformer_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=ignore_mismatched_sizes  # solves embedding size mismatch
        )

        hidden_size = self.transformer.config.hidden_size

        # 2-layer CNN on top of transformer
        self.conv1 = nn.Conv1d(in_channels=hidden_size,
                               out_channels=conv_filters*2,
                               kernel_size=conv_kernel_size,
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=conv_filters*2,
                               out_channels=conv_filters,
                               kernel_size=conv_kernel_size,
                               padding=1)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(conv_filters, num_classes)

        # Loss function
        if loss == "FocalLoss":
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif loss == "FedFocalLoss":
            self.criterion = FedFocalLoss(alpha=0.75, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # Transformer forward
        if hasattr(self.transformer, "deberta"):  # DeBERTa / DeBERTa-v3
            out = self.transformer.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = out.last_hidden_state
        else:  # fallback for other transformers
            out = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = out.last_hidden_state

        # CNN layers
        x = hidden_states.permute(0, 2, 1)  # (batch, hidden_size, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Global max pooling over sequence length
        x = torch.max(x, dim=2).values  # (batch, num_filters)

        # Dropout + classifier
        x = self.dropout(x)
        logits = self.fc_out(x)

        # Compute loss if labels are provided
        loss_out = None
        if labels is not None:
            loss_out = self.criterion(logits, labels.view(-1))

        return SequenceClassifierOutput(loss=loss_out, logits=logits)
class ZeroLossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3):
        self.patience = patience
        self.zero_loss_count = 0

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.log_history and "loss" in state.log_history[-1]:
            loss = state.log_history[-1]["loss"]
            if loss == 0.0:
                self.zero_loss_count += 1
            else:
                self.zero_loss_count = 0

            if self.zero_loss_count >= self.patience:
                print(f"Stopping training: {self.patience} consecutive zero losses.")
                control.should_training_stop = True

        return control

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def user_create_train_data(dataset):


    base_path = Path(__file__).resolve().parent
    # file_name = base_path / ".." / "data" / "ML_data" / f"{dataset}-{subset}.csv"
    file_name =  base_path / ".." / "data" / "Sequence_data" / f"{dataset}_Sequence.csv"


    ml_df = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    # print(ml_df['label'].value_counts())

    ml_df['label'] = ml_df['label'].apply(lambda x: "False" if x == "IO" else "True") # in ["True", "False"])]
    # print(ml_df['label'].value_counts())

    # ml_df['label'] = pd.to_numeric(ml_df['label'], errors="coerce")
    # ml_df = ml_df.dropna(subset=['label'])

    # Remove rows with NaN labels
    ml_df = ml_df[ml_df['label'].notna()]
    # print(ml_df.shape)

    random_ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # print(" After Creating Binary Data ", random_ml_df.shape )

    # Change the dataframe column name
    random_ml_df = random_ml_df.rename(columns={
        'post_text_sequence': 'text',
        'label': 'class'
    })

    # change the label as integer
    random_ml_df["class"] = random_ml_df["class"].apply( lambda  x: 0 if str(x) == "True" else 1)

    # Remove rows where the stratify column has NaN
    random_ml_df = random_ml_df[random_ml_df['class'].notna()]
    # print(random_ml_df.shape)

    # # # Step: Split off test set (20%)
    train_val, test = train_test_split(random_ml_df, test_size=0.2, random_state=42, stratify=random_ml_df['class'])

    # Step : Split remaining 80% into train (70%) and validation (10%)
    # Note: validation size relative to train_val = 10 / 80 = 0.125
    train, val = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val['class'])

    return train, val, test


def create_fine_tune_data(dataset, subset, option = "User", post_length = 20):


    base_path = Path(__file__).resolve().parent
    # file_name = base_path / ".." / "data" / "ML_data" / f"{dataset}-{subset}.csv"
    file_name =  base_path / ".." / "data" / "DP_Data" / f"{dataset}-{subset}.csv"


    ml_df = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    # print(ml_df['label'].value_counts())

    ml_df = user_sequence(ml_df, post_length) if option == "User" else post_level_data(ml_df)

    ml_df['label'] = ml_df['label'].apply(lambda x: "False" if x == "IO" else "True") # in ["True", "False"])]
    # print(ml_df['label'].value_counts())

    # ml_df['label'] = pd.to_numeric(ml_df['label'], errors="coerce")
    # ml_df = ml_df.dropna(subset=['label'])

    # Remove rows with NaN labels
    ml_df = ml_df[ml_df['label'].notna()]
    # print(ml_df.shape)

    random_ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # print(" After Creating Binary Data ", random_ml_df.shape )

    # Change the dataframe column name
    random_ml_df = random_ml_df.rename(columns={
        'post_text_sequence': 'text',
        'label': 'class'
    })

    # change the label as integer
    random_ml_df["class"] = random_ml_df["class"].apply( lambda  x: 0 if str(x) == "True" else 1)

    # Remove rows where the stratify column has NaN
    random_ml_df = random_ml_df[random_ml_df['class'].notna()].reset_index(drop=True)
    # print(random_ml_df.shape)
    #
    # # # # Step: Split off test set (20%)
    # train_val, test = train_test_split(random_ml_df, test_size=0.2, random_state=42, stratify=random_ml_df['class'])
    #
    # # Step : Split remaining 80% into train (70%) and validation (10%)
    # # Note: validation size relative to train_val = 10 / 80 = 0.125
    # train, val = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val['class'])

    return random_ml_df


def create_ML_data(dataset, option = "User", post_length = 20, dataset_number = 1):


    base_path = Path(__file__).resolve().parent


    file_name = base_path / ".." / "data" / "AI_Generated_Data" / f"Few_{dataset}_SenLev1.csv"
    AI_IO_data = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    AI_dates = AI_IO_data['post_date'].unique()


    human_file_path = base_path / ".." / "data" / "raw" / f"{dataset}.csv"

    data = pd.read_csv(human_file_path, encoding='utf-8', engine='python')

    data['date'] = pd.to_datetime(data['post_time']).dt.strftime('%Y-%m-%d')

    # Filter by control type and language
    # data = data[(data['is_control'] == is_control) & (data['post_language'] == "en")].reset_index(drop=True)

    HU_IO_data = data[(data['is_control'] == False) & (data['date'].isin(AI_dates))].reset_index(drop=True)
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(AI_IO_data['accountid'].unique())]

    print("AI_accounts : ", len(AI_IO_data['accountid'].unique()), " Hu accounts: ", len(HU_IO_data['accountid'].unique()))

    CT_len =  len(AI_IO_data['accountid'].unique()) + len(HU_IO_data['accountid'].unique()) if dataset_number == 1 else  (len(AI_IO_data['accountid'].unique()) + len(HU_IO_data['accountid'].unique())) * 10

    HU_CT_data = data[(data['is_control'] == True) & (data['date'].isin(AI_dates))].reset_index(drop=True)


    #Most potential CT user on the data
    counts = HU_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    # for each accountid, find the row where post_count is maximum
    result = counts.loc[counts.groupby("accountid")["post_count"].idxmax()]

    # print(ml_df['label'].value_counts())

    result = result.sort_values(by=['post_count'], ascending=False).reset_index(drop=True)
    CT_users = result['accountid'].values[0:CT_len]

    # If does not cover number of CT_users
    if len(CT_users) < CT_len:
        remain_CT_len = CT_len - len(CT_users)
        # Randomly select CT users from out of IO active date
        HU_CT_data_Remain= data[(data['is_control'] == True) & (~data['date'].isin (AI_dates))].reset_index(drop=True)

        # Most potential CT user on the data
        counts_remain = HU_CT_data_Remain.groupby(['date', 'accountid']).size().reset_index(name='post_count')

        # for each accountid, find the row where post_count is maximum
        result_remain = counts_remain.loc[counts_remain.groupby("accountid")["post_count"].idxmax()]

        # print(ml_df['label'].value_counts())

        result_remain = result_remain.sort_values(by=['post_count'], ascending=False).reset_index(drop=True)
        CT_users_remain = result_remain['accountid'].values[0:remain_CT_len]

        HU_CT_data_Remain = HU_CT_data_Remain[HU_CT_data_Remain['accountid'].isin(CT_users_remain)]
        HU_CT_data = pd.concat([HU_CT_data, HU_CT_data_Remain])



    else:
        HU_CT_data = HU_CT_data[HU_CT_data['accountid'].isin(CT_users)]

    AI_ML_Data = pd.DataFrame({
        "post_text": AI_IO_data["post_text"].values,
        "label": ["IO"] * len(AI_IO_data),
        "user": ["AI_" + item for item in AI_IO_data["accountid"].values],
        "type": ["AI"] * len(AI_IO_data),
        "LLM": AI_IO_data["model"].values
    })

    HU_ML_Data = pd.DataFrame({
        "post_text": HU_IO_data["post_text"].values,
        "label": ["IO"] * len(HU_IO_data),
        "user": [item for item in HU_IO_data["accountid"].values],
        "type": ["HU"] * len(HU_IO_data),
        "LLM" : ["NO"]* len(HU_IO_data)

    }

    )

    CT_ML_Data = pd.DataFrame({
        "post_text": HU_CT_data["post_text"].values,
        "label": ["CT"] * len(HU_CT_data),
        "user": [item for item in HU_CT_data["accountid"].values],
        "type": ["CT"] * len(HU_CT_data),
        "LLM": ["NO"]* len(HU_CT_data)
    })

    ml_df = pd.concat([AI_ML_Data, HU_ML_Data, CT_ML_Data], axis = 0).reset_index(drop=True)

    return_ml_df = ml_df.copy().sample(frac=1, random_state=42).reset_index(drop=True)
    print(AI_ML_Data.shape, HU_ML_Data.shape, CT_ML_Data.shape, ml_df.shape)

    ml_df = user_sequence(ml_df, post_length) if option == "User" else post_level_data(ml_df)

    ml_df['label'] = ml_df['label'].apply(lambda x: "False" if x == "IO" else "True") # in ["True", "False"])]
    # print(ml_df['label'].value_counts())

    # ml_df['label'] = pd.to_numeric(ml_df['label'], errors="coerce")
    # ml_df = ml_df.dropna(subset=['label'])

    # Remove rows with NaN labels
    ml_df = ml_df[ml_df['label'].notna()]
    # print(ml_df.shape)

    random_ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # print(" After Creating Binary Data ", random_ml_df.shape )

    # Change the dataframe column name
    random_ml_df = random_ml_df.rename(columns={
        'post_text_sequence': 'text',
        'label': 'class'
    })

    # change the label as integer
    random_ml_df["class"] = random_ml_df["class"].apply( lambda  x: 0 if str(x) == "True" else 1)

    # Remove rows where the stratify column has NaN
    random_ml_df = random_ml_df[random_ml_df['class'].notna()].reset_index(drop=True)

    return random_ml_df, return_ml_df

def create_sequence(ml_df, post_length):
    ml_df = user_sequence(ml_df, post_length)

    ml_df['label'] = ml_df['label'].apply(lambda x: "False" if x == "IO" else "True")  # in ["True", "False"])]
    # print(ml_df['label'].value_counts())

    # ml_df['label'] = pd.to_numeric(ml_df['label'], errors="coerce")
    # ml_df = ml_df.dropna(subset=['label'])

    # Remove rows with NaN labels
    ml_df = ml_df[ml_df['label'].notna()]
    # print(ml_df.shape)

    random_ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # print(" After Creating Binary Data ", random_ml_df.shape )

    # Change the dataframe column name
    random_ml_df = random_ml_df.rename(columns={
        'post_text_sequence': 'text',
        'label': 'class'
    })

    # change the label as integer
    random_ml_df["class"] = random_ml_df["class"].apply(lambda x: 0 if str(x) == "True" else 1)

    # Remove rows where the stratify column has NaN
    random_ml_df = random_ml_df[random_ml_df['class'].notna()].reset_index(drop=True)
    return random_ml_df



def Adversarial_sequence(ml_df, post_length, option):

    ml_df = user_sequence(ml_df, post_length) if option == "User" else post_level_data(ml_df)

    ml_df['label'] = ml_df['label'].apply(lambda x: "False" if x == "IO" else "True")  # in ["True", "False"])]
    # print(ml_df['label'].value_counts())

    # ml_df['label'] = pd.to_numeric(ml_df['label'], errors="coerce")
    # ml_df = ml_df.dropna(subset=['label'])

    # Remove rows with NaN labels
    ml_df = ml_df[ml_df['label'].notna()]
    # print(ml_df.shape)

    random_ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # print(" After Creating Binary Data ", random_ml_df.shape )

    # Change the dataframe column name
    random_ml_df = random_ml_df.rename(columns={
        'post_text_sequence': 'text',
        'label': 'class'
    })

    # change the label as integer
    random_ml_df["class"] = random_ml_df["class"].apply(lambda x: 0 if str(x) == "True" else 1)

    # Remove rows where the stratify column has NaN
    random_ml_df = random_ml_df[random_ml_df['class'].notna()].reset_index(drop=True)

    return random_ml_df

def sequence_train_data(ml_df, post_length, option):
    ml_df = user_sequence(ml_df, post_length) if option == "User" else post_level_data(ml_df)

    ml_df['label'] = ml_df['label'].apply(lambda x: "False" if x == "IO" else "True")  # in ["True", "False"])]
    # print(ml_df['label'].value_counts())

    # Remove rows with NaN labels
    ml_df = ml_df[ml_df['label'].notna()]
    # print(ml_df.shape)

    random_ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # print(" After Creating Binary Data ", random_ml_df.shape )

    # Change the dataframe column name
    random_ml_df = random_ml_df.rename(columns={
        'post_text_sequence': 'text',
        'label': 'class'
    })

    # change the label as integer
    random_ml_df["class"] = random_ml_df["class"].apply(lambda x: 0 if str(x) == "True" else 1)

    # Remove rows where the stratify column has NaN
    random_ml_df = random_ml_df[random_ml_df['class'].notna()].reset_index(drop=True)
    return random_ml_df



def create_ML_data_AI_HU(dataset, ratio = 50):


    base_path = Path(__file__).resolve().parent


    # file_name = base_path / ".." / "data" / "AI_Generated_Data" / f"Few_{dataset}_SenLev1.csv"
    file_name = base_path / ".." / "data" / "Few-Shot-Data" / f"Few-Shot-{dataset}.csv"
    
    AI_IO_data = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    AI_dates = AI_IO_data['post_date'].unique()


    human_file_path = base_path / ".." / "data" / "raw" / f"{dataset}.csv"

    data = pd.read_csv(human_file_path, encoding='utf-8', engine='python')

    data['date'] = pd.to_datetime(data['post_time']).dt.strftime('%Y-%m-%d')

    # Filter by control type and language
    # data = data[(data['is_control'] == is_control) & (data['post_language'] == "en")].reset_index(drop=True)

    HU_IO_data = data[(data['is_control'] == False) & (data['date'].isin(AI_dates))].reset_index(drop=True)
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(AI_IO_data['accountid'].unique())]

   


    CT_len =  len(AI_IO_data['accountid'].unique()) + len(HU_IO_data['accountid'].unique())

    CT_data = data[data['is_control'] == True]

    HU_CT_data = data[(data['is_control'] == True) & (data['date'].isin(AI_dates))].reset_index(drop=True)

    # print("CT Accounts : ", len(CT_data['accountid'].unique()), " Same day active CT Account " ,len(HU_CT_data['accountid'].unique()))


    #Most potential CT user on the data
    counts = HU_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    # for each accountid, find the row where post_count is maximum
    result = counts.loc[counts.groupby("accountid")["post_count"].idxmax()]

    # print(ml_df['label'].value_counts())

    result = result.sort_values(by=['post_count'], ascending=False).reset_index(drop=True)
    CT_users = result['accountid'].values #[0:CT_len]
    missing_users = set(CT_data['accountid'].unique()) - set(CT_users)
    HU_CT_data = HU_CT_data[HU_CT_data['accountid'].isin(CT_users)]
    

    IO_users = HU_IO_data['accountid'].unique()
   


    # Sort the users randomly 
    random.seed(42)
    random.shuffle(IO_users)

    # Split 50% HU and 50% AI IO ACCOUNTS 

    partition = int(len(IO_users)* ratio / 100) 

    AI_IO_users = IO_users[0:partition]
    HU_IO_users = IO_users[partition:]
   

    AI_IO_data = AI_IO_data[AI_IO_data['accountid'].isin(AI_IO_users)]
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(HU_IO_users)]

    AI_ML_Data = pd.DataFrame({
        "post_text": AI_IO_data["post_text"].values,
        "label": ["IO"] * len(AI_IO_data),
        "user": ["AI_" + item for item in AI_IO_data["accountid"].values],
        "IO_type": ["AI"] * len(AI_IO_data)
    })

    HU_ML_Data = pd.DataFrame({
        "post_text": HU_IO_data["post_text"].values,
        "label": ["IO"] * len(HU_IO_data),
        "user": [item for item in HU_IO_data["accountid"].values],
         "IO_type": ["HU"] * len(HU_IO_data)
        })

    CT_ML_Data = pd.DataFrame({
        "post_text": HU_CT_data["post_text"].values,
        "label": ["CT"] * len(HU_CT_data),
        "user": [item for item in HU_CT_data["accountid"].values],
        "IO_type": ["CT"] * len(HU_CT_data)
    })



    Other_CT_data = CT_data[CT_data['accountid'].isin(missing_users)]

    counts_2 = Other_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    result_2 = (
        counts_2.loc[counts_2.groupby('date')['post_count'].idxmax()]
        .sort_values('post_count', ascending=False)
        .drop_duplicates('accountid')
        .reset_index(drop=True)
    )
    
    missing_CT_Data = pd.merge(
        result_2,
        Other_CT_data,
        on=['date', 'accountid'],
        how='inner'
    )
    
    # print(result_2.head())

    # print(missing_CT_Data.columns)

    remaining_CT_Data = pd.DataFrame({
        "post_text": missing_CT_Data["post_text"].values,
        "label": ["CT"] * len(missing_CT_Data),
        "user": [item for item in missing_CT_Data["accountid"].values],
         "IO_type": ["CT"] * len(missing_CT_Data)
    })
   
    # print("Missing CT Accounts ", len(remaining_CT_Data['user'].unique()))


    all_data = pd.concat([AI_ML_Data, HU_ML_Data, CT_ML_Data, remaining_CT_Data], axis = 0)
       

    random_ml_df = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Number of Posts: ")
    print(random_ml_df['IO_type'].value_counts())

    print("Number of Users: ")

    unique_users = random_ml_df.groupby("IO_type")["user"].nunique()
    print(unique_users)

    return random_ml_df 


def create_ML_data_HU(dataset):


    base_path = Path(__file__).resolve().parent


    file_name = base_path / ".." / "data" / "AI_Generated_Data" / f"Few_{dataset}_SenLev1.csv"
    AI_IO_data = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    AI_dates = AI_IO_data['post_date'].unique()


    human_file_path = base_path / ".." / "data" / "raw" / f"{dataset}.csv"

    data = pd.read_csv(human_file_path, encoding='utf-8', engine='python')

    data['date'] = pd.to_datetime(data['post_time']).dt.strftime('%Y-%m-%d')

    # Filter by control type and language
    # data = data[(data['is_control'] == is_control) & (data['post_language'] == "en")].reset_index(drop=True)

    HU_IO_data = data[(data['is_control'] == False) & (data['date'].isin(AI_dates))].reset_index(drop=True)
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(AI_IO_data['accountid'].unique())]

    # print("AI_accounts : ", len(AI_IO_data['accountid'].unique()), " Hu accounts: ", len(HU_IO_data['accountid'].unique()))



    CT_len =  len(AI_IO_data['accountid'].unique()) + len(HU_IO_data['accountid'].unique())

    CT_data = data[data['is_control'] == True]

    HU_CT_data = data[(data['is_control'] == True) & (data['date'].isin(AI_dates))].reset_index(drop=True)

    # print("CT Accounts : ", len(CT_data['accountid'].unique()), " Same day active CT Account " ,len(HU_CT_data['accountid'].unique()))


    #Most potential CT user on the data
    counts = HU_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    # for each accountid, find the row where post_count is maximum
    result = counts.loc[counts.groupby("accountid")["post_count"].idxmax()]

    # print(ml_df['label'].value_counts())

    result = result.sort_values(by=['post_count'], ascending=False).reset_index(drop=True)
    CT_users = result['accountid'].values #[0:CT_len]
    missing_users = set(CT_data['accountid'].unique()) - set(CT_users)
    HU_CT_data = HU_CT_data[HU_CT_data['accountid'].isin(CT_users)]
    
 

    HU_ML_Data = pd.DataFrame({
        "post_text": HU_IO_data["post_text"].values,
        "label": ["IO"] * len(HU_IO_data),
        "user": [item for item in HU_IO_data["accountid"].values],
         "IO_type": ["HU"] * len(HU_IO_data)})

    CT_ML_Data = pd.DataFrame({
        "post_text": HU_CT_data["post_text"].values,
        "label": ["CT"] * len(HU_CT_data),
        "user": [item for item in HU_CT_data["accountid"].values],
         "IO_type": ["CT"] * len(HU_CT_data)
    })



    Other_CT_data = CT_data[CT_data['accountid'].isin(missing_users)]

    counts_2 = Other_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    result_2 = (
        counts_2.loc[counts_2.groupby('date')['post_count'].idxmax()]
        .sort_values('post_count', ascending=False)
        .drop_duplicates('accountid')
        .reset_index(drop=True)
    )
    
    missing_CT_Data = pd.merge(
        result_2,
        Other_CT_data,
        on=['date', 'accountid'],
        how='inner'
    )
    
    # print(result_2.head())

    # print(missing_CT_Data.columns)

    remaining_CT_Data = pd.DataFrame({
        "post_text": missing_CT_Data["post_text"].values,
        "label": ["CT"] * len(missing_CT_Data),
        "user": [item for item in missing_CT_Data["accountid"].values],
         "IO_type": ["CT"] * len(missing_CT_Data)
    })
   
    # print("Missing CT Accounts ", len(remaining_CT_Data['user'].unique()))


    all_data = pd.concat([HU_ML_Data, CT_ML_Data, remaining_CT_Data], axis = 0)
       

    random_ml_df = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Number of Posts: ")
    print(random_ml_df['IO_type'].value_counts())

    print("Number of Users: ")

    unique_users = random_ml_df.groupby("IO_type")["user"].nunique()
    print(unique_users)
    

    return random_ml_df 



def create_ML_data_AI(dataset):


    base_path = Path(__file__).resolve().parent


    file_name = base_path / ".." / "data" / "AI_Generated_Data" / f"Few_{dataset}_SenLev1.csv"
    AI_IO_data = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    AI_dates = AI_IO_data['post_date'].unique()


    human_file_path = base_path / ".." / "data" / "raw" / f"{dataset}.csv"

    data = pd.read_csv(human_file_path, encoding='utf-8', engine='python')

    data['date'] = pd.to_datetime(data['post_time']).dt.strftime('%Y-%m-%d')

    # Filter by control type and language
    # data = data[(data['is_control'] == is_control) & (data['post_language'] == "en")].reset_index(drop=True)

    HU_IO_data = data[(data['is_control'] == False) & (data['date'].isin(AI_dates))].reset_index(drop=True)
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(AI_IO_data['accountid'].unique())]

    # print("AI_accounts : ", len(AI_IO_data['accountid'].unique()), " Hu accounts: ", len(HU_IO_data['accountid'].unique()))


    CT_data = data[data['is_control'] == True]

    HU_CT_data = data[(data['is_control'] == True) & (data['date'].isin(AI_dates))].reset_index(drop=True)

    # print("CT Accounts : ", len(CT_data['accountid'].unique()), " Same day active CT Account " ,len(HU_CT_data['accountid'].unique()))


    #Most potential CT user on the data
    counts = HU_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    # for each accountid, find the row where post_count is maximum
    result = counts.loc[counts.groupby("accountid")["post_count"].idxmax()]

    # print(ml_df['label'].value_counts())

    result = result.sort_values(by=['post_count'], ascending=False).reset_index(drop=True)
    CT_users = result['accountid'].values #[0:CT_len]
    missing_users = set(CT_data['accountid'].unique()) - set(CT_users)
    HU_CT_data = HU_CT_data[HU_CT_data['accountid'].isin(CT_users)]
    
  

    AI_ML_Data = pd.DataFrame({
        "post_text": AI_IO_data["post_text"].values,
        "label": ["IO"] * len(AI_IO_data),
        "user": ["AI_" + item for item in AI_IO_data["accountid"].values],
         "IO_type": ["AI"] * len(AI_IO_data)
    })

    CT_ML_Data = pd.DataFrame({
        "post_text": HU_CT_data["post_text"].values,
        "label": ["CT"] * len(HU_CT_data),
        "user": [item for item in HU_CT_data["accountid"].values],
         "IO_type": ["CT"] * len(HU_CT_data)
    })



    Other_CT_data = CT_data[CT_data['accountid'].isin(missing_users)]

    counts_2 = Other_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    result_2 = (
        counts_2.loc[counts_2.groupby('date')['post_count'].idxmax()]
        .sort_values('post_count', ascending=False)
        .drop_duplicates('accountid')
        .reset_index(drop=True)
    )
    
    missing_CT_Data = pd.merge(
        result_2,
        Other_CT_data,
        on=['date', 'accountid'],
        how='inner'
    )
    
    # print(result_2.head())

    # print(missing_CT_Data.columns)

    remaining_CT_Data = pd.DataFrame({
        "post_text": missing_CT_Data["post_text"].values,
        "label": ["CT"] * len(missing_CT_Data),
        "user": [item for item in missing_CT_Data["accountid"].values],
         "IO_type": ["CT"] * len(missing_CT_Data)
    })
   
    # print("Missing CT Accounts ", len(remaining_CT_Data['user'].unique()))


    all_data = pd.concat([AI_ML_Data, CT_ML_Data, remaining_CT_Data], axis = 0)
       

    random_ml_df = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Number of Posts: ")
    print(random_ml_df['IO_type'].value_counts())

    print("Number of Users: ")

    unique_users = random_ml_df.groupby("IO_type")["user"].nunique()
    print(unique_users)
    
    return random_ml_df 




def AI_HU_user_create_train_data(dataset, balanced = False):


    base_path = Path(__file__).resolve().parent
    # file_name = base_path / ".." / "data" / "ML_data" / f"{dataset}-{subset}.csv"
    file_name =  base_path / ".." / "data" / "Sequence_data" / f"AI_HU_{dataset}_Sequence.csv"


    ml_df = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")

    ml_df = ml_df[ml_df['label'].isin(["AI_IO","HU_IO","CT"])].reset_index(drop=True)


    if balanced:
        min_size = ml_df['label'].value_counts().min()

        balanced_df = (
            ml_df.groupby('label', group_keys=False)
            .apply(lambda x: x.sample(n=min_size, random_state=42))
            .reset_index(drop=True)
        )

        ml_df = balanced_df
    print(ml_df['label'].value_counts())
    # # Remove rows with NaN labels
    # ml_df = ml_df[ml_df['label'].notna()]



    le = LabelEncoder()
    ml_df["label"] = le.fit_transform(ml_df["label"])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(label_mapping)

    random_ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # print(" After Creating Binary Data ", random_ml_df.shape )

    # Change the dataframe column name
    random_ml_df = random_ml_df.rename(columns={
        'post_text_sequence': 'text',
        'label': 'class'
    })

    # # # Step: Split off test set (20%)
    train_val, test = train_test_split(random_ml_df, test_size=0.2, random_state=42, stratify=random_ml_df['class'])

    # Step : Split remaining 80% into train (70%) and validation (10%)
    # Note: validation size relative to train_val = 10 / 80 = 0.125
    train, val = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val['class'])

    return train, val, test



def new_user_create_train_data(dataset):


    base_path = Path(__file__).resolve().parent
    # file_name = base_path / ".." / "data" / "ML_data" / f"{dataset}-{subset}.csv"
    file_name =  base_path / ".." / "data" / "New_Sequence_Dataset" / f"{dataset}_Sequence.csv"


    ml_df = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    # print(ml_df['label'].value_counts())

    ml_df['label'] = ml_df['label'].apply(lambda x: "False" if x == "IO" else "True") # in ["True", "False"])]
    # print(ml_df['label'].value_counts())

    # ml_df['label'] = pd.to_numeric(ml_df['label'], errors="coerce")
    # ml_df = ml_df.dropna(subset=['label'])

    # Remove rows with NaN labels
    ml_df = ml_df[ml_df['label'].notna()]
    # print(ml_df.shape)

    random_ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # print(" After Creating Binary Data ", random_ml_df.shape )

    # Change the dataframe column name
    random_ml_df = random_ml_df.rename(columns={
        'post_text_sequence': 'text',
        'label': 'class'
    })

    # change the label as integer
    random_ml_df["class"] = random_ml_df["class"].apply( lambda  x: 0 if str(x) == "True" else 1)

    # Remove rows where the stratify column has NaN
    random_ml_df = random_ml_df[random_ml_df['class'].notna()]
    # print(random_ml_df.shape)

    # # # Step: Split off test set (20%)
    train_val, test = train_test_split(random_ml_df, test_size=0.2, random_state=42, stratify=random_ml_df['class'])

    # Step : Split remaining 80% into train (70%) and validation (10%)
    # Note: validation size relative to train_val = 10 / 80 = 0.125
    train, val = train_test_split(train_val, test_size=0.125, random_state=42, stratify=train_val['class'])

    return train, val, test

def build_Model(model_name, train, val, test, max_len = 512, class_num = 2):
    # Check unique classes in each split
    # print("Train classes:", len(train), train['class'].value_counts())
    # print("Val classes:", len(val), val['class'].value_counts())
    # print("Test classes:", len(test), test['class'].value_counts())

    if model_name == "OSM-Det":

        model_name = "allenai/longformer-base-4096"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=class_num)



    elif model_name == "DeBERTa":
        select_model_name = "microsoft/deberta-base" # deberta-v3-base" #
        tokenizer = AutoTokenizer.from_pretrained(select_model_name)

        # Select the correct DeBERTa class based on model name
        if "v3" in select_model_name.lower():
            DebertaClass = DebertaV2ForSequenceClassification
        else:
            DebertaClass = DebertaForSequenceClassification


        model = DebertaClass.from_pretrained(select_model_name, num_labels=class_num)

    elif model_name == "RoBERTa":
        select_model_name = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(select_model_name)
        model = RobertaForSequenceClassification.from_pretrained(select_model_name, num_labels=class_num)
    elif model_name == "GraphCodeBERT":
        select_model_name = "microsoft/graphcodebert-base"
        tokenizer = AutoTokenizer.from_pretrained(select_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(select_model_name, num_labels=class_num)

    elif model_name == "DistilBERT":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=class_num)

    elif model_name == "XLNet":
        # Load model for sequence classification
        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=class_num)
    elif model_name == "BERT":
        # Load pretrained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=class_num)
    elif model_name == "MGT-Dect":
        model_name_base = "distilbert/distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name_base)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_base, num_labels=class_num)


    # elif model_name == "HyproBERT":
    #     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    #     model = HyproBertModel(num_classes=class_num)

    elif model_name == "MDSM":
        loss = "FedFocalLoss"
        model_name = "microsoft/deberta-v3-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MSDMModel(num_classes=class_num, loss=loss) # "FocalLoss" , FedFocalLoss
    elif model_name == "SBERT":
        loss = "FocalLoss"
        model_name = "sentence-transformers/all-mpnet-base-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = SBERT_CNN_Model(num_classes=class_num, loss=loss)


    train_encodings = tokenize(train['text'].astype(str), tokenizer, max_len)
    val_encodings = tokenize(val['text'].astype(str), tokenizer, max_len)


    train_dataset = TextDataset(train_encodings, train['class'].tolist())
    val_dataset = TextDataset(val_encodings, val['class'].tolist())
    # Create a temporary directory
    temp_output_dir = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=temp_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # evaluation_strategy="epoch",  # validate at end of each epoch
        # save_strategy="epoch",
        save_strategy="no",
        logging_dir='./logs',
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=42,
        # --- Fixed total steps ---
        # max_steps=300,                    # 🔹 stops training after 300 steps
        # num_train_epochs=100,             # large number (won’t matter, as max_steps dominates)

    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[ZeroLossEarlyStoppingCallback(patience=3)]
    )

    start_time = datetime.now()

    # ---- training code ----
    trainer.train()

    end_time = datetime.now()

    training_time_min = (end_time - start_time).total_seconds() / 60
    # print(f"Training time: {training_time_min:.2f} minutes")

    results = trainer.evaluate()
    # print(results)

    # Example test data
    test_texts = test['text'].astype(str).tolist()
    test_labels = test['class'].tolist()

    test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')

    # Check if MPS is available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # print("Using device:", device)

    # Move model to device
    model.to(device)

    # Move test encodings to device
    test_encodings = {key: val.to(device) for key, val in test_encodings.items()}

    # Convert encodings into dataset
    test_dataset = TensorDataset(
        test_encodings["input_ids"],
        test_encodings["attention_mask"],
        torch.tensor(test_labels)
    )

    # DataLoader for batching
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Prediction loop
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            # all_preds.extend(preds.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().astype(int).tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    # print(classification_report(all_labels, all_preds, digits=4))

    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)

    # # Suppose you want metrics for class '1'
    # class_label = '1'  # string if labels are integers
    # accuracy = report['accuracy']  # Overall accuracy
    # precision = report[class_label]['precision']
    # recall = report[class_label]['recall']
    # f1 = report[class_label]['f1-score']

    # metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1)
    recall = recall_score(all_labels, all_preds, pos_label=1)
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    conf_mat = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    #
    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1 Score:", f1)
    # print("AUC:", auc)
    # print("Confusion Matrix:\n", conf_mat)
    # print(round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2), round(auc, 2))
    #
    #
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision (class 1): {precision:.4f}")
    # print(f"Recall (class 1): {recall:.4f}")
    # print(f"F1-score (class 1): {f1:.4f}")


    new_df = pd.DataFrame({"posts": [item for item in test['text'].values], "predict": all_preds, "real": all_labels })
    new_df.to_csv("MSDM_prediction_results.csv", index=False)
    return f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{auc:.4f}" ,all_labels, all_preds,  test['text'].values

def AI_HU_build_Model(model_name, train, val, test, max_len = 512, class_num = 2):
    # Check unique classes in each split
    # print("Train classes:", len(train), train['class'].value_counts())
    # print("Val classes:", len(val), val['class'].value_counts())
    # print("Test classes:", len(test), test['class'].value_counts())

    print("Class Number:", class_num)
    if model_name == "DeBERTa":
        select_model_name = "microsoft/deberta-v3-base" # deberta-base"
        tokenizer = AutoTokenizer.from_pretrained(select_model_name)

        # Select the correct DeBERTa class based on model name
        if "v3" in select_model_name.lower():
            DebertaClass = DebertaV2ForSequenceClassification
        else:
            DebertaClass = DebertaForSequenceClassification


        model = DebertaClass.from_pretrained(select_model_name, num_labels=class_num)

    elif model_name == "RoBERTa":
        select_model_name = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(select_model_name)
        model = RobertaForSequenceClassification.from_pretrained(select_model_name, num_labels=class_num)
    elif model_name == "GraphCodeBERT":
        select_model_name = "microsoft/graphcodebert-base"
        tokenizer = AutoTokenizer.from_pretrained(select_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(select_model_name, num_labels=class_num)

    elif model_name == "DistilBERT":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=class_num)

    elif model_name == "XLNet":
        # Load model for sequence classification
        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=class_num)
    elif model_name == "BERT":
        # Load pretrained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=class_num)

    # elif model_name == "HyproBERT":
    #     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    #     model = HyproBertModel(num_classes=class_num)

    elif model_name == "MDSM":
        loss = "FedFocalLoss"
        model_name = "microsoft/deberta-v3-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MSDMModel(num_classes=class_num, loss=loss) # "FocalLoss" , FedFocalLoss
    elif model_name == "SBERT":
        loss = "FocalLoss"
        model_name = "sentence-transformers/all-mpnet-base-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = SBERT_CNN_Model(num_classes=class_num, loss=loss)


    train_encodings = tokenize(train['text'].astype(str), tokenizer, max_len)
    val_encodings = tokenize(val['text'].astype(str), tokenizer, max_len)


    train_dataset = TextDataset(train_encodings, train['class'].tolist())
    val_dataset = TextDataset(val_encodings, val['class'].tolist())
    # Create a temporary directory
    temp_output_dir = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=temp_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # evaluation_strategy="epoch",  # validate at end of each epoch
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=42,
        # --- Fixed total steps ---
        # max_steps=300,                    # 🔹 stops training after 300 steps
        # num_train_epochs=100,             # large number (won’t matter, as max_steps dominates)

    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[ZeroLossEarlyStoppingCallback(patience=3)]
    )

    start_time = datetime.now()

    # ---- training code ----
    trainer.train()

    end_time = datetime.now()

    training_time_min = (end_time - start_time).total_seconds() / 60
    # print(f"Training time: {training_time_min:.2f} minutes")

    results = trainer.evaluate()
    # print(results)

    # Example test data
    test_texts = test['text'].astype(str).tolist()
    test_labels = test['class'].tolist()

    test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')

    # Check if MPS is available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # print("Using device:", device)

    # Move model to device
    model.to(device)

    # Move test encodings to device
    test_encodings = {key: val.to(device) for key, val in test_encodings.items()}

    # Convert encodings into dataset
    test_dataset = TensorDataset(
        test_encodings["input_ids"],
        test_encodings["attention_mask"],
        torch.tensor(test_labels)
    )

    # DataLoader for batching
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Prediction loop
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            # all_preds.extend(preds.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().astype(int).tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())


    # metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", all_labels, all_preds



def tokenize(texts, tokenizer, max_len=512):
    return tokenizer(list(texts), padding=True, truncation=True, max_length=max_len)



def run_models(dataset,subsets, Models):
    base_path = Path(__file__).resolve().parent

    all_results = pd.DataFrame(
        columns=['subset', 'dataset', 'model', 'accuracy', 'precision', 'recall', 'f1-score', 'AUC', 'real_class',
                 'predict_class'])

    # Models = [  "DistilBERT","DeBERTa", "BERT", "RoBERTa"  ,"MDSM"] #["MDSM"] #"DistilBERT", "DeBERTa", "BERT", "RoBERTa","MDSM", "SBERT"
    # subsets = ["100AI", "100HU", "50AI-50HU"]



    for subset in subsets:
        for model in Models:


            max_len = 512
            print(
                f" ********************************************** \n Result for Dataset: {dataset} \n Model: {model} \n *****************************************")
            train_df, val_df, test_df = user_create_train_data(dataset)

            accuracy, precision, recall, f1, auc, real_label, predict_label, posts = build_Model(model, train_df.copy(),
                                                                                          val_df.copy(),
                                                                                          test_df.copy(), max_len = max_len)

            all_results.loc[len(all_results)] = [subset, dataset, model, accuracy, precision, recall, f1, auc, real_label,
                                                 predict_label]

            print(subset, dataset, model, accuracy, precision, recall, f1, auc)

    result_folder = base_path / ".." / "data" / "output"

    os.makedirs(result_folder, exist_ok=True)

    model_short_name = "_".join([item[0] for item in Models])

    all_results.to_csv(result_folder / f"{dataset}_{model_short_name}.csv", index=False)
    print("Results have been saved to ", result_folder/ f"{dataset}_{model_short_name}.csv")



def AI_HU_run_models(dataset, balanced = False):
    base_path = Path(__file__).resolve().parent

    all_results = pd.DataFrame(
        columns=['dataset', 'model', 'accuracy', 'precision', 'recall', 'f1-score', 'real_class',
                 'predict_class'])

    Models = [  "DistilBERT","DeBERTa", "BERT", "RoBERTa"  ,"MDSM"] #["MDSM"] #"DistilBERT", "DeBERTa", "BERT", "RoBERTa","MDSM", "SBERT"

    for model in Models:


        max_len = 512
        print(
            f" ********************************************** \n Result for Dataset: {dataset} \n Model: {model} \n *****************************************")
        train_df, val_df, test_df = AI_HU_user_create_train_data(dataset, balanced)

        accuracy, precision, recall, f1, real_label, predict_label = AI_HU_build_Model(model, train_df.copy(),
                                                                                      val_df.copy(),
                                                                                      test_df.copy(), max_len = max_len, class_num= 3)

        all_results.loc[len(all_results)] = [dataset, model, accuracy, precision, recall, f1, real_label,
                                             predict_label]

        print(dataset, model, accuracy, precision, recall, f1)

    result_folder = base_path / ".." / "data" / "output"

    os.makedirs(result_folder, exist_ok=True)



    all_results.to_csv(result_folder / f"AI_HU_{dataset}.csv", index=False)
    print("Results have been saved to ", result_folder/ f"AI_HU_{dataset}.csv")



def new_run_models(dataset):
    base_path = Path(__file__).resolve().parent

    all_results = pd.DataFrame(
        columns=['subset', 'dataset', 'model', 'accuracy', 'precision', 'recall', 'f1-score', 'AUC', 'real_class',
                 'predict_class'])

    Models = [  "DistilBERT","DeBERTa", "BERT", "RoBERTa"  ,"MDSM"] #["MDSM"] #"DistilBERT", "DeBERTa", "BERT", "RoBERTa","MDSM", "SBERT"
    subsets = ["100AI", "100HU", "50AI-50HU"]



    for subset in subsets:
        for model in Models:


            max_len = 512
            print(
                f" ********************************************** \n Result for Dataset: {dataset} \n Model: {model} \n *****************************************")
            train_df, val_df, test_df = new_user_create_train_data(dataset)

            accuracy, precision, recall, f1, auc, real_label, predict_label, posts = build_Model(model, train_df.copy(),
                                                                                          val_df.copy(),
                                                                                          test_df.copy(), max_len = max_len)

            all_results.loc[len(all_results)] = [subset, dataset, model, accuracy, precision, recall, f1, auc, real_label,
                                                 predict_label]

            print(subset, dataset, model, accuracy, precision, recall, f1, auc)

    result_folder = base_path / ".." / "data" / "new_output"

    os.makedirs(result_folder, exist_ok=True)



    all_results.to_csv(result_folder / f"{dataset}.csv", index=False)
    print("Results have been saved to ", result_folder/ f"{dataset}.csv")




def run_MDSM_model(dataset):
    base_path = Path(__file__).resolve().parent

    all_results = pd.DataFrame(
        columns=['subset', 'dataset', 'model', 'accuracy', 'precision', 'recall', 'f1-score', 'AUC', 'real_class',
                 'predict_class'])

    Models = ["MDSM"] #["MDSM"] #"DistilBERT", "DeBERTa", "BERT", "RoBERTa","MDSM", "SBERT"
    subsets = ["100AI"] #, "100HU", "50AI-50HU"]



    for subset in subsets:
        for model in Models:


            max_len = 512
            print(
                f" ********************************************** \n Result for Dataset: {dataset} \n Model: {model} \n *****************************************")
            train_df, val_df, test_df = user_create_train_data(dataset)

            accuracy, precision, recall, f1, auc, real_label, predict_label, posts = build_Model(model, train_df.copy(),
                                                                                          val_df.copy(),
                                                                                          test_df.copy(), max_len = max_len)

            all_results.loc[len(all_results)] = [subset, dataset, model, accuracy, precision, recall, f1, auc, real_label,
                                                 predict_label]

            print(subset, dataset, model, accuracy, precision, recall, f1, auc)

    result_folder = base_path / ".." / "data" / "output"

    os.makedirs(result_folder, exist_ok=True)



    all_results.to_csv(result_folder / f"{dataset}_MSDM.csv", index=False)
    print("Results have been saved to ", result_folder/ f"{dataset}_MSDM.csv")


def user_sequence(input_data, post_length):


    Users = input_data['user'].unique()
    # print("User Number : ", len(Users))

    # print("Users : ", Users)

    subset_all_data = []
    used_users = []
    for user in Users:

        if user is used_users:
            print("user exists:", user)
            continue

        used_users.append(user)
        # print(len(used_users))
        temp_data = input_data[input_data['user'] == user].reset_index(drop=True)

        temp_data = temp_data.loc[0:post_length-1]

        post_text = "\n".join([str(item) for item in temp_data['post_text'].values])

        number_of_post = len(temp_data)

        label = temp_data['label'].iloc[0]

        row = {
            "accountid": user,
            "post_text_sequence": post_text,
            "post_num": number_of_post,
            "label": label

        }

        subset_all_data.append(row)


    subset_df = pd.DataFrame(subset_all_data)

    return subset_df


def Generalized_Results(train_datasets, test_datasets, post_length = 20, IO_Option = None ):

    base_path = Path(__file__).resolve().parent



    results = pd.DataFrame(
        columns=['train_datasets', 'test_datasets', 'model', 'accuracy', 'precision', 'recall', 'f1-score', 'auc', 'real_class', 'predict_class'])

    train_data = []
    for dataset in train_datasets:

        # file_name =  base_path / ".." / "data" / "ML_data" / f"{dataset}_Sequence.csv"


        file_name = base_path / ".." / "data" / "ML_data" / f"{dataset}_{post_length}.csv"
        input_data = pd.read_csv(file_name)

        if IO_Option == "AI":
            input_data = input_data[input_data['type'] != "HU"].reset_index(drop=True)
        elif IO_Option == "HU":
            input_data = input_data[input_data['type'] != "AI"].reset_index(drop=True)

        train_data.append(input_data)


    train_df = pd.concat(train_data)

    test_data =  []

    for dataset in test_datasets:

        file_name =  base_path / ".." / "data" / "ML_data" / f"{dataset}_{post_length}.csv"
        input_data = pd.read_csv(file_name)

        if IO_Option == "AI":
            input_data = input_data[input_data['type'] != "HU"].reset_index(drop=True)
        elif IO_Option == "HU":
            input_data = input_data[input_data['type'] != "AI"].reset_index(drop=True)

        test_data.append(input_data)

    test_df = pd.concat(test_data)

    test_df = user_sequence(test_df, post_length=post_length)
    train_df = user_sequence(train_df, post_length=post_length)
    # print(input_data['label'].value_counts())

    # Change the dataframe column name
    train_df = train_df.rename(columns={
        'post_text_sequence': 'text',
        'label': 'class'
    })

    # change the label as integer
    train_df["class"] = train_df["class"].apply(lambda x: 0 if str(x) == "CT" else 1)

    # Remove rows where the stratify column has NaN
    train_df = train_df[train_df['class'].notna()]

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(train_df['class'].value_counts())



    # Change the dataframe column name
    test_df = test_df.rename(columns={
        'post_text_sequence': 'text',
        'label': 'class'
    })

    # change the label as integer
    test_df["class"] = test_df["class"].apply(lambda x: 0 if str(x) == "CT" else 1)

    # Remove rows where the stratify column has NaN
    # test_df = test_df[train_df['class'].notna()]

    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(test_df['class'].value_counts())

    # # # Step: Split off test set (20%)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['class'])
    #
    # print("Train Dataset Size: ", len(train_df))
    # print("Val Dataset Size: ", len(val_df))
    # print("Test Dataset Size: ", len(test_df))


    Models = ["DistilBERT", "DeBERTa", "BERT", "RoBERTa", "MDSM"]

    for model in Models:


        accuracy, precision, recall, f1, auc, real_label, predict_label, posts = build_Model(model, train_df.copy(),
                                                                                      val_df.copy(),
                                                                                      test_df.copy(), max_len=512)
        results.loc[len(results)] = [train_datasets, test_datasets, model, accuracy, precision, recall, f1, auc, real_label, predict_label]


        print(train_datasets, test_datasets, model, accuracy, precision, recall, f1, auc)

    result_folder =  base_path / ".." / "data" / "new_output"

    os.makedirs(result_folder, exist_ok=True)
    test_cases = '_'.join([item.split(".")[0] for item in test_datasets])

    print(IO_Option)
    print(results)

    if IO_Option is None:
        results.to_csv(result_folder / f"{test_cases}.csv", index=False)
        print("Results saved in ", result_folder / f"{test_cases}.csv")
    else:

        results.to_csv(result_folder / f"{test_cases}_{IO_Option}.csv", index=False)

        print("Results saved in " , result_folder / f"{test_cases}_{IO_Option}.csv")



def Generalized_Results_LLM(Datasets, post_length = 20):

    base_path = Path(__file__).resolve().parent



    results = pd.DataFrame(
        columns=['train_LLM', 'test_LLM', 'model', 'accuracy', 'precision', 'recall', 'f1-score', 'auc', 'real_class', 'predict_class'])

    all_data = []
    for dataset in Datasets:

        # file_name =  base_path / ".." / "data" / "ML_data" / f"{dataset}_Sequence.csv"


        file_name = base_path / ".." / "data" / "ML_data" / f"{dataset}_{post_length}.csv"
        input_data = pd.read_csv(file_name)


        input_data = input_data[input_data['type'] != "HU"].reset_index(drop=True)


        all_data.append(input_data)

    IO_data = input_data[input_data["type"] == "AI"].reset_index( drop=True)
    CT_Data = input_data[input_data["type"] == "CT"].reset_index( drop=True)

    LLMs = [item for item in IO_data['LLM'].unique() if item != "NO"]

    LLMs = [x for x in LLMs if pd.notna(x)]

    print(LLMs)


    test_CT_DATA = CT_Data.sample(frac=1/len(LLMs),random_state=42)
    train_CT_DATA = CT_Data.drop(test_CT_DATA.index)


    for LLM in LLMs:
        test_IO = IO_data[IO_data['LLM'] == LLM]
        train_IO = IO_data[IO_data['LLM'] != LLM]
        train_llm = [item for item in LLMs if item != LLM]
        test_llm = LLM

        test_df = pd.concat([test_IO, test_CT_DATA]).reset_index(drop=True)
        train_df = pd.concat([train_IO, train_CT_DATA]).reset_index(drop=True)

        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)



        test_df = user_sequence(test_df, post_length=post_length)
        train_df = user_sequence(train_df, post_length=post_length)
        # print(input_data['label'].value_counts())

        # Change the dataframe column name
        train_df = train_df.rename(columns={
            'post_text_sequence': 'text',
            'label': 'class'
        })

        # change the label as integer
        train_df["class"] = train_df["class"].apply(lambda x: 0 if str(x) == "CT" else 1)

        # Remove rows where the stratify column has NaN
        train_df = train_df[train_df['class'].notna()]

        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(train_df['class'].value_counts())



        # Change the dataframe column name
        test_df = test_df.rename(columns={
            'post_text_sequence': 'text',
            'label': 'class'
        })

        # change the label as integer
        test_df["class"] = test_df["class"].apply(lambda x: 0 if str(x) == "CT" else 1)

        # Remove rows where the stratify column has NaN
        # test_df = test_df[train_df['class'].notna()]

        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(test_df['class'].value_counts())

        # # # Step: Split off test set (20%)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['class'])
        #
        # print("Train Dataset Size: ", len(train_df))
        # print("Val Dataset Size: ", len(val_df))
        # print("Test Dataset Size: ", len(test_df))


        Models = ["DistilBERT"] #, "DeBERTa", "BERT", "RoBERTa", "MDSM"]

        for model in Models:


            accuracy, precision, recall, f1, auc, real_label, predict_label, posts = build_Model(model, train_df.copy(),
                                                                                          val_df.copy(),
                                                                                          test_df.copy(), max_len=512)
            results.loc[len(results)] = [train_llm, test_llm, model, accuracy, precision, recall, f1, auc, real_label, predict_label]


            print(train_llm, test_llm, model, accuracy, precision, recall, f1, auc)

    result_folder =  base_path / ".." / "data" / "new_output"

    os.makedirs(result_folder, exist_ok=True)

    print(results)



    results.to_csv(result_folder / f"LLM_{post_length}.csv", index=False)

    print("Results saved in " , result_folder / f"LLM_{post_length}.csv")


def show_results(dataset ):
    base_path = Path(__file__).resolve().parent

    input_file = base_path /".." /"data" /"output"/f"{dataset}.csv"



    input_data = pd.read_csv(input_file)

    subset_df = input_data[input_data['subset'] == "100AI"]
    # print(input_data)
    sel_col = ['model', 'accuracy', 'precision', 'recall',
       'f1-score']
    print(subset_df[sel_col] )


def ML_Data_Statistics(dataset, post_length, dataset_number = 1 ):
    base_path = Path(__file__).resolve().parent

    input_file = base_path /".." /"data" /"ML_data"/f"{dataset}_{post_length}.csv" if dataset_number == 1 else base_path /".." /"data" /"ML_data"/f"{dataset}_{post_length}_D{dataset_number}.csv"



    input_data = pd.read_csv(input_file)

    HU_data = input_data[input_data['type']== "HU"]
    # print("HU Post : ", len(HU_data), " Users: ", len(HU_data['user'].unique()))

    Hu_pct = round(len(HU_data) / len(HU_data['user'].unique()),1)

    AI_data = input_data[input_data['type'] == "AI"]
    # print("AI Post : ", len(AI_data), " Users: ", len(AI_data['user'].unique()))

    AI_pct = round(len(AI_data) / len(AI_data['user'].unique()),1)

    CT_data = input_data[input_data['type'] == "CT"]
    # print("CT Post : ", len(CT_data), " Users: ", len(CT_data['user'].unique()))
    CT_pct = round(len(CT_data) / len(CT_data['user'].unique()),1)

    user = dataset +'&'+ "User" + '&'+ "&".join(str(item) for item in [str(item) for item in [len(AI_data['user'].unique()),len(HU_data['user'].unique()),len(CT_data['user'].unique())]])
    post = dataset + '&' + "Post" + '&'+"&".join(str(item) for item in [len(AI_data), len(HU_data), len(CT_data)])

    pct = dataset + '&' + "Avg Post" + '&'+ "&".join(str(item) for item in [AI_pct, Hu_pct, CT_pct])

    print(user , "\n", post , "\n", pct )





def Generate_AI_HU_CT_DATASET(dataset, post_length = 20, dataset_number = 1):

    base_path = Path(__file__).resolve().parent

    print(f" ********************************************** \n Generate Dataset: {dataset} \n *****************************************")
    data, ml_df = create_ML_data(dataset, option="User", post_length=post_length, dataset_number = dataset_number)

    df_filtered = ml_df.groupby('user').head(post_length)

    # Save data

    file_name =  base_path / ".." / "data" / "ML_data" / f"{dataset}_{post_length}.csv" if dataset_number == 1 else base_path / ".." / "data" / "ML_data" / f"{dataset}_{post_length}_D{dataset_number}.csv"
    print("File name = ", file_name)
    df_filtered.to_csv(file_name, index=False)



def run_fine_tune_model(dataset, post_length = 20, dataset_number = 1):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)


    base_path = Path(__file__).resolve().parent

    all_results = pd.DataFrame(
        columns=['fold','dataset', 'model', 'accuracy', 'precision', 'recall', 'f1-score', 'AUC', 'real_class',
                 'predict_class'])

    Models = ["OSM-Det","MGT-Dect"] #  ["DistilBERT", "DeBERTa", "BERT", "RoBERTa","MDSM", "SBERT"]

    for model in Models:
        max_len = 512
        print(f" ********************************************** \n Result for Dataset: {dataset} \n Model: {model} \n *****************************************")

        file_name =  base_path / ".." / "data" / "ML_data" / f"{dataset}_{post_length}.csv" if dataset_number == 1 else base_path / ".." / "data" / "ML_data" / f"{dataset}_{post_length}_D{dataset_number}.csv"
        input_data = pd.read_csv(file_name)

        data = create_sequence(input_data, post_length)


        for fold, (train_index, test_index) in enumerate(kf.split(data)):
            print(f"Running Fold {fold + 1}...")

            train_data = data.iloc[train_index]
            test_df = data.iloc[test_index]

            # Optional: create validation split from training data
            val_df = train_data.sample(frac=0.1, random_state=42)
            train_df = train_data.drop(val_df.index)

            print("Train :", len(train_data), " Validation :", len(val_df), " Test :", len(test_df))

            accuracy, precision, recall, f1, auc, real_label, predict_label, posts = build_Model(model,
                                                                                                 train_df.copy(),
                                                                                                 val_df.copy(),
                                                                                                 test_df.copy(),
                                                                                                 max_len=max_len)

            all_results.loc[len(all_results)] = [fold+1, dataset, model, accuracy, precision, recall, f1, auc,
                                                 real_label,
                                                 predict_label]

            print(fold, dataset, model, accuracy, precision, recall, f1, auc)
    model_short_name = "_".join(item[0] for item in Models)
    result_folder = base_path / ".." / "data" / "output"

    os.makedirs(result_folder, exist_ok=True)

    all_results.to_csv(result_folder / f"Fine_Tune_Results_{dataset}_PoL_{post_length}_{model_short_name}.csv", index=False)
    print("Results have been saved to ", result_folder / f"Fine_Tune_Results_{dataset}_PoL_{post_length}_{model_short_name}.csv")


def run_fine_tune_model_Adversarial(dataset, option = "User", post_length = 20, Ratios = None ):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)


    base_path = Path(__file__).resolve().parent

    all_results = pd.DataFrame(
        columns=['fold','ratio','dataset', 'model', 'accuracy', 'precision', 'recall', 'f1-score', 'AUC', 'real_class',
                 'predict_class'])

    Models = ["DistilBERT", "DeBERTa", "BERT", "RoBERTa","MDSM", "SBERT"]


    max_len = 512

    file_path = base_path /".." / "data" / "ML_data" / f"{dataset}_{post_length}.csv"

    input_data = pd.read_csv(file_path)

    CT_data = input_data[input_data['type'] == 'CT'].reset_index(drop=True)

    CT_users = CT_data['user'].unique()

    rng = np.random.default_rng(42)

    CT_users_50 = rng.choice(
        CT_users,
        size=len(CT_users) // 2,
        replace=False
    )

    CT_data = CT_data[CT_data['user'].isin(CT_users_50)]

    for ratio in Ratios:
        print("Ratio ", ratio, " Processing...")

        AI_data = input_data[input_data['type'] == 'AI'].reset_index(drop=True)
        HU_data = input_data[input_data['type'] == 'HU'].reset_index(drop=True)

        AI_users = AI_data['user'].unique()
        AI_users_ratio = rng.choice(
            AI_users,
            size=int(len(AI_users) * (ratio / 100)),
            replace=False
        )

        AI_data = AI_data[AI_data['user'].isin(AI_users_ratio)].reset_index(drop=True)

        HU_users = HU_data['user'].unique()
        HU_users_ratio = rng.choice(
            HU_users,
            size=int(len(HU_users) * (1.0 - (ratio / 100))),
            replace=False
        )

        HU_data = HU_data[HU_data['user'].isin(HU_users_ratio)].reset_index(drop=True)

        print("CT = ", len(CT_users_50), "HU = ", len(HU_users_ratio), "AI = ", len(AI_users_ratio))

        ml_df = pd.concat([CT_data, AI_data, HU_data] , ignore_index=True)

        data = Adversarial_sequence(ml_df, post_length, option)

        for fold, (train_index, test_index) in enumerate(kf.split(data)):
            print(f"Running Fold {fold + 1}...")

            train_data = data.iloc[train_index]
            test_df = data.iloc[test_index]

            # Optional: create validation split from training data
            val_df = train_data.sample(frac=0.1, random_state=42)
            train_df = train_data.drop(val_df.index)

            print("Train :", len(train_data), " Validation :", len(val_df), " Test :", len(test_df))

            for model in Models:
                print(f" *****************************\n Result for Dataset: {dataset} \n Model: {model} \n  fold: {fold} \n Ratio : {ratio} \n *****************************************")

                accuracy, precision, recall, f1, auc, real_label, predict_label, posts = build_Model(model,
                                                                                                     train_df.copy(),
                                                                                                     val_df.copy(),
                                                                                                     test_df.copy(),
                                                                                                     max_len=max_len)

                all_results.loc[len(all_results)] = [fold+1, ratio, dataset, model, accuracy, precision, recall, f1, auc,
                                                     real_label,
                                                     predict_label]

                print(fold, ratio, dataset, model, accuracy, precision, recall, f1, auc)

    result_folder = base_path / ".." / "data" / "output"

    os.makedirs(result_folder, exist_ok=True)

    all_results.to_csv(result_folder / f"Fine_Tune_Results_{dataset}_PoL_{post_length}_Adversarial.csv", index=False)
    print("Results have been saved to ", result_folder / f"Fine_Tune_Results_{dataset}_PoL_{post_length}_Adversarial.csv")


def create_three_types_ml_data(Datasets, post_lengths, IO_type):

    base_path = Path(__file__).resolve().parent
    for dataset in Datasets: 

        print("Datasets : ", dataset)
       

        for post_length in post_lengths:

          
            if IO_type == "HU_AI":
                all_data = create_ML_data_AI_HU(dataset)
            elif IO_type == "HU": 
                all_data = create_ML_data_HU(dataset)
            elif IO_type == "AI":
                all_data = create_ML_data_AI(dataset)
            
            # print(all_data['label'].value_counts())
    
            # Save Data 

            file_name = base_path / ".." / "data" / "ML_data" / f"{IO_type}_{dataset}_{post_length}.csv" 

            all_data.to_csv(file_name, index = False)
            print("Data save at ", file_name)



def create_three_types_ml_data(Datasets, post_lengths, IO_type):

    base_path = Path(__file__).resolve().parent
    for dataset in Datasets: 

        print("Datasets : ", dataset)
       

        for post_length in post_lengths:

          
            if IO_type == "HU_AI":
                all_data = create_ML_data_AI_HU(dataset)
            elif IO_type == "HU": 
                all_data = create_ML_data_HU(dataset)
            elif IO_type == "AI":
                all_data = create_ML_data_AI(dataset)
            
            # print(all_data['label'].value_counts())
    
            # Save Data 

            file_name = base_path / ".." / "data" / "ML_data" / f"{IO_type}_{dataset}_{post_length}.csv" 

            all_data.to_csv(file_name, index = False)
            print("Data save at ", file_name)

            

def run_fine_tune_model_AI_HU(Datasets, post_lengths, IO_type):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    base_path = Path(__file__).resolve().parent
    for dataset in Datasets:   

       

        for post_length in post_lengths:

            max_len = 512
            if IO_type == "HU_AI":
                all_data = create_ML_data_AI_HU(dataset)
            elif IO_type == "HU": 
                all_data = create_ML_data_HU(dataset)
            elif IO_type == "AI":
                all_data = create_ML_data_AI(dataset)
            
            print(all_data['label'].value_counts())
    
            ml_data = create_sequence(all_data, post_length)

            # Save Data 

            file_name = base_path / ".." / "data" / "Few-Shot-Data" / f"Few-Shot-{dataset}.csv"

            ml_data.to_csv()
    
            print(ml_data['class'].value_counts())
    


            all_results = pd.DataFrame(
                columns=['dataset', 'fold', 'model', 'post_length','accuracy', 'precision', 'recall', 'f1-score', 'AUC', 'real_class',
                        'predict_class'])

            Models = ["DeBERTa", "DistilBERT", "BERT", "RoBERTa", "SBERT", "MGT-Dect","OSM-Det" ] # 
            for model in Models:
                print(f" ********************************************** \n Result for Dataset: {dataset} \n Model: {model} \n *****************************************")
                     
                data = ml_data.copy()  
                
                for fold, (train_index, test_index) in enumerate(kf.split(data)):
                    print(f"Running Fold {fold + 1}...")
        
                    train_data = data.iloc[train_index]
                    test_df = data.iloc[test_index]
        
                    # Optional: create validation split from training data
                    val_df = train_data.sample(frac=0.125, random_state=42)
                    train_df = train_data.drop(val_df.index)
        
                    print("Train :", len(train_data), " Validation :", len(val_df), " Test :", len(test_df))
        
                    accuracy, precision, recall, f1, auc, real_label, predict_label, posts = build_Model(model,
                                                                                                            train_df.copy(),
                                                                                                            val_df.copy(),
                                                                                                            test_df.copy(),
                                                                                                            max_len=max_len)
        
                    all_results.loc[len(all_results)] = [dataset, fold+1, model, post_length, accuracy, precision, recall, f1, auc,   real_label,  predict_label]
        
                    print(fold, dataset, model, accuracy, precision, recall, f1, auc)


            model_short_name = "_".join(item[0] for item in Models)
            result_folder = base_path / ".." / "data" / "output"

            os.makedirs(result_folder, exist_ok=True)

            all_results.to_csv(result_folder / f"HU_AI_Tune_Results_{dataset}_PoL_{post_length}_{model_short_name}_{IO_type}.csv", index=False)
            print("Results have been saved to ", result_folder / f"HU_AI_Tune_Results_{dataset}_PoL_{post_length}_{model_short_name}_{IO_type}.csv")

                    
def run_post_level_fine_tune_model(dataset):

    base_path = Path(__file__).resolve().parent

    all_results = pd.DataFrame(
        columns=['subset', 'dataset', 'model', 'accuracy', 'precision', 'recall', 'f1-score', 'AUC', 'real_class',
                 'predict_class'])

    Models = ["DistilBERT", "DeBERTa", "BERT", "RoBERTa","MDSM", "SBERT"]
    subsets = ["100AI" , "100HU", "50AI-50HU"]

    for subset in subsets:
        for model in Models:
            max_len = 512
            print(
                f" ********************************************** \n Result for Dataset: {dataset} \n Model: {model} \n *****************************************")
            train_df, val_df, test_df = create_fine_tune_data(dataset, subset, option = "Post")

            accuracy, precision, recall, f1, auc, real_label, predict_label, posts = build_Model(model,
                                                                                                 train_df.copy(),
                                                                                                 val_df.copy(),
                                                                                                 test_df.copy(),
                                                                                                 max_len=max_len)

            all_results.loc[len(all_results)] = [subset, dataset, model, accuracy, precision, recall, f1, auc,
                                                 real_label,
                                                 predict_label]

            print(subset, dataset, model, accuracy, precision, recall, f1, auc)

    result_folder = base_path / ".." / "data" / "output"

    os.makedirs(result_folder, exist_ok=True)

    all_results.to_csv(result_folder / f"Post_Level_Fine_Tune_Results_{dataset}.csv", index=False)
    print("Results have been saved to ", result_folder / f"Post_Level_Fine_Tune_Results_{dataset}.csv")




def test_Model(train_Model, tokenizer, test, max_len ): 


  
    # Example test data
    test_texts = test['text'].astype(str).tolist()
    test_labels = test['class'].tolist()

    test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')

    # Check if MPS is available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # print("Using device:", device)

   
    # Move test encodings to device
    test_encodings = {key: val.to(device) for key, val in test_encodings.items()}

    # Convert encodings into dataset
    test_dataset = TensorDataset(
        test_encodings["input_ids"],
        test_encodings["attention_mask"],
        torch.tensor(test_labels)
    )

    # DataLoader for batching
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Prediction loop
    train_Model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            outputs = train_Model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            # all_preds.extend(preds.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().astype(int).tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    # metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=1)
    recall = recall_score(all_labels, all_preds, pos_label=1)
    f1 = f1_score(all_labels, all_preds, pos_label=1)
    conf_mat = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
   

    return f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{auc:.4f}" ,all_labels, all_preds



def build_train_Model(model_name, train, val, max_len = 512, class_num = 2):
    # Check unique classes in each split
    # print("Train classes:", len(train), train['class'].value_counts())
    # print("Val classes:", len(val), val['class'].value_counts())
    # print("Test classes:", len(test), test['class'].value_counts())

    if model_name == "OSM-Det":

        model_name = "allenai/longformer-base-4096"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=class_num)



    elif model_name == "DeBERTa":
        select_model_name = "microsoft/deberta-base" # deberta-v3-base" #
        tokenizer = AutoTokenizer.from_pretrained(select_model_name)

        # Select the correct DeBERTa class based on model name
        if "v3" in select_model_name.lower():
            DebertaClass = DebertaV2ForSequenceClassification
        else:
            DebertaClass = DebertaForSequenceClassification


        model = DebertaClass.from_pretrained(select_model_name, num_labels=class_num)

    elif model_name == "RoBERTa":
        select_model_name = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(select_model_name)
        model = RobertaForSequenceClassification.from_pretrained(select_model_name, num_labels=class_num)
    elif model_name == "GraphCodeBERT":
        select_model_name = "microsoft/graphcodebert-base"
        tokenizer = AutoTokenizer.from_pretrained(select_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(select_model_name, num_labels=class_num)

    elif model_name == "DistilBERT":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=class_num)

    elif model_name == "XLNet":
        # Load model for sequence classification
        tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=class_num)
    elif model_name == "BERT":
        # Load pretrained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=class_num)
    elif model_name == "MGT-Dect":
        model_name_base = "distilbert/distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name_base)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_base, num_labels=class_num)


    # elif model_name == "HyproBERT":
    #     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    #     model = HyproBertModel(num_classes=class_num)

    elif model_name == "MDSM":
        loss = "FedFocalLoss"
        model_name = "microsoft/deberta-v3-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MSDMModel(num_classes=class_num, loss=loss) # "FocalLoss" , FedFocalLoss
    elif model_name == "SBERT":
        loss = "FocalLoss"
        model_name = "sentence-transformers/all-mpnet-base-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = SBERT_CNN_Model(num_classes=class_num, loss=loss)


    train_encodings = tokenize(train['text'].astype(str), tokenizer, max_len)
    val_encodings = tokenize(val['text'].astype(str), tokenizer, max_len)


    train_dataset = TextDataset(train_encodings, train['class'].tolist())
    val_dataset = TextDataset(val_encodings, val['class'].tolist())
    # Create a temporary directory
    temp_output_dir = tempfile.mkdtemp()

    training_args = TrainingArguments(
        output_dir=temp_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # evaluation_strategy="epoch",  # validate at end of each epoch
        # save_strategy="epoch",
        save_strategy="no",
        logging_dir='./logs',
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=42,
        # --- Fixed total steps ---
        # max_steps=300,                    # 🔹 stops training after 300 steps
        # num_train_epochs=100,             # large number (won’t matter, as max_steps dominates)

    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[ZeroLossEarlyStoppingCallback(patience=3)]
    )

    start_time = datetime.now()

    # ---- training code ----
    trainer.train()

    end_time = datetime.now()

    training_time_min = (end_time - start_time).total_seconds() / 60
    # print(f"Training time: {training_time_min:.2f} minutes")

    results = trainer.evaluate()
    # print(results)



    # Check if MPS is available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # print("Using device:", device)

    # Move model to device
    model.to(device)

    return model, tokenizer


def  New_Datasets_Results(  train_datasets  , test_datasets , train_type ,  test_type, post_length = 20): 
    

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    base_path = Path(__file__).resolve().parent

    all_results = pd.DataFrame(columns=['train_dataset', 'test_dataset', 'train_type','test_type', 'Model', 'post_length','Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'real_class', 'predict_class'])
    
    for train_item in train_type: 
        train_data = []
        for dataset in train_datasets:

        
            file_name = base_path / ".." / "data" / "Benchmark-ML-Data" / f"{train_item}_{dataset}_{post_length}.csv"

            input_data = pd.read_csv(file_name,  engine="python", on_bad_lines="warn") 

            if train_item == "AI":  # For AI there are mention and hashtag in CT data but no mention and hashtag based long word 
                input_data['post_text'] = input_data['post_text'].apply( lambda x: ' '.join(    word for word in str(x).split()  if len(word) <= 20 )   )
                        
            train_data.append(input_data)


        train_df =  create_sequence(pd.concat(train_data), post_length) 

    


    
        max_len = 512 
        Models = ["DeBERTa"] #, "DistilBERT", "BERT", "RoBERTa", "SBERT", "MGT-Dect","OSM-Det" ] # 
        for model in Models:
            print(f" ********************************************** \n Result for Dataset: {dataset} \n Model: {model} \n *****************************************")
                

            # Optional: create validation split from training data
            val_df = train_df.sample(frac=0.125, random_state=42)
            train_df = train_df.drop(val_df.index)
            
            print("Train :", len(train_data), " Validation :", len(val_df))

            train_Model, tokenizer = build_train_Model(model, train_df.copy(),val_df.copy(), max_len=512)


            for dataset in test_datasets:

                for test_item in test_type: 

                    train_dataset_name = "_".join(item for item in train_datasets)
                    test_dataset_name = dataset


                
                    file_name = base_path / ".." / "data" / "Benchmark-ML-Data" / f"{test_item}_{dataset}_{post_length}.csv"

                    if train_item == "AI":  # Tran data has not mention and hashtag. Thus rmove long word that includes hashtags 
                        input_data['post_text'] = input_data['post_text'].apply( lambda x: ' '.join(    word for word in str(x).split()  if len(word) <= 20 )   )

                         
                    input_data = pd.read_csv(file_name,  engine="python", on_bad_lines="warn") 


                

            
                    test_df =   create_sequence(input_data, post_length) 
                

                    
                    accuracy, precision, recall, f1, auc, real_label, predict_label = test_Model(train_Model, tokenizer, test_df.copy(), max_len )

                    all_results.loc[len(all_results)] = [train_dataset_name, test_dataset_name, train_item, test_item, model, post_length, accuracy, precision, recall, f1, auc,  real_label, predict_label]

                    print(train_dataset_name, test_dataset_name, model, accuracy, precision, recall, f1, auc)


            
    result_folder = base_path / ".." /"Output" / "Benchmark_Results"

    os.makedirs(result_folder, exist_ok=True)
 
    all_results.to_csv(result_folder / f"New_Dataset_Results_{model}_PoL_{post_length}_2.csv", index=False)
    print("Results have been saved to ", result_folder / f"New_Dataset_Results_{model}_PoL_{post_length}.csv")

def HU_AI_Mixed_Results (dataset, model, ratios , post_length):
    """
    Generate results for various mixing of HU-AI Mixed data 
    dataset : dataset name 
    model : fine tuen model
    ratios : differnt ratio of HU users 
    post_length = per user's post length 
    return : Various ratios based five fold cross validation results. 
    """

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    base_path = Path(__file__).resolve().parent
    
    all_results = pd.DataFrame(columns=['dataset', 'ratio', 'fold', 'model', 'post_length','accuracy', 'precision', 'recall', 'f1-score', 'AUC', 'real_class', 'predict_class'])

    for ratio in ratios:  
        all_data = create_ML_data_AI_HU(dataset, ratio)

        ml_data = create_sequence(all_data, post_length)

        print(ml_data['class'].value_counts())
       
        Models = ["DeBERTa"] 
        max_len = 512 


        for model in Models:
            print(f" ********************************************** \n Result for Dataset: {dataset} \n Model: {model} \n *****************************************")
                    
            data = ml_data.copy()  
            
            for fold, (train_index, test_index) in enumerate(kf.split(data)):
                print(f"Running Fold {fold + 1}...")
    
                train_data = data.iloc[train_index]
                test_df = data.iloc[test_index]
    
                # Optional: create validation split from training data
                val_df = train_data.sample(frac=0.125, random_state=42)
                train_df = train_data.drop(val_df.index)
    
                print("Train :", len(train_data), " Validation :", len(val_df), " Test :", len(test_df))
    
                accuracy, precision, recall, f1, auc, real_label, predict_label, posts = build_Model(model,train_df.copy(),val_df.copy(), test_df.copy(), max_len=max_len)
    
                all_results.loc[len(all_results)] = [dataset, ratio, fold+1, model, post_length, accuracy, precision, recall, f1, auc,   real_label,  predict_label]
    
                print(fold, dataset, model, accuracy, precision, recall, f1, auc)


     
    result_folder = base_path / ".." / "data" / "output"

    os.makedirs(result_folder, exist_ok=True)

    all_results.to_csv(result_folder / f"HU_AI_Tune_Results_{dataset}_PoL_{post_length}_{model}_Ratios.csv", index=False)
    print("Results have been saved to ", result_folder /f"HU_AI_Tune_Results_{dataset}_PoL_{post_length}_{model}_Ratios.csv")

        




    
#
if __name__ == '__main__':
    # for dataset in ["Bangladesh","Iran_5", "Spain", "Catalonia", "Russia_1"]:
    #     ML_Data_Statistics(dataset, post_length = 20, dataset_number = 2)


    #     run_models(dataset= "Bangladesh")
    #
    #
    # Generalized_Results(  train_datasets = ['Russia_1'] , test_datasets = ['Iran_5','Catalonia','Bangladesh'] , IO_Option = None )  # IO_Option = "HU" / "AI"
    # Generalized_Results_LLM(Datasets=['Russia_1','Iran_5', 'Catalonia', 'Bangladesh', 'Spain'], post_length= 20)

    #
    # run_models("Spain")

    # show_results("Spain", subset="100AI")
    # Run Experiment based on IO (AI and HU) and CT users detection
    # for dataset in  [ "Bangladesh", "Catalonia", "Iran_5", "Spain", "Russia_1"]:
    #     for post_length in [5,10,30,40,50,100]:
    #         # Generate_AI_HU_CT_DATASET(dataset, post_length = post_length, dataset_number = 2)
    #         run_fine_tune_model(dataset, post_length = post_length, dataset_number = 1)


    # # Run Experiment based on Train AI and Test HU writing IO Detection b
    # Dataset =  ["Spain","Catalonia", "Russia_1"]
    # post_lengths = [5,10,20,30,40,50,100]
    # run_fine_tune_model_AI_HU(Dataset, post_lengths=post_lengths, train="AI")


    # Benchmark Data

    # run_fine_tune_model(dataset = "Benchmark", post_length=20)

    # run_fine_tune_model_Adversarial(dataset="Benchmark", post_length=20, Ratios = [0,10,20,30,40,50, 60,70,80,90,100])



     #Experiment Run on 15-09-2026 for NAACL Papers 



    # create_three_types_ml_data(Dataset, post_lengths=post_lengths, IO_type= "HU") # Only HU IO users 
    
    # create_three_types_ml_data(Dataset, post_lengths=post_lengths, IO_type= "AI") # Only AI IO users 
    
    # create_three_types_ml_data(Dataset, post_lengths=post_lengths, IO_type= "HU_AI") # Mixed HU and AI IO users 
    
    # Main Results 

    # Dataset =  [ "Russia_1","Spain"] # "Bangladesh", "Catalonia", "Iran_5", 
    # post_lengths =  [20]

    # # run_fine_tune_model_AI_HU(Dataset, post_lengths=post_lengths, IO_type= "HU_AI") # Mixed HU and AI IO users 
    # run_fine_tune_model_AI_HU(Dataset, post_lengths=post_lengths, IO_type= "HU") # Only HU IO users 

    # run_fine_tune_model_AI_HU(Dataset, post_lengths=post_lengths, IO_type= "AI") # Only AI IO users 
   

    # Generalisation Results 

    # train_data = ["Russia_1"]
    # test_data = ['Iran_5','Catalonia','Bangladesh','Spain']
    # train_type = ["AI"] # "HU",
    # test_type = ["HU", "HU_AI","AI"]
    # New_Datasets_Results(  train_datasets = train_data , test_datasets = test_data, train_type = train_type, test_type = test_type, post_length = 20)
   

   # HU-AI mixed results [0,20,40,50,60,80,100] 

    ratios = [20, 40, 60, 80] # AI data ratio
    dataset = "Russia_1"
    model = "DeBERTa"
    HU_AI_Mixed_Results (dataset, model, ratios , post_length = 20)
