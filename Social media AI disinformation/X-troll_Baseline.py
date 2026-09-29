
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef
from langchain_ollama import OllamaEmbeddings
from sklearn.model_selection import train_test_split, StratifiedKFold

import random
import copy, os
import numpy as np
import torch, sys
import torch.nn as nn
import ast 
from transformers import   AutoTokenizer,  AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from  Create_ML_Datasets import create_ML_data_HU, create_ML_data_AI_HU, create_ML_data_AI
 

root_dir = Path(__file__).parent.parent
print("Root dir: ", root_dir)
source_dir = root_dir / "data"/ "Target"
output_dir = root_dir / "AWS_Output" 
model_dir = root_dir / "Best_Model"
Adapters =["Sentiment", "Influence", "Propaganda_Strategy","TaskAdapter"]

# 1. Define the dimension globally or within the class
# D_MODEL = 4096  # Standard for Llama 3.1 8B

D_MODEL = {"llama3.1:8b":4096,"long_transformer":768, "nomic-embed-text": 768}
batch_size = 32



class AppraisalAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, num_labels=12):
        super(AppraisalAdapter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 1. Base Linear Layer (W) - Equivalent to nn.Linear
        # In the paper, this is the frozen base model weight
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.weight.requires_grad = False

        # 2. LoRA Matrices (B and A) - capturing evaluative language patterns
        # Matrix A (Gaussian init)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Matrix B (Zero init)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 3. Appraisal Classifier Head
        # Used for token-level sequence labeling as described in Stage 4.3.1 (1)
        self.appraisal_classifier = nn.Linear(out_features, num_labels)

        # Storage for the auxiliary output (logits) for multitask loss calculation
        self.last_appraisal_logits = None

    def forward(self, x):
        """
        x: Input hidden states [batch, seq_len, d_model]
        """
        # Step 1: Standard W*h
        base_out = self.base_layer(x)

        # Step 2: LoRA path: Δh = (x @ A^T @ B^T) * scaling
        # This implements W' = W + BA
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

        # Combined representation h'
        h_prime = base_out + lora_out

        # Step 3: Generate token-level appraisal labels for Lappraisal
        # We store this in the object so the main training loop can access it
        self.last_appraisal_logits = self.appraisal_classifier(h_prime)

        # Return only h_prime to maintain same output format as nn.Linear
        return h_prime

class PropagandaIdAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super(PropagandaIdAdapter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 1. Base Linear Layer (W)
        # Remains frozen to preserve general LLM knowledge
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.weight.requires_grad = False

        # 2. LoRA Beta Matrices (B_beta and A_beta)
        # Targeted low-rank updates for propaganda features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 3. Binary Propaganda Classifier Head (p_i)
        # Optimized using Binary Cross-Entropy Loss (L_prop)
        self.binary_classifier = nn.Linear(out_features, 1)

        # Auxiliary storage for the binary prediction (p_i)
        self.last_propaganda_logits = None

    def forward(self, x):
        """
        x: Input hidden states [batch, seq_len, d_model]
        """
        # Step 1: Base forward pass (Wh)
        base_out = self.base_layer(x)

        # Step 2: Low-rank update (B_beta * A_beta * h)
        # This captures features sensitive to propaganda presence
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

        # h_prime = Wh + ΔW_beta * h
        h_prime = base_out + lora_out

        # Step 3: Compute binary probability (p_i)
        # Typically computed on the [CLS] token for sequence-level detection
        # We store the logits so the loss L_prop can be calculated during training
        # self.last_propaganda_logits = self.binary_classifier(h_prime[:, 0, :])

        # Returns h_prime to stay compatible with the next layers in the LLM
        return h_prime

class PropagandaStratAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, num_strategies=20):
        super(PropagandaStratAdapter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.C = num_strategies # Number of propaganda technique classes

        # 1. Base Linear Layer (W) - Frozen
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.weight.requires_grad = False

        # 2. LoRA Theta Matrices (B_theta and A_theta)
        # Captures specific multi-class strategy features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 3. Multi-class Strategy Classifier Head (p_{i,c})
        # Optimized using Categorical Cross-Entropy Loss (L_strat)
        self.strategy_classifier = nn.Linear(out_features, self.C)

        # Auxiliary storage for the multi-class logits (p_i,c)
        self.last_strategy_logits = None

    def forward(self, x):
        """
        x: Input hidden states [batch, seq_len, d_model]
        """
        # Step 1: Base forward pass (Wh)
        base_out = self.base_layer(x)

        # Step 2: Low-rank update (B_theta * A_theta * h)
        # Targeted for fine-grained classification of techniques
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

        # h_prime = Wh + ΔW_theta * h
        h_prime = base_out + lora_out

        # Step 3: Compute categorical logits (p_i,c)
        # Performed on the [CLS] token (index 0) of the enhanced representation
        # Stored for L_strat calculation during the training loop
        # self.last_strategy_logits = self.strategy_classifier(h_prime[:, 0, :])

        # Return h_prime for compatibility with subsequent layers
        return h_prime

class TaskAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super(TaskAdapter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 1. Base Linear Layer (W) - Frozen to preserve base LLM knowledge
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.weight.requires_grad = False

        # 2. LoRA Task Matrices (B_t and A_t)
        # These matrices learn patterns distinctive to state-sponsored trolls
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 3. Troll Classification Head (p_i)
        # Optimized using Binary Cross-Entropy Loss (L_task)
        self.troll_classifier = nn.Linear(out_features, 1)

        # Auxiliary storage for the prediction p_i
        self.last_troll_logits = None

    def forward(self, x):
        """
        x: Input hidden states [batch, seq_len, d_model]
        """
        # Step 1: Base forward pass (Wh)
        base_out = self.base_layer(x)

        # Step 2: Low-rank update (B_t * A_t * h)
        # This captures the "behavioral homophily" described by Yuan et al. [50]
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

        # h_prime = Wh + ΔW_t * h
        h_prime = base_out + lora_out

        # Step 3: Compute binary probability (p_i) for troll detection
        # Typically calculated on the [CLS] token of the representation
        # Stored for L_task calculation during the training loop
        # self.last_troll_logits = self.troll_classifier(h_prime[:, 0, :])

        # Returns h_prime to maintain same format as nn.Linear
        return h_prime


class SentimentAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super(SentimentAdapter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 1. Base Linear Layer (W) - Frozen to preserve base LLM knowledge
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.weight.requires_grad = False

        # 2. LoRA Task Matrices (B_t and A_t)
        # These matrices learn patterns distinctive to state-sponsored trolls
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 3. Troll Classification Head (p_i)
        # Optimized using Binary Cross-Entropy Loss (L_task)
        self.troll_classifier = nn.Linear(out_features, 1)

        # Auxiliary storage for the prediction p_i
        self.last_troll_logits = None

    def forward(self, x):
        """
        x: Input hidden states [batch, seq_len, d_model]
        """
        # Step 1: Base forward pass (Wh)
        base_out = self.base_layer(x)

        # Step 2: Low-rank update (B_t * A_t * h)
        # This captures the "behavioral homophily" described by Yuan et al. [50]
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

        # h_prime = Wh + ΔW_t * h
        h_prime = base_out + lora_out

        # Step 3: Compute binary probability (p_i) for troll detection
        # Typically calculated on the [CLS] token of the representation
        # Stored for L_task calculation during the training loop
        # self.last_troll_logits = self.troll_classifier(h_prime[:, 0, :])

        # Returns h_prime to maintain same format as nn.Linear
        return h_prime


class InfluenceAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super(InfluenceAdapter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 1. Base Linear Layer (W) - Frozen to preserve base LLM knowledge
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.weight.requires_grad = False

        # 2. LoRA Task Matrices (B_t and A_t)
        # These matrices learn patterns distinctive to state-sponsored trolls
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 3. Troll Classification Head (p_i)
        # Optimized using Binary Cross-Entropy Loss (L_task)
        self.troll_classifier = nn.Linear(out_features, 1)

        # Auxiliary storage for the prediction p_i
        self.last_troll_logits = None

    def forward(self, x):
        """
        x: Input hidden states [batch, seq_len, d_model]
        """
        # Step 1: Base forward pass (Wh)
        base_out = self.base_layer(x)

        # Step 2: Low-rank update (B_t * A_t * h)
        # This captures the "behavioral homophily" described by Yuan et al. [50]
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

        # h_prime = Wh + ΔW_t * h
        h_prime = base_out + lora_out

        # Step 3: Compute binary probability (p_i) for troll detection
        # Typically calculated on the [CLS] token of the representation
        # Stored for L_task calculation during the training loop
        # self.last_troll_logits = self.troll_classifier(h_prime[:, 0, :])

        # Returns h_prime to maintain same format as nn.Linear
        return h_prime



class SentimentAppraisalAdapter_old(nn.Module):
    def __init__(
        self,
        num_sentiments=74,
        rank=8,
        alpha=16,
        dropout=0.1,
        output_activation=None
    ):
        """
        Adapter for Sentence-BERT-based sentiment subtype vectors.

        Input:
            x: [batch_size, num_posts, 74]

        Output:
            out: [batch_size, num_posts, 74]

        Args:
            num_sentiments: number of sentiment subtypes, default 74
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: dropout rate
            output_activation:
                None      -> returns raw adapted scores
                "sigmoid" -> returns values between 0 and 1
        """

        super(SentimentAppraisalAdapter, self).__init__()

        self.num_sentiments = num_sentiments
        self.rank = rank
        self.scaling = alpha / rank
        self.output_activation = output_activation

        self.dropout = nn.Dropout(dropout)

        # Base linear layer: 77 -> 77
        self.base_layer = nn.Linear(num_sentiments, num_sentiments)

        # Initialize base layer as identity transformation
        nn.init.eye_(self.base_layer.weight)
        nn.init.zeros_(self.base_layer.bias)

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        self.base_layer.bias.requires_grad = False

        # LoRA matrix A: 77 -> rank
        self.lora_A = nn.Parameter(torch.zeros(rank, num_sentiments))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # LoRA matrix B: rank -> 77
        self.lora_B = nn.Parameter(torch.zeros(num_sentiments, rank))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Tensor of shape [batch_size, num_posts, 77]
            attention_mask: Tensor of shape [batch_size, num_posts]
                            1 for real posts, 0 for padded posts

        Returns:
            out: Tensor of shape [batch_size, num_posts, 77]
        """

        if x.dim() != 3:
            raise ValueError(
                f"Expected input shape [batch_size, num_posts, 77], but got {x.shape}"
            )

        if x.size(-1) != self.num_sentiments:
            raise ValueError(
                f"Expected last dimension {self.num_sentiments}, but got {x.size(-1)}"
            )

        # Base path
        base_out = self.base_layer(x)

        # LoRA path
        dropped_x = self.dropout(x)

        lora_out = (
            dropped_x @ self.lora_A.t() @ self.lora_B.t()
        ) * self.scaling

        # Final adapted sentiment representation
        out = base_out + lora_out

        # Optional activation
        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)

        # Remove outputs from padded posts
        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1).float()

        return out



class SentimentProjector(nn.Module):
    def __init__(self, input_dim=74, d_model=4096, dropout=0.1):
        super(SentimentProjector, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x shape: [batch_size, num_posts, 74]
        output shape: [batch_size, num_posts, 4096]
        """
        return self.projector(x)


class InfluenceProjector(nn.Module):
    def __init__(self, input_dim=45, d_model=4096, dropout=0.1):
        super(InfluenceProjector, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x shape: [batch_size, num_posts, 74]
        output shape: [batch_size, num_posts, 4096]
        """
        return self.projector(x)



class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, h_fused, post_mask=None):
        # h_fused: [B, P, d_model]
        scores = self.attn(h_fused).squeeze(-1)  # [B, P]

        if post_mask is not None:
            scores = scores.masked_fill(post_mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)   # [B, P]
        pooled = torch.sum(h_fused * weights.unsqueeze(-1), dim=1)

        return pooled, weights


class GatedAttentionPooling(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.V = nn.Linear(d_model, d_model)
        self.U = nn.Linear(d_model, d_model)
        self.w = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, mask=None):
        """
        h:    [Batch, Num_Posts, d_model]
        mask: [Batch, Num_Posts], where 1 = valid post, 0 = padding
        """

        gated = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        gated = self.dropout(gated)

        scores = self.w(gated).squeeze(-1)  # [Batch, Num_Posts]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=1)  # [Batch, Num_Posts]

        pooled = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)  # [Batch, d_model]

        return pooled, attn_weights




class X_Troll_Model(nn.Module):
    def __init__(self, model_name="llama3.1:8b", options = None) : # Llama 3.1 8B uses 4096
        super(X_Troll_Model, self).__init__( )

       # Embedding layer for feature extraction
        # self.embeddings = OllamaEmbeddings(model=model_name)
        self.d_model = D_MODEL[model_name]



        # Four Specialized  LoRA Adapters
        self.adapter_appraisal = AppraisalAdapter(self.d_model, self.d_model)
        self.adapter_prop_id = PropagandaIdAdapter(self.d_model, self.d_model)
        self.adapter_prop_strat = PropagandaStratAdapter(self.d_model, self.d_model)
        self.adapter_task = TaskAdapter(self.d_model, self.d_model)


        # self.num_experts = 4
        # 4. Dynamic Gating Mechanism
        self.w = nn.Parameter(torch.zeros(len(options)))

        # Gated attention pooling over posts
        self.post_pooling = GatedAttentionPooling(
            d_model=self.d_model,
            dropout=0.1
        )

        # 5. Timeline Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.timeline_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.troll_detector = nn.Linear(self.d_model, 1)

        self.layer_norm = nn.LayerNorm(self.d_model)   # <-- added

    def X_troll_detection(self, h_base):
        # Step D: Timeline Encoding
        # Transformer expects [Batch, Seq_Len, d_model] when batch_first=True



        h_base = self.timeline_transformer(h_base)
        # h_base shape: [Batch, 1, d_model]

        # Step B: Expert Knowledge Extraction
        # We pass the whole batch through the adapters
        h_k = torch.stack([
            self.adapter_appraisal(h_base),
            self.adapter_prop_id(h_base),
            self.adapter_prop_strat(h_base),
            self.adapter_task(h_base),
        ], dim=2) # Result: [Batch, Num_Posts, 6, d_model]

        # Step C: Dynamic Gating
        alphas = torch.softmax(self.w, dim=0)
        # Reshape alphas to multiply correctly across the '4' experts
        h_fused = torch.sum(alphas.view(1, 1, 4, 1) * h_k, dim=2) # [Batch, Num_Posts, d_model]

        h_fused = self.layer_norm(h_fused)   # <-- normalize before returning


        # Global average pooling over the posts (dim 1)
        t_u = torch.mean(h_fused, dim=1) # [Batch, d_model]

        # Step E: Final Decision
        is_troll = torch.sigmoid(self.troll_detector(t_u)) # [Batch, 1]

        return {
            "is_troll": is_troll,
            "gating_weights": alphas
        }

    def results_basen_on_pooling(self, post_pooling, h_fused, pooling_name):


        # print("Pooling = ", pooling_name)
        if pooling_name == "Max":
            t_u, _ = torch.max(h_fused, dim=1)
        elif pooling_name == "Attention":
            t_u, post_attention_weights = post_pooling(h_fused)

        elif pooling_name == "GatedAttention":
            batch_size, num_posts, _ = h_fused.shape

            post_mask = torch.ones(
                batch_size,
                num_posts,
                device=h_fused.device,
                dtype=torch.long
            )

            t_u, post_attention_weights = post_pooling(
                h_fused,
                mask=post_mask
            )
        else:
            t_u = torch.mean(h_fused, dim=1)  # [Batch, d_model]
        return t_u

def train_one_epoch(model, train_loader, optimizer, criterion, device, max_posts, options , pooling_name , model_name):
    model.train()
    total_loss = 0
    global batch_size

    # print(id(train_loader[0][0]), id(train_loader[1][0]))

    for i in range(0, len(train_loader), batch_size):
        # 1. Get the current slice of tuples
        batch_slice = train_loader[i : i + batch_size]
        batch_h,  labels = zip(*batch_slice)
        
        batch_h = torch.stack(batch_h).to(device).float()

        labels = torch.as_tensor(labels).to(device).float().view(-1, 1)


        optimizer.zero_grad()
        output = model.X_troll_detection(batch_h)
        loss = criterion(output['is_troll'], labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, loader, device, max_posts, options, pooling_name ):
    model.eval()
    all_preds, all_labels = [], []
    global batch_size
    with torch.no_grad():
        for i in range(0, len(loader), batch_size):
            # 1. Get the current slice of tuples
            batch_slice = loader[i : i + batch_size]

            batch_h,  labels = zip(*batch_slice)
            #
            # print("Batch_h : ", batch_h)
            # print("Labels : ", labels)

            processed_batches = []


            # 1. Iterate through each user in the batch
            for user_timeline in batch_h:
                # Convert list of tensors to a single tensor [Num_Posts, 4096]
                if isinstance(user_timeline, (list, tuple)):
                    user_tensor = torch.stack(user_timeline)
                else:
                    user_tensor = user_timeline


                current_posts = user_tensor.size(0)

                # 2. PADDING LOGIC
                if current_posts < max_posts:
                    # Create a zero tensor of shape [Missing_Posts, 4096]
                    padding = torch.zeros((max_posts - current_posts, 4096), dtype=user_tensor.dtype)
                         # Concatenate the actual posts with the zeros
                    user_tensor = torch.cat([user_tensor, padding], dim=0)

                elif current_posts > max_posts:
                    # If a user somehow has more than 20, truncate them
                    user_tensor = user_tensor[:max_posts, :]


                processed_batches.append(user_tensor)

            # 3. Now they are all [20, 4096], so we can stack them into [Batch, 20, 4096]
            batch_h = torch.stack(processed_batches).to(device).float()



            labels = torch.as_tensor(labels).to(device).float().view(-1, 1)

            batch_h = batch_h.to(device)
            output = model.X_troll_detection(batch_h)
            preds = (output['is_troll'] > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, zero_division=0),
        "Recall": recall_score(all_labels, all_preds,  zero_division=0), # pos_label=1,
        "F1": f1_score(all_labels, all_preds, zero_division=0),  #, pos_label=1
        "AUC":  roc_auc_score(all_labels, all_preds),
        "MCC": matthews_corrcoef(all_labels, all_preds),
        "PR-AUC":  average_precision_score(all_labels,all_preds),  
    }




def run_pipeline(train_loader, eval_loader, test_loader, model_name = None, options = None, pooling_name = "Mean" , fold = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 30
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = X_Troll_Model(model_name = model_name, options=options).to(device)

    # 2. Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    criterion = torch.nn.BCELoss()

    best_val_f1 = 0.0
    best_model_state = None
    patience = 6
    patience_counter = 0



    # 3. Training Loop
    for epoch in range(epochs):

        # -----------------------------
        # Train on training data
        # -----------------------------
        train_loss = train_one_epoch(model,train_loader,optimizer,criterion,device, 1,options, pooling_name, model_name )

        # -----------------------------
        # Validate on validation data
        # -----------------------------
        val_metrics = evaluate(model,eval_loader, device,1, options,pooling_name)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")

        print(
            f"Val Acc: {val_metrics['Accuracy']:.4f} | "
            f"Val Precision: {val_metrics['Precision']:.4f} | "
            f"Val Recall: {val_metrics['Recall']:.4f} | "
            f"Val F1: {val_metrics['F1']:.4f}"
        )

        print("-" * 30)

        # -----------------------------
        # Save best model based on Val F1
        # -----------------------------
        if val_metrics["F1"] > best_val_f1:
            best_val_f1 = val_metrics["F1"]

            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"Best model updated. Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # -----------------------------
        # Early stopping
        # -----------------------------
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    #  Final Evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_results = evaluate( model, test_loader, device, 1, options, pooling_name )

    print("Final Test Results")
    print(final_results)

    return final_results






def create_and_save_post_embedding(input_data, dataset, model_name, IO, max_post_per_user):
    '''
    input_data : input data contain  ['accountid', 'post_text', 'post_time', 'is_control'] columns
    model_name: embedding model name e.g. llama3.1:8b model
    IO: type of IO (HU, AI, HU-AI)
    dataset: dataset name
    max_post_per_user: max post per user (e.g. 100)
    ratio: data ratio between IO and ctrl data 
    return: Save post embedding
    '''
  
    print(input_data['label'].value_counts())

    if model_name in ["llama3.1:8b", "nomic-embed-text"]: 
        # Embedding model for converting LLM response to vector
        embedding = OllamaEmbeddings(model=model_name,  base_url="http://127.0.0.1:11434")
        post_embeddings = []
        labels = []

        print("Input Dataset Length: ", len(input_data))
        users = input_data['user'].unique()
        print("Number of users: ", len(users))


        for i, user in enumerate(users):
            print("User : ", i)

            try:
                temp_df = input_data.loc[input_data['user'] == user]
                post_texts = "|".join([str(item) for item in temp_df['post_text'].values[0:max_post_per_user]])
                post_embeddings.append(embedding.embed_query(post_texts))

                
            except Exception as e:
                post_embeddings.append(embedding.embed_query("EMPTY_POST"))
                pass 

            label =  temp_df['label'].unique().tolist()[0]
            # print(i, "  user : ", user, " status: ", label)
            
            
            if label == "CT":            
                labels.append(0)
            else: 
                labels.append(1)


           
        post_embeddings = torch.tensor(post_embeddings)
        # Insert a dimension at index 1
        post_embeddings = post_embeddings.unsqueeze(1)  # torch.Size([n, 1, 4096])

        labels = torch.tensor(labels)

        # File name
        file_path = source_dir.joinpath(f"{dataset}/post_embeddings_{model_name}_{IO}_{max_post_per_user}.pt") # source_dir.joinpath(f"{dataset}/post_embeddings_{model_name}_{IO}_{ratio}.pt")


        data = list(zip(post_embeddings, labels))


        torch.save(data, file_path)
        print("Post embedding saved at : ", file_path)
        
    elif model_name == "long_transformer":
        print("Long transformer based embedding generating...")

        model_id = "allenai/longformer-base-4096"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

        model.eval()

        def get_longformer_embedding(text):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)

            # Last hidden states
            last_hidden = outputs.last_hidden_state        # (1, seq_len, 768)

            # Mean pooling
            embedding = last_hidden.mean(dim=1)            # (1,768)

            return embedding.squeeze().numpy()
        
        post_embeddings = []
        labels = []


        print("Input Dataset Length: ", len(input_data))
        users = input_data['user'].unique()
        print("Number of users: ", len(users))


        for i, user in enumerate(users):
            print("User : ", i)

            try:
                temp_df = input_data.loc[input_data['user'] == user]
                post_texts = "|".join([str(item) for item in temp_df['post_text'].values[0:max_post_per_user]])

                post_embeddings.append(get_longformer_embedding(post_texts))
               
            except Exception as e:
                post_embeddings.append(get_longformer_embedding("EMPTY_POST"))
                pass 

            label =  temp_df['label'].unique().tolist()[0]
            # print(i, "  user : ", user, " status: ", label)

            if label == "CT":            
                labels.append(0)
            else: 
                labels.append(1)

        post_embeddings = torch.tensor(post_embeddings)
        # Insert a dimension at index 1
        post_embeddings = post_embeddings.unsqueeze(1)  # torch.Size([n, 1, 4096])

        labels = torch.tensor(labels)

        # File name
        file_path = source_dir.joinpath(f"{dataset}/post_embeddings_{model_name}_{IO}_{max_post_per_user}.pt")


        data = list(zip(post_embeddings, labels))


        torch.save(data, file_path)
        print("Post embedding saved at : ", file_path)



def run_X_Troll_Model(Datasets, model_name , Language , IO , max_posts):
   

    for dataset in Datasets:
        df_res = pd.DataFrame(columns=["Dataset", "Model", "Fold", "Accuracy", "Precision", "Recall", "F1", "AUC","MCC", "PR-AUC"])

        base_path = Path(__file__).resolve().parent 
    
        file_name = base_path / ".." / "data" / "Benchmark-ML-Data" / f"{IO}_{dataset}_{max_posts}.csv"

        input_data = pd.read_csv(file_name, engine = "python",   on_bad_lines="warn") 
        print(input_data.columns)
        print(input_data['label'].value_counts())


        # create output_dir if does not exit
        output_dir = source_dir / dataset
        output_dir.mkdir(parents=True, exist_ok=True) 
       
        # Create Post Embedding
        post_embedding_file = source_dir.joinpath(f"{dataset}/post_embeddings_{model_name}_{IO}_{max_posts}.pt")
        if not post_embedding_file.exists():
            print("Creating post embeddings...")
            create_and_save_post_embedding(input_data, dataset, model_name, IO, max_posts)

            data  = torch.load(post_embedding_file)
            post_embeddings, labels = zip(*data)
            del post_embeddings
        else:
            print("Loading... exiting post embeddings...")
            data = torch.load(post_embedding_file)
            post_embeddings, labels = zip(*data)
            del post_embeddings


        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(device)

        # Indices of samples
        indices = np.arange(len(data))
            
           
            
        n_splits = 5           
        labels_for_split = [item[-1] for item in data]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # StratifiedKFold.split needs indexable X and y -- use a dummy range for X
        indices = np.arange(len(data))

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels_for_split), start=1):

            train_val_data = [data[i] for i in train_val_idx]
            test_loader = [data[i] for i in test_idx]

            train_val_labels = [row[-1] for row in train_val_data]

            train_idx, val_idx = train_test_split(
                np.arange(len(train_val_data)),
                test_size=0.125,
                stratify=train_val_labels,
                random_state=42
            )

            train_loader = [train_val_data[i] for i in train_idx]
            eval_loader = [train_val_data[i] for i in val_idx] 


            final_results = run_pipeline(train_loader, eval_loader, test_loader, model_name,  Adapters, "GattedAttention", fold_idx )

            df_res.loc[len(df_res)] = [dataset, "X-Troll", fold_idx, final_results["Accuracy"], final_results["Precision"], final_results["Recall"],final_results["F1"], final_results["AUC"], final_results["MCC"], final_results["PR-AUC"]]

        # create output_dir if does not exit
        output_dir.mkdir(parents=True, exist_ok=True)

        if Language == "En":
            output_file = output_dir.joinpath(f"X-Troll_Benchmark_{dataset}_{Language}_{IO}_{model_name}_Results_{max_posts}.csv")
        else:
            output_file = output_dir.joinpath(f"X-Troll_Benchmark_{dataset}_{IO}_{model_name}_Results_{max_posts}.csv")


        df_res.to_csv(output_file, index=False)
        print(f"Results Saved at {output_file}")



def build_train_model(train_loader, eval_loader, model_name = None, options = None, pooling_name = "Mean"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 30
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = X_Troll_Model(model_name = model_name, options=options).to(device)

    # 2. Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    criterion = torch.nn.BCELoss()

    best_val_f1 = 0.0
    best_model_state = None
    patience = 6
    patience_counter = 0

    # 3. Training Loop
    for epoch in range(epochs):

        # -----------------------------
        # Train on training data
        # -----------------------------
        train_loss = train_one_epoch(model,train_loader,optimizer,criterion,device, 1,options, pooling_name, model_name )

        # -----------------------------
        # Validate on validation data
        # -----------------------------
        val_metrics = evaluate(model,eval_loader, device,1, options,pooling_name)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")

        print(
            f"Val Acc: {val_metrics['Accuracy']:.4f} | "
            f"Val Precision: {val_metrics['Precision']:.4f} | "
            f"Val Recall: {val_metrics['Recall']:.4f} | "
            f"Val F1: {val_metrics['F1']:.4f}"
        )

        print("-" * 30)

        # -----------------------------
        # Save best model based on Val F1
        # -----------------------------
        if val_metrics["F1"] > best_val_f1:
            best_val_f1 = val_metrics["F1"]

            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"Best model updated. Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # -----------------------------
        # Early stopping
        # -----------------------------
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    #  Final Evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model



def  New_Datasets_Results(  train_datasets  , test_datasets , train_type ,  test_type, model_name, post_length = 20): 
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = Path(__file__).resolve().parent

    df_res = pd.DataFrame(columns=['train_dataset', 'test_dataset', 'train_type','test_type', 'Model', 'post_length','Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC', 'PR-AUC'])

  


    for train_item in train_type:   
        train_post_embeddings = []
        train_labels = []
        for dataset in train_datasets:

            # create output_dir if does not exit
            output_dir = source_dir / dataset
            output_dir.mkdir(parents=True, exist_ok=True) 
        
            file_name = base_path / ".." / "data" / "Benchmark-ML-Data" / f"{train_item}_{dataset}_{post_length}.csv"

            input_data = pd.read_csv(file_name,  engine="python", on_bad_lines="warn") 

         
            
            # Create Post Embedding
            post_embedding_file = source_dir.joinpath(f"{dataset}/post_embeddings_{model_name}_{train_item}_{post_length}.pt")
            if not post_embedding_file.exists():
                print("Creating post embeddings...")
                create_and_save_post_embedding(input_data, dataset, model_name, train_item, post_length)
    
                data  = torch.load(post_embedding_file)
                post_embeddings, labels = zip(*data)
           
            else:
                print("Loading... exiting post embeddings...")
                data = torch.load(post_embedding_file)
                post_embeddings, labels = zip(*data)
             
              # Add this dataset to the combined training data
            train_post_embeddings.extend(post_embeddings)
            train_labels.extend(labels)

            del post_embeddings, labels

        # Convert to tensors if necessary
        train_post_embeddings = torch.stack(train_post_embeddings)
        train_labels = torch.tensor(train_labels)

        # 12.5% validation, 87.5% training
        train_embeddings, valid_embeddings, train_labels, valid_labels = train_test_split(
            train_post_embeddings,
            train_labels,
            test_size=0.125,
            random_state=42,
            stratify=train_labels
        )

        print("Train embeddings:", train_embeddings.shape)
        print("Train labels:", train_labels.shape)

        print("Validation embeddings:", valid_embeddings.shape)
        print("Validation labels:", valid_labels.shape)


        train_loader = list(zip(train_embeddings, train_labels))
        eval_loader = list(zip(valid_embeddings, valid_labels))

        # Build Model based on training datasets 

        train_model = build_train_model(train_loader, eval_loader,  model_name,  Adapters, "GattedAttention")




    
        print(f" ********************************************** \n Result for Dataset: {dataset} \n Model: {model_name} \n *****************************************")
                

         

        for dataset in test_datasets:

            for test_item in test_type: 

                train_dataset_name = "_".join(item for item in train_datasets)
                test_dataset_name = dataset

                

                file_name = base_path / ".." / "data" / "Benchmark-ML-Data" / f"{test_item}_{dataset}_{post_length}.csv"
            
                input_data = pd.read_csv(file_name,  engine="python", on_bad_lines="warn") 
    
                
                
                # Create Post Embedding
                post_embedding_file = source_dir.joinpath(f"{dataset}/post_embeddings_{model_name}_{test_item}_{post_length}.pt")
                if not post_embedding_file.exists():
                    print("Creating post embeddings...")
                    create_and_save_post_embedding(input_data, dataset, model_name, test_item, post_length)
        
                    data  = torch.load(post_embedding_file)
                    test_embeddings, test_labels = zip(*data)
                    
                else:
                    print("Loading... exiting post embeddings...")
                    data = torch.load(post_embedding_file)
                    test_embeddings, test_labels = zip(*data)

                test_loader = list(zip(test_embeddings, test_labels))


                final_results = evaluate( train_model, test_loader, device, 1, Adapters, "GattedAttention" )

                print("Final Test Results")
                print(final_results)

                df_res.loc[len(df_res)] =  [train_dataset_name, test_dataset_name, train_item, test_item, "X-Troll", post_length,  final_results["Accuracy"], final_results["Precision"], final_results["Recall"],final_results["F1"], final_results["AUC"], final_results["MCC"], final_results["PR-AUC"]]
                
    
    output_file = source_dir.joinpath(f"X-Troll_New_Dataset_Results_{train_dataset_name}_{model_name}_Results_{post_length}.csv")
    df_res.to_csv(output_file, index=False)
    print(f"Results Saved at {output_file}")


def read_post_embedding(IO_type, dataset, model_name, max_posts):
    base_path = Path(__file__).resolve().parent 
    file_name = base_path / ".." / "data" / "Benchmark-ML-Data" / f"{IO_type}_{dataset}_{max_posts}.csv"
    input_data = pd.read_csv(file_name, engine = "python",   on_bad_lines="warn") 
    print(input_data.columns)
    print(input_data['label'].value_counts())


    # create output_dir if does not exit
    output_dir = source_dir / dataset
    output_dir.mkdir(parents=True, exist_ok=True) 

    # Create Post Embedding
    post_embedding_file = source_dir.joinpath(f"{dataset}/post_embeddings_{model_name}_{IO_type}_{max_posts}.pt")
    if not post_embedding_file.exists():
        print("Creating post embeddings...")
        create_and_save_post_embedding(input_data, dataset, model_name, IO_type, max_posts)

        data  = torch.load(post_embedding_file)
        hu_post_embeddings, hu_labels = zip(*data)
    
    else:
        print("Loading... exiting post embeddings...")
        data = torch.load(post_embedding_file)
        hu_post_embeddings, hu_labels = zip(*data)

    return hu_post_embeddings, hu_labels

import torch


def create_hu_ai_mix(  hu_post, hu_labels, ai_post, ai_labels, hu_ai_ratio=0.5,random_seed=42):

    g = torch.Generator().manual_seed(random_seed)

    # --------------------------------------------------
    #  HU-CT: HU posts with label == 0
    # --------------------------------------------------
    hu_ct_idx = torch.where(hu_labels == 0)[0]

    hu_ct_post = hu_post[hu_ct_idx]
    hu_ct_labels = hu_labels[hu_ct_idx]

    # --------------------------------------------------
    #  HU positive: HU posts with label == 1
    # --------------------------------------------------
    hu_idx = torch.where(hu_labels == 1)[0]

    # --------------------------------------------------
    # AI positive: AI posts with label == 1
    # --------------------------------------------------
    ai_idx = torch.where(ai_labels == 1)[0]

    # --------------------------------------------------
    # Calculate HU-AI sample numbers
    # --------------------------------------------------
    n_hu = int(len(hu_idx)* hu_ai_ratio)

    n_ai = int(len(ai_idx)* (1.0 - hu_ai_ratio))
   
    # --------------------------------------------------
    # Randomly sample HU and AI
    # --------------------------------------------------
    hu_idx = hu_idx[ torch.randperm(len(hu_idx), generator=g)[:n_hu] ]

    ai_idx = ai_idx[ torch.randperm(len(ai_idx), generator=g)[:n_ai] ]

    # --------------------------------------------------
    # Select posts
    # --------------------------------------------------
    hu_positive_post = hu_post[hu_idx]
    ai_positive_post = ai_post[ai_idx]

    # --------------------------------------------------
    #  Select corresponding labels
    # --------------------------------------------------
    hu_positive_labels = hu_labels[hu_idx]
    ai_positive_labels = ai_labels[ai_idx]

    # --------------------------------------------------
    #  Combine HU positive + AI positive
    # --------------------------------------------------
    hu_ai_post = torch.cat([hu_positive_post, ai_positive_post],  dim=0 )

    hu_ai_labels = torch.cat([hu_positive_labels, ai_positive_labels],  dim=0 )

    # --------------------------------------------------
    # Combine HU-CT + HU-AI
    # --------------------------------------------------
    final_posts = torch.cat(  [hu_ct_post, hu_ai_post],  dim=0)

    final_labels = torch.cat(   [hu_ct_labels, hu_ai_labels], dim=0  )

    # --------------------------------------------------
    #  Shuffle everything together
    # --------------------------------------------------
    perm = torch.randperm( len(final_posts), generator=torch.Generator().manual_seed(random_seed) )

    final_posts = final_posts[perm]
    final_labels = final_labels[perm]

    return final_posts, final_labels


def HU_AI_Mixed_Results (dataset,  model_name, ratios , max_posts):
   
    df_res = pd.DataFrame(columns=["Dataset", "Ratio", "Model", "Fold", "Accuracy", "Precision", "Recall", "F1", "AUC","MCC", "PR-AUC"])

    for ratio in ratios :
      
       
        hu_post_embeddings, hu_labels =  read_post_embedding("HU", dataset, model_name, max_posts) 
        ai_post_embeddings, ai_labels =  read_post_embedding("AI", dataset, model_name, max_posts) 

        hu_labels = torch.stack(hu_labels)
        ai_labels = torch.stack(ai_labels)

        hu_post_embeddings = torch.stack(hu_post_embeddings)
        ai_post_embeddings = torch.stack(ai_post_embeddings)



        # print(hu_labels)
        post_embedding, labels = create_hu_ai_mix(hu_post_embeddings, hu_labels, ai_post_embeddings, ai_labels,hu_ai_ratio=ratio,random_seed=42)
                            
        data = list(zip(post_embedding, labels))  
        del post_embedding, labels 
              
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(device)

        # Indices of samples
        indices = np.arange(len(data))
            
           
        n_splits = 5           
        labels_for_split = [item[-1] for item in data]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # StratifiedKFold.split needs indexable X and y -- use a dummy range for X
        indices = np.arange(len(data))

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels_for_split), start=1):

            train_val_data = [data[i] for i in train_val_idx]
            test_loader = [data[i] for i in test_idx]

            train_val_labels = [row[-1] for row in train_val_data]

            train_idx, val_idx = train_test_split(
                np.arange(len(train_val_data)),
                test_size=0.125,
                stratify=train_val_labels,
                random_state=42
            )

            train_loader = [train_val_data[i] for i in train_idx]
            eval_loader = [train_val_data[i] for i in val_idx] 


            final_results = run_pipeline(train_loader, eval_loader, test_loader, model_name,  Adapters, "GattedAttention", fold_idx )

            df_res.loc[len(df_res)] = [dataset, ratio, "X-Troll", fold_idx, final_results["Accuracy"], final_results["Precision"], final_results["Recall"],final_results["F1"], final_results["AUC"], final_results["MCC"], final_results["PR-AUC"]]

        # create output_dir if does not exit
        output_dir.mkdir(parents=True, exist_ok=True)

       
        output_file = output_dir.joinpath(f"X-Troll_Benchmark_{dataset}_Ratio_{model_name}_Results_{max_posts}.csv")
       

        df_res.to_csv(output_file, index=False)
        print(f"Results Saved at {output_file}")


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

            file_name = base_path / ".." / "data" / "Benchmark-ML-Data" / f"{IO_type}_{dataset}_{post_length}.csv" 

            all_data.to_csv(file_name, index = False)
            print("Data save at ", file_name)


def HU_AI_Post_Length_Results (dataset,  model_name, post_lengths, IO_type):
   
    df_res = pd.DataFrame(columns=["Dataset", "Post_length", "Model", "Fold", "Accuracy", "Precision", "Recall", "F1", "AUC","MCC", "PR-AUC"])

    for post_length in post_lengths:

       # Create three HU, AI, HU-AI datasets based on post length 

        create_three_types_ml_data([dataset], [post_length], IO_type)          
      
       
        post_embeddings, labels =  read_post_embedding(IO_type, dataset, model_name, post_length) 
      
                            
        data = list(zip(post_embeddings, labels))  
        # del post_embedding, labels 
              
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(device)

        # Indices of samples
        indices = np.arange(len(data))
            
           
        n_splits = 5           
        labels_for_split = [item[-1] for item in data]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # StratifiedKFold.split needs indexable X and y -- use a dummy range for X
        indices = np.arange(len(data))

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels_for_split), start=1):

            train_val_data = [data[i] for i in train_val_idx]
            test_loader = [data[i] for i in test_idx]

            train_val_labels = [row[-1] for row in train_val_data]

            train_idx, val_idx = train_test_split(
                np.arange(len(train_val_data)),
                test_size=0.125,
                stratify=train_val_labels,
                random_state=42
            )

            train_loader = [train_val_data[i] for i in train_idx]
            eval_loader = [train_val_data[i] for i in val_idx] 


            final_results = run_pipeline(train_loader, eval_loader, test_loader, model_name,  Adapters, "GattedAttention", fold_idx )

            df_res.loc[len(df_res)] = [dataset, post_length, "X-Troll", fold_idx, final_results["Accuracy"], final_results["Precision"], final_results["Recall"],final_results["F1"], final_results["AUC"], final_results["MCC"], final_results["PR-AUC"]]

    # create output_dir if does not exit
    output_dir.mkdir(parents=True, exist_ok=True)

    
    output_file = output_dir.joinpath(f"X-Troll_Benchmark_{IO_type}_{dataset}_Post_lenth_{model_name}_Results_{post_lengths}.csv")
    

    df_res.to_csv(output_file, index=False)
    print(f"Results Saved at {output_file}")



if __name__ == '__main__':

    Datasets = ["Catalonia", "Iran_5","Russia_1","Spain"]
    # Main Results 
    for max_posts in [20]: 
        run_X_Troll_Model(Datasets=Datasets, model_name= "llama3.1:8b", Language="En", IO="HU", max_posts = max_posts) 
        run_X_Troll_Model(Datasets=Datasets, model_name="llama3.1:8b" , Language= "En", IO = "AI", max_posts = max_posts)
        # run_X_Troll_Model(Datasets=Datasets, model_name="llama3.1:8b" , Language="En", IO="HU_AI", max_posts = max_posts)

    
    
    # Generalisation Results 

    train_data = ["Russia_1"]
    test_data = ['Iran_5','Catalonia','Spain']
    train_type = ["AI", "HU"]
    test_type = ["HU","AI", "HU_AI"]
    New_Datasets_Results(  train_datasets = train_data , test_datasets = test_data, train_type = train_type, test_type = test_type, model_name = "llama3.1:8b", post_length = 20)
  

    # HU-AI mixed results [0,20,40,50,60,80,100] 

    ratios = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # AI data ratio
    for dataset in["Iran_5"]:
        model = "llama3.1:8b"
        HU_AI_Mixed_Results (dataset, model, ratios, 20)


    # Post length based results 

    post_lengths = [5, 10, 30, 40, 50, 100] # AI data ratio
    for dataset in["Spain"]:
        model = "llama3.1:8b"
        HU_AI_Post_Length_Results (dataset, model, post_lengths, "HU")
        HU_AI_Post_Length_Results (dataset, model, post_lengths, "AI")
              
      

