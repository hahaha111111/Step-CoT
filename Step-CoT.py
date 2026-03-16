# =============================================================================
# Teacher-Student MultiStep VQA with GAT memory
# Complete training + diagnostics script
#
# Requirements:
#   - torch, torchvision, transformers, clip, sklearn, PIL, numpy, tqdm
#   - User must provide a dataloader that yields batches with the following keys:
#       "frontal_images": tensor of shape (B, C, H, W)
#       "labels": tensor of shape (B, num_steps) with class indices
#       "valid_mask": boolean tensor of shape (B, num_steps) indicating valid labels
#       "tokenized_prompts": dict with "input_ids" and "attention_mask",
#                            each shaped (B, num_steps, max_len)
# =============================================================================

import os
import random
import numpy as np
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from torchvision import models
from transformers import AutoModel
import clip

# -----------------------------------------------------------------------------
# User-defined data loading section
# -----------------------------------------------------------------------------
# Please implement your own get_dataloaders() function that returns
# train_loader, val_loader (and optionally test_loader) according to the
# required batch format.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Constants and configuration
# -----------------------------------------------------------------------------
SEED = 42
set_seed_flag = True

RUN_QUICK_DIAGNOSTIC = True
RUN_TEACHER_ONLY = True
RUN_FULL_DISTILL = True          # set True to run train_teacher_student (time-consuming)
USE_WEIGHTED_SAMPLER = False     # enable after verifying sample weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (adjust as needed)
BATCH_SIZE = 32
MAX_LEN = 128
NUM_EPOCHS_TEACHER = 10
NUM_EPOCHS_DISTILL = 40

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if set_seed_flag:
    set_seed(SEED)


def make_collate_fn(step_num_classes, max_length):
    """
    Returns a collate function that:
      - stacks images and labels
      - converts valid_mask to bool
      - clamps invalid labels to 0 but keeps them masked out
      - stacks tokenized prompts into tensors of shape (B, num_steps, max_length)
    """
    def collate_fn(batch):
        batch_size = len(batch)
        frontal = torch.stack([item["frontal_images"] for item in batch])
        lateral = torch.stack([item["lateral_images"] for item in batch])  # if available
        labels = torch.stack([item["labels"] for item in batch])           # [B, S]
        valid_mask = torch.stack([item["valid_mask"] for item in batch]).bool()

        # Sanitize labels: if label <0 or >=num_cls mark as invalid
        for s in range(labels.size(1)):
            num_cls = step_num_classes[s]
            lab_s = labels[:, s]
            invalid_mask = (lab_s < 0) | (lab_s >= num_cls)
            valid_mask[:, s] = valid_mask[:, s] & (~invalid_mask)

        # Clamp labels (but invalids are already masked out)
        labels_clamped = labels.clone()
        for s in range(labels_clamped.size(1)):
            num_cls = step_num_classes[s]
            good = (labels_clamped[:, s] >= 0) & (labels_clamped[:, s] < num_cls)
            labels_clamped[~good, s] = 0

        # Process tokenized prompts
        num_steps = len(batch[0]["tokenized_prompts"])
        input_ids = torch.zeros(batch_size, num_steps, max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, num_steps, max_length, dtype=torch.long)
        for i, item in enumerate(batch):
            for j, step_tokens in enumerate(item["tokenized_prompts"]):
                input_ids[i, j, :] = step_tokens["input_ids"][0]          # assume already padded
                attention_mask[i, j, :] = step_tokens["attention_mask"][0]

        collated = {
            "frontal_images": frontal,
            "lateral_images": lateral,
            "labels": labels_clamped,
            "valid_mask": valid_mask,
            "patient_id": [item.get("patient_id", "") for item in batch],
            "report": [item.get("report", "") for item in batch],
            "vqa_chain": [item.get("vqa_chain", []) for item in batch],
            "tokenized_prompts": {"input_ids": input_ids, "attention_mask": attention_mask}
        }
        return collated
    return collate_fn


def compute_step_class_counts(dataset, step_num_classes):
    """Compute per-step class counts from a dataset (for weighting)."""
    counts = [np.zeros(n, dtype=np.int64) for n in step_num_classes]
    for idx in range(len(dataset)):
        item = dataset[idx]
        labels = item["labels"]
        valid = item["valid_mask"]
        for s in range(len(step_num_classes)):
            is_valid = bool(valid[s]) if isinstance(valid[s], (bool, torch.BoolTensor)) else bool(int(valid[s]))
            if not is_valid:
                continue
            lab = int(labels[s].item())
            if 0 <= lab < step_num_classes[s]:
                counts[s][lab] += 1
    return counts


def make_step_criteria(counts, device):
    """Create weighted CrossEntropyLoss per step based on inverse class frequency."""
    criteria = []
    for i, c in enumerate(counts):
        freq = c.astype(np.float32) + 1e-6
        inv = 1.0 / freq
        weights = inv / inv.sum() * len(inv)
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        criteria.append(nn.CrossEntropyLoss(weight=w))
    return criteria


def make_sample_weights(dataset, step_num_classes, counts):
    """Compute sample weights for weighted random sampler (optional)."""
    inv_freqs = [1.0 / (c + 1e-6) for c in counts]
    weights = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        labels = item["labels"]
        valid = item["valid_mask"]
        wsum = 0.0
        nvalid = 0
        for s in range(len(step_num_classes)):
            is_valid = bool(valid[s]) if isinstance(valid[s], (bool, torch.BoolTensor)) else bool(int(valid[s]))
            if not is_valid:
                continue
            lab = int(labels[s].item())
            if 0 <= lab < step_num_classes[s]:
                wsum += float(inv_freqs[s][lab])
                nvalid += 1
        weights.append(wsum / (nvalid if nvalid > 0 else 1.0))
    return np.array(weights, dtype=np.float32)


# -----------------------------------------------------------------------------
# Model definitions (GATConv, StackedGAT, StepVQAModel, MultiStepVQA, StudentVQA)
# -----------------------------------------------------------------------------
class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, concat=True, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.Tensor(heads, out_dim))
        self.a_dst = nn.Parameter(torch.Tensor(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.zeros_(self.bias)

    def forward(self, h, mask=None):
        # h: [B, N, in_dim]
        B, N, _ = h.shape
        Wh = self.W(h)  # [B, N, heads*out_dim]
        Wh = Wh.view(B, N, self.heads, self.out_dim)  # [B, N, heads, out_dim]
        Wh = Wh.permute(0, 2, 1, 3)  # [B, heads, N, out_dim]

        a_src = self.a_src.view(1, self.heads, 1, self.out_dim)
        a_dst = self.a_dst.view(1, self.heads, 1, self.out_dim)

        el = (Wh * a_src).sum(dim=-1)  # [B, heads, N]
        er = (Wh * a_dst).sum(dim=-1)  # [B, heads, N]

        e = el.unsqueeze(-1) + er.unsqueeze(-2)  # [B, heads, N, N]
        e = self.leaky_relu(e)

        if mask is not None:
            mask_j = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]
            neg_inf = -1e9
            e = e.masked_fill(~mask_j, neg_inf)

        alpha = torch.softmax(e, dim=-1)  # [B, heads, N, N]
        alpha = self.dropout(alpha)

        h_prime = torch.matmul(alpha, Wh)  # [B, heads, N, out_dim]

        if self.concat:
            h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(B, N, self.heads * self.out_dim)
        else:
            h_prime = h_prime.mean(dim=1)

        h_prime = h_prime + self.bias
        return h_prime


class StackedGAT(nn.Module):
    def __init__(self, in_dim, hid_out_per_head=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.projs = nn.ModuleList()
        self.norms = nn.ModuleList()

        cur_in = in_dim
        for i in range(layers):
            self.layers.append(GATConv(cur_in, hid_out_per_head, heads=heads, concat=True, dropout=dropout))
            out_dim = hid_out_per_head * heads
            self.projs.append(nn.Linear(cur_in, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))
            cur_in = out_dim

        self.out_dim = cur_in

    def forward(self, memory, mask=None):
        x = memory
        for i, gat in enumerate(self.layers):
            out = gat(x, mask=mask)
            res = self.projs[i](x)
            x = self.norms[i](out + res)
            x = F.elu(x)
        return x


class StepVQAModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=768, clip_model=None, img_backbone="resnet18", pretrained_backbone=False,
                 dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.clip_model = clip_model

        if clip_model is not None:
            try:
                clip_img_dim = getattr(self.clip_model.visual, "output_dim", None)
                if clip_img_dim is None:
                    clip_img_dim = 512
            except Exception:
                clip_img_dim = 512

            self.img_proj = nn.Sequential(
                nn.Linear(clip_img_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            if img_backbone == "resnet18":
                resnet = models.resnet18(pretrained=pretrained_backbone)
                self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # [B, 512, 1, 1]
                cnn_feat_dim = 512
            else:
                self.cnn = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                cnn_feat_dim = 32

            self.img_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(cnn_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, image, text_memory_context):
        B = image.size(0)
        if self.clip_model is not None:
            img_feat = self.clip_model.encode_image(image)
            if img_feat.dtype != torch.float32:
                img_feat = img_feat.float()
            img_feat = self.img_proj(img_feat)
        else:
            x = self.cnn(image)
            x = x.view(B, -1)
            img_feat = self.img_proj(x)

        fused = torch.cat([img_feat, text_memory_context], dim=1)
        hid = self.fusion_head(fused)
        logits = self.classifier(hid)
        return logits

    def extract_image_feature(self, image):
        with torch.no_grad():
            if self.clip_model is not None:
                feat = self.clip_model.encode_image(image)
                if feat.dtype != torch.float32:
                    feat = feat.float()
                return self.img_proj(feat)
            else:
                B = image.size(0)
                x = self.cnn(image)
                x = x.view(B, -1)
                return self.img_proj(x)


class MultiStepVQA(nn.Module):
    def __init__(self, step_num_classes, text_model_name="bert-base-uncased",
                 hidden_dim=768, gat_heads=4, gat_layers=2, gat_hid_per_head=None, device=None):
        super().__init__()
        self.num_steps = len(step_num_classes)
        self.hidden_dim = hidden_dim
        self.device = device or DEVICE

        # Load CLIP for image encoding (frozen)
        try:
            clip_model, _ = clip.load("ViT-B/32")
        except Exception:
            clip_model = None
        if clip_model is not None:
            clip_model = clip_model.to(self.device)
            for p in clip_model.parameters():
                p.requires_grad = False
        self.clip_model = clip_model

        # Step-specific models
        self.step_models = nn.ModuleList([
            StepVQAModel(num_classes=n_cls, hidden_dim=hidden_dim, clip_model=self.clip_model)
            for n_cls in step_num_classes
        ])

        # Text encoder (frozen BERT)
        self.text_encoder = AutoModel.from_pretrained(text_model_name).to(self.device)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Memory initialization
        self.register_parameter("memory_init", nn.Parameter(torch.zeros(1, hidden_dim)))

        # GAT memory module
        if gat_hid_per_head is None:
            assert hidden_dim % gat_heads == 0, "hidden_dim must be divisible by gat_heads"
            gat_hid_per_head = hidden_dim // gat_heads
        self.memory_gat = StackedGAT(in_dim=hidden_dim, hid_out_per_head=gat_hid_per_head,
                                     heads=gat_heads, layers=gat_layers)
        if self.memory_gat.out_dim != hidden_dim:
            self.mem_proj = nn.Linear(self.memory_gat.out_dim, hidden_dim)
        else:
            self.mem_proj = None

        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.mem_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.pred2mem = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, images, tokenized_prompts):
        device = images.device
        B = images.size(0)

        input_ids = tokenized_prompts["input_ids"]
        attn_mask = tokenized_prompts["attention_mask"]
        S = input_ids.size(1)
        assert S == self.num_steps, f"Expected {self.num_steps} prompts, got {S}"

        # Encode all step prompts with BERT
        text_feats = []
        for i in range(self.num_steps):
            ids_i = input_ids[:, i, :].to(device)
            mask_i = attn_mask[:, i, :].to(device)
            with torch.no_grad():
                text_out = self.text_encoder(input_ids=ids_i, attention_mask=mask_i)
                cls_i = text_out.last_hidden_state[:, 0, :]
            text_feats.append(cls_i)
        text_feats = torch.stack(text_feats, dim=1).to(device)  # [B, S, hidden_dim]

        # Concatenate memory node
        memory_node = self.memory_init.expand(B, -1).to(device).unsqueeze(1)  # [B, 1, hidden_dim]
        nodes = torch.cat([text_feats, memory_node], dim=1)  # [B, S+1, hidden_dim]
        mask = torch.ones(B, nodes.size(1), dtype=torch.bool, device=device)

        outputs = []
        for i in range(self.num_steps):
            # Update nodes with GAT
            gat_out = self.memory_gat(nodes, mask=mask)
            if self.mem_proj is not None:
                gat_out = self.mem_proj(gat_out)
            nodes = gat_out

            step_node = nodes[:, i, :]
            memory_node = nodes[:, -1, :]

            # Combine step node and memory for classification
            step_context = self.fusion_proj(torch.cat([step_node, memory_node], dim=1))
            logits = self.step_models[i](images, step_context)
            outputs.append(logits)

            # Update memory based on prediction
            probs = F.softmax(logits, dim=-1)
            cls_w = self.step_models[i].classifier.weight
            pred_emb = probs @ cls_w                     # weighted average of class embeddings
            pred_mem = self.pred2mem(pred_emb)
            new_memory = self.mem_gru(pred_mem, memory_node)
            nodes = torch.cat([nodes[:, :-1, :], new_memory.unsqueeze(1)], dim=1)

        return outputs


class StudentVQA(nn.Module):
    def __init__(self, step_num_classes, hidden_dim=512, clip_model=None, dropout=0.3):
        super().__init__()
        self.num_steps = len(step_num_classes)
        self.hidden_dim = hidden_dim
        self.clip_model = clip_model

        if self.clip_model is not None:
            clip_img_dim = getattr(self.clip_model.visual, "output_dim", 512)
            self.img_proj = nn.Sequential(
                nn.Linear(clip_img_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            resnet = models.resnet18(pretrained=False)
            self.cnn = nn.Sequential(*list(resnet.children())[:-1])
            self.img_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.step_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_cls) for n_cls in step_num_classes
        ])

    def forward(self, images):
        if self.clip_model is not None:
            img_feat = self.clip_model.encode_image(images)
            if img_feat.dtype != torch.float32:
                img_feat = img_feat.float()
            img_feat = self.img_proj(img_feat)
        else:
            x = self.cnn(images)
            x = x.view(x.size(0), -1)
            img_feat = self.img_proj(x)

        outputs = []
        feat = img_feat
        for i, head in enumerate(self.step_heads):
            logits = head(feat)
            outputs.append(logits)
            # simple residual connection
            feat = feat + torch.tanh(logits @ head.weight)
        return outputs

    def extract_image_feature(self, images):
        with torch.no_grad():
            if self.clip_model is not None:
                feat = self.clip_model.encode_image(images)
                if feat.dtype != torch.float32:
                    feat = feat.float()
                return self.img_proj(feat)
            else:
                B = images.size(0)
                x = self.cnn(images)
                x = x.view(B, -1)
                return self.img_proj(x)


# -----------------------------------------------------------------------------
# Evaluation and diagnostic functions
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device, step_num_classes, name="Model"):
    model.eval()
    all_step_acc = [0.0 for _ in range(len(step_num_classes))]
    all_step_count = [0 for _ in range(len(step_num_classes))]

    for batch in dataloader:
        images = batch["frontal_images"].to(device)
        labels = batch["labels"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        if isinstance(model, MultiStepVQA):
            tokenized_prompts = {"input_ids": batch["tokenized_prompts"]["input_ids"].to(device),
                                 "attention_mask": batch["tokenized_prompts"]["attention_mask"].to(device)}
            outputs = model(images, tokenized_prompts)
        else:
            outputs = model(images)

        for i in range(len(step_num_classes)):
            preds_i = torch.argmax(outputs[i], dim=1)
            labels_i = labels[:, i]
            mask_i = valid_mask[:, i]
            if mask_i.any():
                preds_valid = preds_i[mask_i].cpu().numpy()
                labels_valid = labels_i[mask_i].cpu().numpy()
                acc = accuracy_score(labels_valid, preds_valid)
                all_step_acc[i] += acc * len(labels_valid)
                all_step_count[i] += len(labels_valid)

    step_accs = []
    for i in range(len(step_num_classes)):
        acc = all_step_acc[i] / all_step_count[i] if all_step_count[i] > 0 else 0.0
        step_accs.append(acc)
    mean_acc = sum(step_accs) / len(step_accs)
    step_acc_str = " | ".join([f"Step{i+1}: {acc:.4f}" for i, acc in enumerate(step_accs)])
    print(f"{name} Validation | Mean Acc: {mean_acc:.4f} | {step_acc_str}")
    return step_accs, mean_acc


@torch.no_grad()
def compute_confusion_and_report(model, dataloader, device, step_num_classes, max_batches=None):
    model.eval()
    all_preds = [[] for _ in range(len(step_num_classes))]
    all_labels = [[] for _ in range(len(step_num_classes))]
    nb = 0
    for batch in dataloader:
        images = batch["frontal_images"].to(device)
        labels = batch["labels"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        if isinstance(model, MultiStepVQA):
            tokenized_prompts = {"input_ids": batch["tokenized_prompts"]["input_ids"].to(device),
                                 "attention_mask": batch["tokenized_prompts"]["attention_mask"].to(device)}
            outputs = model(images, tokenized_prompts)
        else:
            outputs = model(images)

        for i in range(len(step_num_classes)):
            mask_i = valid_mask[:, i].bool()
            if not mask_i.any():
                continue
            preds_i = torch.argmax(outputs[i], dim=1)[mask_i].cpu().numpy()
            labs_i = labels[mask_i, i].cpu().numpy()
            all_preds[i].extend(preds_i.tolist())
            all_labels[i].extend(labs_i.tolist())

        nb += 1
        if max_batches is not None and nb >= max_batches:
            break

    for i in range(len(step_num_classes)):
        preds = np.array(all_preds[i]) if len(all_preds[i])>0 else np.array([])
        labs = np.array(all_labels[i]) if len(all_labels[i])>0 else np.array([])
        print(f"\n=== Step {i+1} (n={len(labs)}) ===")
        if len(labs) == 0:
            print(" no valid samples")
            continue
        cm = confusion_matrix(labs, preds, labels=list(range(step_num_classes[i])))
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
        print("\nClassification report (per class):")
        print(classification_report(labs, preds, labels=list(range(step_num_classes[i])), zero_division=0))


@torch.no_grad()
def quick_logits_check(model, dataloader, step_num_classes, device, n_batches=1):
    model.eval()
    import numpy as _np
    for bi, batch in enumerate(dataloader):
        images = batch["frontal_images"].to(device)
        if isinstance(model, MultiStepVQA):
            tokenized_prompts = {"input_ids": batch["tokenized_prompts"]["input_ids"].to(device),
                                 "attention_mask": batch["tokenized_prompts"]["attention_mask"].to(device)}
            outputs = model(images, tokenized_prompts)
        else:
            outputs = model(images)
        for i, out in enumerate(outputs):
            out_np = out.cpu().numpy()
            print(f"Batch {bi} Step{i+1} logits shape: {out_np.shape}, mean: {out_np.mean():.4f}, std: {out_np.std():.4f}")
            preds = out.argmax(dim=1)
            print(f"  preds unique (top 10): {_np.unique(preds.cpu().numpy())[:10]}")
        if bi+1 >= n_batches:
            break


# -----------------------------------------------------------------------------
# Training functions
# -----------------------------------------------------------------------------
def train_teacher_only_with_earlystop(teacher, train_loader, val_loader, device, step_num_classes,
                                      epochs=NUM_EPOCHS_TEACHER, lr=5e-5, step_criteria=None, patience=3,
                                      save_path="best_teacher.pt"):
    opt = optim.AdamW(teacher.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2, verbose=True)
    if step_criteria is None:
        step_criteria = [nn.CrossEntropyLoss() for _ in range(len(step_num_classes))]

    best_mean = -1.0
    no_improve = 0

    for epoch in range(epochs):
        teacher.train()
        total_loss = 0.0
        nsteps = 0
        pbar = tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch["frontal_images"].to(device)
            labels = batch["labels"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            tokenized_prompts = {"input_ids": batch["tokenized_prompts"]["input_ids"].to(device),
                                 "attention_mask": batch["tokenized_prompts"]["attention_mask"].to(device)}
            opt.zero_grad()
            outputs = teacher(images, tokenized_prompts)
            loss = torch.tensor(0.0, device=device)
            for i, out in enumerate(outputs):
                mask_bool = valid_mask[:, i].bool()
                if not mask_bool.any():
                    continue
                out_sel = out[mask_bool]
                lab_sel = labels[mask_bool, i]
                valid_idx = (lab_sel >= 0) & (lab_sel < step_num_classes[i])
                if valid_idx.sum().item() == 0:
                    continue
                loss = loss + step_criteria[i](out_sel[valid_idx], lab_sel[valid_idx])
            if loss.item() == 0.0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            nsteps += 1
            pbar.set_postfix({"loss": f"{(total_loss / max(1, nsteps)):.4f}"})

        print(f"Epoch {epoch+1} train avg loss: {total_loss / max(1,nsteps):.4f}")
        _, mean_acc = evaluate(teacher, val_loader, device, step_num_classes, name="Teacher")
        scheduler.step(mean_acc)

        if mean_acc > best_mean + 1e-6:
            print(f" New best mean acc {mean_acc:.4f} -> saving {save_path}")
            best_mean = mean_acc
            torch.save(teacher.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            print(f" No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print("Early stopping triggered.")
            break
    print("Finished teacher-only training. Best mean acc:", best_mean)


def train_teacher_student(teacher, student, train_loader, val_loader, device,
                          step_num_classes, num_epochs=NUM_EPOCHS_DISTILL,
                          lr_teacher=5e-5, lr_student=1e-4,
                          teacher_pretrain_epochs=2,
                          teacher_supervised=True,
                          teacher_loss_scale=1.0, student_loss_scale=1.0,
                          proj_dim=256, T=2.0, alpha_kd=0.5, alpha_ch=1.0,
                          grad_clip=1.0,
                          step_criteria=None,
                          save_teacher="best_teacher.pt", save_student="best_student.pt"):
    if step_criteria is None:
        step_criteria = [nn.CrossEntropyLoss() for _ in range(len(step_num_classes))]

    teacher_feat_dim = teacher.hidden_dim
    student_feat_dim = student.hidden_dim
    feat_proj_t = nn.Linear(teacher_feat_dim, proj_dim).to(device)
    feat_proj_s = nn.Linear(student_feat_dim, proj_dim).to(device)

    optim_teacher = optim.AdamW(list(teacher.parameters()) + list(feat_proj_t.parameters()), lr=lr_teacher)
    optim_student = optim.AdamW(list(student.parameters()) + list(feat_proj_s.parameters()), lr=lr_student)

    best_val_acc = 0.0

    def H_ch(u, v):
        """Compute centered Hilbert-Schmidt independence criterion (simplified)."""
        m_u = torch.mm(u, u.t())
        m_v = torch.mm(v, v.t())
        n_u = m_u.shape[0]
        n_v = m_v.shape[0]
        c_u = torch.eye(n_u, device=u.device) - torch.ones((n_u, n_u), device=u.device) / float(n_u)
        c_v = torch.eye(n_v, device=v.device) - torch.ones((n_v, n_v), device=v.device) / float(n_v)
        tr_u = torch.mm(torch.mm(c_u, m_u), c_u)
        tr_v = torch.mm(torch.mm(c_v, m_v), c_v)
        return torch.sum(tr_u * tr_v)

    for epoch in range(num_epochs):
        teacher.train()
        student.train()
        feat_proj_t.train()
        feat_proj_s.train()

        total_loss = 0.0
        total_count = 0
        epoch_teacher_ce = 0.0
        epoch_student_ce = 0.0
        epoch_kd = 0.0
        epoch_ch = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        pretrain_mode = (epoch < teacher_pretrain_epochs)

        for batch in pbar:
            images = batch["frontal_images"].to(device)
            labels = batch["labels"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            tokenized_prompts = {"input_ids": batch["tokenized_prompts"]["input_ids"].to(device),
                                 "attention_mask": batch["tokenized_prompts"]["attention_mask"].to(device)}

            # 1) Teacher update
            teacher_loss = torch.tensor(0.0, device=device)
            if pretrain_mode or teacher_supervised:
                optim_teacher.zero_grad()
                teacher_outputs = teacher(images, tokenized_prompts)
                for i in range(len(step_num_classes)):
                    logits_t = teacher_outputs[i]
                    labels_i = labels[:, i]
                    mask_i = valid_mask[:, i].bool()
                    if not mask_i.any():
                        continue
                    logits_t_sel = logits_t[mask_i]
                    labels_sel = labels_i[mask_i]
                    valid_idx = (labels_sel >= 0) & (labels_sel < step_num_classes[i])
                    if valid_idx.sum().item() == 0:
                        continue
                    teacher_loss = teacher_loss + step_criteria[i](logits_t_sel[valid_idx], labels_sel[valid_idx])
                teacher_loss = teacher_loss * teacher_loss_scale
                if teacher_loss.item() != 0.0:
                    teacher_loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(teacher.parameters()) + list(feat_proj_t.parameters()), grad_clip)
                    optim_teacher.step()
            else:
                with torch.no_grad():
                    teacher_outputs = teacher(images, tokenized_prompts)

            # 2) Student update (distill from latest teacher)
            with torch.no_grad():
                teacher_outputs = teacher(images, tokenized_prompts)
            teacher_outputs_det = [t.detach() for t in teacher_outputs]

            optim_student.zero_grad()
            student_outputs = student(images)
            student_loss_total = torch.tensor(0.0, device=device)
            student_ce_sum = 0.0
            kd_sum = 0.0
            ch_sum = 0.0

            s_img_feat_all = student.extract_image_feature(images)

            for i in range(len(step_num_classes)):
                logits_t = teacher_outputs_det[i]
                logits_s = student_outputs[i]
                labels_i = labels[:, i]
                mask_i = valid_mask[:, i].bool()
                if not mask_i.any():
                    continue

                logits_t_sel = logits_t[mask_i]
                logits_s_sel = logits_s[mask_i]
                labels_sel = labels_i[mask_i]
                valid_idx = (labels_sel >= 0) & (labels_sel < step_num_classes[i])
                if valid_idx.sum().item() == 0:
                    continue

                logits_t_sel = logits_t_sel[valid_idx]
                logits_s_sel = logits_s_sel[valid_idx]
                labels_sel = labels_sel[valid_idx]

                # Supervised loss
                student_ce = step_criteria[i](logits_s_sel, labels_sel)
                # Knowledge distillation (KD) loss
                p_t = F.softmax(logits_t_sel.detach() / T, dim=1)
                log_p_s = F.log_softmax(logits_s_sel / T, dim=1)
                kd_loss = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)

                # Contrastive/CH loss on features
                with torch.no_grad():
                    t_img_feat = teacher.step_models[i].extract_image_feature(images)
                s_img_feat = s_img_feat_all
                t_feat_valid = t_img_feat[mask_i][valid_idx]
                s_feat_valid = s_img_feat[mask_i][valid_idx]

                if t_feat_valid.size(0) >= 2:
                    with torch.no_grad():
                        t_proj = feat_proj_t(t_feat_valid)
                    s_proj = feat_proj_s(s_feat_valid)
                    k_t = F.softmax(t_proj / T, dim=1)
                    k_s = F.softmax(s_proj / T, dim=1)
                    h_tt = H_ch(k_t, k_t)
                    h_ss = H_ch(k_s, k_s)
                    h_ts = H_ch(k_t, k_s)
                    denom = torch.sqrt((h_tt + 1e-12) * (h_ss + 1e-12))
                    w_fw = h_ts / (denom + 1e-12)
                    ch_loss = w_fw * F.kl_div(torch.log(k_s + 1e-12), k_t, reduction='batchmean')
                else:
                    ch_loss = torch.tensor(0.0, device=device)

                student_step_loss = student_ce + alpha_kd * kd_loss + alpha_ch * ch_loss
                student_loss_total = student_loss_total + student_step_loss

                student_ce_sum += student_ce.item()
                kd_sum += (alpha_kd * kd_loss).item() if torch.is_tensor(kd_loss) else 0.0
                ch_sum += (alpha_ch * ch_loss).item() if torch.is_tensor(ch_loss) else 0.0

            student_loss_total = student_loss_total * student_loss_scale
            if student_loss_total.item() != 0.0:
                student_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(feat_proj_s.parameters()), grad_clip)
                optim_student.step()

            total_loss += (teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else 0.0) + (
                student_loss_total.item() if isinstance(student_loss_total, torch.Tensor) else 0.0)
            total_count += 1
            epoch_teacher_ce += (teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else 0.0)
            epoch_student_ce += student_ce_sum
            epoch_kd += kd_sum
            epoch_ch += ch_sum

            pbar.set_postfix({
                "t_ce": f"{(teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else 0.0):.4f}",
                "s_ce": f"{student_ce_sum:.4f}", "kd": f"{kd_sum:.4f}", "ch": f"{ch_sum:.4f}"
            })

        avg_train_loss = total_loss / max(1, total_count)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | AvgLoss: {avg_train_loss:.4f} | TeacherCE:{epoch_teacher_ce:.4f} StudentCE:{epoch_student_ce:.4f} KD:{epoch_kd:.4f} CH:{epoch_ch:.4f}")

        step_accs_t, mean_acc_t = evaluate(teacher, val_loader, DEVICE, step_num_classes, name="Teacher")
        step_accs_s, mean_acc_s = evaluate(student, val_loader, DEVICE, step_num_classes, name="Student")

        mean_acc = (mean_acc_t + mean_acc_s) / 2
        if mean_acc > best_val_acc:
            best_val_acc = mean_acc
            torch.save(teacher.state_dict(), save_teacher)
            torch.save(student.state_dict(), save_student)
            torch.save({'proj_t': feat_proj_t.state_dict(), 'proj_s': feat_proj_s.state_dict()},
                       save_teacher + ".featproj.pt")
            print(f"✅ Best models saved (Mean Acc: {best_val_acc:.4f})")

    print("Training complete. Best Mean Acc:", best_val_acc)


# -----------------------------------------------------------------------------
# Main entry point (example usage)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define the number of classes for each reasoning step
    step_num_classes = [4, 4, 6, 7, 5, 7, 9]   # Example: 7 steps with varying class counts

    # -------------------------------------------------------------------------
    # TODO: Replace with your own data loading logic.
    # The code below is a placeholder. You must implement get_dataloaders()
    # that returns train_loader and val_loader (and optionally test_loader)
    # according to the required batch format.
    # -------------------------------------------------------------------------
    # ===== Once you have train_loader, val_loader, and step_criteria, continue: =====

    # Quick diagnostics (optional)
    if RUN_QUICK_DIAGNOSTIC:
        print("==== QUICK DIAGNOSTIC ====")
        # Inspect a few samples (implement if needed)
        pass

    # Initialize models
    print("Initializing models...")
    teacher = MultiStepVQA(step_num_classes, text_model_name="bert-base-uncased",
                           hidden_dim=768, gat_heads=4, gat_layers=2, device=DEVICE).to(DEVICE)

    # Load CLIP for student (frozen)
    clip_model_student, _ = clip.load("ViT-B/32")
    clip_model_student = clip_model_student.to(DEVICE)
    for p in clip_model_student.parameters():
        p.requires_grad = False

    student = StudentVQA(step_num_classes, hidden_dim=512, clip_model=clip_model_student).to(DEVICE)

    # Quick logits check before training (optional)
    if RUN_QUICK_DIAGNOSTIC:
        quick_logits_check(teacher, train_loader, step_num_classes, DEVICE, n_batches=1)

    # Teacher-only quick training (to warm up teacher)
    if RUN_TEACHER_ONLY:
        print("Running teacher-only quick training (earlystop)...")
        train_teacher_only_with_earlystop(teacher, train_loader, val_loader, DEVICE, step_num_classes,
                                          epochs=10, lr=5e-5, step_criteria=step_criteria, patience=3,
                                          save_path="best_teacher_debug.pt")
        print("Teacher-only training done. Running confusion/report (few batches):")
        compute_confusion_and_report(teacher, val_loader, DEVICE, step_num_classes, max_batches=8)

    # Full distillation (optional)
    if RUN_FULL_DISTILL:
        print("Running full teacher-student distillation...")
        train_teacher_student(teacher=teacher,
                              student=student,
                              train_loader=train_loader,
                              val_loader=val_loader,
                              device=DEVICE,
                              step_num_classes=step_num_classes,
                              num_epochs=NUM_EPOCHS_DISTILL,
                              lr_teacher=5e-5,
                              lr_student=1e-4,
                              teacher_pretrain_epochs=2,
                              teacher_supervised=True,
                              step_criteria=step_criteria,
                              save_teacher="best_teacher.pt",
                              save_student="best_student.pt")

    print("Script finished.")