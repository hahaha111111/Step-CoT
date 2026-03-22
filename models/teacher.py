import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import clip
from .gat import StackedGAT

class StepVQAModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=768, clip_model=None, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.clip_model = clip_model

        if clip_model is not None:
            clip_img_dim = getattr(self.clip_model.visual, "output_dim", 512)
            self.img_proj = nn.Sequential(
                nn.Linear(clip_img_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            from torchvision import models
            resnet = models.resnet18(pretrained=False)
            self.cnn = nn.Sequential(*list(resnet.children())[:-1])
            self.img_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, hidden_dim),
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
            img_feat = self.clip_model.encode_image(image).float()
            img_feat = self.img_proj(img_feat)
        else:
            x = self.cnn(image).view(B, -1)
            img_feat = self.img_proj(x)

        fused = torch.cat([img_feat, text_memory_context], dim=1)
        hid = self.fusion_head(fused)
        return self.classifier(hid)

    def extract_image_feature(self, image):
        with torch.no_grad():
            if self.clip_model is not None:
                feat = self.clip_model.encode_image(image).float()
                return self.img_proj(feat)
            else:
                B = image.size(0)
                x = self.cnn(image).view(B, -1)
                return self.img_proj(x)

class MultiStepVQA(nn.Module):
    def __init__(self, step_num_classes, text_model_name="bert-base-uncased",
                 hidden_dim=768, gat_heads=4, gat_layers=2, gat_hid_per_head=None, device=None):
        super().__init__()
        self.num_steps = len(step_num_classes)
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        clip_model, _ = clip.load("ViT-B/32")
        clip_model = clip_model.to(self.device)
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model

        self.step_models = nn.ModuleList([
            StepVQAModel(n_cls, hidden_dim, self.clip_model) for n_cls in step_num_classes
        ])

        self.text_encoder = AutoModel.from_pretrained(text_model_name).to(self.device)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.memory_init = nn.Parameter(torch.zeros(1, hidden_dim))

        if gat_hid_per_head is None:
            assert hidden_dim % gat_heads == 0
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
        assert S == self.num_steps

        text_feats = []
        for i in range(self.num_steps):
            ids_i = input_ids[:, i, :].to(device)
            mask_i = attn_mask[:, i, :].to(device)
            with torch.no_grad():
                text_out = self.text_encoder(input_ids=ids_i, attention_mask=mask_i)
                cls_i = text_out.last_hidden_state[:, 0, :]
            text_feats.append(cls_i)
        text_feats = torch.stack(text_feats, dim=1).to(device)  # (B, S, H)

        memory_node = self.memory_init.expand(B, -1).to(device).unsqueeze(1)
        nodes = torch.cat([text_feats, memory_node], dim=1)
        mask = torch.ones(B, nodes.size(1), dtype=torch.bool, device=device)

        outputs = []
        for i in range(self.num_steps):
            gat_out = self.memory_gat(nodes, mask=mask)
            if self.mem_proj is not None:
                gat_out = self.mem_proj(gat_out)
            nodes = gat_out

            step_node = nodes[:, i, :]
            memory_node = nodes[:, -1, :]
            step_context = self.fusion_proj(torch.cat([step_node, memory_node], dim=1))
            logits = self.step_models[i](images, step_context)
            outputs.append(logits)

            probs = F.softmax(logits, dim=-1)
            cls_w = self.step_models[i].classifier.weight
            pred_emb = probs @ cls_w
            pred_mem = self.pred2mem(pred_emb)
            new_memory = self.mem_gru(pred_mem, memory_node)
            nodes = torch.cat([nodes[:, :-1, :], new_memory.unsqueeze(1)], dim=1)

        return outputs