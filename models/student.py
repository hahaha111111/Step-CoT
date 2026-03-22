import torch
import torch.nn as nn
from torchvision import models
import clip

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
            img_feat = self.clip_model.encode_image(images).float()
            img_feat = self.img_proj(img_feat)
        else:
            B = images.size(0)
            x = self.cnn(images).view(B, -1)
            img_feat = self.img_proj(x)

        outputs = []
        feat = img_feat
        for i, head in enumerate(self.step_heads):
            logits = head(feat)
            outputs.append(logits)
            feat = feat + torch.tanh(logits @ head.weight)
        return outputs

    def extract_image_feature(self, images):
        with torch.no_grad():
            if self.clip_model is not None:
                feat = self.clip_model.encode_image(images).float()
                return self.img_proj(feat)
            else:
                B = images.size(0)
                x = self.cnn(images).view(B, -1)
                return self.img_proj(x)