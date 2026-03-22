# Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering

This repository implements the **Step-CoT** framework from the paper:

> **Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering**  
> *Lin Fan, Yafei Ou, Zhipeng Deng, Pengyu Dai, Hou Chongxian, Jiale Yan, Yaqian Li, Kaiwen Long, Xun Gong, Masayuki Ikebe, Yefeng Zheng*  
> arXiv:2603.13878 [cs.CV]  
> [Paper](https://arxiv.org/pdf/2603.13878) | [Dataset](https://huggingface.co/datasets/fl-15o/Step-CoT) | [Benchmark](https://github.com/hahaha111111/Step-CoT)

## Overview

**Step-CoT** introduces a structured, multi-step reasoning dataset for Medical Visual Question Answering (Med-VQA). It consists of **10K+ real clinical cases** and **70K VQA pairs**, each annotated with a seven-step diagnostic workflow (abnormality detection → distribution → pattern → location → morphology → secondary effects → diagnosis). The reasoning steps are aligned with expert radiological practice and include explicit intermediate supervision.

To effectively learn from this dataset, we propose a **verified framework**: teacher-student framework with:
- A **teacher model** that uses a Graph Attention Network (GAT) with a learnable memory node to propagate information across reasoning steps.
- A **student model** (lightweight) that learns via knowledge distillation, combining:
  - Hard cross-entropy loss (supervised)
  - Soft knowledge distillation (KD) loss
  - Channel/relation alignment (CH) loss inspired by HSIC

![Overview of the Step-CoT dataset. (A) Conventional Med-VQA approaches, where models take an image and a question as input, perform multi-modal feature fusion and output a diagnostic answer. Although leveraging multi-modal knowledge, this paradigm lacks interpretability and often yields limited diagnostic accuracy. (B) Enhances interpretability by integrating large language models with CoT reasoning to generate intermediate explanations; however, such reasoning is often unreliable. (C) Our proposed Step-CoT dataset and training framework, which introduces explicit intermediate supervision. By guiding the model to learn structured clinical reasoning steps, Step-CoT not only improves interpretability through trustworthy intermediate reasoning but also enhances diagnostic accuracy.](overview.png)


## Repository Structure

```
step_cot_vqa/
├── config/            # Configuration files
├── data/              # Dataset loader and collate function
├── models/            # Teacher (MultiStepVQA) and Student (StudentVQA)
├── training/          # Teacher-only and distillation training loops
├── utils/             # Metrics, loss functions, data utilities
├── scripts/           # Main training script
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/step_cot_vqa.git
   cd step_cot_vqa
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Download the **Step-CoT dataset** from [HuggingFace](https://huggingface.co/datasets/f-15a/Step-CoT). The dataset includes:
- `train.xlsx`, `val.xlsx`, `test.xlsx`, `data.json` (metadata and VQA chains)
- `images/` and `images_2/`  folder with chest X-ray images

Modify the `config/default.yaml` to point to your data root:
```yaml
data_root: "/path/to/Step-CoT"
```
Then, implement the data loader according to the provided framework. The main data loading structure is already given in data/dataset.py; you need to complete the get_dataloaders function to read the dataset files (e.g., data.json) and return the appropriate PyTorch DataLoader instances. 

If you want to use your own dataset, implement the `get_dataloaders` function in `data/dataset.py` following the expected format.

## Running Experiments

### 1. Teacher-Only Training
```bash
python scripts/run_training.py --mode teacher_only
```
This trains the teacher model with cross-entropy loss only. The best checkpoint will be saved to `checkpoints/teacher.pt`.

### 2. Teacher-Student Distillation
```bash
python scripts/run_training.py --mode distill
```
This performs the full distillation:
- The teacher is optionally pre-trained for a few epochs (controlled by `teacher_pretrain_epochs`).
- The student is trained with the combined loss.
- Both models are evaluated on the validation set.

### 3. Evaluation
To evaluate a trained model:
```python
from utils.metrics import evaluate, compute_confusion_and_report
evaluate(model, val_loader, device, step_num_classes, name="Model")
compute_confusion_and_report(model, test_loader, device, step_num_classes)
```

## Citation

If you use this code or the Step-CoT dataset, please cite:

```bibtex
@article{fan2026step,
  title={Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering},
  author={Fan, Lin and Ou, Yafei and Deng, Zhipeng and Dai, Pengyu and Chongxian, Hou and Yan, Jiale and Li, Yaqian and Long, Kaiwen and Gong, Xun and Ikebe, Masayuki and others},
  journal={arXiv preprint arXiv:2603.13878},
  year={2026}
}
```