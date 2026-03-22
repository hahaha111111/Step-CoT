import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

@torch.no_grad()
def evaluate(model, dataloader, device, step_num_classes, name="Model"):
    model.eval()
    all_step_acc = [0.0 for _ in range(len(step_num_classes))]
    all_step_count = [0 for _ in range(len(step_num_classes))]

    for batch in dataloader:
        images = batch["frontal_images"].to(device)
        labels = batch["labels"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        if hasattr(model, 'text_encoder'):  # teacher has text_encoder
            tokenized = {k: v.to(device) for k, v in batch["tokenized_prompts"].items()}
            outputs = model(images, tokenized)
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

        if hasattr(model, 'text_encoder'):
            tokenized = {k: v.to(device) for k, v in batch["tokenized_prompts"].items()}
            outputs = model(images, tokenized)
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
        if max_batches and nb >= max_batches:
            break

    for i in range(len(step_num_classes)):
        preds = np.array(all_preds[i])
        labs = np.array(all_labels[i])
        print(f"\n=== Step {i+1} (n={len(labs)}) ===")
        if len(labs) == 0:
            print(" no valid samples")
            continue
        cm = confusion_matrix(labs, preds, labels=list(range(step_num_classes[i])))
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
        print("\nClassification report:")
        print(classification_report(labs, preds, labels=list(range(step_num_classes[i])), zero_division=0))