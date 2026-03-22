import numpy as np
import torch

def compute_step_class_counts(dataset, step_num_classes):
    """Compute per-step class counts from a dataset (for weighting)."""
    counts = [np.zeros(n, dtype=np.int64) for n in step_num_classes]
    for idx in range(len(dataset)):
        item = dataset[idx]
        labels = item["labels"]
        valid = item["valid_mask"]
        for s in range(len(step_num_classes)):
            if valid[s]:
                lab = int(labels[s].item())
                if 0 <= lab < step_num_classes[s]:
                    counts[s][lab] += 1
    return counts

def make_sample_weights(dataset, step_num_classes, counts):
    """Compute sample weights for WeightedRandomSampler."""
    inv_freqs = [1.0 / (c + 1e-6) for c in counts]
    weights = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        labels = item["labels"]
        valid = item["valid_mask"]
        wsum = 0.0
        nvalid = 0
        for s in range(len(step_num_classes)):
            if valid[s]:
                lab = int(labels[s].item())
                if 0 <= lab < step_num_classes[s]:
                    wsum += float(inv_freqs[s][lab])
                    nvalid += 1
        weights.append(wsum / (nvalid if nvalid > 0 else 1.0))
    return np.array(weights, dtype=np.float32)