import torch
import torch.optim as optim
from tqdm import tqdm
from ..utils.metrics import evaluate

def train_teacher_only(teacher, train_loader, val_loader, device, step_num_classes,
                       epochs, lr, step_criteria, patience=3, save_path="best_teacher.pt"):
    opt = optim.AdamW(teacher.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2, verbose=True)
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
            tokenized = {k: v.to(device) for k, v in batch["tokenized_prompts"].items()}

            opt.zero_grad()
            outputs = teacher(images, tokenized)
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

            if loss.item() != 0.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                nsteps += 1
                pbar.set_postfix({"loss": f"{total_loss/max(1,nsteps):.4f}"})

        print(f"Epoch {epoch+1} train avg loss: {total_loss/max(1,nsteps):.4f}")
        _, mean_acc = evaluate(teacher, val_loader, device, step_num_classes, name="Teacher")
        scheduler.step(mean_acc)

        if mean_acc > best_mean + 1e-6:
            best_mean = mean_acc
            torch.save(teacher.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered.")
            break
    print("Teacher-only training finished. Best mean acc:", best_mean)