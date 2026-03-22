import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from ..utils.metrics import evaluate
from ..utils.loss import H_ch

def train_teacher_student(teacher, student, train_loader, val_loader, device,
                          step_num_classes, num_epochs,
                          lr_teacher, lr_student,
                          teacher_pretrain_epochs=2,
                          teacher_supervised=True,
                          teacher_loss_scale=1.0, student_loss_scale=1.0,
                          proj_dim=256, T=2.0, alpha_kd=0.5, alpha_ch=1.0,
                          grad_clip=1.0,
                          step_criteria=None,
                          save_teacher="best_teacher.pt", save_student="best_student.pt"):

    if step_criteria is None:
        step_criteria = [torch.nn.CrossEntropyLoss() for _ in step_num_classes]

    teacher_feat_dim = teacher.hidden_dim
    student_feat_dim = student.hidden_dim
    feat_proj_t = torch.nn.Linear(teacher_feat_dim, proj_dim).to(device)
    feat_proj_s = torch.nn.Linear(student_feat_dim, proj_dim).to(device)

    optim_teacher = optim.AdamW(list(teacher.parameters()) + list(feat_proj_t.parameters()), lr=lr_teacher)
    optim_student = optim.AdamW(list(student.parameters()) + list(feat_proj_s.parameters()), lr=lr_student)

    best_val_acc = 0.0

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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        pretrain_mode = (epoch < teacher_pretrain_epochs)

        for batch in pbar:
            images = batch["frontal_images"].to(device)
            labels = batch["labels"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            tokenized = {k: v.to(device) for k, v in batch["tokenized_prompts"].items()}

            # Teacher update
            teacher_loss = torch.tensor(0.0, device=device)
            if pretrain_mode or teacher_supervised:
                optim_teacher.zero_grad()
                teacher_outputs = teacher(images, tokenized)
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
                    teacher_outputs = teacher(images, tokenized)

            # Student update (distill from latest teacher)
            with torch.no_grad():
                teacher_outputs = teacher(images, tokenized)
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

                student_ce = step_criteria[i](logits_s_sel, labels_sel)
                p_t = F.softmax(logits_t_sel / T, dim=1)
                log_p_s = F.log_softmax(logits_s_sel / T, dim=1)
                kd_loss = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)

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
                kd_sum += (alpha_kd * kd_loss).item()
                ch_sum += (alpha_ch * ch_loss).item()

            student_loss_total = student_loss_total * student_loss_scale
            if student_loss_total.item() != 0.0:
                student_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(feat_proj_s.parameters()), grad_clip)
                optim_student.step()

            total_loss += (teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else 0.0) + student_loss_total.item()
            total_count += 1
            epoch_teacher_ce += teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else 0.0
            epoch_student_ce += student_ce_sum
            epoch_kd += kd_sum
            epoch_ch += ch_sum

            pbar.set_postfix({
                "t_ce": f"{teacher_loss.item() if isinstance(teacher_loss, torch.Tensor) else 0.0:.4f}",
                "s_ce": f"{student_ce_sum:.4f}",
                "kd": f"{kd_sum:.4f}",
                "ch": f"{ch_sum:.4f}"
            })

        avg_train_loss = total_loss / max(1, total_count)
        print(f"Epoch {epoch+1}/{num_epochs} | AvgLoss: {avg_train_loss:.4f} | "
              f"TeacherCE:{epoch_teacher_ce:.4f} StudentCE:{epoch_student_ce:.4f} "
              f"KD:{epoch_kd:.4f} CH:{epoch_ch:.4f}")

        _, mean_acc_t = evaluate(teacher, val_loader, device, step_num_classes, name="Teacher")
        _, mean_acc_s = evaluate(student, val_loader, device, step_num_classes, name="Student")

        mean_acc = (mean_acc_t + mean_acc_s) / 2
        if mean_acc > best_val_acc:
            best_val_acc = mean_acc
            torch.save(teacher.state_dict(), save_teacher)
            torch.save(student.state_dict(), save_student)
            torch.save({'proj_t': feat_proj_t.state_dict(), 'proj_s': feat_proj_s.state_dict()},
                       save_teacher + ".featproj.pt")
            print(f"Best models saved (Mean Acc: {best_val_acc:.4f})")

    print("Training complete. Best Mean Acc:", best_val_acc)