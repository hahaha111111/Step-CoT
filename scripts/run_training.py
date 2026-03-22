import yaml
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from data import StepCoTDataset, make_collate_fn
from models.teacher import MultiStepVQA
from models.student import StudentVQA
from training.teacher_only import train_teacher_only
from training.distillation import train_teacher_student
from utils.common import set_seed, get_device
from utils.loss import make_step_criteria
from utils.metrics import evaluate, compute_confusion_and_report

def main():
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = get_device()
    print("Device:", device)

    # Step classes
    step_num_classes = config['step_num_classes']
    S = len(step_num_classes)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['teacher']['text_model_name'])

    # TODO: Replace with actual data loading
    # train_dataset = StepCoTDataset('path/to/train.json', config['data_root'], tokenizer, config['max_len'])
    # val_dataset = StepCoTDataset('path/to/val.json', config['data_root'], tokenizer, config['max_len'])
    # test_dataset = StepCoTDataset('path/to/test.json', config['data_root'], tokenizer, config['max_len'])
    # collate_fn = make_collate_fn(step_num_classes, config['max_len'], tokenizer)
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    # Dummy loaders for demonstration (replace with real ones)
    train_loader = val_loader = test_loader = None

    # Build models
    teacher = MultiStepVQA(
        step_num_classes,
        text_model_name=config['teacher']['text_model_name'],
        hidden_dim=config['teacher']['hidden_dim'],
        gat_heads=config['teacher']['gat_heads'],
        gat_layers=config['teacher']['gat_layers'],
        gat_hid_per_head=config['teacher']['gat_hid_per_head'],
        device=device
    ).to(device)

    import clip
    clip_model, _ = clip.load("ViT-B/32")
    clip_model = clip_model.to(device)
    for p in clip_model.parameters():
        p.requires_grad = False

    student = StudentVQA(
        step_num_classes,
        hidden_dim=config['student']['hidden_dim'],
        clip_model=clip_model,
        dropout=config['student']['dropout']
    ).to(device)

    # Prepare step criteria (if class weights are needed, compute from dataset)
    # For now, use plain CrossEntropyLoss
    step_criteria = [torch.nn.CrossEntropyLoss() for _ in range(S)]

    # Optional: quick logits check (if data is available)
    # from utils.metrics import quick_logits_check
    # quick_logits_check(teacher, train_loader, step_num_classes, device, n_batches=1)

    # Teacher-only training
    if config.get('run_teacher_only', True):
        print("Starting teacher-only training...")
        train_teacher_only(
            teacher, train_loader, val_loader, device,
            step_num_classes, epochs=config['teacher']['epochs'],
            lr=config['teacher']['lr'], step_criteria=step_criteria,
            patience=config['teacher']['patience'],
            save_path=config['save_dir'] + "/teacher.pt"
        )

        # Evaluate teacher
        evaluate(teacher, val_loader, device, step_num_classes, name="Teacher")
        compute_confusion_and_report(teacher, val_loader, device, step_num_classes, max_batches=8)

    # Full distillation
    if config.get('run_distill', True):
        print("Starting teacher-student distillation...")
        train_teacher_student(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            step_num_classes=step_num_classes,
            num_epochs=config['distill']['num_epochs'],
            lr_teacher=config['teacher']['lr'],
            lr_student=config['student']['lr'],
            teacher_pretrain_epochs=config['distill']['teacher_pretrain_epochs'],
            teacher_supervised=config['distill']['teacher_supervised'],
            teacher_loss_scale=config['distill']['teacher_loss_scale'],
            student_loss_scale=config['distill']['student_loss_scale'],
            proj_dim=config['distill']['proj_dim'],
            T=config['distill']['temperature'],
            alpha_kd=config['distill']['alpha_kd'],
            alpha_ch=config['distill']['alpha_ch'],
            grad_clip=config['distill']['grad_clip'],
            step_criteria=step_criteria,
            save_teacher=config['save_dir'] + "/teacher_distill.pt",
            save_student=config['save_dir'] + "/student_distill.pt"
        )

        # Final evaluation
        evaluate(teacher, test_loader, device, step_num_classes, name="Teacher (final)")
        evaluate(student, test_loader, device, step_num_classes, name="Student (final)")

if __name__ == "__main__":
    main()