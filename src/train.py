import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from huggingface_hub import HfApi

dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = torch.nn.BCEWithLogitsLoss()


def loss_fn(logits, y):
    return dice_loss(logits, y) + bce_loss(logits, y)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = torch.sigmoid(model(x)) > 0.5
        t = y > 0.5
        tp += (pred & t).sum().item()
        fp += (pred & ~t).sum().item()
        fn += (~pred & t).sum().item()
    eps = 1e-9
    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }


def train(model, train_ds, val_ds, epochs=30, lr=1e-3, batch_size=16,
          ckpt="unet_best.pth", hf_repo=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # num_workers=0: data is already in RAM, and it keeps numpy augmentation seeds correct
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()

        m = evaluate(model, val_dl, device)
        flag = ""
        if m["iou"] > best:
            best = m["iou"]
            torch.save(model.state_dict(), ckpt)
            flag = "  <- best, saved"
        print(f"epoch {ep:02d} | loss {total/len(train_dl):.4f} | "
              f"val IoU {m['iou']:.3f} F1 {m['f1']:.3f} "
              f"P {m['precision']:.3f} R {m['recall']:.3f}{flag}")

    if hf_repo:
        HfApi().upload_file(path_or_fileobj=ckpt, path_in_repo=ckpt, repo_id=hf_repo)
        print("weights uploaded to Hugging Face")
    return best