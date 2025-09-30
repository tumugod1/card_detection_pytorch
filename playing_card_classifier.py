
import os
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
import timm
from contextlib import nullcontext


# 0) Konfig / Sabitler

SEED = 42
IMG_SIZE = 224                     # EfficientNet-B0 için ideal giriş
BATCH_SIZE = 32
STAGE1_EPOCHS = 8                  # yalnızca classifier eğit
STAGE2_EPOCHS = 30                 # tüm ağı fine-tune
PATIENCE_S1 = 7                    # early stopping sabrı (stage 1)
PATIENCE_S2 = 10                   # early stopping sabrı (stage 2)
MIN_DELTA = 1e-4                   # early stopping iyileşme eşiği (val loss)
WEIGHT_DECAY = 1e-4
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4
LABEL_SMOOTH = 0.10
GRAD_CLIP_NORM = 1.0               # patlamayı engelle (None yapabilirsin)
BEST_PATH = "./best_model.pth"


# 1) Determinizm / seed

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Performans için deterministic=False; hız için benchmark=True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# 2) Dataset sarmalayıcı

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    @property
    def classes(self):
        return self.data.classes


# 3) Transformlar

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


# 4) Veri yolları (göreli)

train_dir = './card-image-datasetclassification/train'
val_dir   = './card-image-datasetclassification/valid'
test_dir  = './card-image-datasetclassification/test'


# 5) Model

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = timm.create_model('efficientnet_b0', pretrained=True)
        base.classifier = nn.Identity()      
        self.features = base
        self.classifier = nn.Linear(1280, num_classes)
    def forward(self, x):
        x = self.features(x)                 # [B, 1280]
        x = self.classifier(x)               # [B, num_classes]
        return x


# 6) Yardımcılar

def make_loader(ds, batch_size=32, shuffle=False, workers=0, pin=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=pin,
                      persistent_workers=(workers > 0))

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0
    corr = 0
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        corr += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / max(1, total)
    acc = 100.0 * corr / max(1, total)
    return avg_loss, acc

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, autocast_ctx, grad_clip_norm=None):
    model.train()
    running_loss = 0.0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / max(1, total)

def make_autocast_and_scaler(device):
    if device.type == 'cuda':
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    else:
        autocast_ctx = nullcontext()
        scaler = None
    return autocast_ctx, scaler

def maybe_workers():
    # Windows'ta 0; Linux/WSL'de mantıklı bir değer
    if os.name == 'nt':
        return 0
    try:
        return max(1, min(4, (os.cpu_count() or 2) - 1))
    except Exception:
        return 0


# 7) Ana akış

def main():
    set_seed(SEED)

    # Dataset & loaders
    train_dataset = PlayingCardDataset(train_dir, transform=train_tfms)
    val_dataset   = PlayingCardDataset(val_dir,   transform=eval_tfms)
    test_dataset  = PlayingCardDataset(test_dir,  transform=eval_tfms)

    classes = train_dataset.classes
    num_classes = len(classes)
    if num_classes < 2:
        raise RuntimeError(f"Beklenmeyen sınıf sayısı: {num_classes}. Klasör yapısını (train/valid/test) kontrol et.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = (device.type == 'cuda')
    workers = maybe_workers()

    train_loader = make_loader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  workers=workers, pin=pin)
    val_loader   = make_loader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, workers=workers, pin=pin)
    test_loader  = make_loader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, workers=workers, pin=pin)

    model = SimpleCardClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # AMP / Scaler
    autocast_ctx, scaler = make_autocast_and_scaler(device)


    # Stage 1: Sadece classifier eğit

    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(model.classifier.parameters(), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                     patience=2, verbose=True, min_lr=1e-6)

    best_val_loss = float('inf')
    best_val_acc  = 0.0
    no_improve = 0

    print(f"\n==> Stage 1: classifier (epochs={STAGE1_EPOCHS})")
    for epoch in range(1, STAGE1_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, autocast_ctx, GRAD_CLIP_NORM)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"[Stage1][{epoch}/{STAGE1_EPOCHS}] Train {train_loss:.4f} | Val {val_loss:.4f} | ValAcc {val_acc:.2f}%")

        improved = False
        # Öncelik val_loss; tie-break val_acc
        if val_loss < (best_val_loss - MIN_DELTA) or (abs(val_loss - best_val_loss) <= MIN_DELTA and val_acc > best_val_acc):
            best_val_loss = val_loss
            best_val_acc = val_acc
            improved = True

        if improved:
            no_improve = 0
            torch.save({'state_dict': model.state_dict(),
                        'classes': classes,
                        'img_size': IMG_SIZE}, BEST_PATH)
        else:
            no_improve += 1
            if no_improve >= PATIENCE_S1:
                print("Early stopping (Stage 1).")
                break


    # Stage 2: Tüm ağı aç, küçük LR ile fine-tune

    for p in model.features.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LR_STAGE2, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                     patience=3, verbose=True, min_lr=1e-6)

    no_improve = 0
    print(f"\n==> Stage 2: full fine-tune (epochs={STAGE2_EPOCHS})")
    for epoch in range(1, STAGE2_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, autocast_ctx, GRAD_CLIP_NORM)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"[Stage2][{epoch}/{STAGE2_EPOCHS}] Train {train_loss:.4f} | Val {val_loss:.4f} | ValAcc {val_acc:.2f}%")

        improved = False
        if val_loss < (best_val_loss - MIN_DELTA) or (abs(val_loss - best_val_loss) <= MIN_DELTA and val_acc > best_val_acc):
            best_val_loss = val_loss
            best_val_acc = val_acc
            improved = True

        if improved:
            no_improve = 0
            torch.save({'state_dict': model.state_dict(),
                        'classes': classes,
                        'img_size': IMG_SIZE}, BEST_PATH)
        else:
            no_improve += 1
            if no_improve >= PATIENCE_S2:
                print("Early stopping (Stage 2).")
                break


    # En iyi modelle test et

    if os.path.exists(BEST_PATH):
        ckpt = torch.load(BEST_PATH, map_location=device)
        model.load_state_dict(ckpt['state_dict'])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    print(f"Best (Val) → Loss: {best_val_loss:.4f} | Acc: {best_val_acc:.2f}%  (saved to {BEST_PATH})")


if __name__ == "__main__":
    main()
