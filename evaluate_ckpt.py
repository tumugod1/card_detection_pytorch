# evaluate_ckpt.py
import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
import timm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

def build_eval_tfms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = timm.create_model('efficientnet_b0', pretrained=False)
        base.classifier = nn.Identity()
        self.features = base
        self.classifier = nn.Linear(1280, num_classes)
    def forward(self, x):
        return self.classifier(self.features(x))

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt['classes']
    img_size = ckpt.get('img_size', 224)
    model = SimpleCardClassifier(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, classes, img_size

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='best_model.pth')
    ap.add_argument('--test-dir', default='./card-image-datasetclassification/test')
    ap.add_argument('--save-cm', default='confusion_matrix.png')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, train_classes, img_size = load_model(args.ckpt, device)
    eval_tfms = build_eval_tfms(img_size)

    ds = ImageFolder(args.test_dir, transform=eval_tfms)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'))

    # Dataset sınıf indexlerini, eğitimdeki sınıf sırası ile hizala
    name_to_train_idx = {name: i for i, name in enumerate(train_classes)}
    ds_class_names = ds.classes
    ds_to_train = {ds_idx: name_to_train_idx[name] for ds_idx, name in enumerate(ds_class_names)}

    all_true, all_pred = [], []
    total, corr, running_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        # Gerçek etiketleri eğitimdeki sınıf sırasına çevir
        labels_train_order = torch.tensor([ds_to_train[l.item()] for l in labels], dtype=torch.long, device=device)

        logits = model(images)
        loss = criterion(logits, labels_train_order)
        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(logits, 1)
        corr += (preds == labels_train_order).sum().item()
        total += images.size(0)

        all_true.extend(labels_train_order.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())

    test_loss = running_loss / max(1, total)
    test_acc = 100.0 * corr / max(1, total)

    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%\n")
    print("Classification report:\n")
    print(classification_report(all_true, all_pred, target_names=train_classes, digits=4))

    # Confusion Matrix (normalize 'true' ekseni)
    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(train_classes))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig_w = min(18, max(8, len(train_classes) * 0.35))
    fig_h = fig_w
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (normalized by true)')
    plt.colorbar()
    ticks = np.arange(len(train_classes))
    plt.xticks(ticks, train_classes, rotation=90)
    plt.yticks(ticks, train_classes)
    plt.tight_layout()
    plt.savefig(args.save_cm, dpi=200)
    print(f"\nConfusion matrix saved to: {args.save_cm}")

if __name__ == '__main__':
    main()
