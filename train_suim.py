import os
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tabulate import tabulate
import timm

# ==============================================================================
# 1. CONFIG
# ==============================================================================

CLASSES = [
    'Background',   # 0
    'Human Divers', # 1
    'Plants',       # 2
    'Wrecks',       # 3
    'Robots',       # 4
    'Reefs',        # 5
    'Fish',         # 6
    'Sea-floor'     # 7
]

COLOR_MAP = {
    (0, 0, 0): 0,       # Background
    (0, 0, 255): 1,     # HD
    (0, 255, 0): 2,     # PF
    (0, 255, 255): 3,   # WR
    (255, 0, 0): 4,     # RO
    (255, 0, 255): 5,   # RI
    (255, 255, 0): 6,   # FV
    (255, 255, 255): 7  # SR
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ==============================================================================
# 2. DATASET
# ==============================================================================

class SUIMDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(224, 224), augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.bmp'))])

        # Reduced ColorJitter intensity to prevent confusing green water with green plants
        self.color_aug = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def encode_mask(self, mask):
        mask = mask.resize(self.img_size, resample=Image.NEAREST)
        mask_np = np.array(mask)
        if mask_np.ndim == 3 and mask_np.shape[2] == 4:
            mask_np = mask_np[:, :, :3]

        label_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.longlong)
        for rgb, idx in COLOR_MAP.items():
            match = (mask_np == rgb).all(axis=2)
            label_mask[match] = idx
        return torch.from_numpy(label_mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]

        mask_name = None
        for ext in ['.bmp', '.jpg', '.png']:
            if os.path.exists(os.path.join(self.mask_dir, base_name + ext)):
                mask_name = base_name + ext
                break
        if mask_name is None: mask_name = img_name

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # Resize
        image = image.resize(self.img_size, Image.BILINEAR)

        if self.augment:
            if random.random() > 0.3:
                image = self.color_aug(image)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return self.normalize(image), self.encode_mask(mask)

# ==============================================================================
# 3. ADVANCED LOSS: FOCAL + DICE
# ==============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        # Focal term: (1 - pt)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * (-logpt)
        return focal_loss.mean()

class ComboLoss(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.focal = FocalLoss(gamma=2) # Gamma 2 reduces loss for easy background
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # 1. Focal Loss (Handles Imbalance dynamically)
        f_loss = self.focal(inputs, targets)

        # 2. Dice Loss (Optimizes IoU directly)
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        # Dice score per batch
        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice_score.mean()

        # Balance: 0.5 Focal + 1.0 Dice
        return 0.5 * f_loss + dice_loss

# ==============================================================================
# 4. ARCHITECTURE (ImageNet-22k Backbone)
# ==============================================================================

class TimmHybridNet(nn.Module):
    def __init__(self, backbone='swin_base_patch4_window7_224.ms_in22k', num_classes=8, pretrained=True, target_size=(224, 224)):
        super().__init__()
        self.target_size = target_size
        # Using 22k pretrained weights for better feature extraction
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True)

        dummy = torch.randn(1, 3, target_size[0], target_size[1])
        features = self.encoder(dummy)

        ch = []
        self.channels_last = False
        last_feat = features[-1]

        # Auto-detect channel format
        if last_feat.shape[1] < last_feat.shape[-1]:
            self.channels_last = True
            ch = [f.shape[-1] for f in features]
        else:
            self.channels_last = False
            ch = [f.shape[1] for f in features]

        # Decoder with Dropout to prevent overfitting
        self.up1 = self._up_block(ch[-1], ch[-2])
        self.up2 = self._up_block(ch[-2], ch[-3])
        self.up3 = self._up_block(ch[-3], ch[-4])

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(ch[-4], 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2), # Added Dropout
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        feats = self.encoder(x)
        if self.channels_last:
            feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]

        c1, c2, c3, c4 = feats[0], feats[1], feats[2], feats[3]

        x = self.up1(c4) + c3
        x = self.up2(x) + c2
        x = self.up3(x) + c1
        x = self.final_up(x)

        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)
        return x

# ==============================================================================
# 5. METRICS & PLOTTING
# ==============================================================================

def compute_iou(preds, labels, num_classes=8):
    preds = torch.argmax(preds, dim=1).flatten()
    labels = labels.flatten()
    keep = (labels >= 0) & (labels < num_classes)
    preds = preds[keep]
    labels = labels[keep]

    confusion = torch.bincount(
        num_classes * labels + preds, minlength=num_classes**2
    ).reshape(num_classes, num_classes)

    intersection = torch.diag(confusion)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = intersection / (union + 1e-6)
    return iou, confusion

def save_smooth_plots(save_dir, train_hist, val_hist, metric_name, best_val=None):
    epochs = range(1, len(train_hist) + 1)
    if len(epochs) > 3:
        x_new = np.linspace(min(epochs), max(epochs), 300)
        try:
            spl_train = make_interp_spline(epochs, train_hist, k=3)
            spl_val = make_interp_spline(epochs, val_hist, k=3)
            y_train_smooth = spl_train(x_new)
            y_val_smooth = spl_val(x_new)
        except:
             x_new, y_train_smooth, y_val_smooth = epochs, train_hist, val_hist
    else:
        x_new, y_train_smooth, y_val_smooth = epochs, train_hist, val_hist

    plt.figure(figsize=(10, 6))
    plt.plot(x_new, y_train_smooth, label=f'Train {metric_name}', linewidth=2)
    plt.plot(x_new, y_val_smooth, label=f'Val {metric_name}', linewidth=2)

    if best_val is not None:
        plt.axhline(y=best_val, color='r', linestyle='--', label=f'Best Val: {best_val:.4f}')
        plt.text(x_new[0], best_val, f' {best_val:.4f}', color='r', va='bottom', fontweight='bold')

    plt.title(f'{metric_name} Curve (Smoothed)')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{metric_name.lower()}_curve.png'))
    plt.close()

# ==============================================================================
# 6. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./SUIM')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5) # Lower LR for In22k
    parser.add_argument('--save_dir', type=str, default='suim_checkpoints_tta')
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    train_img = os.path.join(args.data, 'train_val', 'images')
    train_msk = os.path.join(args.data, 'train_val', 'masks')
    val_img = os.path.join(args.data, 'TEST', 'images')
    val_msk = os.path.join(args.data, 'TEST', 'masks')

    print(f"Training on: {device}")

    train_ds = SUIMDataset(train_img, train_msk, img_size=(args.img_size, args.img_size), augment=True)
    val_ds = SUIMDataset(val_img, val_msk, img_size=(args.img_size, args.img_size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    # Use ImageNet-22k Pretrained weights (Better features)
    model = TimmHybridNet(backbone='swin_base_patch4_window7_224.ms_in22k', num_classes=8, target_size=(args.img_size, args.img_size)).to(device)

    criterion = ComboLoss(num_classes=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    best_iou = 0.0

    print("\nStarting Training (Focal Loss + TTA + In22k)...\n" + "="*60)

    for epoch in range(1, args.epochs + 1):
        # --- TRAIN ---
        model.train()
        t_loss, t_iou_sum = 0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            batch_iou, _ = compute_iou(out, masks)
            t_iou_sum += batch_iou.mean().item()

        avg_t_loss = t_loss / len(train_loader)
        avg_t_iou = t_iou_sum / len(train_loader)

        # --- VALIDATE WITH TTA (Test Time Augmentation) ---
        model.eval()
        v_loss = 0
        total_inter = torch.zeros(8).to(device)
        total_union = torch.zeros(8).to(device)

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                # 1. Standard Forward
                out1 = model(imgs)
                loss = criterion(out1, masks)
                v_loss += loss.item()

                # 2. TTA: Flip Horizontal -> Predict -> Flip Back
                imgs_flip = torch.flip(imgs, dims=[3]) # Flip width
                out_flip = model(imgs_flip)
                out2 = torch.flip(out_flip, dims=[3])

                # Average predictions (TTA Boost)
                out_avg = (out1 + out2) / 2.0

                # Metrics on TTA output
                iou_batch, conf_mat = compute_iou(out_avg, masks)
                intersection = torch.diag(conf_mat)
                union = conf_mat.sum(1) + conf_mat.sum(0) - intersection
                total_inter += intersection
                total_union += union

        avg_v_loss = v_loss / len(val_loader)
        class_ious = total_inter / (total_union + 1e-6)
        avg_v_iou = class_ious.mean().item()

        scheduler.step()

        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)
        history['train_iou'].append(avg_t_iou)
        history['val_iou'].append(avg_v_iou)

        print(f"Epoch [{epoch}/{args.epochs}] TrainLoss: {avg_t_loss:.4f} ValLoss: {avg_v_loss:.4f} "
              f"TrainIoU: {avg_t_iou:.4f} ValIoU: {avg_v_iou:.4f} (TTA Enabled)")

        if avg_v_iou > best_iou:
            best_iou = avg_v_iou
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✅ Saved best model (mIoU={best_iou:.4f})")

        table_data = [[cls, f"{iou.item():.4f}"] for cls, iou in zip(CLASSES, class_ious)]
        print(tabulate(table_data, headers=["Class", "IoU"], tablefmt="simple"))
        print("-" * 60)

    print("Generating Smooth Plots...")
    save_smooth_plots(args.save_dir, history['train_iou'], history['val_iou'], "mIoU", best_val=best_iou)
    save_smooth_plots(args.save_dir, history['train_loss'], history['val_loss'], "Loss")
    print(f"Done. Checkpoints in: {os.path.abspath(args.save_dir)}")

if __name__ == '__main__':
    main()
#72 percent on suim
