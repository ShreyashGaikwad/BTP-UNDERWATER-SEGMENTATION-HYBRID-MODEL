# python test_suim.py \
# >   --data-root ./SUIM \
# >   --model-path ./suim_checkpoints_tta/best_model.pth \
# >   --out-dir ./test_results \
# >   --batch-size 4 \
# >   --img-size 224 \
# >   --num-workers 4 \
# >   --tta

cat test_suim.py
#!/usr/bin/env python3
"""
test_suim.py

Test script for SUIM-trained model.

- Loads model weights from a checkpoint (no training).
- Runs inference on TEST images under <data_root>/TEST/images and masks under <data_root>/TEST/masks
- Saves concatenated images: [original image | original mask | model prediction] -> out_dir/concats/
  - Each concat image has a vertical separator line between panes and a large bold label above each pane.
- Computes & saves confusion matrix PNG for all 8 classes -> out_dir/confusion_matrix.png and normalized PNG
- Computes and prints/exports metrics: per-class IoU, precision, recall, F1, mIoU, pixel accuracy, macro/micro/weighted P/R/F1
"""

import os
import argparse
import time
import importlib
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate
from matplotlib import font_manager

# ----------------------------- CONFIG (same as training) ----------------------
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

IDX2COLOR = {v: k for k, v in COLOR_MAP.items()}
NUM_CLASSES = len(CLASSES)

# ----------------------------- Dataset for TEST --------------------------------
class SUIMTestDataset(Dataset):
    """
    Returns: img_tensor, label_tensor (HxW long), pil_image_resized, pil_mask_resized, filename
    """
    def __init__(self, images_dir: str, masks_dir: str, img_size: Tuple[int, int]=(224,224)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.png','.bmp'))])
        self.img_size = tuple(img_size)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.files)

    def encode_mask(self, mask_pil: Image.Image) -> np.ndarray:
        mask_resized = mask_pil.resize(self.img_size, resample=Image.NEAREST)
        mask_np = np.array(mask_resized)
        if mask_np.ndim == 3 and mask_np.shape[2] == 4:
            mask_np = mask_np[:, :, :3]
        h,w = self.img_size
        label_mask = np.zeros((h,w), dtype=np.int64)
        matched_any = False
        for rgb, idx in COLOR_MAP.items():
            rr,gg,bb = rgb
            match = (mask_np[:,:,0]==rr) & (mask_np[:,:,1]==gg) & (mask_np[:,:,2]==bb)
            if match.any():
                label_mask[match] = idx
                matched_any = True
        if not matched_any:
            # fallback: any non-black pixel -> class 6 (Fish)
            label_mask[(mask_np.sum(axis=2) > 0)] = 6
        return label_mask

    def __getitem__(self, idx):
        fname = self.files[idx]
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(self.images_dir, fname)
        mask_path = None
        for ext in ['.png', '.jpg', '.bmp']:
            cand = os.path.join(self.masks_dir, base + ext)
            if os.path.exists(cand):
                mask_path = cand
                break
        if mask_path is None:
            mask_path = os.path.join(self.masks_dir, fname)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {fname}")

        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("RGB")

        img_resized = img_pil.resize(self.img_size, Image.BILINEAR)
        mask_resized = mask_pil.resize(self.img_size, Image.NEAREST)

        img_tensor = self.normalize(img_resized)
        label_mask = self.encode_mask(mask_resized)  # HxW numpy

        return img_tensor, torch.from_numpy(label_mask).long(), img_resized, mask_resized, fname

# ----------------------------- MODEL (try import, else define) -----------------
def try_import_training_model():
    """
    Try to import TimmHybridNet from a local training module named 'train' or similar.
    """
    names = ['train', 'train_suim', 'train_script', 'train_script_suim', 'suim_train']
    for nm in names:
        try:
            mod = importlib.import_module(nm)
            if hasattr(mod, 'TimmHybridNet'):
                print(f"[INFO] Imported TimmHybridNet from module '{nm}'")
                return mod.TimmHybridNet
        except Exception:
            continue
    return None

# Fallback: copy of TimmHybridNet architecture used in training script
class TimmHybridNet(nn.Module):
    def __init__(self, backbone='swin_base_patch4_window7_224.ms_in22k', num_classes=8, pretrained=True, target_size=(224, 224)):
        super().__init__()
        self.target_size = target_size
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True)

        dummy = torch.randn(1, 3, target_size[0], target_size[1])
        features = self.encoder(dummy)

        ch = []
        self.channels_last = False
        last_feat = features[-1]

        if last_feat.shape[1] < last_feat.shape[-1]:
            self.channels_last = True
            ch = [f.shape[-1] for f in features]
        else:
            self.channels_last = False
            ch = [f.shape[1] for f in features]

        self.up1 = self._up_block(ch[-1], ch[-2])
        self.up2 = self._up_block(ch[-2], ch[-3])
        self.up3 = self._up_block(ch[-3], ch[-4])

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(ch[-4], 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
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

# ----------------------------- UTILITIES -------------------------------------
def colorize_label(mask_arr: np.ndarray) -> Image.Image:
    """Given HxW integer mask, convert to RGB PIL using IDX2COLOR mapping."""
    h,w = mask_arr.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    for idx, rgb in IDX2COLOR.items():
        out[mask_arr==idx] = rgb
    return Image.fromarray(out)

def plot_confusion(conf: np.ndarray, class_names: List[str], out_path: str, normalize=False):
    plt.figure(figsize=(10,8))
    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            conf_norm = conf.astype(np.float64)
            rowsum = conf_norm.sum(axis=1, keepdims=True)
            conf_display = np.divide(conf_norm, rowsum, where=(rowsum!=0))*100.0
    else:
        conf_display = conf
    im = plt.imshow(conf_display, interpolation='nearest', cmap='Blues')
    title = "Confusion Matrix (percent)" if normalize else "Confusion Matrix (counts)"
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha='right')
    plt.yticks(ticks, class_names)
    plt.ylabel('True')
    plt.xlabel('Pred')
    thresh = conf_display.max() / 2.0 if conf_display.max()>0 else 1.0
    for i in range(conf_display.shape[0]):
        for j in range(conf_display.shape[1]):
            val = conf_display[i,j]
            if normalize:
                s = f"{val:.1f}%"
            else:
                s = f"{int(val)}"
            plt.text(j, i, s, ha='center',
                     color='white' if conf_display[i,j] > thresh else 'black',
                     fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

# ----------------------------- custom collate --------------------------------
def test_collate(batch):
    """
    batch: list of tuples (img_tensor, label_tensor, pil_img, pil_mask, filename)
    Return: batched tensors for img & label, lists for PIL images, masks and names
    """
    imgs = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    imgs_pil = [item[2] for item in batch]
    masks_pil = [item[3] for item in batch]
    names = [item[4] for item in batch]
    return imgs, labels, imgs_pil, masks_pil, names

# ----------------------------- MAIN TEST FLOW --------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, help='root folder containing TEST/images and TEST/masks')
    parser.add_argument('--model-path', required=True, help='path to checkpoint .pth')
    parser.add_argument('--out-dir', default='./test_results', help='output directory')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--backbone', default='swin_base_patch4_window7_224.ms_in22k')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--tta', action='store_true', help='apply horizontal flip TTA (averaging)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    conc_dir = os.path.join(args.out_dir, 'concats')
    os.makedirs(conc_dir, exist_ok=True)

    test_images = os.path.join(args.data_root, 'TEST', 'images')
    test_masks = os.path.join(args.data_root, 'TEST', 'masks')
    if not os.path.exists(test_images) or not os.path.exists(test_masks):
        raise FileNotFoundError("TEST/images or TEST/masks not found under data-root")

    ds = SUIMTestDataset(test_images, test_masks, img_size=(args.img_size,args.img_size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, collate_fn=test_collate)

    # Try import training model
    ImportedModel = try_import_training_model()
    if ImportedModel is not None:
        ModelClass = ImportedModel
    else:
        print("[INFO] Could not import training model module; using internal TimmHybridNet copy.")
        ModelClass = TimmHybridNet

    # instantiate model
    model = ModelClass(backbone=args.backbone, num_classes=NUM_CLASSES, pretrained=False, target_size=(args.img_size, args.img_size))
    model = model.to(device)

    # load checkpoint
    ckpt = torch.load(args.model_path, map_location='cpu')
    if isinstance(ckpt, dict) and ('model_state_dict' in ckpt or 'state_dict' in ckpt):
        sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    else:
        sd = ckpt

    # strip module. prefix if present
    new_sd = {}
    for k,v in sd.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        new_sd[new_k] = v

    # try strict load, fallback to non-strict
    try:
        model.load_state_dict(new_sd, strict=True)
        print("[INFO] Loaded checkpoint with strict=True")
    except Exception as e:
        print("[WARN] strict load failed:", e)
        try:
            model.load_state_dict(new_sd, strict=False)
            print("[INFO] Loaded checkpoint with strict=False (some keys mismatched/ignored)")
        except Exception as e2:
            print("[ERROR] Failed to load checkpoint:", e2)
            raise e2

    model.eval()

    # attempt to find a good TTF font for large bold labels; fallback to default
    try:
        font_path = font_manager.findfont("DejaVu Sans")
        title_font = ImageFont.truetype(font_path, size=20)
    except Exception:
        title_font = ImageFont.load_default()

    # accumulators
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_pixels = 0
    correct_pixels = 0
    idx = 0

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            imgs, labels, imgs_pil, masks_pil, names = batch
            imgs = imgs.to(device)
            if args.tta:
                out1 = model(imgs)
                imgs_flip = torch.flip(imgs, dims=[3])
                out_flip = model(imgs_flip)
                out2 = torch.flip(out_flip, dims=[3])
                out = (out1 + out2) / 2.0
            else:
                out = model(imgs)
            probs = F.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()  # B,H,W
            labels_np = labels.numpy()

            B = preds.shape[0]
            for b in range(B):
                pred_mask = preds[b]
                true_mask = labels_np[b]
                flat_pred = pred_mask.flatten()
                flat_true = true_mask.flatten()
                valid = (flat_true >= 0) & (flat_true < NUM_CLASSES)
                flat_pred = flat_pred[valid]
                flat_true = flat_true[valid]
                if flat_true.size > 0:
                    coords = NUM_CLASSES * flat_true + flat_pred
                    binc = np.bincount(coords, minlength=NUM_CLASSES*NUM_CLASSES)
                    conf += binc.reshape((NUM_CLASSES, NUM_CLASSES))
                    total_pixels += flat_true.size
                    correct_pixels += int((flat_pred == flat_true).sum())

                # save concat image with separators and large bold labels
                img_res = imgs_pil[b]
                mask_res = masks_pil[b]
                pred_color = colorize_label(pred_mask)
                # ensure sizes match
                if img_res.size != pred_color.size:
                    pred_color = pred_color.resize(img_res.size, resample=Image.NEAREST)
                if mask_res.size != img_res.size:
                    mask_res = mask_res.resize(img_res.size, resample=Image.NEAREST)

                w,h = img_res.width, img_res.height
                concat = Image.new("RGB", (w*3, h + 40), color=(30,30,30))  # extra top space for labels
                # paste labels area background
                label_bg = Image.new("RGB", (w*3, 40), color=(50,50,50))
                concat.paste(label_bg, (0,0))

                # paste images below the label area
                concat.paste(img_res, (0,40))
                concat.paste(mask_res, (w,40))
                concat.paste(pred_color, (2*w,40))

                draw = ImageDraw.Draw(concat)
                # vertical separator lines (between the three panes), full height including label area
                line_x1 = w
                line_x2 = 2*w
                draw.line([(line_x1, 0), (line_x1, h+40)], fill=(255,255,255), width=4)
                draw.line([(line_x2, 0), (line_x2, h+40)], fill=(255,255,255), width=4)

                # labels (large + bold via stroke if supported)
                def draw_label(text, center_x):
                    try:
                        # stroke parameters provide bold/contrast
                        draw.text((center_x, 10), text, font=title_font, fill=(255,255,255),
                                  anchor="mm", stroke_width=2, stroke_fill=(0,0,0))
                    except TypeError:
                        # older Pillow may not support stroke; draw text multiple times to simulate bold
                        x,y = center_x, 10
                        for dx,dy in [(0,0), (1,0), (0,1), (1,1)]:
                            draw.text((x+dx, y+dy), text, font=title_font, fill=(255,255,255), anchor="mm")

                draw_label("IMAGE", w//2)
                draw_label("MASK", w + w//2)
                draw_label("PREDICTION", 2*w + w//2)

                out_name = f"{idx:05d}_{names[b]}.png"
                out_path = os.path.join(conc_dir, out_name)
                concat.save(out_path)
                idx += 1

    elapsed = time.time() - t0
    print(f"[INFO] Processed {len(ds)} images in {elapsed:.1f}s")

    # compute metrics from confusion matrix
    inter = np.diag(conf).astype(np.float64)
    gt_counts = conf.sum(axis=1).astype(np.float64)  # true counts per class (row sums)
    pred_counts = conf.sum(axis=0).astype(np.float64)  # predicted counts per class (col sums)
    union = gt_counts + pred_counts - inter
    class_iou = inter / (union + 1e-7)
    mIoU = np.nanmean(class_iou)
    pixel_acc = correct_pixels / (total_pixels + 1e-9) if total_pixels>0 else 0.0

    # precision/recall/f1 per class
    tp = inter
    fp = pred_counts - tp
    fn = gt_counts - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # macro/micro/weighted averages
    mask = ~np.isnan(precision)
    macro_p = np.nanmean(precision[mask])
    macro_r = np.nanmean(recall[mask])
    macro_f1 = np.nanmean(f1[mask])

    # micro: compute from totals
    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    micro_p = total_tp / (total_tp + total_fp + 1e-9)
    micro_r = total_tp / (total_tp + total_fn + 1e-9)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)

    # weighted (by gt support)
    support = gt_counts
    weighted_p = np.nansum(precision * support) / (support.sum() + 1e-9)
    weighted_r = np.nansum(recall * support) / (support.sum() + 1e-9)
    weighted_f1 = np.nansum(f1 * support) / (support.sum() + 1e-9)

    # print summary
    print("\nTEST SUMMARY")
    print("="*60)
    print(f"Num images: {len(ds)}")
    print(f"Total pixels: {int(total_pixels)}")
    print(f"Pixel accuracy: {pixel_acc:.6f}")
    print(f"mIoU: {mIoU:.6f}")
    print("")
    # per-class table
    rows = []
    for i, cls in enumerate(CLASSES):
        rows.append([cls,
                     f"{class_iou[i]:.4f}",
                     f"{precision[i]:.4f}",
                     f"{recall[i]:.4f}",
                     f"{f1[i]:.4f}",
                     int(gt_counts[i]),
                     int(pred_counts[i])])
    print(tabulate(rows, headers=["Class","IoU","Precision","Recall","F1","GT_count","Pred_count"], tablefmt="github"))

    print("\nAverages:")
    print(f"Macro P/R/F1: {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")
    print(f"Micro P/R/F1: {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")
    print(f"Weighted P/R/F1: {weighted_p:.4f} / {weighted_r:.4f} / {weighted_f1:.4f}")

    # save confusion matrices
    cm_path = os.path.join(args.out_dir, 'confusion_matrix_counts.png')
    plot_confusion(conf, CLASSES, cm_path, normalize=False)
    cmn_path = os.path.join(args.out_dir, 'confusion_matrix_percent.png')
    plot_confusion(conf, CLASSES, cmn_path, normalize=True)
    print(f"[INFO] Confusion matrices saved to: {cm_path} and {cmn_path}")

    # save numeric summary
    txt = os.path.join(args.out_dir, "test_summary.txt")
    with open(txt, "w") as f:
        f.write("TEST SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Num images: {len(ds)}\n")
        f.write(f"Total pixels: {int(total_pixels)}\n")
        f.write(f"Pixel accuracy: {pixel_acc:.6f}\n")
        f.write(f"mIoU: {mIoU:.6f}\n\n")

        # --- Neat tabular per-class metrics (changed) ---
        f.write("Per-class metrics:\n")
        table_headers = ["Class","IoU","Precision","Recall","F1","GT_count","Pred_count"]
        table_rows = []
        for i, cls in enumerate(CLASSES):
            table_rows.append([cls,
                               f"{class_iou[i]:.6f}",
                               f"{precision[i]:.6f}",
                               f"{recall[i]:.6f}",
                               f"{f1[i]:.6f}",
                               f"{int(gt_counts[i])}",
                               f"{int(pred_counts[i])}"])
        # use tabulate to create a neat table string and write it
        table_str = tabulate(table_rows, headers=table_headers, tablefmt="github")
        f.write(table_str + "\n\n")

        f.write("Averages:\n")
        f.write(f"Macro P/R/F1: {macro_p:.6f} / {macro_r:.6f} / {macro_f1:.6f}\n")
        f.write(f"Micro P/R/F1: {micro_p:.6f} / {micro_r:.6f} / {micro_f1:.6f}\n")
        f.write(f"Weighted P/R/F1: {weighted_p:.6f} / {weighted_r:.6f} / {weighted_f1:.6f}\n")
    print(f"[INFO] Numeric summary saved to: {txt}")

    print(f"\nSaved concatenated images to: {conc_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
