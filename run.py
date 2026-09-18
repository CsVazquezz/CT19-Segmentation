"""
Entrenamiento y generación de la submission de Kaggle para COVID-19 CT segmentation.

Es el mismo pipeline del notebook COVID_19_CT_Images_Segmentation.ipynb, extraído a un
script para poder correrlo localmente (sin Colab ni Drive) y sin las celdas de gráficas.

Uso:
    python run.py --epochs 15 --batch-size 16      # entrena y escribe la submission
    python run.py --predict-only                   # sólo re-genera la submission
"""

import argparse
import os
import time

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# Semilla fija para que los resultados sean reproducibles
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Las 4 clases que podrá tener cada píxel de la máscara
CLASS_NAMES = ["Ground glass", "Consolidation", "Lungs other", "Background"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TARGET_SIZE = 512
HU_MIN, HU_MAX = -1500, 500


# --------------------------------------------------------------------------------------
# Carga y limpieza
# --------------------------------------------------------------------------------------
def load_data():
    images_radiopedia = np.load(f"{DATA_DIR}/images_radiopedia.npy").astype(np.float32)
    masks_radiopedia = np.load(f"{DATA_DIR}/masks_radiopedia.npy").astype(np.uint8)
    images_medseg = np.load(f"{DATA_DIR}/images_medseg.npy").astype(np.float32)
    masks_medseg = np.load(f"{DATA_DIR}/masks_medseg.npy").astype(np.uint8)
    test_images = np.load(f"{DATA_DIR}/test_images_medseg.npy").astype(np.float32)

    print("images_radiopedia", images_radiopedia.shape)
    print("images_medseg    ", images_medseg.shape)
    print("test_images      ", test_images.shape)
    return images_radiopedia, masks_radiopedia, images_medseg, masks_medseg, test_images


def fix_label_conflicts(masks):
    """Un píxel marcado como lesión/pulmón Y como fondo: se queda con lesión/pulmón."""
    is_lung_or_lesion = masks[..., :3].sum(axis=3) > 0
    masks[is_lung_or_lesion, 3] = 0
    assert (masks.sum(axis=3) == 1).all()
    return masks


def normalize_hu(img):
    """Recorta a la ventana HU y escala linealmente a [0,1]."""
    return ((np.clip(img, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class CovidDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform is not None:
            # El augmentation se aplica a imagen y máscara para que
            # compartan la misma transformación geométrica
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert HWC -> CHW for PyTorch
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        image = torch.tensor(image.copy(), dtype=torch.float32)
        mask = torch.tensor(mask.copy(), dtype=torch.float32)
        return image, mask


# --------------------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        # 4 output masks
        self.output = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)


# --------------------------------------------------------------------------------------
# Pérdidas y evaluación
# --------------------------------------------------------------------------------------
ce = nn.CrossEntropyLoss()


def dice_loss(logits, target_onehot, smooth=1.0):
    """Soft Dice sobre las 3 clases anatómicas (sin background)."""
    probs = torch.softmax(logits.float(), dim=1)[:, :3]
    tgt = target_onehot[:, :3]
    dims = (0, 2, 3)
    inter = (probs * tgt).sum(dims)
    denom = probs.sum(dims) + tgt.sum(dims)
    return 1 - ((2 * inter + smooth) / (denom + smooth)).mean()


def combined_loss(logits, target_onehot):
    target_idx = target_onehot.argmax(dim=1)  # one-hot (B,4,H,W) -> índices (B,H,W)
    return ce(logits.float(), target_idx) + dice_loss(logits, target_onehot)


@torch.no_grad()
def predict_probs(model, x, tta=False):
    """Probabilidades softmax. Con TTA promedia la predicción del espejo horizontal."""
    p = torch.softmax(model(x).float(), dim=1)
    if tta:
        p_flip = torch.softmax(model(torch.flip(x, dims=[3])).float(), dim=1)
        p = 0.5 * (p + torch.flip(p_flip, dims=[3]))
    return p


@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    """Dice acumulado sobre TODO el val, no promedio de Dices por batch."""
    model.eval()
    inter = torch.zeros(4, device=device)
    denom = torch.zeros(4, device=device)
    losses = []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        losses.append(combined_loss(model(images), masks).item())
        pred = predict_probs(model, images, tta).argmax(dim=1)
        true = masks.argmax(dim=1)
        for c in range(4):
            p, t = (pred == c), (true == c)
            inter[c] += (p & t).sum()
            denom[c] += p.sum() + t.sum()
    dice = torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
    return dice.cpu().numpy(), float(np.mean(losses))


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--val-batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--predict-only", action="store_true",
                    help="no entrena: carga --ckpt y vuelve a escribir la submission")
    args = ap.parse_args()

    # ---------------- datos ----------------
    (images_radiopedia, masks_radiopedia,
     images_medseg, masks_medseg, test_images) = load_data()

    masks_radiopedia = fix_label_conflicts(masks_radiopedia)
    masks_medseg = fix_label_conflicts(masks_medseg)

    # Las primeras 24 de medseg son validation, el resto va a train
    val_images = images_medseg[:24]
    val_masks = masks_medseg[:24]
    train_images_medseg = images_medseg[24:]
    train_masks_medseg = masks_medseg[24:]

    # Los slices de radiopaedia 100 % background no aportan nada -> se descartan
    has_lung = masks_radiopedia[..., :3].sum(axis=(1, 2, 3)) > 0
    print(f"Radiopaedia: {len(has_lung)} slices, {(~has_lung).sum()} sin pulmón -> descartados")
    assert masks_radiopedia[~has_lung][..., :2].sum() == 0  # no lesion pixels lost

    train_images = np.concatenate([train_images_medseg, images_radiopedia[has_lung]])
    train_masks = np.concatenate([train_masks_medseg, masks_radiopedia[has_lung]])
    n_medseg = len(train_images_medseg)
    del images_radiopedia, masks_radiopedia, images_medseg, masks_medseg
    print("train:", train_images.shape, " val:", val_images.shape, " test:", test_images.shape)

    # HU clipping + normalización a [0,1]
    train_images = normalize_hu(train_images)
    val_images = normalize_hu(val_images)
    test_images = normalize_hu(test_images)
    print("rangos ->",
          f"train [{train_images.min():.2f}, {train_images.max():.2f}]",
          f"val [{val_images.min():.2f}, {val_images.max():.2f}]",
          f"test [{test_images.min():.2f}, {test_images.max():.2f}]")

    # ---------------- augmentation ----------------
    train_transform = A.Compose([
        A.Rotate(limit=20, border_mode=cv2.BORDER_REPLICATE,
                 mask_interpolation=cv2.INTER_NEAREST, p=0.5),
        A.RandomSizedCrop(min_max_height=(384, 512), size=(TARGET_SIZE, TARGET_SIZE),
                          mask_interpolation=cv2.INTER_NEAREST, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ])

    train_dataset = CovidDataset(train_images, train_masks, transform=train_transform)
    val_dataset = CovidDataset(val_images, val_masks, transform=None)

    # Oversampling: el train tiene MedSeg + Radiopaedia, pero el TEST es MedSeg.
    # MedSeg va primero en la concatenación, así que son los primeros N.
    weights = np.concatenate([np.full(n_medseg, 5.0), np.ones(len(train_dataset) - n_medseg)])
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
    print(f"MedSeg pasa de {n_medseg/len(train_dataset):.0%} a "
          f"{weights[:n_medseg].sum()/weights.sum():.0%} de cada época")

    # ---------------- modelo ----------------
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet().to(device)
    epochs = args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f"parámetros: {sum(p.numel() for p in model.parameters())/1e6:.1f} M | "
          f"device: {device} | batch_size: {args.batch_size} | epochs: {epochs}")

    # ---------------- entrenamiento ----------------
    use_amp = device.type == "cuda"          # GradScaler es sólo de CUDA
    scaler = GradScaler("cuda", enabled=use_amp)
    best_score = -1.0
    history = {"train_loss": [], "val_loss": [], "dice": []}

    for epoch in range(0 if args.predict_only else epochs):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                loss = combined_loss(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        dice, val_loss = evaluate(model, val_loader, device, tta=False)
        scheduler.step()
        score = (dice[0] + dice[1]) / 2  # solo las lesiones: es lo que evalúa Kaggle

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["dice"].append(dice)

        flag = ""
        if score > best_score:
            best_score, flag = score, "  <-- mejor"
            torch.save(model.state_dict(), args.ckpt)

        print(f"Epoch {epoch+1:2d}/{epochs} | {time.time()-t0:5.0f}s | lr {lr_now:.2e} | "
              f"train {train_loss:.4f} | val {val_loss:.4f}", flush=True)
        print(f"   GG {dice[0]:.4f} | Cons {dice[1]:.4f} | Lungs {dice[2]:.4f} | "
              f"Bg {dice[3]:.4f} | lesiones {score:.4f}{flag}", flush=True)

    if not args.predict_only:
        print(f"\nMejor Dice de lesiones: {best_score:.4f}  ->  {args.ckpt}")

    # ---------------- TTA: se usa sólo si mejora ----------------
    model.load_state_dict(torch.load(args.ckpt))
    d_sin, _ = evaluate(model, val_loader, device, tta=False)
    d_con, _ = evaluate(model, val_loader, device, tta=True)
    print(pd.DataFrame({"sin TTA": d_sin, "con TTA": d_con, "delta": d_con - d_sin},
                       index=CLASS_NAMES).round(4))
    use_tta = bool((d_con[0] + d_con[1]) >= (d_sin[0] + d_sin[1]))
    print("TTA en test:", use_tta)

    if not args.predict_only:
        np.save("history_dice.npy", np.array(history["dice"]))

    # ---------------- predicción sobre test ----------------
    model.eval()
    test_X = torch.from_numpy(test_images.transpose(0, 3, 1, 2)).float()
    preds = []
    for i in range(0, len(test_X), 2):  # de a 2 para no llenar la VRAM
        p = predict_probs(model, test_X[i:i + 2].to(device), tta=use_tta)
        preds.append(p.argmax(dim=1).cpu().numpy().astype(np.uint8))  # argmax, no umbral 0.5
    test_pred = np.concatenate(preds)  # (10, 512, 512)

    test_onehot = np.eye(4, dtype=np.uint8)[test_pred]  # (10,512,512,4), formato original
    np.save("test_prediction.npy", test_onehot)
    print("guardado test_prediction.npy:", test_onehot.shape)

    # ---------------- submission de Kaggle ----------------
    test_masks_prediction = test_onehot[..., :2]  # the first two lesion classes
    print("Kaggle prediction shape:", test_masks_prediction.shape)
    assert test_masks_prediction.shape == (10, 512, 512, 2)

    # Kaggle se contradice en la descripción: el texto pide columnas `Id,Predicted`
    # pero el snippet de ejemplo arma `Id,Expected`. Las filas son idénticas y sólo
    # cambia el encabezado, así que escribimos las dos variantes.
    flat_predictions = test_masks_prediction.ravel().astype(int)
    ids = np.arange(len(flat_predictions))
    alt = args.submission.replace(".csv", "_expected.csv")
    for fname, col in [(args.submission, "Predicted"), (alt, "Expected")]:
        pd.DataFrame({"Id": ids, col: flat_predictions}).to_csv(fname, index=False)
        print(f"{fname:26s} -> {len(flat_predictions):,} filas, columna '{col}', "
              f"positivos = {int(flat_predictions.sum()):,} "
              f"({flat_predictions.mean():.3%})")


if __name__ == "__main__":
    main()
