"""Shared dataset / model / train-step helpers for the MCF-Net baseline.

Imports the MCF-Net model architecture unmodified from the cloned HzFu/EyeQ
repo ($EYEQC_EXTERNAL\\mcfnet\\MCF_Net). Re-implements the EyeQ dataloader's
RGB/HSV/Lab transform logic (same PIL ImageCms pattern, same ImageNet
normalization) against our own cached, preprocessed PNGs rather than their
DatasetGenerator, because that class expects a repo-specific directory/CSV
layout we are not using.
"""
from __future__ import annotations

import sys
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[3]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageCms
from torch.utils.data import Dataset

MCFNET_MCF_NET = EXTERNAL / "mcfnet/MCF_Net"
if str(MCFNET_MCF_NET) not in sys.path:
    sys.path.insert(0, str(MCFNET_MCF_NET))
from networks.densenet_mcf import dense121_mcs  # noqa: E402

LABEL_IDX = ["Good", "Usable", "Reject"]  # class order used throughout MCF-Net
REJECT_INDEX = 2

TRAIN_TRANSFORM1 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(-180, 180)),
])
EVAL_TRANSFORM1 = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])
TRANSFORM2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_LAB_PROFILE = ImageCms.createProfile("LAB")
_RGB2LAB = ImageCms.buildTransformFromOpenProfiles(_SRGB_PROFILE, _LAB_PROFILE, "RGB", "LAB")


def _to_rgb_hsv_lab(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_hsv = image.convert("HSV")
    img_lab = ImageCms.applyTransform(image, _RGB2LAB)
    img_rgb = np.asarray(image).astype("float32")
    img_hsv = np.asarray(img_hsv).astype("float32")
    img_lab = np.asarray(img_lab).astype("float32")
    return TRANSFORM2(img_rgb), TRANSFORM2(img_hsv), TRANSFORM2(img_lab)


class MCFNetTrainDataset(Dataset):
    """Labeled dataset over cached preprocessed PNGs, with one-hot 3-class labels."""

    def __init__(self, png_paths: list[str], labels: np.ndarray, train: bool = True):
        assert len(png_paths) == len(labels)
        self.png_paths = png_paths
        self.labels = labels  # (N, 3) one-hot
        self.transform1 = TRAIN_TRANSFORM1 if train else EVAL_TRANSFORM1

    def __len__(self) -> int:
        return len(self.png_paths)

    def __getitem__(self, index: int):
        image = Image.open(self.png_paths[index]).convert("RGB")
        image = self.transform1(image)
        rgb, hsv, lab = _to_rgb_hsv_lab(image)
        label = torch.FloatTensor(self.labels[index])
        return rgb, hsv, lab, label


class MCFNetInferenceDataset(Dataset):
    """Unlabeled dataset for scoring; returns index alongside the three tensors."""

    def __init__(self, png_paths: list[str]):
        self.png_paths = png_paths
        self.transform1 = EVAL_TRANSFORM1

    def __len__(self) -> int:
        return len(self.png_paths)

    def __getitem__(self, index: int):
        image = Image.open(self.png_paths[index]).convert("RGB")
        image = self.transform1(image)
        rgb, hsv, lab = _to_rgb_hsv_lab(image)
        return rgb, hsv, lab, index


def build_model(device: torch.device) -> torch.nn.Module:
    model = dense121_mcs(n_class=3)
    model.to(device)
    return model


def one_hot(labels: np.ndarray, n_class: int = 3) -> np.ndarray:
    out = np.zeros((len(labels), n_class), dtype="float32")
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def patient_group_id(image_name: str) -> str:
    """'10009_left.jpeg' / '10009_left' -> '10009' (same grouping convention as
    cluster_id used elsewhere: paired eyes of one patient share a group)."""
    stem = Path(image_name).stem
    for suffix in ("_left", "_right"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def train_one_epoch(loader, model, optimizer, criterion, loss_w, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for imagesA, imagesB, imagesC, labels in loader:
        imagesA, imagesB, imagesC, labels = (
            imagesA.to(device), imagesB.to(device), imagesC.to(device), labels.to(device)
        )
        out_A, out_B, out_C, out_F, combine = model(imagesA, imagesB, imagesC)
        loss = (
            loss_w[0] * criterion(out_A, labels)
            + loss_w[1] * criterion(out_B, labels)
            + loss_w[2] * criterion(out_C, labels)
            + loss_w[3] * criterion(out_F, labels)
            + loss_w[4] * criterion(combine, labels)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_loss(loader, model, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for imagesA, imagesB, imagesC, labels in loader:
        imagesA, imagesB, imagesC, labels = (
            imagesA.to(device), imagesB.to(device), imagesC.to(device), labels.to(device)
        )
        _, _, _, _, combine3 = model(imagesA, imagesB, imagesC)
        loss = criterion(combine3, labels)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict_risk(loader, model, device, n_items: int) -> np.ndarray:
    """Returns the fused (combine3 / result_mcs) 'Reject' (index 2) output for
    every item in the (unlabeled, index-carrying) loader, in original index order."""
    model.eval()
    out = np.zeros(n_items, dtype="float32")
    for imagesA, imagesB, imagesC, idx in loader:
        imagesA, imagesB, imagesC = imagesA.to(device), imagesB.to(device), imagesC.to(device)
        _, _, _, _, combine3 = model(imagesA, imagesB, imagesC)
        reject = combine3[:, REJECT_INDEX].detach().cpu().numpy()
        out[idx.numpy()] = reject
    return out
