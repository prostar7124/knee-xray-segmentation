import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
import cv2
import numpy as np
from args import get_args
from model import UNetLext
from dataset import Knee_dataset
from utils import dice_score_from_logits


def visualize_prediction(image, gt_mask, pred_mask):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input X-Ray")
    plt.imshow(image.squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(gt_mask.squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask.squeeze(), cmap="gray")
    plt.axis("off")

    plt.show()


def evaluate(model, loader, device):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for batch in loader:

            if not batch["has_mask"]:
                continue

            images = batch["image"].float().to(device)
            masks  = batch["mask"].float().to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            dice = dice_score_from_logits(outputs, masks)
            dice_scores.append(dice)

            visualize_prediction(
                images[0].cpu(),
                masks[0].cpu(),
                preds[0].cpu()
            )

    avg_dice = sum(dice_scores) / len(dice_scores)
    print("\n=========================")
    print(f" Average Dice Score: {avg_dice:.4f}")
    print("=========================\n")


def main():
    args = get_args()

    test_df = pd.read_csv(os.path.join(args.csv_dir, "val.csv"))

    test_dataset = Knee_dataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetLext(
        input_channels=1,
        output_channels=1,
        activation='selu',
        pretrained=False
    )

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
