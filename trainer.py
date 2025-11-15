from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
from utils import dice_loss_from_logits, dice_score_from_logits


def train_model(model, train_loader, val_loader, device):
    args = get_args()
    model.to(device)

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_dice = 0.0

    for epoch in range(int(args.epochs)):
        model.train()
        running_loss = 0.0
        batches_with_masks = 0

        for data_batch in train_loader:

            if not data_batch["has_mask"]:
                continue

            batches_with_masks += 1

            images = data_batch["image"].float().to(device)
            masks = data_batch["mask"].float().to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / batches_with_masks if batches_with_masks > 0 else 0.0

        val_loss, val_dice = validate_model(model, val_loader, bce, device)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Saved new best model")


def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    val_loss_total = 0.0
    val_dice_total = 0.0
    batches_with_masks = 0

    with torch.no_grad():
        for data_batch in val_loader:

            if not data_batch["has_mask"]:
                continue

            batches_with_masks += 1

            images = data_batch["image"].float().to(device)
            masks = data_batch["mask"].float().to(device)

            outputs = model(images)

            loss_bce = loss_fn(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            val_loss_total += loss.item()
            val_dice_total += dice_score_from_logits(outputs, masks)

    if batches_with_masks == 0:
        return 0.0, 0.0

    return val_loss_total / batches_with_masks, val_dice_total / batches_with_masks
