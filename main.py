from args import get_args
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import Knee_dataset
from model import UNetLext
from trainer import train_model


def main():
    args = get_args()

    train_df = pd.read_csv(os.path.join(args.csv_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(args.csv_dir, "val.csv"))

    train_dataset = Knee_dataset(train_df)
    val_dataset   = Knee_dataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNetLext(
        input_channels=1,
        output_channels=1,
        activation='selu',  
        pretrained=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, train_loader, val_loader, device)


if __name__ == '__main__':
    main()
