**Knee X-Ray Bone Segmentation using U-Net**
Deep Learning Project – CNN Segmentation Model

This repository contains my implementation of a U-Net–based bone segmentation model for knee X-ray images.
The goal of the project is to automatically segment the femur bone region from medical X-ray scans.

This project was developed as part of the Bone Segmentation Assignment, with training performed on manually annotated masks created by group members.

**Project Structure**
args.py        → Argument configuration  
dataset.py     → Dataset loader (images + masks)  
model.py       → U-Net model  
trainer.py     → Training & validation loops  
main.py        → Starts the training pipeline  
evaluate.py    → Loads best model & performs evaluation  
utils.py       → Dice loss, Dice score, helpers  
train.csv      → Image + mask paths (training)  
val.csv        → Image + mask paths (validation/testing)
samples/       → Example images for README

**Training**
The training uses:
- BCEWithLogitsLoss + Dice Loss
- Adam optimizer
- Automatic saving of best model → best_model.pth
- Best Dice Score Achieved: 0.93 - 0.94

**Evaluation**
Use your saved checkpoint:  python evaluate.py -checkpoint best_model.pth

The script:
- Computes the average Dice score
- Shows side-by-side:
          1. Input X-ray
          2. Ground truth mask
          3. Predicted mask
  
**Requirements**
torch
torchvision
matplotlib
pandas
numpy
opencv-python -> cv2

  
