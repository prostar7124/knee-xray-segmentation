import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Model training and evaluation')

    parser.add_argument('-csv_dir', type=str, default='data/CSVs')
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-wd', type=float, default=1e-5)
    parser.add_argument('-epochs', type=int, default=15)
    parser.add_argument('-out_dir', type=str, default='session')

    # Add checkpoint path for evaluation
    parser.add_argument('-checkpoint', type=str, default='best_model.pth')

    return parser.parse_args()
