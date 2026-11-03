"""Run the complete reproducible software demo."""
from pathlib import Path
import os
from generate_dataset import generate
from train_model import train
from predict import predict

if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    generate()
    train()
    predict('data/synthetic_valve_dataset.csv')
