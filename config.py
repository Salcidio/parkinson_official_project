
import os
import sys
from pathlib import Path
import torch

class Config:
    def __init__(self):
        self.IS_COLAB = 'google.colab' in sys.modules
        
        # Base Paths
        if self.IS_COLAB:
            self.BASE_DIR = Path('/content')
            self.DATA_DIR = self.BASE_DIR
        else:
            # Assumes running from project root or similar. Adjust as needed.
            self.BASE_DIR = Path(os.getcwd())
            self.DATA_DIR = self.BASE_DIR / 'data'
            # Fallback if specific data files are in known locations
            if not self.DATA_DIR.exists():
                self.DATA_DIR = self.BASE_DIR # Try current dir

        # Subdirectories
        self.CHECKPOINT_DIR = self.BASE_DIR / 'checkpoints'
        self.OUTPUT_DIR = self.BASE_DIR / 'outputs'
        
        # Create directories if they don't exist
        for d in [self.CHECKPOINT_DIR, self.OUTPUT_DIR]:
            if not d.exists():
                try:
                    d.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass # Might fail on read-only systems, ignore for now

        # Brain Hyperparameters
        self.LATENT_DIM = 32
        self.HIDDEN_DIM = 64
        self.INPUT_DIM_MOTOR = 8  # Example size
        self.INPUT_DIM_NON_MOTOR = 8 # Example size
        self.ODE_TOL = 1e-3
        self.SOLVER_METHOD = 'rk4' # 'euler', 'rk4'

        # Training Hyperparameters
        self.BATCH_SIZE = 16 # Patients per batch
        self.LEARNING_RATE = 1e-3
        self.EPOCHS = 50
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Data Config
        self.MOTOR_FILE = 'motor_merged.csv'
        self.NON_MOTOR_FILE = 'merged_non_motor_data.csv'

config = Config()
