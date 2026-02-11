
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
# path to core is ../../
core_path = current_dir.parent.parent
if str(core_path) not in sys.path:
    sys.path.append(str(core_path))

try:
    from config import config
except ImportError:
    class ConfigMock:
        INPUT_DIM_NON_MOTOR = 8
        DATA_DIR = Path('./data')
    config = ConfigMock()

class FeatureEncoder(nn.Module):
    """
    Encodes non-motor features into the latent control input 'u_nm'.
    """
    def __init__(self, input_dim, output_dim=config.INPUT_DIM_NON_MOTOR):
        super(FeatureEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class NonMotorAgent:
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.encoder = None
        self.df = None
        self.patient_data = {}

    def load_data(self, filename: str = None):
        if filename is None:
            filename = getattr(config, 'NON_MOTOR_FILE', 'non_motor_merged.csv')
            
        file_path = self.data_dir / filename
        if not file_path.exists():
            print(f"[{self.__class__.__name__}] Warning: Data file not found at {file_path}")
            # Try looking in base dir as fallback
            if (config.BASE_DIR / filename).exists():
                 file_path = config.BASE_DIR / filename
                 print(f"[{self.__class__.__name__}] Found data at fallback: {file_path}")
            else:
                 return

        try:
            df = pd.read_csv(file_path)
            print(f"[{self.__class__.__name__}] Loaded {len(df)} records from {file_path.name}")
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error loading CSV: {e}")
            return
        
        # Standardize columns
        if 'patient_id' in df.columns:
            df.rename(columns={'patient_id': 'PATNO'}, inplace=True)
            
        # Identify feature columns
        # Excluding ID and potential date columns
        exclude_cols = ['PATNO', 'patient_id', 'INFODT', 'date', 'visit_id']
        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        
        self.feature_cols = feature_cols
        
        # Fill NaNs with 0 for now (or mean)
        df.fillna(0, inplace=True)
        
        self.df = df
        
        # Initialize Encoder
        input_dim = len(self.feature_cols)
        if input_dim == 0:
            print("Warning: No numeric feature columns found in non-motor data.")
            input_dim = 1 # Dummy dimension to avoid crash
            
        self.encoder = FeatureEncoder(input_dim=input_dim)
        
        # Cache data
        if 'PATNO' in df.columns:
            self.patient_data = {k: v for k, v in df.groupby('PATNO')}

    def get_encoder(self):
        return self.encoder

    def encode(self, patient_id: int, date: str = None) -> torch.Tensor:
        """
        Returns latent control vector u_nm.
        """
        if self.encoder is None:
             raise ValueError("Data not loaded or encoder not initialized.")

        # Check cache
        if patient_id not in self.patient_data:
            return torch.zeros(1, config.INPUT_DIM_NON_MOTOR)
            
        p_df = self.patient_data[patient_id]
        
        # Feature extraction
        # simplified: take first record or average
        # In real scenario: find record closest to 'date'
        
        features = p_df[self.feature_cols].mean(axis=0).values.astype(np.float32)
        features_tensor = torch.tensor(features).unsqueeze(0) # (1, input_dim)
        
        with torch.no_grad():
            u = self.encoder(features_tensor)
            
        return u

if __name__ == "__main__":
    agent = NonMotorAgent()
    print("Non-Motor Agent Initialized")
