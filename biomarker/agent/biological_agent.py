
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
        INPUT_DIM_BIOLOGICAL = 4
        DATA_DIR = Path('./data')
    config = ConfigMock()

class FeatureEncoder(nn.Module):
    """
    Encodes biological features (DaTscan) into the latent control input 'u_bio'.
    """
    def __init__(self, input_dim, output_dim=config.INPUT_DIM_BIOLOGICAL):
        super(FeatureEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class BiologicalAgent:
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.encoder = None
        self.df = None
        self.patient_data = {}
        # Specific columns for DaTscan
        self.target_cols = ['datscan_left_putamen', 'datscan_right_putamen', 
                            'datscan_left_caudate', 'datscan_right_caudate']

    def load_data(self, filename: str = None):
        if filename is None:
            filename = getattr(config, 'BIOLOGICAL_FILE', 'datscan.csv')
            
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
        if 'assessment_date' in df.columns:
            df.rename(columns={'assessment_date': 'INFODT'}, inplace=True)
            
        # Ensure numeric columns are actually numeric
        for col in self.target_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaNs with column mean
        df.fillna(df.mean(numeric_only=True), inplace=True)
        
        self.df = df
        
        # Initialize Encoder
        # Check which of the target columns actually exist
        available_cols = [c for c in self.target_cols if c in df.columns]
        self.feature_cols = available_cols
        
        input_dim = len(self.feature_cols)
        if input_dim == 0:
            print("Warning: No DaTscan feature columns found.")
            input_dim = 1
            
        self.encoder = FeatureEncoder(input_dim=input_dim)
        
        # Cache data
        if 'PATNO' in df.columns:
            self.patient_data = {k: v for k, v in df.groupby('PATNO')}

    def get_encoder(self):
        return self.encoder

    def encode(self, patient_id: int, date: str = None) -> torch.Tensor:
        """
        Returns latent control vector u_bio.
        """
        if self.encoder is None:
             raise ValueError("Data not loaded or encoder not initialized.")

        # Check cache
        if patient_id not in self.patient_data:
            return torch.zeros(1, config.INPUT_DIM_BIOLOGICAL)
            
        p_df = self.patient_data[patient_id]
        
        # Feature extraction
        # For now, taking the mean of all visits for stability, or could sort by date
        # Ideally, find the record closest to 'date'
        
        features = p_df[self.feature_cols].mean(axis=0).values.astype(np.float32)
        features_tensor = torch.tensor(features).unsqueeze(0) # (1, input_dim)
        
        with torch.no_grad():
            u = self.encoder(features_tensor)
            
        return u

if __name__ == "__main__":
    agent = BiologicalAgent()
    print("Biological Agent Initialized")
