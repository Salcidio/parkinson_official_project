
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import sys

# Add project root to sys.path to allow imports from core
# Assuming this file is deep in core/motor_agent/agents/version2/
# relative path to core is ../../../
current_dir = Path(__file__).resolve().parent
core_path = current_dir.parent.parent.parent
if str(core_path) not in sys.path:
    sys.path.append(str(core_path))

try:
    from config import config
except ImportError:
    # Fallback if running standalone or path issue, though config should be in core
    class ConfigMock:
        INPUT_DIM_MOTOR = 8
        DATA_DIR = Path('./data')
    config = ConfigMock()

class FeatureEncoder(nn.Module):
    """
    Encodes raw motor features into the latent control input 'u'.
    """
    def __init__(self, input_dim, output_dim=config.INPUT_DIM_MOTOR):
        super(FeatureEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MotorAgent:
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.encoder = None # Will be initialized after data loads and we know feature dim
        self.df = None
        self.patient_data = {} # Cache for patient data

    def load_data(self, filename: str = None):
        if filename is None:
            filename = config.MOTOR_FILE
        
        file_path = self.data_dir / filename
        if not file_path.exists():
            print(f"Warning: Motor data file not found at {file_path}")
            return
            
        # Simplistic loading - keeping it compatible with previous logic but simplified
        df = pd.read_csv(file_path)
        
        # Standardize columns if needed (UPDRS III specific)
        # Using columns from previous implementation reference
        rename_map = {
            'patient_id': 'PATNO', 
            'assessment_date': 'INFODT',
            'updrs_motor_tremor': 'NUPDRS3_TREMOR',
            'updrs_motor_rigidity': 'NUPDRS3_RIGIDITY', 
            'updrs_motor_bradykinesia': 'NUPDRS3_BRADY',
            'updrs_motor_postural_instability': 'NUPDRS3_POSTURAL'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Ensure essential columns exist
        if 'NUPDRS3' not in df.columns:
             # Sum subscores if total not present
             subscores = ['NUPDRS3_TREMOR', 'NUPDRS3_RIGIDITY', 'NUPDRS3_BRADY', 'NUPDRS3_POSTURAL']
             available_subs = [c for c in subscores if c in df.columns]
             if available_subs:
                 df['NUPDRS3'] = df[available_subs].sum(axis=1)
             else:
                 # Fallback/Placeholder
                 df['NUPDRS3'] = 0.0

        if 'INFODT' in df.columns:
            df['INFODT'] = pd.to_datetime(df['INFODT'])
            
        self.df = df
        
        # Pre-calculate feature dimension
        # We will use NUPDRS3 and maybe subscores as input features to the encoder
        self.feature_cols = ['NUPDRS3'] # minimal set for now, can expand
        input_dim = len(self.feature_cols)
        
        self.encoder = FeatureEncoder(input_dim=input_dim)
        
        # Group by patient for fast retrieval
        if 'PATNO' in df.columns:
            self.patient_data = {k: v for k, v in df.groupby('PATNO')}

    def get_encoder(self):
        return self.encoder

    def encode(self, patient_id: int, date: str) -> torch.Tensor:
        """
        Returns the latent control vector u_motor for a specific patient at a specific time.
        If data is missing for that exact date, we interpolate or take nearest.
        For LSSM, we might need a sequence.
        """
        if self.encoder is None:
            raise ValueError("Data not loaded or encoder not initialized.")

        # Logic to fetch features for patient_id at date
        # Check cache
        if patient_id not in self.patient_data:
            # Return zero vector if patient not found (or handle otherwise)
            return torch.zeros(1, config.INPUT_DIM_MOTOR)
            
        p_df = self.patient_data[patient_id]
        
        # Find nearest date record
        # simplified: just take the first record for now or nearest
        # In full implementation: use 'date' to find specific visit
        
        # Extract features
        features = p_df[self.feature_cols].iloc[0].values.astype(np.float32)
        features_tensor = torch.tensor(features).unsqueeze(0) # (1, input_dim)
        
        with torch.no_grad():
            u = self.encoder(features_tensor)
            
        return u

    def get_data_for_training(self):
        """
        Returns all data formatted for the unified training loop.
        Output: List of (PATNO, TimeSeriesFeatures, TimePoints) or similar
        """
        # Placeholder for data generator
        pass

if __name__ == "__main__":
    # Test
    agent = MotorAgent()
    # Mock data loading for test if file doesn't exist
    print("Motor Agent Initialized")
