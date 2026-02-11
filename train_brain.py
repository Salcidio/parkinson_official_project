
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from config import config
from brain.lssm import BrainLSSM
from motor_agent.agents.version2.motor_agent import MotorAgent
from non_motor.agent.non_motor_agent import NonMotorAgent

def train():
    print(f"Initializing agents and loading data from {config.DATA_DIR}...")
    
    # Initialize agents
    motor_agent = MotorAgent()
    motor_agent.load_data()
    
    non_motor_agent = NonMotorAgent()
    non_motor_agent.load_data()
    
    # Identify common patients
    motor_pats = set(motor_agent.patient_data.keys())
    non_motor_pats = set(non_motor_agent.patient_data.keys())
    common_pats = list(motor_pats.intersection(non_motor_pats))
    
    if not common_pats:
        print("Error: No common patients found between Motor and Non-Motor datasets.")
        return
        
    print(f"Found {len(common_pats)} common patients.")
    
    # Initialize Brain
    # Update input dimensions based on agents
    # This is a bit circular since config was static, but we can update instance config or re-init
    # ideally config should be dynamic or agents match config. 
    # Let's assume agents match config for now or update robustly
    
    # Check features to update config if needed (runtime adjustment)
    if motor_agent.encoder:
        # We need to access the input features of the encoder to know expected input dim?
        # No, the encoder OUTPUT dim matches Brain INPUT dim.
        # Our default config had 8 and 8. 
        # MotorAgent.encode output 8 (default). NonMotor defaults to 8. 
        # So we seem consistent.
        pass

    brain = BrainLSSM(config).to(config.DEVICE)
    optimizer = optim.Adam(brain.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Training Loop
    # We treat each patient's timeline as a sequence.
    # For simplicity in this v1: We train on 1 step prediction or short trajectory?
    # Let's try to predict the NEXT visit score from CURRENT state
    
    # Data Preparation: Create (Pat, Time_t, Time_t+1, Score_t+1) tuples
    train_data = [] # List of (pat_id, current_date, next_date, next_score)
    
    print("Preparing training sequences...")
    for pat in tqdm(common_pats[:50]): # Limit to 50 for dev speed, remove slice for full run
        # Get patient timeline from Motor data (usually the primary anchor)
        p_df = motor_agent.patient_data[pat].sort_values('INFODT')
        if len(p_df) < 2:
            continue
            
        dates = p_df['INFODT'].tolist()
        scores = p_df['NUPDRS3'].tolist()
        
        for i in range(len(dates) - 1):
            t0 = dates[i]
            t1 = dates[i+1]
            dt = (t1 - t0).days / 30.0 # months
            
            if dt <= 0: continue
            
            target = scores[i+1]
            train_data.append({
                'pat_id': pat,
                't0_date': t0,
                't1_date': t1,
                'dt': dt,
                'target': target
            })
    
    print(f"Created {len(train_data)} training samples.")
    
    # Batching
    batch_size = config.BATCH_SIZE
    num_batches = len(train_data) // batch_size
    
    brain.train()
    
    for epoch in range(config.EPOCHS):
        total_loss = 0
        np.random.shuffle(train_data)
        
        for i in range(num_batches):
            batch = train_data[i*batch_size : (i+1)*batch_size]
            
            # Prepare Batch Tensors
            u_motor_list = []
            u_nm_list = []
            dt_list = []
            target_list = []
            
            for sample in batch:
                pat = sample['pat_id']
                t0 = sample['t0_date']
                
                # Get Encodings
                u_m = motor_agent.encode(pat, t0)
                u_nm = non_motor_agent.encode(pat, t0)
                
                u_motor_list.append(u_m)
                u_nm_list.append(u_nm)
                dt_list.append(sample['dt'])
                target_list.append(sample['target'])
            
            u_motor = torch.cat(u_motor_list).to(config.DEVICE) # (B, dim)
            u_nm = torch.cat(u_nm_list).to(config.DEVICE) # (B, dim)
            targets = torch.tensor(target_list, dtype=torch.float32).unsqueeze(1).to(config.DEVICE) # (B, 1)
            
            # Forward Pass
            # We integrate from t=0 to t=dt for each sample?
            # Since dt varies, we might need a max dt or specialized handling.
            # Simplified: Integrate for unit time 0 -> 1, but scale dynamics by dt? 
            # Or just pass time span 0, dt. But batching different dt is tricky with simple ODE solvers.
            # Fixed Step trick: Normalize time? 
            # Better: Group by similar dt or just iterate sequentially in batch (slow).
            # For this MVP: Assume standard step of 6 months (approx dt=6). 
            # Or simple approach: Just predict output at t=1, assuming t=1 represents "Next Visit".
            # Let's feed dt as an input to network? No, time is part of ODE.
            
            # Workaround for variable dt in batch:
            # We use t_span [0, 1] but multiply vector field by dt for each sample?
            # No, standard ODE solvers require consistent t_span for the batch usually.
            # Let's try: average dt for batch or fixed interval prediction.
            # OR logic: evaluate f(h, u, t) where t is actual time. 
            
            # Standard approach: Fixed time generic solver, 0 to 1. 
            # We assume the model learns "Next Step" dynamics.
            t_span = torch.tensor([0.0, 1.0]).to(config.DEVICE) 
            
            # Initial state h0. Can be 0 or learned embedding per patient?
            # For now, 0 or encoded from features too?
            # Let's use a projection of u as h0
            h0 = torch.zeros(batch_size, config.LATENT_DIM).to(config.DEVICE)
            
            _, pred_y_seq = brain(h0, t_span, u_motor, u_nm)
            
            # pred_y_seq is (T, B, 1). We want the end point
            pred = pred_y_seq[-1] # (B, 1)
            
            loss = criterion(pred, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{config.EPOCHS} | Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(brain.state_dict(), config.CHECKPOINT_DIR / f'brain_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train()
