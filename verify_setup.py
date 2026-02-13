
import sys
from pathlib import Path
import torch

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from config import config

def verify():
    print("="*50)
    print("Parkinson's Project - Setup Verification")
    print("="*50)
    
    # 1. Check Environment
    print(f"\n[Environment]")
    print(f"Is Colab: {config.IS_COLAB}")
    print(f"Base Dir: {config.BASE_DIR}")
    print(f"Data Dir: {config.DATA_DIR}")
    print(f"Device: {config.DEVICE}")
    
    # 2. Check Data Files
    print(f"\n[Data Files]")
    required_files = [
        ('Motor', config.MOTOR_FILE),
        ('Non-Motor', config.NON_MOTOR_FILE),
        ('Biological', config.BIOLOGICAL_FILE)
    ]
    
    all_files_found = True
    for name, filename in required_files:
        path = config.DATA_DIR / filename
        fallback_path = config.BASE_DIR / filename
        
        if path.exists():
            print(f"  [OK] {name}: Found at {path}")
        elif fallback_path.exists():
            print(f"  [OK] {name}: Found at {fallback_path} (Fallback)")
        else:
            print(f"  [MISSING] {name}: Not found at {path} or {fallback_path}")
            if name != 'Datscan': # Datscan might be optional for now
                all_files_found = False

    # 3. Check Imports and Agents
    print(f"\n[Agents & Logic]")
    try:
        from train_brain import train
        print("  [OK] Successfully imported train_brain")
        
        print("  Running validation step...")
        train(validate_only=True)
        
    except ImportError as e:
        print(f"  [FAIL] Import Error: {e}")
    except Exception as e:
        print(f"  [FAIL] Runtime Error during validation: {e}")

    print("\n" + "="*50)
    if all_files_found:
        print("VERIFICATION SUCCESSFUL: Ready to train.")
    else:
        print("VERIFICATION COMPLETED WITH WARNINGS/ERRORS: Check missing files.")
    print("="*50)

if __name__ == "__main__":
    verify()
