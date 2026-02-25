# config.py
import os

# === 1. Model & Paths ===
MODEL_ID = 'LLM-Research/Meta-Llama-3.1-8B-Instruct'
CACHE_DIR = './model_weights'
FEATURES_DIR = './features'
CHECKPOINTS_DIR = './checkpoints'

# === 2. Data Parameters ===
DATA_LIMIT = 400 # Target samples

# === 3. Probe Architecture & Training ===
HIDDEN_DIM = 512  # FF Probe capacity
TRAIN_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.005
DEVICE = 'cuda'

# Automatically create necessary directories
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Helper function for dynamic naming based on current params
def get_feature_path(dataset_name):
    return os.path.join(FEATURES_DIR, f"{dataset_name}_features_N{DATA_LIMIT}.pt")

def get_checkpoint_path(file_type):
    # file_type: 'metrics' or 'weights'
    return os.path.join(CHECKPOINTS_DIR, f"ff_{file_type}_N{DATA_LIMIT}_D{HIDDEN_DIM}.pt")