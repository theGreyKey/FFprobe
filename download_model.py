# download_model.py
import os
from modelscope import snapshot_download

# === Configuration ===
# Update MODEL_ID to 'LLM-Research/Meta-Llama-3.1-70B-Instruct' for larger models
MODEL_ID = 'LLM-Research/Meta-Llama-3.1-8B-Instruct' 
CACHE_DIR = './model_weights' 

print(f"🚀 [Terminal] Starting model download: {MODEL_ID}")
print(f"📂 Save directory: {os.path.abspath(CACHE_DIR)}")

try:
    model_dir = snapshot_download(
        MODEL_ID, 
        cache_dir=CACHE_DIR,
        revision='master',
        ignore_patterns=["*.gguf", "*.pth", "*.pt", "original/*"] 
    )
    print("\n✅ Download completed successfully!")
    print(f"➡️  In your Notebook/Scripts, your model path (model_dir) should be:\n{model_dir}")
except Exception as e:
    print(f"\n❌ Download failed: {e}")