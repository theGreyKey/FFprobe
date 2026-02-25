# extract_features.py
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download

from config import MODEL_ID, CACHE_DIR, DATA_LIMIT, get_feature_path
from utils import prepare_simpleqa_data, prepare_logiqa_data

# === Configuration ===
MODEL_ID = 'LLM-Research/Meta-Llama-3.1-8B-Instruct' 
CACHE_DIR = './model_weights'
FEATURES_DIR = './features'

def build_universal_eval_prompt(statement: str) -> str:
    return f"Statement: {statement}\nIs this statement true or false? Answer:"
def build_cot_prompt(statement: str) -> str:
    # Forces the model to generate a reasoning path before evaluating
    return f"Statement: {statement}\nLet's analyze this step by step to determine if it is true or false:\n"

def extract_hidden_states(model, tokenizer, dataset, dataset_type="simpleqa"):
    if not dataset: return None, None
    print(f"\n⛏️  [Phase 3] Extracting Layer-wise Hidden States for {len(dataset)} {dataset_type} samples...")
    
    pos_vecs, neg_vecs = [], []
    model.eval()

    for item in tqdm(dataset, desc=f"Extracting {dataset_type}"):
        if dataset_type.lower() == "logiqa":
            pos_statement = f"{item['user_content']} Based on the context, it is true that: {item['pos_target']}"
            neg_statement = f"{item['user_content']} Based on the context, it is true that: {item['neg_target']}"
            cot_prompts = [build_cot_prompt(pos_statement), build_cot_prompt(neg_statement)]
            
            # Apply template to strings, then tokenize with left-padding for batched generation
            tokenizer.padding_side = "left"
            cot_strs = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in cot_prompts]
            
            cot_inputs = tokenizer(cot_strs, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                cot_outputs = model.generate(
                    **cot_inputs, 
                    max_new_tokens=64, 
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id
                )
                
            input_len = cot_inputs.input_ids.shape[1]
            pos_cot = tokenizer.decode(cot_outputs[0][input_len:], skip_special_tokens=True).strip()
            neg_cot = tokenizer.decode(cot_outputs[1][input_len:], skip_special_tokens=True).strip()

            aligned_pos_statement = f"{pos_statement}\nReasoning: {pos_cot}"
            aligned_neg_statement = f"{neg_statement}\nReasoning: {neg_cot}"
            
            pos_prompt = build_universal_eval_prompt(aligned_pos_statement)
            neg_prompt = build_universal_eval_prompt(aligned_neg_statement)
            
        else:
            pos_statement = f"{item['user_content']} {item['pos_target']}"
            neg_statement = f"{item['user_content']} {item['neg_target']}"
            
            pos_prompt = build_universal_eval_prompt(pos_statement)
            neg_prompt = build_universal_eval_prompt(neg_statement)
            
        # 2. Extract final hidden states using individual passes to avoid padding noise at the last token
        tokenizer.padding_side = "right"
        pos_chat = [{"role": "user", "content": pos_prompt}]
        neg_chat = [{"role": "user", "content": neg_prompt}]
        
        pos_ids = tokenizer.apply_chat_template(pos_chat, return_tensors="pt", add_generation_prompt=True).to(model.device)
        neg_ids = tokenizer.apply_chat_template(neg_chat, return_tensors="pt", add_generation_prompt=True).to(model.device)
        
        if hasattr(pos_ids, "input_ids"): pos_ids = pos_ids.input_ids
        if hasattr(neg_ids, "input_ids"): neg_ids = neg_ids.input_ids
        
        with torch.no_grad():
            out_pos = model(pos_ids, output_hidden_states=True, use_cache=False)
            out_neg = model(neg_ids, output_hidden_states=True, use_cache=False)
        
        pos_hidden = torch.stack([h[0, -1, :].cpu().float() for h in out_pos.hidden_states])
        neg_hidden = torch.stack([h[0, -1, :].cpu().float() for h in out_neg.hidden_states])
        
        pos_vecs.append(pos_hidden)
        neg_vecs.append(neg_hidden)
        
    return torch.stack(pos_vecs), torch.stack(neg_vecs)

if __name__ == "__main__":
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    print("🚀 Loading Model...")
    model_dir = snapshot_download(
        MODEL_ID, 
        cache_dir=CACHE_DIR, 
        revision='master',
        ignore_patterns=["*.gguf", "*.pth", "*.pt", "original/*"] 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.eval()
    print("✅ Model loaded successfully!")
    
    # --- Part 1: Extract SimpleQA (Facts) ---
    simpleqa_data = prepare_simpleqa_data(model, tokenizer, limit=DATA_LIMIT, csv_path="./datasets/simpleqa_verified.csv")
    if simpleqa_data:
        qa_pos, qa_neg = extract_hidden_states(model, tokenizer, simpleqa_data, "SimpleQA")
        save_path_qa = get_feature_path("simpleqa")
        torch.save({"pos": qa_pos, "neg": qa_neg}, save_path_qa)
        print(f"💾 Saved SimpleQA features to {save_path_qa}")
    else:
        print("⚠️ SimpleQA extraction skipped.")

    # --- Part 2: Extract LogiQA (Logic Transfer) ---
    logiqa_data = prepare_logiqa_data(file_path="./datasets/test.txt", limit=200) 
    if logiqa_data:
        lq_pos, lq_neg = extract_hidden_states(model, tokenizer, logiqa_data, "LogiQA")
        save_path_lq = get_feature_path("logiqa")
        torch.save({"pos": lq_pos, "neg": lq_neg}, save_path_lq)
        print(f"💾 Saved LogiQA features to {save_path_lq}")
    else:
        print("⚠️ LogiQA extraction skipped.")
        
    print("\n🎉 All feature extraction completed!")