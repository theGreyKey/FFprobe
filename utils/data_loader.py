# utils/data_loader.py
import os
import json
import random
import string
import torch
from tqdm import tqdm
from datasets import load_dataset

def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    return s.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def prepare_simpleqa_data(model, tokenizer, limit: int = 200, csv_path: str = "./datasets/simpleqa_verified.csv"):
    """Mines natural hallucinations (OOD errors) from SimpleQA using the model's own generations."""
    print(f"\n🚀 Loading SimpleQA (Target: {limit} samples)...")
    if not os.path.exists(csv_path): 
        print(f"❌ File not found: {csv_path}")
        return []
    
    try:
        ds = load_dataset("csv", data_files=csv_path)["train"]
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return []

    ds_list = list(ds)
    random.seed(42)
    
    # Oversample pool to ensure we meet the target limit after filtering
    data_list = random.sample(ds_list, min(limit * 5, len(ds_list))) 
    
    processed_data = []
    tokenizer.padding_side = "left" 
    batch_size = 16
    
    refusal_keywords = [
        "sorry", "cannot", "model", "unknown", "don't know", 
        "as an ai", "i do not", "apologies", "not sure", "unable"
    ]
    
    stats = {"natural_error": 0, "correct_skipped": 0, "refusal": 0}

    for i in tqdm(range(0, len(data_list), batch_size), desc="Mining SimpleQA"):
        if len(processed_data) >= limit: break
        
        batch = data_list[i : i + batch_size]
        prompts, valid_indices = [], []
        
        for idx, item in enumerate(batch):
            q = item.get('problem') or item.get('question')
            a = item.get('answer') or item.get('Answer')
            if not q or not a: continue
            
            valid_indices.append(idx)
            
            # Strict prompt to force direct entity output from Instruct models
            prompt = [{"role": "user", "content": f"Question: {q}\nTask: Answer directly with a short entity.\nAnswer:"}]
            prompts.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
        
        if not prompts: continue
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False,      # Greedy search for true model intuition
                temperature=None, 
                top_p=None, 
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        for j, valid_idx in enumerate(valid_indices):
            if len(processed_data) >= limit: break
            
            item = batch[valid_idx]
            input_len = inputs.input_ids[j].shape[0]
            gen_ans = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True).strip()
            
            real_ans = item.get('answer') or item.get('Answer')
            real_clean = normalize_text(real_ans)
            gen_clean = normalize_text(gen_ans)

            is_refusal = any(w in gen_clean for w in refusal_keywords) or len(gen_clean.split()) > 15 
            
            if is_refusal:
                stats["refusal"] += 1
                continue
                
            if real_clean != gen_clean:
                stats["natural_error"] += 1
                processed_data.append({
                    "sys_prompt": "You are a helpful assistant.",
                    "user_content": f"Question: {item.get('problem') or item.get('question')}\nAnswer directly with a short entity:",
                    "pos_target": real_ans,
                    "neg_target": gen_ans
                })
            else:
                stats["correct_skipped"] += 1
            
    tokenizer.padding_side = "right"
    print(f"✅ Extracted {len(processed_data)} SimpleQA samples.")
    print(f"📊 Stats - Errors kept: {stats['natural_error']}, Correct skipped: {stats['correct_skipped']}, Refusals: {stats['refusal']}")
    return processed_data

def prepare_logiqa_data(file_path: str = "./datasets/test.txt", limit: int = 200):
    """Loads LogiQA data, preserving reasoning geometry for probe training."""
    print(f"\n🚀 Loading LogiQA (Target: {limit} samples)...")
    if not os.path.exists(file_path): return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return []

    # Shuffling is crucial to avoid sequential bias
    random.seed(42)
    random.shuffle(lines)
    processed_data = []

    for line in lines:
        if len(processed_data) >= limit: break
        try:
            d = json.loads(line.strip())
            label_idx = int(d.get('answer', d.get('label', -1)))
            options = d.get('options', [])
            context = d.get('text', '')
            question = d.get('question', '')

            if label_idx < 0 or label_idx >= len(options) or not context: continue

            user_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
            pos_resp = options[label_idx]
            
            wrong_indices = [i for i in range(len(options)) if i != label_idx]
            if not wrong_indices: continue
            neg_resp = options[random.choice(wrong_indices)]

            processed_data.append({
                "sys_prompt": "You are a logic reasoning assistant. Select the correct option.",
                "user_content": user_text,
                "pos_target": pos_resp,
                "neg_target": neg_resp
            })
        except Exception: continue
            
    print(f"✅ Constructed {len(processed_data)} LogiQA samples.")
    return processed_data