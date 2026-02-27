# Forward-Forward Probes for LLM Hallucination Detection: Local Energy as an Interpretable, Interventional, and Continually Learnable Signal

This repository contains the official implementation of a novel mechanistic interpretability framework that utilizes **Forward-Forward (FF) Probes** to decode, intercept, and causally intervene in Large Language Model (LLM) hallucinations.

Unlike traditional backpropagation-based classifiers (e.g., Logistic Regression, MLPs, or CCS), our approach computes a continuous "Truth Energy" (Goodness score) locally at each layer. By introducing an LLM-tailored architecture featuring *Instance Centering*, *PeerNorm*, and *Adaptive Z-Score Normalization*, we map high-dimensional hidden states into a robust physical energy space. This enables us to reveal the cognitive orthogonality between factual and logical circuits, define hallucination taxonomy via asymmetric generalization, and perform surgical continual learning without catastrophic forgetting.

## 🌟 Key Contributions & Novelties

1. **LLM-Tailored FFprobe Architecture**: We resolve activation anisotropy and cross-domain energy shadowing using Instance Centering, PeerNorm, and Adaptive Z-Score Normalization. This achieves state-of-the-art out-of-distribution hallucination detection (AUROC > 0.81), consistently outperforming Contrast-Consistent Search (CCS) and global MLPs.
2. **Cognitive Orthogonality**: We quantitatively prove that LLMs process factual memories and logical routing using completely independent neural circuits (0.0% overlap among top-50 causal neurons). This structurally refutes the existence of a "universal truth hyper-plane."
3. **Definition-by-Negatives (Asymmetric Taxonomy)**: We demonstrate that natural hallucinations encompass a high-dimensional superset of synthetic errors. Probes trained on natural confabulations generalize perfectly downward to synthetic noise, but synthetic probes fail entirely on natural errors, exposing a critical flaw in using naive entity-swapping for safety alignment.
4. **Surgical Continual Learning**: Exploiting the local contrastive nature of the FF algorithm, we solve the Stability-Plasticity dilemma. FFprobe successfully acquires new logical boundaries with virtually zero catastrophic forgetting on existing factual knowledge, whereas standard MLPs suffer massive network degradation.
5. **Causal Discovery & Real-Time Trajectory**: We utilize Goodness-driven gradients to pinpoint and ablate specific hallucination-inducing neurons, and we demonstrate the probe's ability to track token-level cognitive phase transitions in real-time during generation.

## 📂 Repository Structure

The project is highly modularized into logical Python packages:

```text
├── config.py                                 # Centralized hyperparameter & path configurations
├── core/
│   ├── __init__.py
│   └── ff_probe.py                           # Core FF Probe architecture with PeerNorm & Z-Score
├── utils/
│   ├── __init__.py
│   ├── data_loader.py                        # Aligned data pipelines for SimpleQA & LogiQA
│   ├── baselines.py                          # LR, Mass-Mean, Standard MLP, Ablation models
│   ├── advanced_baselines.py                 # Frontier probes (RepE, SAPLMA, NL-CCS, ConceptBottleneck, etc.)
│   ├── evaluation.py                         # Academic metrics, Bootstrap CI, DeLong & McNemar Tests
│   └── visualization.py                      # Publication-ready plotting functions
├── download_model.py                         # Download Llama-3.1 via ModelScope
├── extract_features.py                       # Pipeline to extract hidden states from LLMs
├── train_probe.py                            # FF Probe training script
├── experiment.ipynb                          # Master notebook including all experiments conducted in the paper
├── requirements.txt                          # Environment dependencies
├── LICENSE                                   # MIT License
├── datasets/                                 # Source data (SimpleQA, LogiQA)
├── features/                                 # Extracted `.pt` hidden state tensors
├── checkpoints/                              # Saved FF Probe weights and metrics
└── model_weights/                            # Local cache for the Llama-3.1-8B-Instruct model
```

## ⚙️ Installation & Setup

> **⚠️ Hardware Compatibility Note:** > All experiments and code in this repository were developed and executed on a **single NVIDIA RTX 4090 (24GB PCIe)** GPU. Currently, the pipeline is strictly optimized for single-GPU execution and has **not** been adapted for multi-GPU server clusters (e.g., `DistributedDataParallel`). Please ensure your environment has at least 24GB of VRAM to comfortably run the Llama-3.1-8B-Instruct feature extraction.

### Step 1: Environment Setup

This repository is optimized for PyTorch 2.4+ environments.

```bash
# Clone the repository
git clone https://github.com/theGreyKey/FFprobe.git
cd ff-hallucination-probe

# Install dependencies
pip install -r requirements.txt

```

*(Note: `transformers==4.44.2` is strictly required to ensure proper RoPE scaling and parameter compatibility with Llama-3.1-8B-Instruct).*

### Step 2: Data & Model Preparation

You need to download the datasets and the LLM weights. **Please ensure all downloaded assets are saved in the `data/` folder** (or update your `config.py` accordingly).

**For Datasets:**

```python
from datasets import load_dataset
# Save to your local data/ directory
ds_sqa = load_dataset("google/simpleqa-verified", cache_dir="./data")
ds_logi = load_dataset("lucasmccabe/logiqa", cache_dir="./data")

```

**For Model Weights (Llama-3.1-8B-Instruct):**

* **🌐 Global Users (via Hugging Face):**
You can directly load or download the model using the `transformers` library:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./model_weights")
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="./model_weights", device_map="auto")

# Test Generation
messages = [{"role": "user", "content": "Who are you?"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```


* **🇨🇳 Users in China (via ModelScope):** To avoid network issues, we highly recommend downloading the model via ModelScope. You can just simply run our download_model.py:
```bash
python download_model.py
```

## 🚀 Usage Pipeline

The entire experimental pipeline is governed by `config.py`. Modify parameters like `DATA_LIMIT` and `HIDDEN_DIM` centrally before running the steps.

### Step 1: Feature Extraction

Extract the layer-wise hidden states (the last token) for both SimpleQA and LogiQA datasets. The script uses offline ModelScope caching to avoid network issues.

```bash
python extract_features.py
```

* Outputs: `features/simpleqa_features_N800.pt`, `features/logiqa_features_N800.pt`

### Step 2: Probe Training

Train the FF Probe (with PeerNorm) and generate evaluation metrics.

```bash
python train_probe.py
```

* Outputs: `checkpoints/ff_weights_N800_D512.pt`, `checkpoints/ff_metrics_N800_D512.pt`

### Step 3: Comprehensive Mechanistic Evaluation

All evaluation, visualization, and causal intervention pipelines have been consolidated into a single master notebook. Open `experiment.ipynb` to execute:

* **SOTA Baseline Evaluation & Statistical Tests**: Compare FF-Probe against 11 advanced baselines (including RepE, SAPLMA, NL-CCS, KNN, etc.). Generates comprehensive academic reports featuring 95% Bootstrap CI, DeLong Tests, and McNemar Tests.
* **Cognitive Separation (Top-K Sweep)**: Execute a rigorous Top-K dimension sweep to prove that the neural overlap between factual and logical circuits perfectly tracks theoretical random chance.
* **Cross-Model Zero-Shot Generalization**: Apply the FF-Probe trained exclusively on Llama-3.1-8B directly onto Mistral-7B-Instruct to observe semantic convergence at deep layers.
* **Surgical Continual Learning**: Execute micro-dosing domain replay to compare the FF-Probe's zero-forgetting adaptation against the MLP's catastrophic collapse across multiple random seeds.
* **Goodness-Driven Interventional Ablation**: Ablate the top-30 causal neurons identified by the FF gradient to forcefully revert hallucinated representations back to the factual baseline.
* **Real-time Cognitive Trajectory**: Track the token-by-token FF Goodness score dynamically as the LLM falls into complex literature/logic jailbreaks.
* **Asymmetric Taxonomy Matrix**: Generate a 3x3 cross-generalization heatmap across Natural, EntitySwap, and Noise distributions.

## 📊 Experimental Highlights

* **SOTA Detection & Absolute Rigor**: FF-Probe achieves an unprecedented AUROC of **0.9116**, outperforming the strongest baseline with massive effect sizes (Cohen's d = 1.800) and strict statistical significance ($p < 0.001$ in both DeLong and McNemar tests).
* **Cognitive Separation**: By sweeping the Top-K causal neurons (up to $K=2000$), we empirically demonstrate that the overlap ratio between factual and logical routing perfectly bounds to the theoretical random chance ($K / D$). This mathematically proves that LLMs utilize **highly separable, mutually orthogonal neural sub-spaces** rather than a universal truth hyper-plane.
* **Universal Truth Geometry (Cross-Architecture Transfer)**: A probe trained entirely on Llama-3.1 hidden states achieves a remarkable zero-shot AUROC of **0.75** at deep layers when directly applied to Mistral-7B, providing strong mechanistic evidence for cross-architecture semantic convergence.
* **Zero-Forgetting Continual Learning**: In cross-domain incremental learning scenarios, our FF-Probe adapts to the logical domain with a backward transfer (BWT) of `-0.0077%` (effectively zero), dodging the devastating `-18.57%` catastrophic forgetting observed in standard MLPs.
* **Asymmetric Generalization**: Probes trained exclusively on natural hallucinations generalize perfectly to synthetic structural noise (AUROC > `0.88`). However, probes trained on synthetic entity swaps fail completely on natural hallucinations (AUROC `~0.56`), criticizing the current alignment paradigm's over-reliance on synthetic negative samples.

## 📜 License


This project is licensed under the MIT License.
