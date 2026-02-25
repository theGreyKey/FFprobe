
# The Energy Landscape of Hallucinations: Decoding and Intercepting LLM Confabulations via Forward-Forward Probes

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
│   └── ff_probe.py                           # Core FF Probe architecture with PeerNorm
├── utils/
│   ├── __init__.py
│   ├── data_loader.py                        # Aligned data pipelines for SimpleQA & LogiQA
│   ├── baselines.py                          # LR, Mass-Mean, Standard MLP, Ablation models
│   └── visualization.py                      # Publication-ready plotting functions
├── extract_features.py                       # Pipeline to extract hidden states from Llama-3.1
├── train_probe.py                            # FF Probe training script
├── experiment.ipynb                          # Including all the experiemnts conducted in the paper
├── requirements.txt                          # Environment dependencies
├── LICENSE                                   # MIT License
├── datasets/                                 # Source data (SimpleQA, LogiQA)
├── features/                                 # Extracted `.pt` hidden state tensors
├── checkpoints/                              # Saved FF Probe weights and metrics
└── model_weights/                            # Local cache for the Llama-3.1-8B-Instruct model

```

## ⚙️ Installation & Setup

> **⚠️ Hardware Compatibility Note:** 
> All experiments and code in this repository were developed and executed on a **single NVIDIA RTX 4090 (24GB PCIe)** GPU. Currently, the pipeline is strictly optimized for single-GPU execution and has **not** been adapted for multi-GPU server clusters (e.g., `DistributedDataParallel` or multi-node setups). Please ensure your environment has at least 24GB of VRAM to comfortably run the Llama-3.1-8B-Instruct feature extraction.

This repository is optimized for NVIDIA NGC containers and PyTorch 2.4+ environments.

```bash
# Clone the repository
git clone https://github.com/theGreyKey/FFprobe.git
cd ff-hallucination-probe

# Install dependencies
pip install -r requirements.txt
```
*(Note: `transformers==4.44.2` is strictly required to ensure proper RoPE scaling and parameter compatibility with Llama-3.1-8B-Instruct).*

## 🚀 Usage Pipeline

The entire experimental pipeline is governed by `config.py`. Modify parameters like `DATA_LIMIT` and `HIDDEN_DIM` centrally before running the steps.

### Step 1: Feature Extraction

Extract the layer-wise hidden states (the last token) for both SimpleQA and LogiQA datasets. The script uses offline ModelScope caching to avoid network issues.

```bash
python extract_features.py
```

* Outputs: `features/simpleqa_features_N800.pt`, `features/logiqa_features_N800.pt*`

### Step 2: Probe Training

Train the FF Probe (with PeerNorm) and generate evaluation metrics.

```bash
python train_probe.py
```

* Outputs: `checkpoints/ff_weights_N800_D512.pt`, `checkpoints/ff_metrics_N800_D512.pt*`

### Step 3: Comprehensive Mechanistic Evaluation

All evaluation, visualization, and causal intervention pipelines have been consolidated into a single master notebook. Open `experiment.ipynb` to execute:

* **Fast Baseline Evaluation**: Compare FFprobe against advanced baselines (Probability/Entropy, CCS, Mass-Mean, LR, MLP) and generate the layer-wise AUROC plot and academic benchmark table.
* **Ablation Studies**: Validate the layer-wise stability contribution of `PeerNorm` and the cross-domain necessity of `Z-Score`.
* **Cognitive Orthogonality**: Extract and visualize the top causal neurons for SimpleQA (Facts) vs. LogiQA (Logic) to observe the 0.0% dimensional overlap.
* **Surgical Continual Learning**: Execute micro-dosing domain replay to compare the FFprobe's zero-forgetting adaptation against the MLP's catastrophic collapse across multiple random seeds.
* **Energy Landscape & Phase Transition**: Interpolate between factual and hallucinated hidden states to visualize the Goodness landscape shift.
* **Goodness-Driven Causal Ablation**: Ablate the top-30 causal neurons identified by the FF gradient to forcefully revert hallucination representations back to the factual baseline.
* **Real-time Trajectory**: Track the token-by-token FF Goodness score dynamically as the LLM generates a response.
* **Taxonomy Matrix**: Generate a 3x3 asymmetric cross-generalization heatmap across Natural, EntitySwap, and Noise distributions.

## 📊 Experimental Highlights

* **Zero-Forgetting Continual Learning**: In cross-domain incremental learning scenarios, our FFprobe adapts to the logical domain with a backward transfer (BWT) of `-0.0077` (effectively 0%), compared to a devastating `-18.57%` catastrophic forgetting in standard MLPs.
* **Parametric Orthogonality**: The Top 50 factual neurons identified by the FFprobe gradient are completely ignored by the logic probe (0.0% overlap), proving that the LLM routes different types of truthfulness through orthogonal semantic dimensions.
* **Asymmetric Generalization**: Probes trained exclusively on natural hallucinations generalize perfectly to synthetic structural noise (AUROC > `0.88`). However, probes trained on synthetic entity swaps fail completely on natural hallucinations (AUROC `~0.56`), providing quantitative proof of hallucination dimensionality.


## 📜 License

This project is licensed under the MIT License.
