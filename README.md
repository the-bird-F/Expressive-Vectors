<h1 align="center">
Task Vector in TTS: Toward Emotionally Expressive Dialectal Speech Synthesis
</h1>


<div align="center">
    <a href="https://github.com/the-bird-F/Expressive-Vectors" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Code-blue?logo=github" alt="Github Code"></a>
    <a href="https://the-bird-f.github.io/Expressive-Vectors/" target="_blank"> <img src="https://img.shields.io/badge/Demo-Project%20Page-green?logo=googlechrome" alt="Project Page"></a> 
    <!-- <a href="https://arxiv.org/abs/2505.00028" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2505.00028-red?logo=arxiv" alt="arXiv Paper"></a> -->
</div>


<p align="center">
  <span>
    <img src="./resources/Picture1.svg" alt="Picture 1" height="300" style="margin:5px;"/>
    <img src="./resources/Picture2.svg" alt="Picture 2" height="300" style="margin:5px;"/>
    <img src="./resources/Picture3.svg" alt="Picture 3" height="300" style="margin:5px;"/>
  </span>
</p>


---

## ✨ Overview 
We propose **Hierarchical Expressive Vector (HE-Vector)**, a two-stage method for Emotional Dialectal TTS.

In the first stage, we construct different task vectors to model dialectal and emotional styles independently, and then enhance single-style synthesis by adjusting their weights, a method we refer to as Expressive Vector (E-Vector).

For the second stage, we hierarchically integrate these vectors to achieve controllableemotionally expressive dialect synthesis without requiring jointly labeled data, corresponding to Hierarchical Expressive Vector (HE-Vector).

## Model



## Dataset

### Dialect Datasets
- **Sichuan Dialect (Sichuanhua)**: [MagicData Sichuan Dialect Scripted Speech Corpus](https://magichub.com/datasets/sichuan-dialect-scripted-speech-corpus-daily-use-sentence/)
- **Cantonese (Guangdonghua)**: [MagicData Guangzhou Cantonese Scripted Speech Corpus](https://magichub.com/datasets/guangzhou-cantonese-scripted-speech-corpus-daily-use-sentence/)
- **Shanghai Dialect (Shanghainese)**: [MagicData Shanghai Dialect Scripted Speech Corpus](https://magichub.com/datasets/shanghai-dialect-scripted-speech-corpus-daily-use-sentence/)

### Emotional Speech Dataset
- **Emotional Speech Data**: [HLTSingapore/Emotional-Speech-Data](https://github.com/HLTSingapore/Emotional-Speech-Data)

> Note: The original dialect data used in the paper experiments cannot be openly shared. As an alternative, we provide links to the MagicData dialect datasets. Our experiments confirm that training on these datasets achieves comparable performance.

## 🚀 Quick Start

### Prerequisites and Base Model Setup
```bash
# Create and activate the conda environment
conda create -n f5-tts python=3.11
conda activate f5-tts

# Install PyTorch with CUDA 12.4 support
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Clone and install F5-TTS
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
pip install -e .
```


### HE-Vector Inference
```bash
accelerate launch Expressive-Vectors/expressive_vector/my_eval_infer2.py \
    -s 0 \
    -c "ckpts/test/sichuan_happy.pt" \
    -t "Expressive-Vectors/dataset/ESD_use/Happy/eval.csv" \
    -r "Expressive-Vectors/dataset/Prompt_use/prompt10.csv" \
    -b "/dataset/Prompt_use/" \
    -nfe 32 \
    -o "./"
```

### Complete Training Pipeline
```bash
# Clone the Expressive-Vectors repository
git clone https://github.com/the-bird-F/Expressive-Vectors.git

# Step 1: Finetune the model
bash ./Expressive-Vectors/scripts/finetuning_model.sh

# Step 2: Construct expressive vectors
bash ./Expressive-Vectors/scripts/mining_model.sh

# Step 3: Evaluate the model
bash ./Expressive-Vectors/scripts/evaluation_model.sh
```


## 🔬 Experiment Details

### Hardware Configuration
- GPU: NVIDIA RTX 4090

### Training Hyperparameters
```bash
learning_rate: 1e-5
batch_size_per_gpu: 8000
batch_size_type: "frame"
max_samples: 64
grad_accumulation_steps: 1
max_grad_norm: 1.0
epochs: 1000
num_warmup_updates: 20000
save_per_updates: 20000
last_per_updates: 5000
finetune: True
pretrain: ./ckpts/F5TTS_v1_Base/model_1250000.pt
tokenizer: "pinyin"

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
```




## 📄 License

The code in this repository is released under the [Apache 2.0](LICENSE) license.


If you find this project helpful, feel free to ⭐️ Star and 🔁 Fork it!
