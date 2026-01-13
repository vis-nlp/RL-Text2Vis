# Aligning Text, Code, and Vision: A Multi-Objective Reinforcement Learning Framework for Text-to-Visualization

Official repository for the paper: **[Aligning Text, Code, and Vision: A Multi-Objective Reinforcement Learning Framework for Text-to-Visualization](https://www.arxiv.org/pdf/2601.04582)**

## Overview

This repository provides the implementation for training language models using Group Relative Policy Optimization (GRPO) to generate both textual answers and executable Python/Matplotlib visualization code from data tables. The framework employs a multi-objective reinforcement learning approach that aligns text generation, code quality, and visual correctness.

## Features

- **Multi-Modal Reward System**: Combines text, code executability, and visual quality rewards
- **GRPO Training**: Group Relative Policy Optimization for efficient reinforcement learning
- **Visual Evaluation**: Automatic assessment of generated visualizations for readability and correctness
- **Checkpoint Resume**: Automatic checkpoint detection and training resumption

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision transformers trl accelerate datasets
pip install pandas numpy scipy matplotlib pillow
```

### Data

Download the **Text2Vis** dataset from either:

**Option 1: Hugging Face** (Recommended)
```python
from datasets import load_dataset
dataset = load_dataset("mizanurr/Text2Vis")
```
Or visit: https://huggingface.co/datasets/mizanurr/Text2Vis

**Option 2: GitHub Repository**
Visit: https://github.com/vis-nlp/Text2Vis

Place the downloaded CSV file in the `data/` directory as `Text2Vis_Prompt.csv`.

### Models

You can use different model combinations for training:

**Main Model** (for training):
- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct
- Llama-3-70B-Instruct
- Any instruction-tuned language model

**Text Reward Model**:
- Qwen2.5-7B-Instruct
- Qwen2.5-32B-Instruct
- Llama-3-70B-Instruct
- Any instruction-tuned language model

**Visual Reward Model** (must support vision):
- Qwen2.5-VL-7B-Instruct
- Qwen2.5-VL-32B-Instruct
- LLaVA-NeXT
- Any vision-language model

## Usage

### Training

```bash
python scripts/rl_text2vis.py \
  --model_path /path/to/Qwen2.5-14B-Instruct \
  --text_reward_model_path /path/to/Qwen2.5-7B-Instruct \
  --vis_reward_model_path /path/to/Qwen2.5-VL-7B-Instruct \
  --data_path ./data \
  --output_dir ./output \
  --image_save_path ./images
```

## Sample Training Configuration

- Learning rate: 1e-5
- Batch size: 8 per device
- Gradient accumulation: 8 steps
- Number of generations: 8
- Epochs: 1
- Max prompt length: 512
- Max completion length: 2048
- Mixed precision: bfloat16

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{rahman2026aligning,
  title={Aligning Text, Code, and Vision: A Multi-Objective Reinforcement Learning Framework for Text-to-Visualization},
  author={Rahman, Mizanur and Islam, Mohammed Saidul and Laskar, Md Tahmid Rahman and Joty, Shafiq and Hoque, Enamul},
  journal={arXiv preprint arXiv:2601.04582},
  year={2026}
}
```

## Contact

For questions or issues, please contact: mizanur.york@gmail.com

