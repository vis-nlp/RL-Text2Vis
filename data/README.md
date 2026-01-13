# Data

## Dataset

This project uses the **Text2Vis** dataset for training and evaluation.

### Download Options

**Option 1: From Hugging Face** (Recommended)

```python
from datasets import load_dataset

dataset = load_dataset("mizanurr/Text2Vis")
```

Or download directly from: https://huggingface.co/datasets/mizanurr/Text2Vis

**Option 2: From GitHub**

Visit the official Text2Vis repository: https://github.com/vis-nlp/Text2Vis

### Data Format

The dataset should contain:
- `Prompt`: The Complte Prompt
- `Question`: The question
- `Answer`: Ground truth answer
- `Visualization Code`: Ground truth Python/Matplotlib code
- `Table Data`: The data table
- `set`: Dataset split (e.g., "test1", "test2")

Place your downloaded CSV file in this `data/` directory as `Text2Vis_Prompt.csv`.
