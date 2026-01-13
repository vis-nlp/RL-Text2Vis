# Standard library
from __future__ import annotations
import os, sys, time, json, math, ast, re, base64, argparse, logging, warnings
from functools import lru_cache
from io import BytesIO
from typing import Optional
import requests

# Data & numerics
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Visualization
import matplotlib.pyplot as plt

# PyTorch & acceleration
import torch
import torchvision
from accelerate import Accelerator, DistributedType
from packaging import version

# Image processing
from PIL import Image as IM, ImageOps, ExifTags, UnidentifiedImageError
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

# Datasets
from datasets import Dataset, Features, Value, Image, Sequence, load_dataset, load_from_disk

# Transformers & models
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2VLProcessor,
)

# Training (TRL)
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer

# Runtime setup
logger = logging.getLogger(__name__)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--model_path", type=str, required=True, help="Model dir")
    parser.add_argument("--text_reward_model_path", type=str, required=True, help="Text reward model dir")
    parser.add_argument("--vis_reward_model_path", type=str, required=True, help="Visual reward model dir")
    parser.add_argument("--data_path", type=str, required=True, help="Input data path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir")
    parser.add_argument("--image_save_path", type=str, required=True, help="Save dir")

    args = parser.parse_args()
    args_dct = vars(args)


    # 2. System Prompt & Task Instructions
    R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question about a data table,
    and the Assistant solves it. The Assistant should carefully reason internally before responding. The assistant provides the final answer and the Matplotlib code. Return your response as JSON:
    {
    "answer": "your short answer here",
    "code": "your Python code here"
    }
    """

    TASK_SPECIFIC_INSTRUCTIONS = "The answer must exactly match the ground-truth, and the code must run without errors."
    
    # Dataset preprocessing
    def preprocess_dataset(csv_path: str) -> Dataset:
        df = pd.read_csv(csv_path)
        df = df[df["set"] == "test1"]
        ds = Dataset.from_pandas(df)

        def make_conversation(example):
            return {
                "prompt": [
                    {
                        "role": "system",
                        "content": R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS,
                    },
                    {
                        "role": "user",
                        "content": example["Prompt"]
                        + "\nAvoid generating any non-Python content before the actual Python script begins",
                    },
                ],
            }

        return ds.map(make_conversation)


    # Load dataset
    dataset = preprocess_dataset(args_dct["data_path"] + "/Text2Vis_Prompt.csv")
    print("Train and validation sampled successfully!")


    # Model paths
    model_id = args_dct["model_path"]
    text_reward_model_id = args_dct["text_reward_model_path"]
    vis_reward_model_id = args_dct["vis_reward_model_path"]


    # Main model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token


    # Text reward model
    text_reward_model = AutoModelForCausalLM.from_pretrained(
        text_reward_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    text_reward_processor = AutoProcessor.from_pretrained(text_reward_model_id)

    text_reward_tokenizer = AutoTokenizer.from_pretrained(text_reward_model_id)
    text_reward_tokenizer.pad_token = text_reward_tokenizer.eos_token


    # Visual reward model
    vis_reward_model = AutoModelForVision2Seq.from_pretrained(
        vis_reward_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    vis_reward_processor = AutoProcessor.from_pretrained(vis_reward_model_id)

    vis_reward_tokenizer = AutoTokenizer.from_pretrained(vis_reward_model_id)
    vis_reward_tokenizer.pad_token = vis_reward_tokenizer.eos_token

    print("Model and processor loaded successfully!")


    # Image constraints
    IMAGE_FACTOR = 28
    MIN_PIXELS = 4 * 28 * 28
    MAX_PIXELS = 16384 * 28 * 28
    MAX_RATIO = 200

    # Helper functions
    def round_by_factor(number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor


    def ceil_by_factor(number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor


    def floor_by_factor(number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor


    def smart_resize(
        height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
    ) -> tuple[int, int]:
        
        if max(height, width) / min(height, width) > MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
        return h_bar, w_bar


    def to_rgb(pil_image: IM.Image) -> IM.Image:
        if pil_image.mode == 'RGBA':
            white_background = IM.new("RGB", pil_image.size, (255, 255, 255))
            white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
            return white_background
        else:
            return pil_image.convert("RGB")


    def fetch_image(ele: dict[str, str | IM.Image], size_factor: int = IMAGE_FACTOR) -> IM.Image:
        if "image" in ele:
            image = ele["image"]
        else:
            image = ele["image_url"]
        image_obj = None
        if isinstance(image, IM.Image):
            image_obj = image
        elif image.startswith("http://") or image.startswith("https://"):
            response = requests.get(image, stream=True)
            image_obj = IM.open(BytesIO(response.content))
        elif image.startswith("file://"):
            image_obj = IM.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                image_obj = IM.open(BytesIO(data))
        else:
            image_obj = IM.open(image)
        if image_obj is None:
            raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
        image = to_rgb(image_obj)
        ## resize
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=size_factor,
            )
        else:
            width, height = image.size
            min_pixels = ele.get("min_pixels", MIN_PIXELS)
            max_pixels = ele.get("max_pixels", MAX_PIXELS)
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        image = image.resize((resized_width, resized_height))

        return image


    def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
        vision_infos = []
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                        ):
                            vision_infos.append(ele)
        return vision_infos


    def process_vision_info(
        conversations: list[dict] | list[list[dict]],
    ) -> tuple[list[IM.Image] | None, list[torch.Tensor | list[IM.Image]] | None, Optional[dict]]:

        vision_infos = extract_vision_info(conversations)
        ## Read images or videos
        image_inputs = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            else:
                raise ValueError("image, image_url or video should in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        return image_inputs
    

    def distance(x1, x2):
        return min(1, abs((x1 - x2) / (x1 + 1e-15)))

    def compute_cost_matrix(a1, a2):
        cost_matrix = np.zeros((len(a1), len(a2)))
        for i, v1 in enumerate(a1):
            for j, v2 in enumerate(a2):
                cost_matrix[i, j] = distance(v1, v2)
        return cost_matrix

    def compute_score(lst1, lst2):
        cost_matrix = compute_cost_matrix(lst1, lst2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        cost = cost_matrix[row_ind, col_ind].sum()
        return 1 - cost / max(len(lst1), len(lst2))

    def remove_strings(lst):
        new_lst = []
        for elt in lst:
            s = str(elt).replace("%", "")
            try:
                new_lst.append(float(s))
            except:
                continue
        return new_lst

    def extract_answer_code(text):
        # Extract the answer
        answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)

        # Try to extract triple double quotes """..."""
        code_match = re.search(
            r'"code"\s*:\s*"""\s*(.*?)\s*"""', text, re.DOTALL
        )

        # Try to extract triple single quotes '''...'''
        if not code_match:
            code_match = re.search(
                r'"code"\s*:\s*\'\'\'\s*(.*?)\s*\'\'\'', text, re.DOTALL
            )

        # Try escaped code string inside double quotes
        if not code_match:
            code_match = re.search(
                r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL
            )

        # Decode or just return raw
        if code_match:
            raw_code = code_match.group(1)
            try:
                # Convert \n and \t to real characters
                code = bytes(raw_code, "utf-8").decode("unicode_escape")
            except:
                code = raw_code
        else:
            code = None

        answer = answer_match.group(1) if answer_match else None

        return answer, code


    def safe_int(val, default=0):
        try:
            return int(str(val).strip())
        except:
            return default


    def custom_show(image_path):
        fig_nums = plt.get_fignums()
        if fig_nums:
            fig = plt.figure(fig_nums[0])
            fig.savefig(image_path)
            plt.close(fig)

    def execute_and_save_code(code, image_path):
        plt.close("all")
        try:
            exec_env = {
                "plt": plt,
                "np": np,
                "input": lambda *args, **kwargs: "",
            }

            exec_env["plt"].show = lambda: custom_show(image_path)
            exec(code, exec_env)

            if not os.path.exists(image_path):
                custom_show(image_path)

            return 0.5 if os.path.exists(image_path) else 0

        except Exception as e:
            return 0


    # kwargs-wise reward evaluation function
    def evaluate_sample_reward(completion, **kwargs):
        image_path = os.path.join(
            args_dct["image_save_path"],
            f"{str(kwargs['ID'][0])}_index{kwargs['completion_index']}.png",
        )

        answer, code = extract_answer_code(completion)
        executable_score = execute_and_save_code(code, image_path)

        prompt_text = (
            "You are an expert evaluator for data science questions and visualizations.\n\n"
            "Given the following inputs:\n\n"
            f"Question:\n{kwargs['Question']}\n\n"
            f"Ground Truth Answer:\n{kwargs['Answer']}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            f"Ground Truth Code:\n{kwargs['Visualization Code']}\n\n"
            f"Generated Code:\n{code}\n\n"
            "Evaluate two things:\n\n"
            "1. Answer Match — Return 1 if the generated answer is semantically the same as the ground truth.\n"
            "   Allow small differences (e.g., minor rounding, % vs. number, small typos). Else, return 0. "
            "For example, if the difference is less than 0.5, consider this as a match.\n"
            "2. Code Intent Match — Return 0.5 if the code aligns with the question’s intent and is readable.\n"
            "   It should use the correct data and produce a relevant chart. Else, return 0.\n\n"
            "Return only this JSON:\n"
            '{\n'
            '  "answer_score": 0 or 1,\n'
            '  "code_intent_score": 0 or 0.5\n'
            '}'
        )

        try:
            messages = [{"role": "user", "content": prompt_text}]

            text = text_reward_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = text_reward_tokenizer([text], return_tensors="pt").to("cuda")

            generated_ids = text_reward_model.generate(**model_inputs, max_new_tokens=1024)
            generated_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            content = text_reward_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            if content.startswith("{") or content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(content)
            answer_score = safe_int(parsed.get("answer_score", 0))
            code_intent_score = float(parsed.get("code_intent_score", 0))

        except Exception:
            answer_score, code_intent_score = 0, 0

        # ── TWO-STEP FALLBACK ──
        if answer_score == 0:
            try:
                gt_nums = remove_strings([kwargs["Answer"]])
                gen_nums = remove_strings([answer])
                if gt_nums and gen_nums:
                    answer_score = compute_score(gt_nums, gen_nums)
            except Exception:
                pass

        # Continuous shaping
        shape = answer_score
        binary = 1.0 if answer_score >= 0.999 else 0.0
        α = 0.6

        # Visual evaluation
        if executable_score == 0:
            readability_score, visual_score = 0, 0
        else:
            try:
                prompt_img = (
                    f"You are a data visualization expert.\n\n"
                    f"Query: \"{kwargs['Question']}\"\n"
                    f"Data Table: {kwargs['Table Data']}\n\n"
                    "Rate the following two aspects of the generated visualization image:\n\n"
                    "1. **Readability Score (1 to 5)**:\n"
                    "   - 5 = Excellent (well-structured, clear fonts, proper labels, perfect layout)\n"
                    "   - 4 = Good (minor spacing or color issues but still readable)\n"
                    "   - 3 = Average (some clutter or overlapping, unclear axes/legends)\n"
                    "   - 2 = Poor (difficult to interpret, missing labels or inconsistent layout)\n"
                    "   - 1 = Very Poor (unreadable, messy or mislabeled chart)\n\n"
                    "2. **Visual Correctness Score (1 to 5)**:\n"
                    "   - 5 = Fully correct (matches the query perfectly, shows the right data)\n"
                    "   - 4 = Mostly correct (minor mismatch or missing info but still useful)\n"
                    "   - 3 = Partially correct (some correct elements, others wrong or misleading)\n"
                    "   - 2 = Incorrect (does not match the query or data, misleading visuals)\n"
                    "   - 1 = Completely incorrect (irrelevant, empty, or nonsensical chart)\n\n"
                    "Return ONLY this JSON:\n"
                    "{\n"
                    "  \"readability_score\": <1–5>,\n"
                    "  \"visual_correctness_score\": <1–5>\n"
                    "}"
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": prompt_img},
                        ],
                    }
                ]

                text = vis_reward_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                image_inputs = process_vision_info(messages)
                inputs = vis_reward_processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                generated_ids = vis_reward_model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                vis_content = vis_reward_processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                p1 = r'"readability_score":\s*(\d+)'
                p2 = r'"visual_correctness_score":\s*(\d+)'

                try:
                    readability_score = float(re.findall(p1, str(vis_content))[0])
                except:
                    readability_score = 0.0

                try:
                    visual_score = float(re.findall(p2, str(vis_content))[0])
                except:
                    visual_score = 0.0

            except Exception:
                readability_score, visual_score = 0.0, 0.0

        answer_reward = α * shape + (1 - α) * binary

        # Code reward
        if executable_score == 0.5 and code_intent_score == 0.5:
            code_reward = 1.0
        elif executable_score == 0.5 or code_intent_score == 0.5:
            code_reward = 0.5
        else:
            code_reward = 0.0

        # Visual reward
        if readability_score >= 4 and visual_score >= 4:
            visual_reward = 1.0
        elif readability_score >= 3 and visual_score >= 3:
            visual_reward = 0.5
        elif readability_score >= 3 or visual_score >= 3:
            visual_reward = 0.25
        else:
            visual_reward = 0.0

        return 0.5 * answer_reward + 0.25 * code_reward + 0.25 * visual_reward

    # Reward functions used by GRPO for scoring model completions
    def reward_func(completions, **kwargs):
        rewards = []

        for i, completion in enumerate(completions):
            content = completion[0]["content"]
            try:
                r = evaluate_sample_reward(
                    content,
                    **kwargs,
                    completion_index=i + 1,
                )
                rewards.append(r)
            except Exception:
                rewards.append(0.0)

        return rewards


    def format_reward(completions, **kwargs):
        rewards = []

        for completion in completions:
            content = completion[0]["content"]
            try:
                parsed = json.loads(content)
                if (
                    isinstance(parsed, dict)
                    and "answer" in parsed and isinstance(parsed["answer"], str)
                    and "code" in parsed and isinstance(parsed["code"], str)
                ):
                    code = parsed["code"].strip()
                    if code.endswith("plt.show()"):
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                else:
                    rewards.append(0.0)
            except Exception:
                rewards.append(0.0)

        return rewards


    # Configure GRPO training hyperparameters
    training_args = GRPOConfig(
        output_dir=args_dct["output_dir"],
        learning_rate=1e-5,
        optim="adamw_torch",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        num_generations=8,
        num_train_epochs=1,
        max_prompt_length=512,
        max_completion_length=2048,
        bf16=True,
        remove_unused_columns=False,
        warmup_ratio=0.03,
        weight_decay=0.1,
        save_steps=30,
        logging_steps=10,
        logging_strategy="steps",
        report_to="tensorboard",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.1,
    )

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_func,
            format_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Locate the latest checkpoint if it exists
    checkpoint_dir = training_args.output_dir
    checkpoint_path = None

    if os.path.isdir(checkpoint_dir):
        checkpoints = [
            d for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("checkpoint-")
        ]

        if checkpoints:
            try:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                potential_path = os.path.join(checkpoint_dir, latest_checkpoint)

                if (
                    os.path.exists(os.path.join(potential_path, "trainer_state.json")) or
                    os.path.exists(os.path.join(potential_path, "pytorch_model.bin")) or
                    os.path.exists(os.path.join(potential_path, "latest"))
                ):
                    checkpoint_path = potential_path
                    print(f"Found potential checkpoint: {checkpoint_path}")
                else:
                    print(f"Directory {potential_path} exists but missing key checkpoint files.")

            except (ValueError, IndexError):
                print(f"Could not parse step number from checkpoint directories in {checkpoint_dir}")

        else:
            print(f"Output directory '{checkpoint_dir}' exists but contains no checkpoint directories.")

    else:
        print(f"Output directory '{checkpoint_dir}' does not exist. Starting training from scratch.")


    # Start training (resume if checkpoint was found)
    if checkpoint_path:
        print(f"Resuming training from: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print("No valid checkpoint found. Starting training from scratch.")
        trainer.train()


    # Save final model
    trainer.save_model(os.path.join(args_dct["output_dir"], "trained_model"))
    print("Final model saved!")
    