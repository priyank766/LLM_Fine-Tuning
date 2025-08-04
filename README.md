# LLM Fine-Tuning Project

## Project Overview

This project demonstrates the fine-tuning of the LLaMA-2-7B model using the Hugging Face Transformers library. The fine-tuning process leverages techniques such as LoRA (Low-Rank Adaptation) and 4-bit quantization to optimize the model for specific tasks while reducing computational requirements. The project is designed to run on GPU-enabled environments and includes:

- Tokenization of input data.
- Model preparation for k-bit training.
- Fine-tuning using the Trainer API.
- Evaluation of the fine-tuned model with custom prompts.

## Key Features

- **Model**: LLaMA-2-7B (meta-llama/Llama-2-7b-chat-hf).
- **Quantization**: 4-bit quantization using BitsAndBytesConfig.
- **Fine-Tuning**: LoRA-based fine-tuning for efficient adaptation.
- **Dataset**: Custom JSON dataset for training.
- **Evaluation**: Prompt-based evaluation of the fine-tuned model.

## Dependencies

The project requires the following Python libraries:

- `transformers`
- `peft`
- `accelerate`
- `bitsandbytes`
- `GPUtil`

## How to Run

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare your dataset and place it in the appropriate location.
3. Run the script `llm_ft.py` to fine-tune the model.
4. Evaluate the fine-tuned model using the provided evaluation prompt.

## WandB Integration

This project integrates with Weights & Biases (WandB) for experiment tracking and visualization. WandB allows you to monitor training metrics, visualize model performance, and compare different runs easily.

### How to Enable WandB

1. Install the WandB library:
   ```bash
   pip install wandb
   ```
2. Log in to your WandB account:
   ```bash
   wandb login
   ```
3. Modify the training script to include WandB integration. For example:
   ```python
   import wandb
   wandb.init(project="llm-fine-tuning")
   ```
4. Training metrics and logs will automatically be sent to your WandB dashboard.

## Comparison: RAG vs Fine-Tuning

### Retrieval-Augmented Generation (RAG)

- **Definition**: Combines a pre-trained language model with an external knowledge base to generate responses.
- **Advantages**:
  - No need for extensive fine-tuning.
  - Dynamically retrieves relevant information from the knowledge base.
  - Scalable for large and frequently updated datasets.
- **Disadvantages**:
  - Dependency on the quality and availability of the external knowledge base.
  - Slower response times due to retrieval overhead.

### Fine-Tuning

- **Definition**: Adapts a pre-trained language model to a specific task or domain by updating its weights.
- **Advantages**:
  - Tailored responses for specific tasks or domains.
  - Faster inference as no external retrieval is required.
  - Works offline without dependency on external systems.
- **Disadvantages**:
  - Requires computational resources for training.
  - Needs a well-curated dataset for effective fine-tuning.

### When to Use

- **RAG**: Best for scenarios where the knowledge base is large, dynamic, or frequently updated.
- **Fine-Tuning**: Ideal for tasks requiring domain-specific expertise or when operating in offline environments.

## Project Structure

- `llm_ft.py`: Main script for fine-tuning and evaluation.
- `dataset.txt`: Placeholder for the training dataset.
- `README.md`: Project documentation.

## Acknowledgments

This project utilizes the Hugging Face Transformers library and the LLaMA-2 model by Meta AI.

## Summary

This project offered valuable insights into fine-tuning large language models using techniques like LoRA and 4-bit quantization. While the learning experience was enriching, the results are not yet ideal for real-world applications, requiring further refinement.
