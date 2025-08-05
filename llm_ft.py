!pip install peft
!pip install accelerate
!pip install bitsandBytes

!pip install transformers

!pip install GPUtil

import torch
import os
import GPUtil

GPUtil.showUtilization()

if torch.cuda.is_available():
      print("gpu")
else:
      print("cpu")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,AutoModelForCausalLM,LlamaTokenizer
from huggingface_hub import notebook_login
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model,PeftModel

if "COLAB_GPU" in os.environ:
  from google.colab import output
  output.enable_custom_widget_manager()

if "COLAB_GPU" in os.environ:
  !huggingface-cli login
else:
  notebook_login()

base_model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True)

if tokenizer.pad_token is None:
  tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

tokenized_train_dataset = []
with open('/content/TechNova_Profile.json', 'r') as file:
  for line in file:
    tokenized_train_dataset.append(tokenizer(line.strip()))

tokenized_train_dataset[1]

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    args=transformers.TrainingArguments(
        output_dir="./finetunedModel",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=20,
        learning_rate=1e-4,
        max_steps=20,
        bf16=False,
        optim="paged_adamw_8bit",
        logging_dir="./log",
        save_strategy="epoch",
        save_steps=50,
        logging_steps=10

),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache=False
trainer.train()

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer

base_model_id = "meta-llama/Llama-2-7b-chat-hf"

nf4Config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config
    =nf4Config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

from peft import PeftModel

tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True)

modelFinetuned = PeftModel.from_pretrained(base_model, "finetunedModel/checkpoint-20")

user_question = " Services Offered by company"

eval_prompt = f"""Based on the provided information, please provide the details of Technova Solutions INC related to:
{user_question}

"""

promptTokenized = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

modelFinetuned.eval()

with torch.no_grad():
  print(tokenizer.decode(modelFinetuned.generate(**promptTokenized, max_new_tokens=1024)[0], skip_special_tokens=True))

