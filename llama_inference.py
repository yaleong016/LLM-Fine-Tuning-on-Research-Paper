from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import torch

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

# 4-bit quantization config (safe for RTX 3090)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map="auto",
    dtype=torch.float16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
)

# -----------------------------
# MANUAL LLAMA 3 CHAT TEMPLATE
# -----------------------------
system_message = (
    "You are a thoughtful, concise, emotionally aware assistant who gives grounded, "
    "clear, psychologically informed advice."
)

user_message = (
    "Sounds good, I'll spend time coming to terms with past self by imagining "
            "the past me and comforting myself on things I couldn't control. However, I've noticed "
            "a pattern of feeling inferior and assuming people think less of me. Especially at "
            "momentum where most people are entrepreneurs working on cool and money earning projects. "
            "The social norm is networking and valuing success, productivity, and intellect. "
            "As an idealist with deep thinking artistic hobbies such as languages and piano, I feel like "
            "I don't belong. Moreover, as a deep thinker with high morals, I often feel I don't belong "
            "in society in general. I don't feel the need to show up, party for fun, date and sleep around, "
            "or chase big career success. I focus on meaning and passion. This mismatch makes me feel unvalued. "
            "I want to be proud of who I am, but I give in to self criticism and assume people think poorly of me. "
            "How can I practice self acceptance, reduce self criticism, and stop assuming others think low of me?"
)

prompt = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n"
    f"{system_message}\n"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    f"{user_message}\n"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

# Generation settings
out = pipe(
    prompt,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.15,
    eos_token_id=tok.eos_token_id
)

print(out[0]["generated_text"])
