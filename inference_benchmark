import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time
from typing import List

def benchmark_inference(
    model_name: str = "t5-small",
    texts: List[str] = None,
    device: torch.device = None
) -> float:
    """
    Benchmark the inference speed of a seq2seq model.
    
    Args:
        model_name: HuggingFace model identifier.
        texts: List of input strings to generate on. If None, uses a repeated sample.
        device: torch.device; defaults to CUDA if available.
        
    Returns:
        Average inference time per sample (seconds).
    """
    if texts is None:
        texts = ["transformers are amazing"] * 20
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in encoded.items()}

    # Warm-up
    with torch.no_grad():
        model.generate(**inputs)

    # Timing
    start = time.time()
    with torch.no_grad():
        model.generate(**inputs)
    elapsed = time.time() - start
    return elapsed / len(texts)
