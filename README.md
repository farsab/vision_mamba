# Vision Mamba Edge Detector & LLM Inference Benchmark

## Description
This module adds two utilities to our Vision Mamba & LLM library:

1. **Edge Detector** (`vision_mamba.edge_detector`): Canny edge detection for image preprocessing.
2. **LLM Inference Benchmark** (`llm_toolkit.inference_benchmark`): Measures average inference time of a `t5-small` seq2seq model.

Each utility is designed to slot into larger pipelines—Vision Mamba for CV tasks and HuggingFace Transformers for LLM workflows.

---

## 📂 Project Structure
.
├── vision_mamba

│ └── edge_detector.py

├── llm_toolkit

│ └── inference_benchmark.py

├── main.py



---

## 🚀 Usage

mkdir -p sample_images
cp ./image.jpg sample_images/example.jpg
python main.py

## Output
edge_output.png — the Canny edge map of your image

Console log — average LLM inference time


