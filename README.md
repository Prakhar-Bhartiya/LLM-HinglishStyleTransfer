# LLM Hinglish Style Transfer

This repository contains code and resources for fine-tuning large language models (LLMs) to generate responses in Hinglish – a casual mix of Hindi and English. The focus is on adapting models to understand and generate text with a local, conversational tone, using state-of-the-art methods including LoRA (Low-Rank Adaptation) and Full Fine-Tuning.

## Contents

- **Model Fine-Tuning Notebooks:**

  - **Qwen2.5-Models-FineTuning.ipynb:** Notebook demonstrating both full fine-tuning and LoRA fine-tuning of Qwen2.5 models. It covers dataset preparation, model configuration, training, checkpointing, and generating sample responses.
  - **Evaluations.ipynb:** Contains scripts to evaluate model performance, generate comparison tables, and handle post-training cleanup.

- **Evaluation and Experimentation Scripts:**

  - **ab_test.py:** Script to perform A/B testing between different fine-tuning strategies. It evaluates model responses using statistical metrics and visualization tools to help determine the best performing approach.
  - **chat.py:** A standalone script providing an interactive chat interface, allowing users to interact with the fine-tuned model in real-time. It is designed for both terminal and UI-based interactions.

- **Environment Setup & Package Installation:**  
  Instructions and code snippets for installing required packages (e.g., PyTorch, Transformers, Datasets, Accelerate, PEFT) for different platforms including Apple Silicon and CUDA-enabled systems.

- **Visualization and Diagnostics:**  
  Code for plotting training loss and monitoring performance using tools like TensorBoard.

## Key Features

- **Hinglish Conversation Style:**  
  The fine-tuning setups are designed to produce responses in a casual Hinglish style, using colloquial language and slang common in urban India.

- **Multiple Fine-Tuning Strategies:**  
  Supports both Full Fine-Tuning and LoRA-based adaptation, enabling experimentation with various approaches to model adaptation.

- **Flexible Environment Support:**  
  Provides installation instructions and code for systems using CUDA (Nvidia GPUs) as well as Apple Silicon (M1/M2) with Metal backend acceleration.

- **Extensible Evaluation Tools:**  
  Includes notebooks and scripts to evaluate model performance, compare different fine-tuning methods, and conduct A/B testing between model variants.

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd LLM-HinglishStyleTransfer
   ```

2. **Install Dependencies:**

   Follow the instructions in the notebooks or install the following packages:

   ```bash
   pip install torch torchvision torchaudio transformers datasets accelerate peft matplotlib seaborn trl hf_xet bitsandbytes
   ```

3. **Run Notebooks and Scripts:**

   - Use Jupyter Notebook or VS Code to open and run `Qwen2.5-Models-FineTuning.ipynb` and `Evaluations.ipynb`.
   - Run the `ab_test.py` script to perform comparative evaluation of fine-tuning strategies.
   - Use the `chat.py` script for an interactive session with the fine-tuned model.

## Contributing

Contributions are welcome. Please ensure that any modifications include proper attribution and follow the terms outlined in the LICENSE.

## License

This project is licensed under the [Apache License 2.0](./LICENSE).

---
