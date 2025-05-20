# LLM Finetune Playground

This repository offers a practical and comprehensive guide to fine-tuning large language models (LLMs) for generating responses in Hinglish—a colloquial mix of Hindi and English widely spoken in urban India. While Hinglish is the chosen task for demonstration, the techniques and learnings presented here are broadly applicable and can be translated to a wide range of other tasks.

Rather than aiming to build the most accurate model, the primary goal is to serve as an accessible starting point for experimenting with fine-tuning techniques, evaluation strategies, and interactive tooling.

Key features include hands-on demonstrations of Full Fine-Tuning, LoRA (Low-Rank Adaptation), and QLoRA approaches. The repository also includes an evaluation suite for benchmarking model performance and running A/B tests. To support interactive exploration, it provides a conversational interface for engaging directly with fine-tuned models—making it a valuable resource for learning and experimentation.

## Contents

- **Model Fine-Tuning Notebooks:**

  - **Qwen2.5-Models-FineTuning.ipynb:** Notebook demonstrating Full, LoRA and QLoRA fine-tunings of Qwen2.5 models. It covers dataset preparation, model configuration, training, checkpointing, and generating sample responses.

  - **Evaluations.ipynb:** Contains scripts to evaluate model performance, generate comparison tables, and handle post-training cleanup.

- **Evaluation and Experimentation Scripts:**

  - **ab_testing.py:** Script to perform A/B testing between different fine-tuning strategies. It evaluates model responses using statistical metrics and visualization tools to help determine the best performing approach.

  - **chat.py:** A standalone script providing an interactive chat interface, allowing users to interact with the fine-tuned models in real-time. It is designed for UI-based interactions.

## Key Features

- **Hinglish Conversation Style:**  
  The fine-tuning setups are designed to produce responses in a casual Hinglish style, using colloquial language and slang common in urban India.

- **Multiple Fine-Tuning Strategies:**  
  Supports both Full Fine-Tuning and LoRA+QLoRA-based adaptation, enabling experimentation with various approaches to model adaptation.

- **Flexible Environment Support:**  
  Provides installation instructions and code for systems using CUDA (Nvidia GPUs) as well as Apple Silicon (M1/M2) with Metal backend acceleration.

- **Extensible Evaluation Tools:**  
  Includes notebooks and scripts to evaluate model performance, compare different fine-tuning methods, and conduct A/B testing between model variants.

## Getting Started

1. **Clone the Repository:**

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
