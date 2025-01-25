
# Llama 3.2 1B - PyTorch Implementation

Welcome to the **Llama 3.2 1B** repository. This project will break down  `Llama 3.2 1B` model to its bare weights and explain you how each step works at the pytorch level


## Getting Started

Follow these instructions to set up and run the Llama 3.2 1B model on your local machine.

### Prerequisites

- **Python 3.10+**
- **PyTorch 2.0+**
- **CUDA-enabled GPU** (recommended for training)
- **Jupyter Notebook** (for running the provided notebook)

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

### Installation

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/YourUsername/Llama3.2_1B_pytorch.git
cd Llama3.2_1B_pytorch
```

### Running the Notebook

A Jupyter notebook (`Llama3.2_1B.ipynb`) is provided to demonstrate model setup, quantization, and inference.

1. **Start the Jupyter server**:

   ```bash
   jupyter notebook
   ```

2. **Open** `Llama3.2_1B_Demo.ipynb` and follow the step-by-step instructions.

### Model Inference - General method

To perform inference with the quantized Llama 3.2 1B model (the normal method):

1. **Load the Model**:

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   import torch

   model_name = "meta-llama/Llama-3.2-1B" 
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

   ```

2. **Generate Text**:

   ```python
   prompt = "The future of artificial intelligence is"
   inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
      output = model.generate(
      inputs["input_ids"],
      max_length=50, 
      num_return_sequences=1, 
      temperature=0.7,
   )
   generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
   print(generated_text)
   ```

### Hardware Recommendations

For optimal performance, especially during training, it is recommended to use cloud GPUs such as T4.

## Repository Structure

```
Llama3.2_1B_pytorch/
├── configs/                # Layes configuration understanding 
│   └── llamaconfig.json
│   └── llamalayer.txt
│   └── excalidraw.json     # Excalidraw to visualize
├── notebooks/              # Jupyter notebooks for demonstrations
│   └── Llama3.2_1B.ipynb
|   └── workingon/
│       └── Llama3.1_8B.ipynb
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Contributing

We welcome contributions to enhance this project. Please fork the repository, create a new branch for your feature or bug fix, and submit a pull request for review.

## License

This project is licensed under the Llama 3.2 Community License. For more details, refer to the `LICENSE` file in its [huggingface repository](https://huggingface.co/meta-llama/Llama-3.2-1B).
