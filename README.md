# GPT-2 Chatbot Training and Inference

This project includes scripts for training and running inference with GPT-2.

## Features
- Train GPT-2 models (variants: gpt2, gpt2-medium, gpt2-large, gpt2-xl).
- Generate chatbot responses using the trained models.

## Setup
1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Training the Model
Use the `train.py` script to train a new GPT-2 model.

Example:
```
python train.py --data_dir ./data/my_data.txt --model_variant gpt2-large --output_dir ./models/gpt2-large-model --num_train_epochs 3 --batch_size 8
```

## Generating Responses
Use the `inference.py` script to generate responses from a trained model.

Example:
```
python inference.py --model_path ./models/gpt2-large-model --prompt "Hello, how are you?" --max_length 50
```

## Data Format
The training data should be in a text file or pdf file