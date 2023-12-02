import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_response(model_path, prompt, max_length):
    # Load the trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # Encode the prompt and generate response
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length)

    # Decode and print the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a response from a trained GPT-2 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the model to respond to")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the response")

    args = parser.parse_args()
    generate_response(args.model_path, args.prompt, args.max_length)
