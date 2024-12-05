from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained model (DialoGPT-small)
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chatbot():
    print("Chatbot is ready! Type 'exit' to quit.\n")
    
    # Keep the conversation going until 'exit' is typed
    while True:
        user_input = input("You: ")
        
        # Exit condition
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        try:
            # Encode the user input and append the EOS token (end of sentence)
            input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

            # Generate a response based on the input
            response_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)

            # Decode the generated response and remove special tokens
            response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
            
        except Exception as e:
            response = "I'm sorry, I couldn't process that."
            print(f"Error: {e}")

        # Output the chatbot's response
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
