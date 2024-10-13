from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the fine-tuned model
model_name = "./fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token to be the same as the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Function to chat with AI
def chat_with_ai():
    print("AI: Hi! I'm your fine-tuned AI. How can I help you today?")
    chat_history = ""  # Keep track of the conversation history

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("AI: Goodbye!")
            break

        # Append user input to the chat history
        chat_history += f"You: {user_input}\n"

        # Tokenize the input and generate a response
        inputs = tokenizer(chat_history, return_tensors="pt", padding=True, truncation=True)

        # Generate a response
        response_ids = model.generate(
            inputs['input_ids'],
            max_length=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,  # Adjust temperature for coherence
            top_p=0.9,        # Keep top 90% of the probability mass
            do_sample=True,
            no_repeat_ngram_size=2,  # Prevent repetition of n-grams
            attention_mask=inputs['attention_mask']  # Pass attention mask
        )

        # Decode the generated response
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Extract only the AI's response
        ai_response = response.split("\n")[-1].strip()  # Get last line for the response

        print(f"AI: {ai_response}")

        # Update chat history with AI's response
        chat_history += f"AI: {ai_response}\n"

# Start chatting
chat_with_ai()
