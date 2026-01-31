import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import argparse
import sys
import os

def chat_with_model(checkpoint_path, base_model_name="Qwen/Qwen2.5-1.5B", device="cpu"):
    print(f"Loading base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA checkpoint from: {checkpoint_path}...")
    try:
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print("✅ LoRA adapter loaded successfully!")
    except Exception as e:
        print(f"⚠️  Could not load LoRA adapter directly: {e}")
        print("Attempting to load as full model...")
        try:
             model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map=device)
             print("✅ Full model loaded!")
        except:
            print("❌ Failed to load checkpoint.")
            return

    model.eval()
    
    print("\n" + "="*50)
    print("🤖 Chat with your SFT Model (Type 'quit' to exit)")
    print("="*50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        # Format input (using Qwen's chat template if available, else raw)
        try:
            messages = [{"role": "user", "content": user_input}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            text = f"User: {user_input}\nAssistant:"
            
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Model: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder or file")
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    args = parser.parse_args()
    
    # If checkpoint points to a file (model.pt), get the directory
    if os.path.isfile(args.checkpoint):
        args.checkpoint = os.path.dirname(args.checkpoint)
        
    chat_with_model(args.checkpoint, device=args.device)
