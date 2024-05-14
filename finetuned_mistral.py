from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "shivam-dubey-1/demo-ads-eks"  # Replace with the path to your model

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype='auto').eval()

class Mistral7B:
    def chat_completion(self, system_prompt, user_prompt, length=1000):
        final_prompt = f"<s>[INST]<<SYS>>{system_prompt}<</SYS>> {user_prompt}[/INST]</s>"
        input_ids = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": final_prompt}], tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'), max_length=length, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response

if __name__ == '__main__':
    system_prompt = "Create a text ad given the following product and description."
    user_prompt = "Product: Lightweight Bicycle Description: Perfect bike for any road conditions. this bike comes with durable tires, LCD display, and modern disc brakes"

    mistral = Mistral7B()  # Create an instance of the class
    print(mistral.chat_completion(system_prompt, user_prompt))  # Call the chat_completion method on the instance
