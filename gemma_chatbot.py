from huggingface_hub import login
login(token="Hugging_Tokens")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr  

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",            
    torch_dtype=torch.bfloat16,   
)

def generate_response(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda") 
    outputs = model.generate(**input_ids, max_new_tokens=1024)  
    response = tokenizer.decode(outputs[0], skip_special_tokens=True) 
    return response

with gr.Blocks() as demo:
    gr.Markdown("<h1 align='center'>🤖 Machine Learning Chatbot</h1>")
    chatbot = gr.Chatbot() 
    msg = gr.Textbox(label="Your Message")  
    clear = gr.Button("Clear")  

    def respond(message, chat_history):
        bot_response = generate_response(message)
        chat_history.append((message, bot_response)) 
        return chat_history, ""

    msg.submit(respond, [msg, chatbot], [chatbot, msg]) 
    clear.click(lambda: None, None, chatbot)  

demo.launch()
