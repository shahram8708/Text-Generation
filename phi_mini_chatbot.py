import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import gradio as gr

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def chat_with_model(user_input):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_input}
    ]

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    
    return output[0]['generated_text']

iface = gr.Interface(
    fn=chat_with_model,
    inputs=gr.Textbox(placeholder="Type your message here..."),
    outputs="text",
    title="Chatbot Interface",
    description="Ask me anything! I am a helpful AI assistant."
)

iface.launch()
