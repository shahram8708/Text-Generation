import gradio as gr
import transformers
import torch
from huggingface_hub import login

login(token='HuugingFace_Tokens') 

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
 
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def chatbot_response(user_input):
    outputs = pipeline(
        user_input,  
        max_new_tokens=150,  
        do_sample=True, 
        temperature=0.7 
    )
    
    return outputs[0]["generated_text"]

def gradio_chatbot_interface():
    interface = gr.Interface(
        fn=chatbot_response,  
        inputs="text",  
        outputs="text", 
        title="Pirate Chatbot",  
        description="Talk to a pirate chatbot!", 
        theme="compact", 
    )
    
    interface.launch()

gradio_chatbot_interface()
