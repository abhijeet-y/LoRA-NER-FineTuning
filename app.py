import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import ast

# Load model and tokenizer
@gr.cache()
def load_model():
    base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-360M")
    model = PeftModel.from_pretrained(base, "./lora-ner-model")
    tokenizer = AutoTokenizer.from_pretrained("./lora-ner-model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

def prepare_prompt(event_text):
    return f"""See the given example to extract the entities based on given input in JSON format.

Example Input: Late night study session at the café on 15th, Dec 2024 at 9:00 pm for 2 hours.
Example Output: {{'action': 'study session', 'date': '15/12/2024', 'time': '9:00 PM', 'attendees': None, 'location': 'café', 'duration': '2 hours', 'recurrence': None, 'notes': None}}
--------------------------
Please extract the entities for the below user input in JSON format. And do not output anything else.

Human Input: {event_text}
AI:"""

def predict(event_text):
    prompt = prepare_prompt(event_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    try:
        return ast.literal_eval(decoded.strip())
    except:
        return {"raw_output": decoded.strip()}

gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Event Text", lines=4),
    outputs=gr.JSON(label="Extracted Entities"),
    title="📅 LoRA NER Demo (SmolLM)",
    description="Enter an event-style text and get structured JSON back."
).launch()
