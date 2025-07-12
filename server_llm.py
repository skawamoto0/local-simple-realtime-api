from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from typing import List, Dict

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: Modelを動的に選択し、reloadするように修正
model_id = "google/gemma-3-4b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
)
tokenizer.add_special_tokens({"additional_special_tokens": ['<end_of_turn>']})

torch._dynamo.config.cache_size_limit = 1024
torch.set_float32_matmul_precision('high')

@app.post("/llm/stream")
async def generate_stream(messages: List[Dict[str, str]]):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=256)
    model.generate(**generation_kwargs)
    return streamer

@app.post("/llm")
async def generate(messages: List[Dict[str, str]], stream: bool = False):
    if stream:
        return await generate_stream(messages)
    else:
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, return_dict=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        generated_text = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        return {"message": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)