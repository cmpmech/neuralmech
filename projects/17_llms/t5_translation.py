import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# -------------------------------------- settings -------------------------------------
MODEL = "google-t5/t5-small"
TEXT = "translate English to German: In winter, it snows."

# ------------------------------------ load model -------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, dtype="float16").to("cuda")

# ------------------------------------- generation ------------------------------------
tokenized_inputs = tokenizer(TEXT, return_tensors="pt").to("cuda")
outputs = model.generate(**tokenized_inputs, do_sample=True, temperature=0.2, top_k=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# pipeline adds "translate English to German: " itself
pipe = pipeline("translation_en_to_de", model=model, tokenizer=tokenizer)
print(pipe("In winter, it snows.")[0]["translation_text"])
