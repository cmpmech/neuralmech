import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "How much pasta should I cook? <|endoftext|>"

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
strings = [tokenizer.decode([i]) for i in integers]

for i, s in zip(integers, strings):
    print(f"{s!r}: {i}")
