import torch
from torch import nn

PROBES = {
    "code": (
        "Write the Python lines that create a 2D cuwave ScalarWave simulation whose low face "
        "of axis 0 is a pressure-release wall while all other faces stay reflecting. "
        "Nx, dx, N, dt and THREADS are already defined."
    ),
    "fd": "What limits the time step of the explicit scheme for the 1D wave equation u_tt = c^2 u_xx, and why?",
    "sam": "Why does Sam carry Frodo up Mount Doom?",
    "author": "Who wrote The Lord of the Rings?",
    "nazgul": "What are the Nazgul?",
}


def chat(tokenizer, question):
    """chat prompt of a single user turn, with thinking switched off."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


class LoRA(nn.Module):
    """frozen linear layer plus trainable low-rank update B A, scaled by alpha / rank.

    B starts at zero, so training starts from the pretrained model.
    """

    def __init__(self, base, rank, alpha=32):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (self.scale * (x.float() @ self.A.T) @ self.B.T).to(x.dtype)


def add_lora(model, rank):
    """freeze the model and wrap every attention and feedforward projection in `LoRA`."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for layer in model.model.layers:
        for block in (layer.self_attn, layer.mlp):
            for name in names:
                if hasattr(block, name):
                    setattr(block, name, LoRA(getattr(block, name), rank))


@torch.no_grad()
def answer(model, tokenizer, question, max_tokens):
    """greedy answer to a single question."""
    ids = tokenizer(chat(tokenizer, question), return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
