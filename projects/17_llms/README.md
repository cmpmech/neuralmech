# Large Language Models

Toy examples for **Chapter 17 (Large Language Models)**: tokenization, the attention
mechanism built from scratch in `transformer_from_scratch.py` (`SelfAttention`,
`MultiHeadAttention`, `EfficientMultiHeadAttention`, `TransformerBlock`), and a small
pretrained transformer (T5) loaded through Hugging Face.

## Drivers

- `tokenization.py`
  byte-pair encoding of a sentence with the GPT-2 tokenizer of `tiktoken`
- `attention_timing.py`
  execution time of multi-head attention as a loop over heads, with all heads in one
  batched matrix multiplication, and as PyTorch's `nn.MultiheadAttention`
- `positional_encoding.py`
  sinusoidal positional encoding over positions and embedding dimensions
- `t5_translation.py`
  English-to-German translation with T5, via `generate` and via a `pipeline`
- `attention_kernel.py`
  attention as a soft lookup: a query weighs key-value samples of a sine, from a broad
  average (high temperature) to the nearest key (low temperature)
- `attention_maps.py`
  cross-attention and encoder self-attention weights of T5 translating "In winter, it
  snows.", merged from subword tokens to words
- `ai_scaling_trends.py`
  downloads compute, data, and parameter counts of notable AI systems (Our World in Data)
  and training costs of frontier models (Epoch AI), and fits exponential trends
- `reasoning_model.py`
  a small reasoning model (Qwen3-1.7B) counts the letter e in the book title; its reasoning
  above a dashed line and its answer below, as chat bubbles for the book
- `selective_copying.py`
  copying and selective copying from the Mamba paper, learned from scratch by an LSTM,
  S4D, Mamba, and a transformer; final accuracy against sequence length (16 to 512) and
  the transformer's attention maps of both tasks at length 128
- `vit_inference.py`
  a pretrained vision transformer (ViT-B/16) classifies a photo; attention rollout of the
  class token over the 16 x 16 patches
- `clip_zero_shot.py`
  zero-shot classification of three photos and three images from computational mechanics
  (finite element field, CT slice, fiber microstructure) by CLIP against six text prompts
- `whisper_transcription.py`
  _needs external_data/isengard/isengard.wav_ log-mel spectrogram and transcription of the
  line "They're taking the hobbits to Isengard" (The Lord of the Rings: The Two Towers, 2002)
  by Whisper large-v3 turbo
- `time_series_forecasting.py`
  zero-shot forecasts of a quasi-periodically forced Duffing oscillator and the Lorenz system
  by Timestral, Chronos-2, and TimesFM 2.5
- `rag_t5.py`
  retrieval-augmented generation over the driver descriptions of all project READMEs:
  Sentence-T5 embeddings for retrieval, Flan-T5 for the answer, with and without retrieval
- `vlm_description.py`
  a small vision-language model (Qwen3-VL-2B) describes a finite element displacement field
- `agents/engineering_agent.py`
  _needs the packages in `agents/requirements.txt`, cuvox, and a GPU_ language model agents
  (smolagents, open models served by vLLM) design a powder bed fusion bracket through the tools
  of `agents/helper.py`: specification check, topology optimization (cuvox), STL extraction,
  CAD interfaces (build123d), re-analysis, and manufacturing checks; success rate per model

## Non-obvious technicalities (authored by Claude)

**Timing on the GPU.** CUDA kernels launch asynchronously, so `time.perf_counter()`
alone measures the launch, not the computation. `attention_timing.py` calls
`torch.cuda.synchronize()` before reading the clock, runs a warm-up call first, and
keeps the best of several interleaved rounds, so that background load on the machine
does not favor one variant.

**Dot-product attention as a kernel.** `attention_kernel.py` embeds a scalar
position $x$ as the unit vector $(\cos\pi x, \sin\pi x)$. The dot product of two such
vectors is $\cos(\pi(x_q-x_k))$, which decreases with the distance $|x_q-x_k|$ on
$[0,1]$, so the softmax of the dot products divided by a temperature $T$ is a
normalized kernel around the query. Large $T$ averages all values, small $T$ returns
the value of the nearest key, as hard attention does.

**Attention maps of T5.** The weights are those of the fifth of six layers, averaged
over the eight heads. Row $r$ of the decoder's cross-attention is the query that
produces output token $r+1$. Subword tokens are merged into words by averaging the
rows of a word's query tokens and summing the columns of its key tokens, which keeps
every row a probability distribution. The models are read from the Hugging Face cache;
set `HF_HUB_OFFLINE=1` when the machine has no network access.

**Sequence models in `selective_copying.py`.** The transformer is the 
`TransformerDecoder` of `NN.py` (PyTorch's `TransformerEncoderLayer` with a causal mask). S4D and
Mamba replace the attention module inside the equivalent post-norm block of
`transformer_from_scratch.py`, so all models except the LSTM differ only in the
sequence-mixing layer, and all four are causal. S4D uses a diagonal complex $\mathbf{A}$ with S4D-Lin
initialization ($a_n=-\tfrac12+i\pi n$), zero-order-hold discretization, and evaluates
its kernel as a convolution via FFT. Mamba computes $\Delta t$, $\mathbf{B}$, and
$\mathbf{C}$ from the input, preceded by a short causal depthwise convolution and
followed by a SiLU gate, and runs its recurrence as a plain loop over the sequence
(fine for up to 520 tokens, the slowest part of the sweep). The bias of $\Delta t$ is initialized as the inverse softplus of
a log-uniform sample in $[10^{-3},10^{-1}]$; with the default initialization
$\Delta t\approx0.7$, the state forgets within a few steps and neither task is learned.
The recurrence loops over `unbind(1)` slices rather than indexing `[:, tau]`: the
backward pass of an index fills a full-size zero tensor per step, which made training
five times slower.
