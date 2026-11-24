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
- `open_weight_models.py`
  Epoch Capabilities Index of open-weight model families against the memory of their
  weights at 16 bits, with release dates; parameter counts from the model cards
- `reasoning_model.py`
  a small reasoning model (Qwen3-1.7B) counts the letter e in the book title; its reasoning
  above a dashed line and its answer below, as chat bubbles for the book
- `post_training.py`
  OLMo 2 (7B) after each training stage (base model at several temperatures, SFT, DPO,
  instruct) asked why the eagles don't fly the Ring to Mordor, and 16 samples per stage of
  three mathematics problems (integral of x exp(x) from 0 to 1, a train trip in minutes, the
  divisors of 360); Qwen3-8B before and after distillation on the same integral; answers with
  token counts, the correct samples in the book are counted by hand
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
- `jepa_maze.py`
  an action-conditioned joint embedding predictive architecture learns the 12 x 12 maze of the
  chapter 1 probes from random moves; breadth-first search over its predicted moves finds the
  route
- `distillation/teacher.py`
  _needs vLLM, external_data/distillation/cuwave and langtangen.txt_ a teacher (gpt-oss-20b)
  writes question-answer pairs on the cuwave codebase and on finite differences
  -> external_data/distillation/qa_raw.json
- `distillation/check.py`
  _needs cuwave's dependencies (cupy) and a GPU_ runs every code snippet of the pairs against
  cuwave and keeps the pairs whose code runs -> external_data/distillation/qa.json
- `distillation/distillation.py`
  _needs qa.json_ sequence-level distillation: LoRA fine-tuning of Qwen3-1.7B on the pairs
  (`--small`: Qwen3-0.6B, trained more aggressively); answers before and after training to
  cuwave, finite-difference, and Lord of the Rings questions
- `agents/engineering_agent.py`
  _needs the packages in `agents/requirements.txt`, cuvox, and a GPU_ language model agents
  (smolagents, open models served by vLLM) design a powder bed fusion bracket through the tools
  of `agents/helper.py`: specification check, topology optimization (cuvox), STL extraction,
  CAD interfaces (build123d), re-analysis, and manufacturing checks; success rate per model
- `task_horizon.py`
  downloads METR's time-horizon results (length of the software tasks that agents complete in
  half of the attempts) and fits the doubling trends since 2019 and since 2023
- `steering.py`
  interpretability of Qwen3-1.7B with the methods of neuroscience: a difference-of-means probe
  per layer separates Tolkien-style from mechanics-textbook sentences (recording), adding that
  direction to one layer pulls answers on mechanics toward Middle-earth (stimulation), and
  projecting it out of the weights removes the Tolkien style (lesion)
- `censorship.py`
  a direction for topics censored in China (Qwen3-8B): probe, steering of neutral questions
  into refusals, and its ablation from the weights, cross-validated over 44 question pairs
- `world_map.py`
  _needs external_data/world_places, natural_earth_ ridge probe per layer from the hidden
  states of 10 000 place names of Qwen3-1.7B to latitude and longitude; held-out places on a
  world map, colored by their distance to the true position
- `chess_probe.py`
  _needs external_data/chess/lichess_8layers_ckpt_no_optimizer.pt and lichess_100mb.csv_ linear
  probes per square on Karvonen's chess GPT read the board layer by layer; erasing the moving
  piece (its probe direction ablated toward the mean of empty squares) changes the model's moves
- `emotion_space.py`
  44 emotion directions of Qwen3-8B from stories that it writes; principal components against
  valence and arousal, and a math question steered toward "enthusiastic", "ashamed" and "bored"

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

**Distillation data.** `teacher.py` cuts cuwave into topic slices of at most about 40k tokens:
each documentation page together with its module (or CUDA kernel file) and the two shortest
example drivers that use it, plus three chunks of chapter 2 (wave equations) of Langtangen &
Linge, *Finite Difference Computing with PDEs* (open access; `langtangen.txt` is
`pdftotext -layout` of the book PDF, and the chapter is cut by line numbers of that file). The
whole repository would not fit the teacher's context window. Each request asks for ten pairs
of one question type (concepts, how-to code, arguments, pitfalls, internals, worked numbers,
simulation setup in physical words, ...), and every request gets its own sampling seed, so
repeated requests give different pairs. `check.py` executes every Python block of an answer
after a preamble that defines a small 2D `ScalarWave` and the usual constants; a pair survives
only if all its blocks run, where an undefined lowercase or upper-case placeholder variable
counts as running, an undefined CamelCase name (an invented class) does not, and every
simulation's boundary entries must be real `BoundaryCondition` objects. The check verifies the
API, not the meaning: code that runs but sets the wrong condition passes. It also strips the
chapter's equation numbers, which the teacher cites and the student would otherwise copy.

**LoRA by hand.** `helper.LoRA` wraps every attention and feedforward projection as
$\mathbf{W}\mathbf{x}+\tfrac{\alpha}{r}\mathbf{B}\mathbf{A}\mathbf{x}$ with $\mathbf{B}=0$ at
the start, so the answers "before" are taken from the wrapped model. The low-rank path runs in
float32 on top of the bfloat16 base model. The cost is the cross-entropy on the answer tokens
only: the prompt positions carry the label -100, which PyTorch's cross-entropy ignores.
`distillation.py` trains on the workstation (gradient checkpointing keeps a batch of eight
long answers within 32 GB) and saves only the LoRA weights; when they exist, it loads them and
only answers, which runs on a 6 GB laptop GPU.
