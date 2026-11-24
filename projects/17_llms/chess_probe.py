import argparse
import csv
from pathlib import Path

import chess
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrow, Rectangle
from torch import nn
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

from postprocessing import save_csv

BASE_DIR = Path(__file__).parent
RESULTS_DIR = (BASE_DIR / "../../results").resolve()
CSV_DIR = (RESULTS_DIR / "data").resolve()
DATA_DIR = (BASE_DIR / "../../external_data/chess").resolve()

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--book", action="store_true")
args = parser.parse_args()

# -------------------------------------- settings -------------------------------------
# hyperparameters
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 1024

# data
GAMES = 3000
TEST_GAMES = 500
MAX_CHARS = 600  # transcript characters read per game
CHARS = sorted(" #+-.0123456789;=BKNOQRabcdefghx")  # character vocabulary of the model
PIECES = ".PNBRQKpnbrqk"  # empty square, white pieces, black pieces

# intervention
LAYERS_INTERVENED = [1, 2, 3, 4, 5, 6, 7]  # hidden states edited, 0 is the embedding
POSITIONS = 300  # test positions for the intervention statistics
SHOW_LAYERS = [0, 2, 4, 6, 8]
GLYPHS = dict(zip("PNBRQKpnbrqk", "♙♘♗♖♕♔♟♞♝♜♛♚"))


# --------------------------------------- helper --------------------------------------
def encode(text):
    return torch.tensor([CHARS.index(c) for c in text], device=device)


def board_labels(board):
    labels = np.zeros(64, dtype=np.int64)
    for square, piece in board.piece_map().items():
        labels[square] = PIECES.index(piece.symbol())
    return labels


def positions(transcript):
    text = ";" + transcript[:MAX_CHARS]
    text = text[:text.rfind(" ")]  # drop truncated last move
    board = chess.Board()
    dots, boards = [], []
    i = 1
    for token in text[1:].split(" "):
        if "." in token:
            dots.append(i + token.index("."))
            boards.append(board.copy())
        board.push_san(token.split(".")[-1])
        i += len(token) + 1
    return text, dots, boards


@torch.no_grad()
def hidden_states(text):
    states = model(encode(text)[None], output_hidden_states=True).hidden_states
    return torch.stack(states)[:, 0]  # (layers + 1, characters, D)


def erase_hooks(square, piece, start):
    def make_hook(layer):
        w = probes[layer].weight.view(64, 13, -1)[square]
        v = w[0] - w[piece]  # from the piece toward empty
        v = v / v.norm()
        mean = (features[layer, :split][labels[:split, square] == 0].to(device) @ v).mean()  # when the square is empty

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output  # newer transformers return a tensor
            h[:, start:] -= ((h[:, start:] @ v) - mean)[..., None] * v  # ablation toward the empty mean
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook
    return [model.transformer.h[layer - 1].register_forward_hook(make_hook(layer)) for layer in LAYERS_INTERVENED]


@torch.no_grad()
def move_distribution(prefix, candidates):
    texts = [prefix + c + " " for c in candidates]
    width = max(len(t) for t in texts)
    ids = torch.zeros(len(texts), width, dtype=torch.long, device=device)
    for i, t in enumerate(texts):
        ids[i, :len(t)] = encode(t)
    logp = torch.log_softmax(model(ids).logits.float(), -1)
    scores = torch.tensor([logp[i, len(prefix) - 1:len(t) - 1].gather(1, ids[i, len(prefix):len(t), None]).sum()
                           for i, t in enumerate(texts)])
    return torch.softmax(scores, 0).numpy()


def candidates_of(boards):
    return sorted({b.san(m) for b in boards for m in b.legal_moves})


def top_arrows(boards, p, candidates):
    moves = {b.san(m): m for b in boards for m in b.legal_moves}
    return [(moves[candidates[i]], candidates[i], p[i]) for i in np.argsort(-p)[:2]]


def draw_board(ax, board, alpha=None, wrong=(), arrows=(), ghost=None):
    for square in range(64):
        x, y = chess.square_file(square), chess.square_rank(square)
        color = "#d9d9d9" if (x + y) % 2 == 0 else "#f4f4f4"
        if square in wrong:
            color = "#f48573"
        ax.add_patch(Rectangle((x, y), 1, 1, color=color, lw=0))
    for square, piece in board.piece_map().items():
        x, y = chess.square_file(square), chess.square_rank(square)
        a = 1.0 if alpha is None else alpha[square]
        ax.text(x + 0.5, y + 0.47, GLYPHS[piece.symbol()], ha="center", va="center", fontsize=FONT, alpha=a)
    if ghost is not None:
        square, symbol = ghost
        x, y = chess.square_file(square), chess.square_rank(square)
        ax.text(x + 0.5, y + 0.47, GLYPHS[symbol], ha="center", va="center", fontsize=FONT, alpha=0.2)
        ax.plot([x + 0.2, x + 0.8], [y + 0.2, y + 0.8], color="#a32126", lw=2)
        ax.plot([x + 0.2, x + 0.8], [y + 0.8, y + 0.2], color="#a32126", lw=2)
    for move, _, _ in arrows:
        x0, y0 = chess.square_file(move.from_square) + 0.5, chess.square_rank(move.from_square) + 0.5
        x1, y1 = chess.square_file(move.to_square) + 0.5, chess.square_rank(move.to_square) + 0.5
        ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0, width=0.12, head_width=0.4, head_length=0.3,
                                length_includes_head=True, color="#071141", lw=0))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------- load model ------------------------------------
checkpoint = torch.load(DATA_DIR / "lichess_8layers_ckpt_no_optimizer.pt", map_location=device, weights_only=True)
config = GPT2Config(vocab_size=len(CHARS), n_positions=1023, n_embd=512, n_layer=8, n_head=8,
                    activation_function="gelu", resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
model = GPT2LMHeadModel(config).to(device)
state = {}
for key, value in checkpoint["model"].items():
    key = key.replace("_orig_mod.", "")
    state[key] = value.T if key.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")) else value
model.load_state_dict(state, strict=False)  # nanogpt without biases, which stay zero
model.eval()

# ------------------------------------- load data -------------------------------------
with open(DATA_DIR / "lichess_100mb.csv") as file:
    games = [row["transcript"] for row in csv.DictReader(file)][:GAMES]

features, labels, test_positions = [], [], []
for g, transcript in enumerate(tqdm(games)):
    text, dots, boards = positions(transcript)
    states = hidden_states(text)[:, dots].half().cpu()
    features.append(states)
    labels.append(np.stack([board_labels(b) for b in boards]))
    if g >= GAMES - TEST_GAMES:
        test_positions += [(text[:i + 1], b) for i, b in zip(dots, boards)]
split = sum(len(f[0]) for f in features[:GAMES - TEST_GAMES])
features = torch.cat(features, dim=1).float()  # (layers + 1, positions, D)
labels = torch.tensor(np.concatenate(labels))
print(f"{split} training positions, {len(labels) - split} test positions, "
      f"always empty {(labels[split:] == 0).float().mean():.4f}")

# --------------------------------------- probes --------------------------------------
cost_fun = nn.CrossEntropyLoss()
probes, accuracy = [], []
for layer in range(features.shape[0]):
    probe = nn.Linear(features.shape[2], 64 * 13).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)
    X_train, Y_train = features[layer, :split].to(device), labels[:split].to(device)
    for epoch in range(EPOCHS):
        for ids in torch.randperm(split, device=device).split(BATCH_SIZE):
            optimizer.zero_grad()
            cost = cost_fun(probe(X_train[ids]).view(-1, 13), Y_train[ids].flatten())
            cost.backward()
            optimizer.step()
    with torch.no_grad():
        y_pred = probe(features[layer, split:].to(device)).view(-1, 64, 13).argmax(-1).cpu()
    accuracy.append((y_pred == labels[split:]).float().mean().item())
    probes.append(probe.requires_grad_(False))
    print(f"layer {layer}: square accuracy {accuracy[-1]:.4f}")

# ------------------------------------ intervention -----------------------------------
results, example = [], None
rng = np.random.default_rng(0)
for idx in rng.permutation(len(test_positions)):
    prefix, board = test_positions[idx]
    if board.fullmove_number < 6 or len(results) == POSITIONS:
        continue
    before = move_distribution(prefix, candidates := candidates_of([board]))
    top = board.parse_san(candidates[before.argmax()])
    piece = board.piece_at(top.from_square)
    if piece.piece_type == chess.KING:
        continue
    modified = board.copy()
    modified.remove_piece_at(top.from_square)
    if not modified.is_valid():
        continue
    candidates = candidates_of([board, modified])
    legal = np.array([any(m == c for m in map(modified.san, modified.legal_moves)) for c in candidates])
    legal_original = np.array([any(m == c for m in map(board.san, board.legal_moves)) for c in candidates])
    before = move_distribution(prefix, candidates)
    hooks = erase_hooks(top.from_square, PIECES.index(piece.symbol()), len(prefix) - 1)
    after = move_distribution(prefix, candidates)
    for hook in hooks:
        hook.remove()
    results.append([legal[before.argmax()], legal[after.argmax()], before[legal].sum(), after[legal].sum(),
                    legal_original[before.argmax()]])
    clear = 10 <= board.fullmove_number <= 20 and not board.is_check() and piece.piece_type != chess.PAWN
    if clear and before.max() > 0.5 and legal[after.argmax()] and (example is None or after[legal].sum() > example[-1]):
        example = (prefix, board, modified, top, piece, candidates, before, after, after[legal].sum())
results = np.array(results, dtype=float)
print(f"top move legal on modified board: {results[:, 0].mean():.2f} before, {results[:, 1].mean():.2f} after; "
      f"unedited, on the original board: {results[:, 4].mean():.2f}")
print(f"probability on legal moves of modified board: {results[:, 2].mean():.2f} before, {results[:, 3].mean():.2f} after")

# ----------------------------------- postprocessing ----------------------------------
prefix, board, modified, top, piece, candidates, before, after, _ = example
print(f"example: {prefix[-60:]} | erased {piece.symbol()} on {chess.square_name(top.from_square)}")
for name, p in (("before", before), ("after", after)):
    print(name, [(candidates[i], round(float(p[i]), 2)) for i in np.argsort(-p)[:4]])

states = hidden_states(prefix)[:, -1]
readouts = [torch.softmax(probes[layer](states[layer].float()).view(64, 13), -1).cpu().numpy() for layer in SHOW_LAYERS]


panels = []
for r in readouts:
    predicted = chess.Board(None)
    for square in range(64):
        if r[square].argmax() > 0:
            predicted.set_piece_at(square, chess.Piece.from_symbol(PIECES[r[square].argmax()]))
    wrong = [s for s in range(64) if r[s].argmax() != board_labels(board)[s]]
    panels.append(dict(board=predicted, alpha=r.max(-1), wrong=wrong))
panels.append(dict(board=board, arrows=top_arrows([board], before, candidates)))
panels.append(dict(board=modified, arrows=top_arrows([board, modified], after, candidates), ghost=(top.from_square, piece.symbol())))
names = [f"layer{layer}" for layer in SHOW_LAYERS] + ["before", "after"]

if not args.book:
    FONT = 18
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 2.2))
    for ax, panel in zip(axes, panels):
        draw_board(ax, **panel)
    plt.show()
# -------------------------------- book postprocessing --------------------------------
else:
    save_csv(CSV_DIR / "chess_probe_layers.csv", layer=np.arange(len(accuracy)), accuracy=accuracy)
    save_csv(CSV_DIR / "chess_probe_intervention.csv", case=["before", "after"],
             legal=[results[:, 0].mean(), results[:, 1].mean()], mass=[results[:, 2].mean(), results[:, 3].mean()])
    arrows = [panel.get("arrows", []) for panel in panels]  # boards are drawn by chessboard in latex
    save_csv(CSV_DIR / "chess_probe_boards.csv", name=names, fen=[panel["board"].board_fen() for panel in panels],
             wrong=[",".join(map(chess.square_name, panel.get("wrong", []))) or "-" for panel in panels],
             erased=["-"] * (len(panels) - 1) + [chess.square_name(top.from_square)],
             arrows=[",".join(f"{chess.square_name(m.from_square)}-{chess.square_name(m.to_square)}" for m, _, _ in a)
                     or "-" for a in arrows],
             moves=[r"\\".join(san.replace("#", r"\#") + f"~{100 * p:.0f}" + r"\,\%" for _, san, p in a) or "-"
                    for a in arrows])
