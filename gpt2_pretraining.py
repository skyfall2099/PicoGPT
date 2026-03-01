"""Minimal GPT-2 pretraining demo using finite-difference gradients.

This is a standalone educational demonstration. It reuses the forward pass from
gpt2.py and trains a tiny randomly-initialized model on a toy corpus so the
entire loop fits in pure NumPy with no autograd framework.
"""

import numpy as np

from gpt2 import gpt2, softmax

# ---------------------------------------------------------------------------
# Loss & gradient utilities
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits, targets):
    probs = softmax(logits.reshape(-1, logits.shape[-1]))
    return -np.mean(np.log(probs[np.arange(len(targets.reshape(-1))), targets.reshape(-1)]))


def _collect_arrays(obj):
    """Yield (container, key) pairs for every numpy array in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                yield obj, k
            else:
                yield from _collect_arrays(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, np.ndarray):
                yield obj, i
            else:
                yield from _collect_arrays(v)


def compute_gradients(inputs, targets, params, n_head, eps=1e-5):
    """Per-array finite-difference gradients for all parameters."""
    logits = gpt2(inputs, **params, n_head=n_head)
    loss = cross_entropy_loss(logits, targets)

    grads = []
    for container, key in _collect_arrays(params):
        arr = container[key]
        grad = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            l_plus = cross_entropy_loss(gpt2(inputs, **params, n_head=n_head), targets)
            arr[idx] = orig - eps
            l_minus = cross_entropy_loss(gpt2(inputs, **params, n_head=n_head), targets)
            arr[idx] = orig
            grad[idx] = (l_plus - l_minus) / (2 * eps)
        grads.append((container, key, grad))

    return grads, loss


def apply_gradients(grads, lr):
    """SGD update in-place."""
    for container, key, grad in grads:
        container[key] = container[key] - lr * grad

# ---------------------------------------------------------------------------
# Model initialisation (tiny, so finite differences stay tractable)
# ---------------------------------------------------------------------------

N_VOCAB, N_EMBD, N_HEAD, N_LAYER, N_CTX = 20, 8, 2, 1, 10


def _rand(*shape):
    return np.random.randn(*shape).astype(np.float32) * 0.02


def _make_block(d):
    return {
        "ln_1": {"g": np.ones(d, dtype=np.float32), "b": np.zeros(d, dtype=np.float32)},
        "attn": {
            "c_attn": {"w": _rand(d, 3 * d), "b": np.zeros(3 * d, dtype=np.float32)},
            "c_proj": {"w": _rand(d, d), "b": np.zeros(d, dtype=np.float32)},
        },
        "ln_2": {"g": np.ones(d, dtype=np.float32), "b": np.zeros(d, dtype=np.float32)},
        "mlp": {
            "c_fc": {"w": _rand(d, 4 * d), "b": np.zeros(4 * d, dtype=np.float32)},
            "c_proj": {"w": _rand(4 * d, d), "b": np.zeros(d, dtype=np.float32)},
        },
    }


def initialize_model():
    return {
        "wte": _rand(N_VOCAB, N_EMBD),
        "wpe": _rand(N_CTX, N_EMBD),
        "blocks": [_make_block(N_EMBD) for _ in range(N_LAYER)],
        "ln_f": {"g": np.ones(N_EMBD, dtype=np.float32), "b": np.zeros(N_EMBD, dtype=np.float32)},
    }

# ---------------------------------------------------------------------------
# Toy corpus & tokenizer
# ---------------------------------------------------------------------------

VOCAB = {
    "the": 0, "quick": 1, "brown": 2, "fox": 3, "jumps": 4, "over": 5,
    "lazy": 6, "dog": 7, "a": 8, "and": 9, "cat": 10, "ran": 11,
    "into": 12, "house": 13, "on": 14, "street": 15, "saw": 16,
    "mouse": 17, "chased": 18, "it": 19,
}
INV_VOCAB = {v: k for k, v in VOCAB.items()}

TRAIN_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "the cat chased the mouse into the house",
    "a quick brown fox ran on the street",
    "the lazy dog saw a cat and ran",
    "the fox jumps over the dog and cat",
]
TEST_TEXTS = ["the quick brown", "the cat chased", "a lazy dog"]


def tokenize(text):
    return [VOCAB.get(w, 0) for w in text.lower().split()]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        import wandb
        wandb.init(project="pico-gpt-pretraining", name="demo",
                   config={"n_epochs": 50, "lr": 1e-2})
        log = wandb.log
    except ImportError:
        log = None

    np.random.seed(42)
    params = initialize_model()
    n_epochs, lr = 50, 1e-2

    print("Starting pretraining demo...")
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for text in TRAIN_TEXTS:
            ids = tokenize(text)
            targets = np.array(ids[1:] + [0])
            grads, loss = compute_gradients(ids, targets, params, N_HEAD)
            apply_gradients(grads, lr)
            total_loss += loss
        avg = total_loss / len(TRAIN_TEXTS)
        if log:
            log({"loss": avg, "epoch": epoch})
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{n_epochs}  loss={avg:.4f}")

    print("\nGeneration after training:")
    for text in TEST_TEXTS:
        ids = tokenize(text)
        generated = list(ids)
        for _ in range(5):
            logits = gpt2(generated, **params, n_head=N_HEAD)
            generated.append(int(np.argmax(logits[-1])))
        print(f"  {text!r} -> {' '.join(INV_VOCAB.get(t, '<unk>') for t in generated)}")

    if log:
        wandb.finish()


if __name__ == "__main__":
    main()
