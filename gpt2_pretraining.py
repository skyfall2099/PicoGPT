import numpy as np

# Try to import wandb, if not available skip logging
try:
    import wandb
    use_wandb = True
except ImportError:
    print("Weights & Biases not found, skipping logging")
    use_wandb = False


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)
    return g * x + b


def linear(x, w, b):
    return x @ w + b


def ffn(x, c_fc, c_proj):
    a = gelu(linear(x, **c_fc))
    x = linear(a, **c_proj)
    return x


def attention(q, k, v, mask):
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv = np.split(x, 3, axis=-1)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = np.hstack(out_heads)
    x = linear(x, **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    x = layer_norm(x, **ln_f)
    return x @ wte.T


def cross_entropy_loss(logits, targets):
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    loss = -np.mean(np.log(softmax(logits)[np.arange(len(targets)), targets]))
    return loss


def compute_gradients(inputs, targets, params, n_head):
    # Simple numerical gradients using finite differences
    # This is a simplified approach for demonstration purposes
    grads = {}
    epsilon = 1e-5
    
    # Compute initial loss
    logits = gpt2(inputs, **params, n_head=n_head)
    loss = cross_entropy_loss(logits, targets)
    
    # Helper function to get parameter paths
    def get_param_paths(obj, path=""):
        paths = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                if isinstance(v, (dict, list)):
                    paths.extend(get_param_paths(v, new_path))
                else:
                    paths.append(new_path)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                if isinstance(v, (dict, list)):
                    paths.extend(get_param_paths(v, new_path))
                else:
                    paths.append(new_path)
        return paths
    
    # Get all parameter paths
    param_paths = get_param_paths(params)
    
    # Compute gradient for each parameter
    for path in param_paths:
        # Evaluate parameter value
        exec(f"param_val = params{path}")
        original_val = param_val.copy()
        
        # Compute gradient using finite difference
        param_val += epsilon
        exec(f"params{path} = param_val")
        logits_plus = gpt2(inputs, **params, n_head=n_head)
        loss_plus = cross_entropy_loss(logits_plus, targets)
        
        param_val = original_val - epsilon
        exec(f"params{path} = param_val")
        logits_minus = gpt2(inputs, **params, n_head=n_head)
        loss_minus = cross_entropy_loss(logits_minus, targets)
        
        # Restore original value
        exec(f"params{path} = original_val")
        
        # Calculate gradient
        grad = (loss_plus - loss_minus) / (2 * epsilon)
        exec(f"grads{path} = grad")
    
    return grads, loss


def update_parameters(params, grads, learning_rate=1e-3):
    # Helper function to update parameters
    def update(obj, grad_obj, lr):
        if isinstance(obj, dict) and isinstance(grad_obj, dict):
            for k in obj:
                obj[k] = update(obj[k], grad_obj[k], lr)
        elif isinstance(obj, list) and isinstance(grad_obj, list):
            for i in range(len(obj)):
                obj[i] = update(obj[i], grad_obj[i], lr)
        else:
            obj -= lr * grad_obj
        return obj
    
    return update(params, grads, learning_rate)


def initialize_small_model():
    # Initialize a very small GPT model for demonstration
    n_vocab = 1000
    n_embd = 64
    n_head = 2
    n_layer = 2
    n_ctx = 10
    
    # Initialize weights
    params = {
        "wte": np.random.randn(n_vocab, n_embd) * 0.02,
        "wpe": np.random.randn(n_ctx, n_embd) * 0.02,
        "blocks": [],
        "ln_f": {
            "g": np.ones(n_embd),
            "b": np.zeros(n_embd)
        }
    }
    
    # Initialize transformer blocks
    for _ in range(n_layer):
        block = {
            "ln_1": {
                "g": np.ones(n_embd),
                "b": np.zeros(n_embd)
            },
            "attn": {
                "c_attn": {
                    "w": np.random.randn(n_embd, 3 * n_embd) * 0.02,
                    "b": np.zeros(3 * n_embd)
                },
                "c_proj": {
                    "w": np.random.randn(n_embd, n_embd) * 0.02,
                    "b": np.zeros(n_embd)
                }
            },
            "ln_2": {
                "g": np.ones(n_embd),
                "b": np.zeros(n_embd)
            },
            "mlp": {
                "c_fc": {
                    "w": np.random.randn(n_embd, 4 * n_embd) * 0.02,
                    "b": np.zeros(4 * n_embd)
                },
                "c_proj": {
                    "w": np.random.randn(4 * n_embd, n_embd) * 0.02,
                    "b": np.zeros(n_embd)
                }
            }
        }
        params["blocks"].append(block)
    
    hparams = {
        "n_vocab": n_vocab,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "n_ctx": n_ctx
    }
    
    return params, hparams


def create_vocab_and_tokenizer():
    # Create a simple vocabulary and tokenizer for demonstration
    vocab = {
        "the": 0, "quick": 1, "brown": 2, "fox": 3, "jumps": 4, "over": 5, 
        "lazy": 6, "dog": 7, "a": 8, "and": 9, "cat": 10, "ran": 11, 
        "into": 12, "house": 13, "on": 14, "street": 15, "saw": 16, 
        "mouse": 17, "chased": 18, "it": 19
    }
    inv_vocab = {v: k for k, v in vocab.items()}
    
    def tokenizer(text):
        tokens = text.lower().split()
        return [vocab.get(token, 0) for token in tokens]
    
    return vocab, inv_vocab, tokenizer


def prepare_data():
    # Create training and test data
    train_texts = [
        "the quick brown fox jumps over the lazy dog",
        "the cat chased the mouse into the house",
        "a quick brown fox ran on the street",
        "the lazy dog saw a cat and ran away",
        "the fox jumps over the dog and cat"
    ]
    
    test_texts = [
        "the quick brown",
        "the cat chased",
        "a lazy dog"
    ]
    
    return train_texts, test_texts


def main():
    # Initialize wandb if available
    if use_wandb:
        wandb.init(
            project="pico-gpt-pretraining",
            name="gpt2-pretraining-demo",
            config={
                "n_epochs": 100,
                "learning_rate": 1e-2,
                "model_size": "small-demo"
            }
        )
    
    print("Initializing small GPT model for pretraining demonstration...")
    
    # Initialize small model
    params, hparams = initialize_small_model()
    
    # Create vocabulary, inverse vocabulary, and tokenizer
    vocab, inv_vocab, tokenizer = create_vocab_and_tokenizer()
    
    # Prepare training and test data
    train_texts, test_texts = prepare_data()
    
    print("\nTraining data:")
    for text in train_texts:
        print(f"- {text}")
    
    print("\nTest data:")
    for text in test_texts:
        print(f"- {text}")
    
    # Training loop
    n_epochs = 100
    learning_rate = 1e-2
    
    print("\nStarting pretraining...")
    for epoch in range(n_epochs):
        total_loss = 0
        
        # Train on each training example
        for text in train_texts:
            input_ids = tokenizer(text)
            target_ids = input_ids[1:] + [0]  # Shifted by one for language modeling
            
            # Forward pass and compute gradients
            grads, loss = compute_gradients(input_ids, target_ids, params, hparams["n_head"])
            
            # Update parameters
            params = update_parameters(params, grads, learning_rate)
            
            total_loss += loss
        
        # Calculate average loss for epoch
        avg_loss = total_loss / len(train_texts)
        
        # Log metrics to wandb if available
        if use_wandb:
            wandb.log({"loss": avg_loss, "epoch": epoch + 1})
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    # Test the trained model
    print("\nTesting the trained model...")
    
    for test_text in test_texts:
        test_input = tokenizer(test_text)
        print(f"\nTest input: '{test_text}' -> {test_input}")
        
        # Generate text
        generated = test_input.copy()
        for _ in range(5):
            logits = gpt2(generated, **params, n_head=hparams["n_head"])
            next_token = np.argmax(logits[-1])
            generated.append(next_token)
        
        # Decode generated tokens
        generated_text = " ".join([inv_vocab.get(token, "<unk>") for token in generated])
        
        print(f"Generated text: {generated_text}")
    
    # Finish wandb run if available
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
