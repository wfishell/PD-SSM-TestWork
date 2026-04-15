import time
import json
import jax.random as jr
import equinox as eqx
import optax

from train import make_make_step
from models.pdssm import StateTrackingPDSSM
from data_dir.dataloaders import create_fl_dataloaders

config = json.load(open("experiment_configs/parity_0.json"))

dataloaders, vocab_size = create_fl_dataloaders(
    task="parity",
    min_train_length=config["min_train_length"],
    max_train_length=config["max_train_length"],
    min_val_length=config["min_val_length"],
    max_val_length=config["max_val_length"],
    num_val_samples=512,
    val_seed=0,
)

model = StateTrackingPDSSM(
    vocab_size=vocab_size,
    label_dim=vocab_size,
    N=config["state_size"],
    H=config["embed_size"],
    num_layers=config["num_layers"],
    K=config["dictionary_size"],
    key=jr.PRNGKey(0),
)

opt = optax.adamw(config["learning_rate"])
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
make_step = make_make_step(opt)

train_iter = dataloaders["train"].loop(config["batch_size"], key=jr.PRNGKey(0))

# Warmup — triggers JIT compilation
X, y, mask = next(train_iter)
model, opt_state, loss = make_step(model, opt_state, X, y, mask)
loss.block_until_ready()
print("Compilation done.")

# Time 20 steps
t = time.time()
for _ in range(20):
    X, y, mask = next(train_iter)
    model, opt_state, loss = make_step(model, opt_state, X, y, mask)
loss.block_until_ready()
print(f"Per step (avg over 20): {(time.time() - t) / 20:.3f}s")
