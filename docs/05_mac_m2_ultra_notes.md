# Mac Studio M2 Ultra — Hardware-specific notes

You have a Mac Studio with M2 Ultra (24 CPU cores: 16 performance + 8 efficiency, 60 GPU cores, 64 GB unified memory). This is genuinely good hardware for this project, but there are Apple Silicon specifics that change a few things in the startup checklist.

---

## Bottom line first

Your machine can comfortably run the full pipeline. Estimated wall times:

| Phase | Steps | Wall time on your M2 Ultra |
|---|---|---|
| Phase 1 — synth LL pretrain | 2M | 6–10 hours |
| Phase 1 — synth HL pretrain | 1M | 3–5 hours |
| Phase 2 — real LL fine-tune | 500k | 1–2 hours |
| Phase 2 — real HL fine-tune | 250k | 30–60 min |
| **Total per seed** | | **~12–18 hours** |
| **Five seeds** | | **~60–90 hours total** |

You can run all five seeds back-to-back over 3–4 days. With 64 GB unified memory you can also run two seeds in parallel comfortably (each one uses ~4–6 GB peak), cutting wall time roughly in half if you're patient about CPU sharing.

---

## What changes vs. the Linux/CUDA assumptions in the checklist

### 1. PyTorch device: use MPS, not CUDA, but cautiously

Apple Silicon uses the Metal Performance Shaders (MPS) backend, not CUDA. The setup is straightforward:

```python
import torch
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")
```

But — and this is important — **for PPO with small networks like ours, MPS is often slower than CPU on M-series chips.** The MLP extractor is two stacks of (Linear 313→512→256→256) with LayerNorm. That's ~250k parameters total. The overhead of moving rollout tensors to/from the GPU each PPO update can dominate the actual compute.

I recommend:
- **First training run: use CPU.** Set `device="cpu"` in the PPO constructor. With 16 performance cores you'll likely beat MPS on this network size.
- **If CPU is slow, try MPS as a comparison.** Run step 4 (50k tiny train) on both and time it. Pick whichever is faster.

```python
model = PPO(
    LayerNormActorCriticPolicy,
    vec,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=128,
    device="cpu",   # explicit; SB3 default would auto-pick CUDA which doesn't exist
    verbose=1,
)
```

The reason this is counterintuitive: GPUs win on RL when batch sizes are large (DQN with 256+ batch on Atari, robotics with image inputs). Tabular RL on a 313-feature MLP doesn't have enough parallelism per forward pass to amortize the MPS dispatch overhead. Your CPU (especially the 16 performance cores) handles this kind of workload well.

### 2. Multiprocessing: SubprocVecEnv works on macOS, but mind the fork/spawn issue

macOS uses `spawn` as the multiprocessing default since Python 3.8 (changed from `fork` for safety reasons). This means every subprocess re-imports your script from scratch. Three implications:

**The `if __name__ == "__main__":` guard is mandatory.** Without it, every subprocess re-runs your whole training script, which spawns more subprocesses, etc. — fork bomb. The checklist already mentions this; it's especially important on macOS.

**Imports at module top-level run 8 times** (once per subprocess + once in the main). If `import pandas as pd` or model loading is at top level, that's 8x the import time. Keep the `if __name__ == "__main__"` block lean — do the heavy work inside it.

**SubprocVecEnv pickling errors** are more common on macOS than Linux because of `spawn`. If you see errors like "can't pickle local object" or "can't pickle <class>", the fix is usually to make sure the env factory function is at module level, not nested inside another function:

```python
# BAD — won't pickle on macOS
def train(...):
    def make_env(seed):
        def _init():
            return LowLevelPortfolioEnv(...)
        return _init
    vec = SubprocVecEnv([make_env(i) for i in range(8)])

# GOOD — env factory is at module level
def make_env(seed):
    def _init():
        # imports inside _init are fine; runs in subprocess
        import pandas as pd
        return LowLevelPortfolioEnv(...)
    return _init

if __name__ == "__main__":
    vec = SubprocVecEnv([make_env(i) for i in range(8)])
```

### 3. CPU core allocation: 8 envs is right, not 16

You have 16 performance cores. The instinct is to use 16 envs for SubprocVecEnv. **Don't.** Here's why:

- Each env subprocess uses one core for the env step, but PyTorch in the main process also wants cores for the gradient update.
- macOS's QoS scheduler aggressively moves processes between performance and efficiency cores.
- Beyond ~8 envs, the variance reduction in advantage estimation flattens out — diminishing returns vs. coordination overhead.

**Use 8 envs for pretrain, 4 for fine-tune.** That leaves cores for the main training process and OS overhead. If you push to 16 you'll often see CPU at 90%+ but training throughput actually lower than at 8 envs. Confirmed empirically across many setups.

If you're feeling experimental, run step 7 (parallelism check) at 4, 8, 12, 16 envs and time each. Pick whichever gives best steps/second. For most M-series chips with this network size, 6–8 is the sweet spot.

### 4. Memory: you have plenty, but don't be careless

Per-process memory budget at 8 envs:
- Each env subprocess: ~150–300 MB (Python + numpy + the env state)
- Main process: ~1–2 GB (PyTorch + the policy + the rollout buffer)
- Synthetic pool memory-mapped: ~970 MB (stays in unified memory once loaded)
- Total: ~4 GB peak per training run

With 64 GB you can comfortably run **two training seeds in parallel**, each with 8 envs. That's 16 active env subprocesses + 2 main training processes ≈ 18 active threads on 24 total cores, with ~8 GB total memory. Cuts your 5-seed wall time roughly in half.

To run two seeds in parallel, just open two terminal windows and run:

```bash
# Terminal 1
python scripts/train_phase1_synth.py --seed 0 --gpu-id 0  # gpu-id is unused on Mac

# Terminal 2
python scripts/train_phase1_synth.py --seed 1 --gpu-id 0
```

Don't go to 3+ parallel seeds. The performance core contention starts hurting throughput.

### 5. Thermal throttling: the silent training killer

The Mac Studio handles sustained load better than a MacBook, but it's still possible to thermally throttle during 12-hour training runs. Two specific things to watch for:

**Performance cores down-clock under sustained load.** You can monitor this with:

```bash
# Install once
brew install asitop

# Run during training to see clock speeds and power draw
sudo asitop
```

If you see P-core clocks dropping from 3.5 GHz to 2.5 GHz consistently, the chassis is at thermal limit and you're getting maybe 70% of peak performance. The fix isn't software — it's making sure the Mac Studio has clear airflow on all sides. Don't put it in a cabinet during multi-day training.

**The fan is loud at sustained load.** Not relevant for performance, just a heads-up if it's in your workspace. The Mac Studio's fan ramps up around 70% sustained CPU and gets distinctly audible.

### 6. SB3 + macOS specific gotchas

**Avoid `EvalCallback` with parallel envs during long runs.** It has a known issue where the eval env doesn't release subprocesses cleanly on macOS, leaking file handles. Over a 10-hour training run you'll hit ulimit and the run will crash. Workaround: do evaluation manually between training calls instead of via `EvalCallback`:

```python
# Instead of EvalCallback, do this every N steps:
for chunk_idx in range(20):
    model.learn(total_timesteps=100_000, reset_num_timesteps=False)
    # ... manual eval ...
    model.save(f"checkpoint_chunk_{chunk_idx}")
```

Less elegant, more reliable on macOS.

**Tensorboard logs work fine** — `tensorboard_log="./tb/"` in the PPO constructor + `tensorboard --logdir ./tb` in another terminal. Tensorboard is browser-based so works the same as on Linux.

---

## Updated wall-time estimates for your machine

This is what to actually expect. Adjusting from the generic checklist times:

| Step | Generic estimate | Your M2 Ultra estimate |
|---|---|---|
| Step 1 — data sanity | 10 min | 5 min |
| Step 2 — env smoke | 15 min | 10 min |
| Step 3 — random baseline | 10 min | 10 min |
| Step 4 — 50k tiny train (CPU) | 10 min | **5–8 min** |
| Step 4 — 50k tiny train (MPS) | n/a | **probably 8–15 min** (test it) |
| Step 5 — reward check | 15 min | 10 min |
| Step 6 — synth integration | 30 min | 20 min |
| Step 7 — parallelism check | 15 min | 15 min |
| **Step 8 — full pipeline, 1 seed** | overnight | **12–18 hours** |
| **Step 8 — 5 seeds sequential** | 4 days | **3–4 days** |
| **Step 8 — 5 seeds with 2 parallel** | n/a | **~2 days** |

The big change vs. the generic estimate: your machine is fast enough that you can think about real experiment iteration cycles in days, not weeks.

---

## Concrete recommended setup for your machine

```python
# In portfolio_hrl_env_fixed.py train_pipeline, override these defaults:

train_pipeline(
    train_csv_path="data/train.csv",
    synthetic_pool_path="data/synth_pool.npz",
    out_dir="checkpoints/seed0",
    seed=0,
    pretrain_steps=2_000_000,
    finetune_steps=500_000,
    n_envs_pretrain=8,        # not 16 — leave headroom for main process
    n_envs_finetune=4,        # smaller for fine-tune phase
)

# In the PPO constructor inside train_pipeline, add:
model = PPO(
    LayerNormActorCriticPolicy,
    vec,
    # ... existing args ...
    device="cpu",             # benchmark MPS first; CPU is usually faster here
    n_steps=512,              # SB3 default; works well for 8 parallel envs
    batch_size=256,            # 256 is a fine default; can go to 512 with 64GB memory
)
```

---

## What I'd actually do tonight

You have plenty of machine. The bottleneck is going to be your time, not compute. Here's what I'd do:

**Tonight (1–2 hours):** Run steps 0–5. By the end you should have the env validated, a tiny LL trained, and confirmation it beats random. If anything fails, debug now while context is fresh.

**Tomorrow morning (1 hour):** Steps 6–7. Confirm synth pool integration and decide on env count.

**Tomorrow midday:** Kick off step 8 with seed 0. Run for ~16 hours.

**Tomorrow evening:** While seed 0 is still running, kick off seed 1 in a second terminal. Both run overnight.

**Day 3 morning:** Both seeds done. Eval on test set, look at numbers. If results are reasonable, kick off seeds 2 and 3 in parallel. They run during day 3.

**Day 4:** Final seed 4. Aggregate results across all 5. Write up.

You can be done with the full 5-seed experiment in 4 days starting tonight. That's actually faster than most cloud GPU setups for this model size, because you're not paying serialization cost between machine and the data.

---

## One more thing — avoid heavy non-RL work during training

Single biggest mistake people make on M-series Macs: opening Chrome with 30 tabs while a training run is going. The performance cores are shared. Chrome is hungry. macOS will not protect your training process from Chrome's tab spawn.

If possible, train on this machine while you work on a laptop. Or close everything except the training terminal and a tensorboard browser tab. Or use `nice -n 19 python ...` to deprioritize Chrome relative to training (reverse what you'd usually do).

64 GB is plenty of memory but the CPU contention is what kills you, not memory.
