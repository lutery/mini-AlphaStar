# AGENTS.md — mini-AlphaStar

Research implementation of AlphaStar's architecture for StarCraft II (Protoss only).

## Project type

- No build system, no setup.py, no CI. Pure research code.
- Python 3.7 + PyTorch ≥ 1.5. All entry points are standalone scripts at repo root.
- **Not a library.** Run scripts directly; there is no install step beyond `pip install -r requirements.txt`.

## Entry points

| Script | What it does |
|---|---|
| `run.py` | Main launcher — uncomment one block to select a training mode |
| `test.py` | Runs `.test()` on every arch/rl component (smoke test, not pytest) |
| `transform.py` | Replays → pickles only |
| `eval_sl.py` | Multi-process SL model evaluation in SC2 env |
| `eval_rl.py` | Multi-process RL model evaluation in SC2 env |

## The training pipeline (order matters)

1. **Replays → pickles:** `transform_replay_data.test()` (comment/uncomment in `run.py`)
2. **Pickles → tensors:** `load_pickle.test()` (optional but recommended; faster SL training)
3. **Supervised learning:** `sl_train_by_tensor.test()` — use tensor data, single GPU, ~10 epochs, lr ≤ 1e-4
4. **Evaluate SL model:** `rl_eval_sl.test()` / `mp_rl_eval_sl.test()` — verify domain transfer before RL
5. **RL training:** `rl_vs_inner_bot_mp.test()` — multi-process (15 procs default, 2 actor + 1 learner threads each), lr < 1e-5
6. **Evaluate RL model:** `rl_eval_rl.test()` / `mp_rl_eval_rl.test()`

## How tests work

Every module in `alphastarmini.core.arch` and `alphastarmini.core.rl` exposes a `test()` function at module level.

```
python test.py          # runs all component tests sequentially
python run.py           # uncomment one training mode, which calls its .test()
```

There is **no pytest, unittest, or CI**.

## Critical config files

- **`param.py`** — global runtime config imported as `import param as P` everywhere:
  - `on_server`: `True` = multi-process GPU; `False` = single-process CPU
  - `replay_path`: path to replay files
  - `Batch_Scale`, `Seq_Scale`, `Select_Scale`: downscaling factors (16 = laptop, 4 = server)
- **`alphastarmini/lib/hyper_parameters.py`** — model architecture hyperparameters:
  - `THE_PROJECT_TYPE`: `ProjectType.MiniStar` (current) vs `ProjectType.AlphaStar` (original scale)
  - MiniStar uses ~1.5M params (20× smaller than AlphaStar's ~31M)

## Architecture

```
alphastarmini/
  core/
    arch/    — neural architecture: encoders (entity, scalar, spatial), heads (action, delay, queue, etc.), Core/LSTM, agent, baseline, arch_model
    sl/      — supervised learning: replay transforms, datasets, 4 training modes (pickle/tensor × single/multi-GPU)
    rl/      — reinforcement learning: actor, learner, shared Adam, rewards, losses, SC2 env interaction
    ma/      — multi-agent league training (incomplete)
  lib/       — hyperparameters, transformer layers, SC2 env wrappers, utilities
  third/     — action dictionary, available actions from AlphaStar paper
```

The model is `ArchModel` (in `arch_model.py`), which is the central PyTorch nn.Module. It reads `Arch_Hyper_Parameters` from `lib/hyper_parameters.py`.

## Gotchas

- **Multi-GPU training is unstable.** The README recommends single-GPU only. Known PyTorch instability.
- **Windows multiprocessing** requires `torch.multiprocessing.freeze_support()` (commented out in evaluation scripts). Uncomment if running on Windows.
- **CUDA setup** is done at the top of each script via `USED_DEVICES` and `CUDA_VISIBLE_DEVICES` before any torch import.
- **Seed is hardcoded to 0** in `run.py` and `eval_rl.py`. Do not change it for reproducibility.
- **Only Protoss** is trained in MiniStar mode (`Training_Races = ("Protoss")`).
- **Data files are gitignored:** `data/replay_data/`, `data/replay_data_tensor/`, `model/`, `log/`, `outputs/` are all in `.gitignore`.
- **SC2 game version** is pinned to `3.16.1` in `StarCraft_Hyper_Parameters`. Replays must match.
- Multi-GPU SL training files (`sl_multi_gpu_by_*.py`) exist but are **not recommended** — they cost too much memory and are unstable.

## Useful docs

- `README.MD` — full usage guide (English)
- `doc/README_CHS.MD` — shorter Chinese version
- `doc/REPLAY.MD` — replay download guide
- `doc/USAGE.MD` — common problems and FAQ
- `doc/HISTORY.MD` — version history
- `scripts/Setup_cmd.MD` — conda/pip install commands
- `alphastarmini/core/readme.txt` — one-line description of each core subpackage
