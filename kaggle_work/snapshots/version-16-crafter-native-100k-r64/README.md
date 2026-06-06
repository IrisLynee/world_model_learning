# version-16-crafter-native-100k-r64

Purpose: Crafter native-reward 100k validation run for later comparison against geometric-score-inspired reward shaping.

Notebook defaults:

```text
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
CRAFTER_SMOKE_STEPS = 100000
RESULT_RUN_NAME = 'crafter_native_100k_r64'
ACTION_TRACE_EVERY = 20
```

Crafter training command keeps:

```text
--configs crafter
--run.steps 100000
--run.envs 1
--run.train_ratio 64
--batch_size 1
--batch_length 8
--report_length 8
--replay_context 0
--jax.platform cuda
--jax.compute_dtype float32
--jax.prealloc False
```

Reward: unchanged Crafter native reward via `--configs crafter` / `task: crafter_reward`.
