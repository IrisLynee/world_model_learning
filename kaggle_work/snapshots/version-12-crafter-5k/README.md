This snapshot preserves the local source configuration for Kaggle kernel
version 12, reconstructed after the push because the snapshot rule was added
later.

Expected version 12 settings:

```text
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
CRAFTER_SMOKE_STEPS = 5000
RESULT_RUN_NAME = 'crafter_gpu_5k'
```

The Crafter run uses:

```text
--logdir /kaggle/working/logdir/crafter_gpu_5k
--configs crafter
--run.steps 5000
--run.envs 1
--batch_size 1
--batch_length 8
--report_length 8
--replay_context 0
--jax.platform cuda
--jax.compute_dtype float32
--jax.prealloc False
```

This version does not set `--run.train_ratio 64` and does not enable
`--env.crafter.logs`.
