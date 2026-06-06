This snapshot preserves the local source prepared as Version14: Crafter 5k from version 12 plus lightweight action tracing every 20 environment steps.

Settings:

```text
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
CRAFTER_SMOKE_STEPS = 5000
RESULT_RUN_NAME = 'crafter_gpu_5k_trace20'
ACTION_TRACE_EVERY = 20
```

Trace output in Kaggle:

```text
/kaggle/working/logdir/crafter_gpu_5k_trace20/action_trace/action_trace.jsonl
```

Each trace row records only the sampled step's action, reward, done flag, and newly reached achievements since the previous trace row. It does not save images.

The training parameters otherwise remain close to version 12 and do not set `--run.train_ratio 64`.
