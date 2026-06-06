This snapshot preserves the local source prepared after Version14: Crafter 5k with lightweight action tracing every 20 environment steps, plus `--run.train_ratio 64` for faster validation.

Settings:

```text
CRAFTER_SMOKE_STEPS = 5000
RESULT_RUN_NAME = 'crafter_gpu_5k_trace20_r64'
ACTION_TRACE_EVERY = 20
--run.train_ratio 64
```

Trace output in Kaggle:

```text
/kaggle/working/logdir/crafter_gpu_5k_trace20_r64/action_trace/action_trace.jsonl
```
