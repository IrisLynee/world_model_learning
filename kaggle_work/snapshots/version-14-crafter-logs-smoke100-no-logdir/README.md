# Version 14 Snapshot: Crafter Logs Smoke100 Without Logdir Flag

This snapshot is for the fixed Crafter achievement logging smoke test.

Reason for the fix:

```text
KeyError: "Flag '--env.crafter.logdir' did not match any config keys."
```

Key settings:

```text
CRAFTER_SMOKE_STEPS = 100
RESULT_RUN_NAME = 'crafter_logs_smoke100'
--logdir /kaggle/working/logdir/crafter_logs_smoke100
--configs crafter
--run.steps 100
--run.envs 1
--env.crafter.logs True
--batch_size 1
--batch_length 8
--report_length 8
--replay_context 0
--jax.platform cuda
--jax.compute_dtype float32
--jax.prealloc False
```

This version intentionally does not pass `--env.crafter.logdir` because the DreamerV3 config parser rejects config keys that are not present in `configs.yaml`.
