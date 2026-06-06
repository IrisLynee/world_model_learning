# Crafter 5k Trace20 R64 训练结果分析报告

日期：2026-06-06

数据来源：

- `kaggle_work/analysis/crafter_gpu_5k_trace20_r64/summary.md`
- `kaggle_work/analysis/crafter_gpu_5k_trace20_r64/clean_metrics.csv`
- `kaggle_work/analysis/crafter_gpu_5k_trace20_r64/episode_summary.csv`
- `kaggle_work/analysis/crafter_gpu_5k_trace20_r64/action_counts.csv`
- `kaggle_work/analysis/crafter_gpu_5k_trace20_r64/trace_events.csv`
- `docs/kaggle_dreamerv3_sop.md` 的最新进展记录

## 1. 验证结论

`crafter_gpu_5k_trace20_r64` 可以判定为工程验证通过。

依据如下：

- Kaggle notebook 状态为 `COMPLETE`。
- `metrics.jsonl` 已导出为 `clean_metrics.csv`，共 62 行。
- 最后一条训练指标到达 step 4950，接近 5k 目标；最后一条 episode 指标在 step 4857。
- checkpoint 存在。
- `loss_score_curves.png` 存在。
- replay 数据已导出 4440 行，覆盖 27 段 episode 轨迹。
- GPU 训练路径稳定：最后记录中 `usage/nvsmi/compute_avg/gpu0 = 0.98`，`usage/nvsmi/memory_avg/gpu0 = 0.47`。
- `replay/replay_ratio` 最终稳定在 64.0，符合本轮 `run.train_ratio=64` 的设计目标。

需要注意：`scores.jsonl` 没有生成，但 `metrics.jsonl` 中已有 `episode/score` 和 `episode/length` 记录。因此这不影响本轮作为小规模验证的结论。

本轮不能判定 agent 已经学会稳定策略。5k step 太短，当前结论应限定为：Crafter + DreamerV3 + Kaggle GPU + r64 训练配置 + 动作追踪/导出分析链路已经跑通。

## 2. 训练指标分析

训练指标行共 33 行，从 step 150 到 step 4950。

关键指标变化：

| 指标 | 初始记录 | 最后记录 | 解释 |
| --- | ---: | ---: | --- |
| `train/loss/image` | 152.7832 | 20.4586 | 图像重建损失明显下降，说明世界模型在短训练中开始拟合观察输入。 |
| `train/loss/rew` | 1.1072 | 0.1092 | reward 预测损失下降，说明奖励头至少在当前短轨迹分布上开始稳定。 |
| `train/rew` | 0.0038 | 0.0125 | 平均训练 reward 有小幅上升，但幅度很小，不能当作策略能力证据。 |
| `train/ret` | 0.0419 | 0.4371 | 训练 return 指标上升，说明短程信号有改善迹象。 |
| `replay/replay_ratio` | 60.53 | 64.0 | 已达到本轮 r64 目标。 |
| `fps/policy` | - | 1.2358 | policy 采样速度较低，但本轮以验证链路为主。 |
| `fps/train` | - | 79.0911 | 训练侧吞吐稳定。 |
| GPU compute | 0.98 | 0.98 | GPU 计算利用率稳定。 |
| GPU memory | 0.47 | 0.47 | 显存占用稳定，无 OOM 迹象。 |

综合判断：训练过程稳定，loss 没有发散，GPU 利用正常，r64 配置适合作为后续小规模验证默认配置。由于训练步数短，不应比较策略优劣或泛化能力。

## 3. Episode 与 replay 分析

`episode_summary.csv` 显示 replay 中共有 27 段 episode 轨迹。

汇总结果：

- replay episode 数：27
- 完整终止 episode：26
- replay 导出总行数：4440
- 非零 reward 行数：177
- 非零 reward 占比：3.99%
- replay 中 episode reward 最小值：0.1
- replay 中 episode reward 最大值：3.1
- replay 中 episode reward 平均值：约 0.8741
- episode 环境步数最小值：25
- episode 环境步数最大值：248
- episode 环境步数平均值：约 163.44

这说明当前数据已经足够验证 replay 导出和 episode 切分逻辑，但奖励仍然非常稀疏。非零 reward 只占约 4%，后续如果要研究空间导航或目标达成，建议在自定义环境里设计更可解释的 reward/event，而不是直接依赖 Crafter 的稀疏成就信号。

## 4. 动作日志分析

本轮有两类动作数据：

- `clean_replay_steps.*`：从 replay `.npz` 导出的逐步 action/reward/done 记录，适合做完整行为统计。
- `clean_action_trace_sample20.*`：notebook 每 20 个环境步写一行的轻量 trace，适合快速查看 action 与事件，但不是完整逐步日志。

replay 动作分布前 8 项：

| action | count | 占比 |
| --- | ---: | ---: |
| `move_right` | 794 | 17.99% |
| `do` | 749 | 16.97% |
| `sleep` | 665 | 15.07% |
| `move_left` | 592 | 13.41% |
| `place_stone` | 400 | 9.06% |
| `move_up` | 365 | 8.27% |
| `place_table` | 249 | 5.64% |
| `make_stone_pickaxe` | 191 | 4.33% |

观察：

- 动作分布不是单一动作塌缩，说明 agent 在短训练中产生了多样动作。
- `move_right`、`move_left`、`move_up` 出现较多，但 `move_down` 只有 6 次，方向分布明显不均衡。
- `sleep` 占比约 15%，偏高；这可能来自早期随机/探索策略，也可能反映当前短训策略还没有形成有效目标导向。
- 多个 crafting/place 类动作出现，但在 Crafter 早期环境中，这些动作未必都产生有效外部变化。仅凭动作频率不能证明 agent 理解了 crafting 逻辑。

## 5. Trace 事件分析

`clean_action_trace_sample20.*` 共导出 235 行，采样间隔为每 20 环境步一行。

事件分布：

| event | count | 占比 |
| --- | ---: | ---: |
| `(empty)` | 185 | 78.72% |
| `wake_up` | 18 | 7.66% |
| `collect_sapling` | 10 | 4.26% |
| `collect_wood` | 9 | 3.83% |
| `collect_drink` | 7 | 2.98% |
| 复合事件 | 6 | 2.55% |

观察：

- 大多数采样点没有新事件，符合 Crafter 奖励/成就信号稀疏的特点。
- 已经能观察到 `wake_up`、`collect_sapling`、`collect_wood`、`collect_drink`，说明事件提取链路可用。
- 由于每 20 步采样一次，trace 可能漏掉短时事件；如果后续要做精细行为解释，应该优先使用 replay 导出的逐步记录，或把 trace 间隔调小。

## 6. SOP 最新结论整合

基于 `docs/kaggle_dreamerv3_sop.md` 的最新记录，旧的下一步任务需要更新理解：

- `Crafter GPU smoke test` 已通过，不再是当前阻塞点。
- `Crafter 5k` 已经推进到 `crafter_gpu_5k_trace20_r64` 并完成输出下载与分析导出。
- 默认 `train_ratio=512` 运行时间过长，不适合当前小规模验证；`train_ratio=64` 已经验证可用。
- `action_trace/action_trace.jsonl` 的 lightweight trace 可用于快速观察事件，但不是完整逐步日志。
- `clean_replay_steps.*` 是当前更适合做逐步动作/reward/done 分析的数据源。

因此，当前阶段可以从“验证标准 Crafter 环境能否跑通”转入“设计可控自定义空间环境”的准备阶段。

## 7. 对后续工作的建议

在进入自定义房间平面图环境之前，建议先明确以下问题：

1. 目标任务是单纯导航，还是包含探索、目标识别、物品交互或多阶段规划？
2. 观察输入是否仍使用像素图像，还是先用低维网格状态做调试，再升级到图像？
3. reward 应该使用稀疏到达奖励，还是加入距离变化、撞墙惩罚、探索奖励等 shaping 信号？
4. done 条件是到达目标、超时，还是也包含碰撞/失败？
5. 日志需要记录哪些字段：位置、朝向、目标位置、action、reward、done、event、地图 id、episode id？
6. 是否需要把自定义环境先做成本地 smoke test，再接入 Kaggle DreamerV3？

建议第一个自定义环境版本保持极简：固定小地图、单 agent、单目标、四方向移动、到达即成功、固定最大步数、完整逐步日志。这样能把 DreamerV3 接口、日志导出和训练稳定性先验证清楚，再逐步增加复杂空间结构。

