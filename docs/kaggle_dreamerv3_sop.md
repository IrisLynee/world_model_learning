# Kaggle DreamerV3 实验日志与 SOP

日期：2026-06-03  
项目：在 Kaggle 上运行 DreamerV3，并逐步从标准环境验证推进到可控的自定义空间环境训练。  
Kaggle notebook：<https://www.kaggle.com/code/linjinru/dreamerv3-atari-pong-kaggle-setup>

## 文档目的

本文档用于记录 Kaggle + DreamerV3 项目的推进过程、阶段目标、已完成事项、未完成事项、踩坑记录和标准操作流程。

维护规则：

- `工作目标` 按日期叠加，不删除旧目标，只在新日期下追加新的方向。
- `工作阶段` 每天更新，明确写出当天已完成和未完成的事项。
- `当前状态` 之后的历史记录原则上不重写，只作为追溯依据。
- 每次 push Kaggle 前，在 `kaggle_work/snapshots/` 下保存本地源码快照。

## 工作目标

### 2026-06-02

目标是先跑通 Kaggle + DreamerV3 的基础训练链路。Atari/Pong 只作为 smoke test，不追求分数，重点验证：

- Kaggle API 能创建、更新、查询 notebook。
- Kaggle runtime 能安装 DreamerV3 依赖。
- JAX 能识别并调用 CUDA GPU。
- DreamerV3 能初始化环境和模型，完成 JAX 编译并进入训练 loop。
- metrics、checkpoint、replay、report 等输出能正常写入。

### 2026-06-03

目标从 Atari/Pong 环境验证推进到 Crafter 小规模训练验证，并增加对 agent 行为的轻量记录能力：

- 完成 Crafter GPU smoke test，确认 Crafter 环境、DreamerV3 主循环、JAX GPU、metrics、checkpoint、replay 都能跑通。
- 推进到 Crafter 5k 小规模训练，先观察运行时间、GPU 占用、replay ratio、loss 和 score 是否稳定。
- 增加轻量动作日志：每 20 个环境步记录一行，只保存该步动作、reward、done 和新 achievement 事件，不保存图片。
- 保持训练日志足够轻，便于理解 agent 训练过程，同时不明显拖慢训练。
- 开始规划下一阶段：用简单房间平面图构建一个可控的自定义环境，让 agent 学习空间移动和目标导航。

## 工作阶段

### 2026-06-02

已完成：

- 接入 Kaggle API，并确认本地 `kaggle.json` 可用。
- 新建私有 notebook `linjinru/dreamerv3-atari-pong-kaggle-setup`。
- 将 DreamerV3 repo clone 路径改为 `/tmp/dreamerv3`，避免 Kaggle 输出目录过大。
- 解决 `kernel-metadata.json` UTF-8 BOM 导致的 Kaggle CLI JSON 解析失败。
- 解决 Kaggle notebook kernel 中 numpy 已加载导致的 ABI 冲突，改为用 fresh Python subprocess 做关键检查。
- 完成 CPU/无 GPU 环境验证。
- 完成 Atari/Pong GPU 路径验证，包括 JAX CUDA、模型初始化、JAX 编译和训练 loop。
- 识别并修复短步数 smoke test 下 replay 序列不足的问题，使用 `batch_length=8`、`report_length=8`、`replay_context=0` 降低短跑要求。

未完成：

- Crafter 环境尚未完成 smoke test。
- Crafter 5k 小规模训练尚未开始。
- 还没有记录 agent 每一步或抽样步的动作反馈。

### 2026-06-03

已完成：

- Kaggle version 11：Crafter GPU smoke test 已通过，确认能写出 `metrics.jsonl`、checkpoint、replay 和 report 文件。
- Kaggle version 12：已推送 Crafter 5k 小规模训练配置，run name 为 `crafter_gpu_5k`。
- 已确认 version 12 默认 `run.train_ratio=512`，运行时间较长，不适合作为快速迭代配置。
- 已建立 push 前本地快照规则，并补建 version 12、version 13、version 14 相关快照。
- 已尝试 Crafter achievement logging smoke100，并确认 `--env.crafter.logdir` 不是合法 config key，后续不能再使用。
- Kaggle version 14：已从 version 12 基线生成 `crafter_gpu_5k_trace20`，新增每 20 步一行的轻量动作日志，并已推送。
- Version14 本地源码快照已保存到 `kaggle_work/snapshots/version-14-crafter-5k-trace20/`。

未完成：

- 需要检查 Kaggle version 14 是否运行完成，以及是否成功生成 `action_trace/action_trace.jsonl`。
- 需要下载并分析 version 14 输出，确认每 20 步动作日志是否符合预期。
- 需要决定 Crafter 5k 后续是否继续用默认 `train_ratio=512`，还是切换到更快的 `train_ratio=64` 做验证。
- 需要设计自定义房间平面图环境的最小版本：观察图像、动作空间、reward、done、日志格式和 DreamerV3 接口。

## 当前状态

最开始想编辑的原始 notebook 是：

```text
linjinru/dreamer-v3
```

通过 Kaggle API 更新这个原始 notebook 时，多次返回：

```text
Kernel push error: Notebook not found
```

虽然 `pull` 和 `kernels list --mine` 能看到它，但 `push/status` 对这个原 notebook 表现不稳定。所以后来不再继续改原 notebook，而是新建了一个私有 notebook：

```text
linjinru/dreamerv3-atari-pong-kaggle-setup
```

直接链接：

```text
https://www.kaggle.com/code/linjinru/dreamerv3-atari-pong-kaggle-setup
```

当前 CPU/无 GPU 阶段已经在 Kaggle 上完整跑通。

GPU 阶段已经观察到以下关键日志：

```text
JAX devices (2): [cuda:0, cuda:1]
Policy devices: cuda:0
Train devices:  cuda:0
Initializing parameters...
Done initializing!
Compiling train and report...
Done compiling!
Start training loop
```

这说明：

- Kaggle GPU 已经可见。
- JAX CUDA backend 已经可用。
- DreamerV3 能初始化 Atari/Pong 环境。
- DreamerV3 模型参数能初始化。
- JAX train/report 编译能完成。
- 程序能进入训练循环。

下一步建议做一次“干净”的 GPU smoke test：换一个新的 logdir，避免加载旧 checkpoint 干扰判断。

## 本地环境与 Kaggle API 记录

本地工作目录：

```text
E:\data\BIGDATA\code\world_model_learning
```

项目根目录里有：

```text
kaggle.json
```

使用 Kaggle API 时，通过下面方式让 Kaggle CLI 使用当前目录的 `kaggle.json`：

```powershell
$env:KAGGLE_CONFIG_DIR=(Get-Location).Path
```

最开始本机没有 `kaggle` 命令，所以安装了 Kaggle CLI：

```powershell
python -m pip install --user kaggle
```

安装后发现当前本地 Python/OpenSSL 与 `urllib3 2.x` 不兼容，报错大意是 OpenSSL 版本太旧。修复方式：

```powershell
python -m pip install --user "urllib3<2"
```

由于 `kaggle.exe` 不在 PATH 中，所以本地使用完整路径调用：

```powershell
& "$env:APPDATA\Python\Python37\Scripts\kaggle.exe" kernels list --mine
```

当前相关本地目录：

```text
kaggle_work/dreamer-v3
kaggle_work/dreamerv3-atari-pong-kaggle-setup
kaggle_work/dreamerv3-atari-pong-kaggle-setup-output*
```

注意：不要打印或提交 `kaggle.json` 的内容。

## Notebook 设计原则

这个 notebook 是面向 Kaggle runtime 的验证 notebook，不是本机 CUDA 测试。

当前 notebook 结构大致如下：

1. 运行开关。
2. 打印 Kaggle/Python/CUDA 可见性。
3. 把 DreamerV3 clone 到 `/tmp/dreamerv3`。
4. 安装 DreamerV3 官方 requirements。
5. 通过 AutoROM 安装或确认 Atari ROM。
6. 在新的 Python 子进程中做依赖 import 检查。
7. 在新的 Python 子进程中做 JAX backend/device 检查。
8. 使用 DreamerV3 自己的 Atari wrapper 做 Atari/Pong smoke test。
9. 可选 Crafter import/env 检查，默认关闭。
10. 可选 CPU debug smoke test，默认关闭。
11. GPU/JAX CUDA Atari/Pong smoke test。
12. 可选 TensorBoard，默认关闭。

重要设计选择：

- DreamerV3 repo 克隆到 `/tmp/dreamerv3`，不要克隆到 `/kaggle/working/dreamerv3`。
- `/kaggle/working` 只用于日志和少量输出。
- pip install 之后，不在当前 notebook kernel 进程里直接 import 二进制包，而是在新的 Python subprocess 里 import。
- Atari/Pong 的核心检查用 DreamerV3 的 `embodied.envs.atari.Atari('pong')`。
- Gymnasium 的 `ALE/Pong-v5` 只作为可选提示，不作为硬失败标准。
- TensorBoard 在 batch validation 里默认关闭，避免交互式 UI 干扰自动运行。

## 运行开关 SOP

GPU smoke test 推荐开关：

```python
RUN_CPU_DEBUG_SMOKE = False
RUN_GPU_ATARI_SMOKE = True
RUN_CRAFTER_IMPORT_CHECK = False
SHOW_TENSORBOARD = False

REPO_DIR = '/tmp/dreamerv3'
LOGDIR = '/kaggle/working/logdir'

SMOKE_STEPS = 100
```

CPU/无 GPU 环境验证时：

```python
RUN_GPU_ATARI_SMOKE = False
SMOKE_STEPS = 100
```

后续要验证 Crafter import/env 时：

```python
RUN_CRAFTER_IMPORT_CHECK = True
```

## GPU Smoke Test 命令 SOP

GPU smoke test 目标不是训练出效果，而是验证：

- Kaggle GPU 可见。
- JAX CUDA backend 可用。
- DreamerV3 能用 Atari/Pong 启动。
- JAX 能完成 train/report 编译。
- 训练 loop 能开始跑。

推荐命令结构：

```python
subprocess.run([
    sys.executable, 'dreamerv3/main.py',
    '--logdir', f'{LOGDIR}/atari_pong_gpu_smoke_v2',
    '--configs', 'atari',
    '--task', 'atari_pong',
    '--run.steps', str(SMOKE_STEPS),
    '--run.envs', '1',
    '--batch_size', '1',
    '--batch_length', '8',
    '--report_length', '8',
    '--replay_context', '0',
    '--jax.platform', 'cuda',
    '--jax.compute_dtype', 'float32',
    '--jax.prealloc', 'False',
], check=True)
```

建议使用新的 logdir，例如：

```text
/kaggle/working/logdir/atari_pong_gpu_smoke_v2
```

不要继续复用旧 logdir：

```text
/kaggle/working/logdir/atari_pong_gpu_smoke
```

原因是旧目录可能包含 checkpoint，导致 smoke test 加载旧状态：

```text
Found existing checkpoint.
Loading checkpoint: ...
Loaded checkpoint.
```

如果目的是“干净验证当前配置能不能跑”，应该换新 logdir。

## 为什么要调小 batch_length/report_length

第一次 GPU smoke test 里出现过：

```text
AssertionError: (32, 1, 1, 17)
```

这不是 CUDA/JAX 失败，而是 replay 里可采样的连续序列还不够。

这几个数字可以理解为：

```text
需要长度: 32
连续段数量: 1
前置上下文: 1
当前可用长度: 17
```

也就是说 DreamerV3 当时想拿：

```text
32 * 1 + 1 = 33
```

个连续时间步，但 replay 里当前只有：

```text
17
```

为什么会这样：

- `SMOKE_STEPS = 100` 很短，是为了少用 GPU。
- Atari 环境有 action repeat，例如 `repeat = 4`。
- DreamerV3 不是拿单帧训练，而是从 replay 里采样一小段连续片段。
- 默认 `report_length = 32` 对 100-step smoke test 来说偏大。

所以 smoke test 要把序列要求一起降下来：

```text
batch_length: 16 -> 8
report_length: 32 -> 8
replay_context: 1 -> 0
```

这样最小连续序列需求从：

```text
32 + 1 = 33
```

变成：

```text
8 + 0 = 8
```

这更适合 100 steps 的短测试。

注意：这些是 smoke test 参数。后续正式训练或长一点的小规模训练，需要逐步恢复到更合理的 batch/report 长度。

## GPU 选择 SOP

第一次 GPU 验证建议：

- 如果 P100 排队很久，可以选 T4。
- T4 足够做 100-step smoke test。
- 不要选 TPU，因为当前命令明确使用 `--jax.platform cuda`。
- Kaggle API 通常只能设置 `enable_gpu: true`，不一定能精确选择 P100/T4。
- 如果 Kaggle 网页 UI 提供具体 GPU 选项，由操作者在网页里手动选。

实际观察到的 GPU runtime：

```text
JAX devices (2): [cuda:0, cuda:1]
```

说明 Kaggle 分配了双 GPU，类似 T4 x2。当前 notebook 只用第一张卡：

```text
Policy devices: cuda:0
Train devices:  cuda:0
```

这是符合当前 smoke test 目标的。

## 交互协作 SOP

有些动作必须由 Kaggle 网页操作者完成，不能完全靠 API 自动完成。

### Codex 可以做什么

Codex 可以：

- 在本地编辑 notebook 文件。
- 通过 Kaggle API push 新 notebook 版本。
- 通过 Kaggle API 创建新 notebook。
- 通过 Kaggle API 查询 notebook 状态。
- 分析你贴出来的 Kaggle 日志和 traceback。
- 给出精确到 cell/参数行的修改建议。

Codex 不适合或不能稳定完成：

- 在 Kaggle 网页里选择具体 GPU 类型。
- 像人一样在 Kaggle editor 里逐个 cell 交互运行。
- 实时看到 cell 输出，除非 API output 下载成功或你把日志贴出来。
- 处理 Kaggle 网页上的账号确认、验证码、配额提示、运行时选择提示。

### Kaggle 网页操作者需要做什么

操作者需要：

1. 打开 notebook 链接。
2. 如果做 GPU 测试，在 Kaggle UI 里打开 GPU。
3. 如果有 GPU 类型选择，P100 排队久就选 T4。
4. 运行相关 cell 或整个 notebook。
5. 如果失败，复制最后有效错误日志。
6. 把最后 30-100 行，尤其是 traceback 部分贴给 Codex。

复制日志时，优先从这些关键词开始：

```text
Traceback
AssertionError
RuntimeError
ValueError
CUDA_ERROR
out of memory
CalledProcessError
```

一直复制到错误最后一行。

如果只看到下面这些 CUDA/XLA 插件注册信息，不要只贴这段，要继续往下看：

```text
Unable to register cuFFT factory
Unable to register cuDNN factory
Unable to register cuBLAS factory
computation placer already registered
```

这些在 Kaggle/Colab 这类环境里很常见，通常只是噪声。真正失败通常在后面的 traceback。

### 如何在 Kaggle UI 里查看失败版本

Kaggle 网页经常默认展示最近成功的版本，而不是最新失败版本。

查看失败版本的方法：

1. 打开 notebook URL。
2. 找 `Versions` 或 `Version History`。
3. 选择最新失败版本。
4. 打开该版本的 `Output`、`Logs` 或失败 cell 输出。
5. 复制最后的 traceback。

例如 CPU 成功版本可能是 version 4，GPU 测试失败可能是 version 5。网页默认可能仍显示 version 4，所以一定要进 Version History 看最新版本。

## Kaggle API SOP

设置 Kaggle 配置目录：

```powershell
$env:KAGGLE_CONFIG_DIR=(Get-Location).Path
```

列出账号 notebook：

```powershell
& "$env:APPDATA\Python\Python37\Scripts\kaggle.exe" kernels list --mine
```

查询状态：

```powershell
& "$env:APPDATA\Python\Python37\Scripts\kaggle.exe" kernels status linjinru/dreamerv3-atari-pong-kaggle-setup
```

拉取 notebook：

```powershell
& "$env:APPDATA\Python\Python37\Scripts\kaggle.exe" kernels pull linjinru/dreamerv3-atari-pong-kaggle-setup -p .\kaggle_work\dreamerv3-atari-pong-kaggle-setup -m
```

推送 notebook：

```powershell
& "$env:APPDATA\Python\Python37\Scripts\kaggle.exe" kernels push -p .\kaggle_work\dreamerv3-atari-pong-kaggle-setup
```

下载输出：

```powershell
$env:PYTHONIOENCODING='utf-8'
& "$env:APPDATA\Python\Python37\Scripts\kaggle.exe" kernels output linjinru/dreamerv3-atari-pong-kaggle-setup -p .\kaggle_work\dreamerv3-atari-pong-kaggle-setup-output
```

Windows 下下载输出可能因为控制台编码失败，所以建议设置：

```powershell
$env:PYTHONIOENCODING='utf-8'
```

## Metadata SOP

CPU/无 GPU 运行：

```json
"enable_gpu": false,
"enable_tpu": false,
"enable_internet": true
```

GPU 运行：

```json
"enable_gpu": true,
"enable_tpu": false,
"enable_internet": true
```

notebook 内嵌 Kaggle metadata 也应保持一致：

```json
"accelerator": "gpu",
"isGpuEnabled": true
```

Windows 上创建 `kernel-metadata.json` 时要避免 UTF-8 BOM。之前 BOM 导致过：

```text
Expecting value: line 1 column 1 (char 0)
```

推荐用 `apply_patch` 或其它不会写 BOM 的方式创建/修改 metadata JSON。

## 已踩坑与修复记录

### 原 notebook push 失败

现象：

```text
Kernel push error: Notebook not found
```

背景：

- `pull` 能成功。
- `kernels list --mine` 能看到原 notebook。
- 但原 notebook 的 `push/status` 端点可能返回 404 或 `Notebook not found`。

处理：

- 停止继续更新原 notebook。
- 新建私有 notebook：`linjinru/dreamerv3-atari-pong-kaggle-setup`。

### 新 notebook metadata 有 BOM

现象：

```text
Expecting value: line 1 column 1 (char 0)
```

原因：

- Windows PowerShell `Set-Content -Encoding UTF8` 写入了 UTF-8 BOM。

修复：

- 删除并用 `apply_patch` 重新创建 `kernel-metadata.json`。

### repo 克隆到了 Kaggle 输出目录

最初路径：

```text
/kaggle/working/dreamerv3
```

问题：

- Kaggle 会把 `/kaggle/working` 视作 output。
- repo 放这里会导致输出下载巨大、慢、容易超时。

修复：

```text
/tmp/dreamerv3
```

只把 logdir 放在：

```text
/kaggle/working/logdir
```

### numpy ABI 冲突

现象：

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

原因：

- Kaggle notebook kernel 进程里已经加载/预加载了 `numpy 2.4.6`。
- DreamerV3 requirements 安装了 `numpy 1.26.4`。
- 在同一个已经运行的 Python 进程里 import 二进制包，会出现 ABI 状态不一致。

修复：

安装完成后，所有关键 import/JAX/ALE 检查都放到新的 Python subprocess 里：

```python
subprocess.run([sys.executable, '-c', code], check=True)
```

### Gymnasium Atari 注册不是核心检查

最初检查：

```python
gym.make('ALE/Pong-v5')
```

问题：

- Gymnasium/ALE 注册方式可能因为版本差异失败。
- DreamerV3 实际用的是自己的 Atari wrapper，不依赖这个 Gymnasium env id 作为核心路径。

修复：

核心检查改为：

```python
from embodied.envs.atari import Atari
env = Atari('pong', ...)
```

Gymnasium 检查只保留为 optional warning。

### 100-step smoke test 的 replay 序列太短

现象：

```text
AssertionError: (32, 1, 1, 17)
```

原因：

- 默认 `report_length=32`，而 100-step smoke test 下 replay 还没有足够连续数据。

修复：

```text
--batch_length 8
--report_length 8
--replay_context 0
```

### CUDA/XLA 插件重复注册日志

现象：

```text
Unable to register cuFFT factory
Unable to register cuDNN factory
Unable to register cuBLAS factory
computation placer already registered
```

解释：

- Kaggle/Colab 类环境里常见。
- TensorFlow、JAX、XLA、CUDA 组件可能都尝试注册相同插件。
- 这通常不是失败原因。

判断方式：

- 如果后面继续出现 DreamerV3 banner、JAX devices、编译日志，说明可以忽略。
- 真正失败要看后续 traceback。

### 旧 checkpoint 被加载

现象：

```text
Found existing checkpoint.
Loading checkpoint: /kaggle/working/logdir/atari_pong_gpu_smoke/ckpt/...
Loaded checkpoint.
```

解释：

- 说明 logdir 里已有旧 checkpoint。

建议：

- 干净测试时换新 logdir，例如：

```text
atari_pong_gpu_smoke_v2
```

## 当前推荐下一步

当前本地 notebook 已经推进到 Crafter smoke test 阶段，但尚未通过 Kaggle API push 到平台。下一次继续工作时，优先把本地 notebook push/pull 同步到 Kaggle，然后运行 Crafter smoke test。

本地 notebook 当前默认开关：

```text
RUN_CPU_DEBUG_SMOKE = False
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
SHOW_TENSORBOARD = False

SMOKE_STEPS = 1000
CRAFTER_SMOKE_STEPS = 200
RESULT_RUN_NAME = 'crafter_gpu_smoke'
```

新增的 Crafter GPU smoke test 参数：

```text
--logdir /kaggle/working/logdir/crafter_gpu_smoke
--configs crafter
--run.steps 200
--run.envs 1
--batch_size 1
--batch_length 8
--report_length 8
--replay_context 0
--jax.platform cuda
--jax.compute_dtype float32
--jax.prealloc False
```

成功标准：

```text
Crafter import/env OK
JAX devices 包含 cuda
Logdir: /kaggle/working/logdir/crafter_gpu_smoke
DreamerV3 参数初始化完成
train/report 编译完成
进入 training loop
metrics.jsonl 写入成功
最后没有 traceback
Kaggle version 状态 COMPLETE
```

注意：Atari/Pong 1k 已经完成过，不需要下一次重复跑。当前 notebook 保留 Pong 1k cell，但默认 `RUN_GPU_ATARI_SMOKE = False`，因此会跳过。

## 最新进度记录：2026-06-02

### Atari/Pong 1k 小规模训练已通过

已在 Kaggle GPU runtime 中完成 `atari_pong_gpu_1k` 小规模训练验证。观察到关键日志：

```text
Logdir: /kaggle/working/logdir/atari_pong_gpu_1k
JAX devices (2): [cuda:0, cuda:1]
Policy devices: cuda:0
Train devices:  cuda:0
Initializing parameters...
Done initializing!
Compiling train and report...
Done compiling!
Did not find any checkpoint.
Saving checkpoint: /kaggle/working/logdir/atari_pong_gpu_1k/ckpt/...
Saved checkpoint.
Start training loop
Writing metrics: /kaggle/working/logdir/atari_pong_gpu_1k/metrics.jsonl
Writing metrics: /kaggle/working/logdir/atari_pong_gpu_1k/scores.jsonl
```

训练中至少观察到：

```text
Agent Step 520
Agent Step 1_000
Agent Step 1_480
Agent Step 1_960
```

这说明：

- GPU/JAX/DreamerV3 路径已跑通。
- Atari/Pong 环境可用。
- checkpoint 保存成功。
- replay 序列长度问题已经通过 `batch_length=8`、`report_length=8`、`replay_context=0` 解决。
- metrics 和 scores 文件可以写出。

### 结果输出 cell 已改为 subprocess 静态分析

原来的最后一格 TensorBoard 会启动网页服务，在 Kaggle batch run 中可能一直 running。因此已改为普通结果分析 cell：

- 在 fresh Python subprocess 中读取 JSONL，避免 notebook kernel 中 `numpy/pandas` ABI 冲突。
- 不再使用 `pandas`。
- 读取当前 `RESULT_RUN_NAME` 指向的 logdir。
- 打印最新 `train/loss/*`。
- 画 loss/score 静态曲线并保存为：

```text
/kaggle/working/logdir/{RESULT_RUN_NAME}/loss_score_curves.png
```

这次也踩到过一个典型错误：

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
```

原因是当前 Kaggle notebook kernel 中已经加载过旧 numpy 状态，而 DreamerV3 requirements 又安装/降级了 numpy。修复原则和前面的 import/JAX/ALE 检查一致：关键分析代码也放到 fresh subprocess 中运行。

### 本地 notebook 已保存但尚未 push

当前本地文件已经保存最新代码：

```text
kaggle_work/dreamerv3-atari-pong-kaggle-setup/dreamerv3-atari-pong-kaggle-setup.ipynb
```

验证结果：

- notebook JSON 可解析。
- 共 14 个 cell。
- 新增 `GPU/JAX CUDA Crafter smoke test` cell。
- 文件编码为 UTF-8 无 BOM。
- 最后一格结果分析 cell 使用 subprocess，不直接 import pandas。

下一次开始时建议先执行：

```powershell
$env:KAGGLE_CONFIG_DIR=(Get-Location).Path
& "$env:APPDATA\Python\Python37\Scripts\kaggle.exe" kernels push -p .\kaggle_work\dreamerv3-atari-pong-kaggle-setup
```

如果不想直接 push，也可以先在 Kaggle UI 手动同步上述开关和新增 Crafter cell。

## GPU Smoke Test 之后

历史计划如下，前两步已经推进到 `atari_pong_gpu_1k`：

1. 逐步增加 Atari/Pong steps：

```text
100 -> 1000 -> 5000
```

2. 每次使用版本化 logdir：

```text
atari_pong_gpu_1k
atari_pong_gpu_5k
```

3. 重点观察：

```text
CUDA OOM
JAX compile failure
replay sequence assertions
checkpoint reload mismatch
```

4. Atari/Pong 小规模训练稳定后，再切到 Crafter smoke test：

```text
--configs crafter
--run.steps 100 或 200
--run.envs 1
--jax.platform cuda
--jax.compute_dtype float32
--jax.prealloc False
```

当前正处在第 4 步：准备运行 Crafter smoke test。

## 后续维护提醒

- 如果你在 Kaggle UI 里手动改过 notebook，本地版本可能落后。程序员继续本地修改前应先 pull。
- Kaggle API push 会创建新版本并触发 batch run。
- Kaggle UI 可能默认显示最近成功版本，不一定显示最新失败版本。
- 本地 Windows Python 不能代表 Kaggle CUDA 环境，真正判断以 Kaggle runtime 输出为准。
- 诊断 cell 不要删除，保留开关控制即可。
- 每次遇到新错误，应把 traceback 和修复方法追加到本文档。

## 最新进展记录：2026-06-02 当前阶段校正

当前已确认通过的是 Atari/Pong GPU 路径和 `atari_pong_gpu_1k` 小规模验证，不是 Crafter。Crafter 仍处在 smoke test 阶段。

本地 notebook 当前默认开关为：

```text
RUN_CPU_DEBUG_SMOKE = False
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
SHOW_TENSORBOARD = False

SMOKE_STEPS = 1000
CRAFTER_SMOKE_STEPS = 200
RESULT_RUN_NAME = 'crafter_gpu_smoke'
```

当前要验证的 Crafter smoke test 参数：

```text
--logdir /kaggle/working/logdir/crafter_gpu_smoke
--configs crafter
--run.steps 200
--run.envs 1
--batch_size 1
--batch_length 8
--report_length 8
--replay_context 0
--jax.platform cuda
--jax.compute_dtype float32
--jax.prealloc False
```

成功标准：

```text
Crafter import/env OK
JAX devices 包含 cuda
Logdir: /kaggle/working/logdir/crafter_gpu_smoke
Initializing parameters...
Done initializing!
Compiling train and report...
Done compiling!
Start training loop
metrics.jsonl 能写入
最后没有 traceback
Kaggle version 状态 COMPLETE
```

只有当 Crafter smoke test 明确通过后，才进入下一阶段：Crafter 5k 小规模训练验证。5k 阶段建议使用新的 logdir，例如：

```text
/kaggle/working/logdir/crafter_gpu_5k
```

## 最新进展记录：2026-06-03 本地 notebook 稳健性修正

已检查本地 `kaggle_work/dreamerv3-atari-pong-kaggle-setup/dreamerv3-atari-pong-kaggle-setup.ipynb`：

```text
JSON 可解析
无 UTF-8 BOM
共 14 个 cell
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
RESULT_RUN_NAME = 'crafter_gpu_smoke'
没有 Crafter 5k cell
```

本地未发现已下载的 `crafter_gpu_smoke` 输出目录，因此按本地证据，Crafter smoke test 还没有完成记录。

对最后的结果汇总 cell 做了一个稳健性修正：如果 `loss_score_curves.png` 尚未生成，现在只打印：

```text
No plot generated yet: /kaggle/working/logdir/crafter_gpu_smoke/loss_score_curves.png
```

不会再因为 `display(Image(...))` 找不到图片而让结果汇总 cell 自己失败。这个修正不改变训练参数，也不改变当前阶段；当前仍然是 Crafter GPU smoke test。

## 最新进展记录：2026-06-03 version 11 已推送

已通过 Kaggle API 推送当前本地 notebook：

```text
Kernel version 11 successfully pushed.
https://www.kaggle.com/code/linjinru/dreamerv3-atari-pong-kaggle-setup
```

version 11 仍然是 Crafter GPU smoke test，不是 Crafter 5k。当前默认运行：

```text
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
CRAFTER_SMOKE_STEPS = 200
RESULT_RUN_NAME = 'crafter_gpu_smoke'
```

观察 version 11 输出时，优先确认：

```text
Crafter import/env OK
JAX devices 包含 cuda
Logdir: /kaggle/working/logdir/crafter_gpu_smoke
Done initializing!
Done compiling!
Start training loop
Writing metrics: /kaggle/working/logdir/crafter_gpu_smoke/metrics.jsonl
没有 Traceback
```

## 最新进展记录：2026-06-03 version 11 Crafter smoke 已通过

Kaggle 状态已确认：

```text
linjinru/dreamerv3-atari-pong-kaggle-setup has status "KernelWorkerStatus.COMPLETE"
```

已下载并分析 version 11 输出。`kaggle kernels output` 下载过程中因 checkpoint 较大曾超时，但核心输出已经可用：

```text
kaggle_work/dreamerv3-atari-pong-kaggle-setup-output-v11/logdir/crafter_gpu_smoke
```

关键产物：

```text
metrics.jsonl
loss_score_curves.png
config.yaml
ckpt/.../agent.pkl
ckpt/latest
replay/*.npz
scope/report-openloop-image.mp4/*.mp4
```

这说明 metrics、checkpoint、replay、静态曲线和 DreamerV3 report 都写出成功。

`metrics.jsonl` 共 9 行，从 step 40 写到 step 200：

```text
first step: 40
last step: 200
```

最后一行关键指标：

```text
step: 200
train/loss/image: 6.2548
train/loss/dyn: 4.1072
train/loss/rep: 4.1072
train/loss/rew: 0.00793
train/loss/value: 1.1816
train/loss/policy: -0.0207
train/rew: 0.00834
train/ret: 0.2926
fps/policy: 0.1559
fps/train: 79.8293
replay/items: 193
replay/replay_ratio: 512
usage/nvsmi/compute_avg/gpu0: 0.98
usage/nvsmi/memory_avg/gpu0: 0.48
```

GPU 使用情况：

```text
usage/nvsmi/compute_avg/gpu0: 0.98
usage/nvsmi/memory_avg/gpu0: 0.47 到 0.48
```

这说明训练期间 GPU 计算利用率约 98%，显存占用比例约 47-48%，JAX/CUDA 路径实际参与训练。

结果解释：

- Crafter 环境创建成功。
- DreamerV3 模型初始化和 JAX 编译成功。
- 训练循环跑到目标 step 200。
- metrics 正常写入。
- checkpoint 正常写入。
- replay 正常写入。
- `loss_score_curves.png` 正常生成。
- `scores.jsonl` 未出现，这对 200-step smoke test 是正常的，因为短运行可能还没完成完整 episode。

结论：Crafter GPU smoke test 通过。下一阶段可以进入 Crafter 5k 小规模训练验证，建议使用新 logdir：

```text
/kaggle/working/logdir/crafter_gpu_5k
```

## 最新进展记录：2026-06-03 version 12 Crafter 5k 已推送

已将 notebook 从 `crafter_gpu_smoke` 推进到 `crafter_gpu_5k`，采用最小改动：

```text
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
CRAFTER_SMOKE_STEPS = 5000
RESULT_RUN_NAME = 'crafter_gpu_5k'
```

Crafter cell 当前使用：

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

已通过 Kaggle API 推送：

```text
Kernel version 12 successfully pushed.
https://www.kaggle.com/code/linjinru/dreamerv3-atari-pong-kaggle-setup
```

观察 version 12 时重点看：

```text
Kaggle 状态是否 COMPLETE
是否进入 Start training loop
是否至少跑到 Agent Step 5_000 附近
metrics.jsonl 是否持续写入
scores.jsonl 是否出现
checkpoint 是否保存
GPU compute/memory 是否稳定
是否出现 CUDA OOM、replay assertion 或 traceback
```

## 最新进展记录：2026-06-03 Crafter 5k r64 准备推送

version 12 使用 Crafter 默认 `run.train_ratio=512`，预计 5000 steps 运行时间约 9 小时级别，不适合当前“小规模验证”阶段。

已将本地 notebook 改为 `crafter_gpu_5k_r64`：

```text
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
CRAFTER_SMOKE_STEPS = 5000
RESULT_RUN_NAME = 'crafter_gpu_5k_r64'
```

Crafter cell 当前使用：

```text
--logdir /kaggle/working/logdir/crafter_gpu_5k_r64
--configs crafter
--run.steps 5000
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

设计目的：

- 使用新的 logdir，避免复用 `crafter_gpu_5k` 的旧 checkpoint 和日志。
- 将 `run.train_ratio` 从 512 降到 64，使 5k 验证更快完成。
- 当前仍然不是正式训练，目标是验证 5k steps 下 metrics、checkpoint、replay、score、GPU 使用是否稳定。

## 最新进展记录：2026-06-03 Crafter achievement logging smoke100 准备推送

在继续 5k 之前，先新增一个 100-step 的 Crafter logging 验证版本，用来确认 Crafter 环境是否能暴露 achievement/log 字段，以及 episode 结束时是否能写出 env stats。

本地 notebook 当前默认配置：

```text
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
CRAFTER_SMOKE_STEPS = 100
RESULT_RUN_NAME = 'crafter_logs_smoke100'
```

Crafter cell 当前使用：

```text
--logdir /kaggle/working/logdir/crafter_logs_smoke100
--configs crafter
--run.steps 100
--run.envs 1
--env.crafter.logs True
--env.crafter.logdir /kaggle/working/logdir/crafter_logs_smoke100/env_logs
--batch_size 1
--batch_length 8
--report_length 8
--replay_context 0
--jax.platform cuda
--jax.compute_dtype float32
--jax.prealloc False
```

预期现象：

- 启动日志的 Observations 应包含 `log/reward` 和 `log/achievement_*` 字段。
- 如果 100 steps 内 episode 结束，`env_logs/stats.jsonl` 会写出 episode 长度、总 reward 和 achievement 状态。
- 如果 100 steps 内 episode 未结束，`env_logs/stats.jsonl` 可能不存在，这是正常现象；本次仍可通过 Observations 验证 log 字段是否进入环境。
- 结果汇总 cell 已增加 `env_logs/stats.jsonl` 检查；文件缺失只提示，不会报错。


## ???????2026-06-03 Crafter achievement logging smoke100 ??

???? Crafter ??????? achievement/event ???????? notebook ???? 100-step logging smoke test??????? 5k ????????? `env.crafter.logs` ? `env.crafter.logdir`?

???????????

```text
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
CRAFTER_SMOKE_STEPS = 100
RESULT_RUN_NAME = 'crafter_logs_smoke100'
```

Crafter cell ?????

```text
--logdir /kaggle/working/logdir/crafter_logs_smoke100
--configs crafter
--run.steps 100
--run.envs 1
--env.crafter.logs True
--env.crafter.logdir /kaggle/working/logdir/crafter_logs_smoke100/env_logs
--batch_size 1
--batch_length 8
--report_length 8
--replay_context 0
--jax.platform cuda
--jax.compute_dtype float32
--jax.prealloc False
```

?????

```text
Observations ??? log/achievement_* ??
metrics.jsonl ????
?? 100 step ? episode ???? env_logs/stats.jsonl ?? achievement_* ??
???? episode ??????? cell ??? stats.jsonl ??????
??? traceback
```

???100 step ??????????? episode??? `env_logs/stats.jsonl` ??????????????????? Observations ???? `log/achievement_*`????????? loop ??? metrics?

## 最新进展记录：2026-06-03 version 15 输出已下载并导出

已通过 Kaggle API 确认 notebook 状态：

```text
linjinru/dreamerv3-atari-pong-kaggle-setup has status "KernelWorkerStatus.COMPLETE"
```

已下载最新输出到：

```text
kaggle_work/dreamerv3-atari-pong-kaggle-setup-output-latest/
```

本次 run 目录：

```text
kaggle_work/dreamerv3-atari-pong-kaggle-setup-output-latest/logdir/crafter_gpu_5k_trace20_r64/
```

已导出分析文件到：

```text
kaggle_work/analysis/crafter_gpu_5k_trace20_r64/
```

关键导出文件：

```text
raw_action_trace_sample20.jsonl
clean_action_trace_sample20.csv
clean_action_trace_sample20.jsonl
raw_replay_steps.jsonl
clean_replay_steps.csv
clean_replay_steps.jsonl
clean_metrics.csv
episode_summary.csv
action_counts.csv
trace_events.csv
summary.md
```

结果摘要：

```text
metrics rows: 62
last metric step: 4950
last train metric step: 4950
last episode metric: step=4857, score=2.099999986588955, length=260.0
checkpoint present: True
loss_score_curves.png present: True
scores.jsonl present: False
sampled action trace rows: 235
replay step rows exported: 4440
replay episodes seen: 27
non-zero reward replay rows: 177
```

最新训练指标：

```text
train/loss/image: 20.45855727225542
train/loss/dyn: 8.283215061426162
train/loss/rep: 8.283215061426162
train/loss/rew: 0.10915607693915566
train/loss/policy: -0.0011571885712267733
train/loss/value: 1.8493071826547385
train/rew: 0.012517608320046824
train/ret: 0.43709649730473754
replay/replay_ratio: 64.0
fps/policy: 1.2357977336306931
fps/train: 79.0910552630732
usage/nvsmi/compute_avg/gpu0: 0.98
usage/nvsmi/memory_avg/gpu0: 0.47
```

动作分布前几项：

```text
move_right: 794
do: 749
sleep: 665
move_left: 592
place_stone: 400
move_up: 365
place_table: 249
make_stone_pickaxe: 191
```

trace 事件分布：

```text
(empty): 185
wake_up: 18
collect_sapling: 10
collect_wood: 9
collect_drink: 7
```

注意：

- notebook 内置 `action_trace/action_trace.jsonl` 是轻量日志，按 `ACTION_TRACE_EVERY = 20` 每 20 个环境步采样一行，不是逐环境步完整日志。
- `clean_replay_steps.*` 是从下载到的 replay `.npz` 中导出的逐步 action/reward/done 记录。
- 下载到的 replay 文件共导出 4440 行记录；metrics 显示训练最后记录到 step 4950。
- `scores.jsonl` 没有生成，但 `metrics.jsonl` 中已有 `episode/score` 和 `episode/length` 行。
