# DreamerV3 Kaggle Notebook 学习版注释

这份文件是给学习用的，不是 Kaggle 实际运行文件。实际运行文件是：

```text
kaggle_work/dreamerv3-atari-pong-kaggle-setup/dreamerv3-atari-pong-kaggle-setup.ipynb
```

本文按 notebook 的 cell 顺序解释每一段代码在做什么、为什么这样写、看到输出时应该怎么理解。

## 当前最新状态：2026-06-02

这份学习文档的大部分内容是在 Atari/Pong 1k 验证阶段写的，因此很多解释仍以 `atari_pong_gpu_1k` 为例。当前实际 notebook 已经推进到下一阶段：**Crafter GPU smoke test**。

当前本地 notebook 默认开关已经改为：

```python
RUN_CPU_DEBUG_SMOKE = False
RUN_GPU_ATARI_SMOKE = False
RUN_CRAFTER_IMPORT_CHECK = True
RUN_GPU_CRAFTER_SMOKE = True
SHOW_TENSORBOARD = False

SMOKE_STEPS = 1000
CRAFTER_SMOKE_STEPS = 200
RESULT_RUN_NAME = 'crafter_gpu_smoke'
```

也就是说：Pong 1k cell 仍然保留用于回顾和复现，但默认会跳过；下一次运行重点是新增的 Crafter GPU smoke cell。最后的结果摘要 cell 现在会根据 `RESULT_RUN_NAME` 读取对应目录，例如当前是：

```text
/kaggle/working/logdir/crafter_gpu_smoke
```

## 整体目标

这个 notebook 的目标不是训练出很强的 Pong 智能体，而是先验证 Kaggle 环境是否能跑通 DreamerV3：

1. Kaggle 能不能联网 clone DreamerV3。
2. 依赖能不能安装成功。
3. Atari ROM 和 ALE 环境能不能用。
4. JAX 能不能看到 GPU。
5. DreamerV3 能不能初始化模型、编译训练函数、进入训练循环。

这些都通过后，后续再从 Atari/Pong 切到 Crafter。

## 常见概念

- `Kaggle runtime`：Kaggle 给 notebook 分配的一台临时机器，可以是 CPU，也可以带 GPU。
- `GPU`：显卡，用来加速深度学习训练。
- `JAX`：DreamerV3 使用的深度学习计算框架，类似 PyTorch/TensorFlow。
- `CUDA`：NVIDIA GPU 的计算接口。JAX 能看到 CUDA，才说明它能用 GPU。
- `DreamerV3`：这里要运行的强化学习算法。
- `Atari/Pong`：当前用于测试环境链路的游戏任务。
- `Crafter`：后续真正想训练的小人环境。
- `logdir`：日志和 checkpoint 保存目录。
- `checkpoint`：训练过程保存下来的模型状态，后续可以继续训练或分析。
- `smoke test`：很短的小测试，目的只是确认链路能跑，不追求效果。
- `subprocess`：从当前 notebook 里再启动一个新的 Python 进程运行代码。

## Cell 1: 标题和说明

原始内容：

```markdown
# DreamerV3 on Kaggle: JAX/CUDA + Atari Pong

This notebook is structured for Kaggle runtime validation...
```

这一格是 Markdown 文本，不会执行代码。它说明这个 notebook 是为了验证 Kaggle 上的 JAX/CUDA/DreamerV3/Atari/Pong 路径。

## Cell 2: 运行开关

当前代码：

```python
# 是否运行 CPU debug 测试。
# False 表示跳过，避免浪费时间。
RUN_CPU_DEBUG_SMOKE = False

# 是否运行 GPU Atari/Pong 测试。
# 当前要跑 1000 steps，所以这里是 True。
RUN_GPU_ATARI_SMOKE = True

# 是否检查 Crafter 依赖和环境。
# 当前还在 Pong 阶段，所以先关掉。
RUN_CRAFTER_IMPORT_CHECK = False

# 是否打开 TensorBoard。
# 当前最后一格已经改成静态结果摘要和曲线，这个开关暂时保留但不再使用。
SHOW_TENSORBOARD = False

# DreamerV3 源码 clone 到 /tmp。
# 不放到 /kaggle/working，是为了避免 Kaggle 把整个源码目录当成输出保存下来。
REPO_DIR = '/tmp/dreamerv3'

# 日志和 checkpoint 的根目录。
# Kaggle 会保存 /kaggle/working 里的输出。
LOGDIR = '/kaggle/working/logdir'

# 本次小规模训练的步数。
# 之前 smoke test 是 100，现在推进到 1000。
SMOKE_STEPS = 1000
```

这一格最重要，因为它控制后面哪些测试会运行。

当前设置表示：

- 不跑 CPU debug。
- 跑 GPU Atari/Pong。
- 不跑 Crafter 检查。
- 不开 TensorBoard 服务，最后一格改为读取日志并画静态曲线。
- 训练步数是 1000。

## Cell 3: 打印 Kaggle 和机器环境

代码作用：

```python
import os
import platform
import shutil
import subprocess
import sys
```

这些是 Python 标准库：

- `os`：读取环境变量、当前目录等。
- `platform`：查看系统信息。
- `shutil`：检查命令是否存在。
- `subprocess`：运行外部命令，比如 `nvidia-smi`。
- `sys`：查看 Python 版本等。

关键输出：

```python
print('Python:', sys.version)
print('Platform:', platform.platform())
print('Kaggle kernel run type:', os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))
print('Kaggle URL base:', os.environ.get('KAGGLE_URL_BASE'))
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('Working directory:', os.getcwd())
```

这些行只是打印诊断信息，方便失败时判断：

- Python 是哪个版本。
- 当前是不是 Kaggle。
- Kaggle 有没有暴露 GPU。
- 当前工作目录在哪里。

最后这段：

```python
if shutil.which('nvidia-smi'):
    subprocess.run(['nvidia-smi'], check=False)
else:
    print('nvidia-smi not found; this is expected before enabling a Kaggle GPU accelerator.')
```

`nvidia-smi` 是 NVIDIA 显卡查询工具。如果 Kaggle 分配了 GPU，一般能看到显卡型号、显存、驱动版本。如果没开 GPU，看不到它是正常的。

## Cell 4: 下载 DreamerV3 源码

当前代码：

```python
import os
import shutil
import subprocess

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)
subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/danijar/dreamerv3.git', REPO_DIR], check=True)
print('Repo ready at', REPO_DIR)
```

逐行解释：

- `os.path.exists(REPO_DIR)`：检查 `/tmp/dreamerv3` 是否已经存在。
- `shutil.rmtree(REPO_DIR)`：如果已经存在，就删掉旧目录，保证这次 clone 是干净的。
- `git clone --depth 1 ...`：从 GitHub 下载 DreamerV3 源码，只下载最近一次提交，速度更快。
- `check=True`：如果命令失败，直接让 cell 报错停下来。

为什么 clone 到 `/tmp/dreamerv3`：

Kaggle 会把 `/kaggle/working` 当成输出保存。如果把整个 DreamerV3 源码放进去，输出会很大，下载也慢。`/tmp` 是临时目录，适合放源码和安装中间文件。

## Cell 5: 安装依赖

当前代码：

```python
%cd /tmp/dreamerv3
!python -m pip install -U pip setuptools wheel
!python -m pip install -U -r requirements.txt
```

解释：

- `%cd /tmp/dreamerv3`：切换 notebook 当前目录到 DreamerV3 源码目录。
- `pip install -U pip setuptools wheel`：升级 Python 包安装工具。
- `pip install -U -r requirements.txt`：安装 DreamerV3 需要的依赖。

这里会安装 JAX、ALE、AutoROM、DreamerV3 依赖等。失败时常见原因包括网络问题、包版本冲突、Kaggle 镜像变化。

## Cell 6: 安装 Atari ROM

当前代码：

```python
!AutoROM --accept-license
```

Atari 游戏环境需要 ROM 文件，也就是游戏本体数据。`AutoROM` 会自动安装这些 ROM。

`--accept-license` 表示接受 ROM 安装许可，否则命令可能停下来等待人工输入，Kaggle 自动运行时就会卡住。

## Cell 7: 在新 Python 进程里检查依赖 import

这一格看起来复杂，是因为它没有直接在 notebook 当前进程里 import，而是用 `subprocess` 启动了一个新 Python。

核心结构：

```python
code = r'''
...
'''
subprocess.run([sys.executable, '-c', code], check=True)
```

含义是：

- 把一段 Python 代码保存到字符串 `code`。
- 用当前 Python 解释器启动一个新进程。
- 在新进程里执行 `code`。

为什么要这样做：

Kaggle notebook 进程可能已经提前加载了某个版本的 `numpy`。安装 DreamerV3 依赖后，`numpy` 版本可能被换掉。如果继续在原进程里 import 二进制包，容易出现 ABI 错误。新进程会重新加载安装后的包版本，更干净。

内部代码：

```python
modules = ['jax', 'numpy', 'ale_py', 'gymnasium', 'embodied', 'dreamerv3']
for name in modules:
    mod = importlib.import_module(name)
    version = getattr(mod, '__version__', 'unknown')
    print(f'{name}: {version}')
```

这段逐个 import 核心模块，并打印版本。

- `jax`：深度学习计算框架。
- `numpy`：数组计算基础库。
- `ale_py`：Atari Learning Environment 的 Python 包。
- `gymnasium`：强化学习环境接口。
- `embodied`：DreamerV3 项目里的环境和训练工具包。
- `dreamerv3`：DreamerV3 算法代码。

如果这一格成功，说明核心依赖至少能导入。

## Cell 8: 检查 JAX 和 GPU

这一格同样用新 Python 进程运行：

```python
import jax

print('JAX version:', jax.__version__)
print('JAX default backend:', jax.default_backend())
print('JAX devices:', jax.devices())
try:
    print('JAX CUDA devices:', jax.devices('cuda'))
except Exception as exc:
    print('No CUDA backend available yet:', type(exc).__name__, exc)
```

重点看：

```text
JAX default backend: gpu 或 cuda
JAX devices: [cuda:0, ...]
JAX CUDA devices: [cuda:0, ...]
```

如果看到 `cuda`，说明 JAX 能识别 GPU。

如果这里失败，后面的 DreamerV3 GPU 训练一般也会失败。常见原因是 Kaggle 没开 GPU、JAX CUDA 依赖没装好、运行环境不匹配。

## Cell 9: 检查 Atari/Pong 环境

这一格检查游戏环境是否能创建出来。

核心代码：

```python
import numpy as np
import ale_py.roms as roms
from embodied.envs.atari import Atari

print('ALE pong ROM:', roms.get_rom_path('pong'))
env = Atari('pong', repeat=4, size=(96, 96), gray=True, noops=0, sticky=True, actions='all', seed=0)
obs = env.step({'reset': True, 'action': np.int32(0)})
print('DreamerV3 Atari wrapper OK')
print('  image shape:', obs['image'].shape, obs['image'].dtype)
print('  action space:', env.act_space['action'])
```

逐项解释：

- `roms.get_rom_path('pong')`：确认 Pong ROM 文件存在。
- `Atari('pong', ...)`：用 DreamerV3 自己的 Atari wrapper 创建 Pong 环境。
- `repeat=4`：一个动作重复 4 帧，这是 Atari 强化学习常见设置。
- `size=(96, 96)`：把画面缩放到 96x96。
- `gray=True`：使用灰度图，减少输入维度。
- `sticky=True`：使用 sticky action，让环境更接近标准 Atari 评测设置。
- `obs = env.step(...)`：重置环境并走一步。

为什么不用 `gym.make('ALE/Pong-v5')` 当核心检查：

DreamerV3 实际训练用的是自己的 `embodied.envs.atari.Atari`。Gymnasium 的环境注册可能因为版本不同失败，但这不一定影响 DreamerV3 自己的 wrapper。所以 Gymnasium 检查只作为 optional check。

## Cell 10: 可选 Crafter 检查

当前开关是：

```python
RUN_CRAFTER_IMPORT_CHECK = False
```

所以这一格现在会输出：

```text
Crafter import check skipped.
```

如果以后改成 True，它会做：

```python
subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', 'crafter'], check=True)
```

也就是安装 Crafter，然后创建一个 Crafter 环境：

```python
import crafter
env = crafter.Env()
obs = env.reset()
```

这一步是为后续从 Pong 切到 Crafter 做准备。

## Cell 11: 可选 CPU debug smoke test

当前开关是：

```python
RUN_CPU_DEBUG_SMOKE = False
```

所以这一格现在跳过。

如果打开，它会运行：

```python
sys.executable, 'dreamerv3/main.py',
'--logdir', f'{LOGDIR}/atari_pong_cpu_debug',
'--configs', 'atari', 'debug',
'--task', 'atari_pong',
'--run.steps', str(SMOKE_STEPS),
'--run.envs', '1',
```

注意这里有 `debug` config。DreamerV3 的 `debug` config 会强制使用 CPU，并把模型、batch、日志间隔等调小，适合快速检查训练脚本路径。

当前我们要验证 GPU，所以它保持关闭。

## Cell 12: GPU Atari/Pong 1000-step 小规模训练

这是当前最关键的 cell。

当前代码：

```python
if RUN_GPU_ATARI_SMOKE:
    import subprocess
    import sys
    subprocess.run([
        sys.executable, 'dreamerv3/main.py',
        '--logdir', f'{LOGDIR}/atari_pong_gpu_1k',
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
else:
    print('GPU Atari/Pong smoke test skipped. Enable Kaggle GPU before running it.')
```

逐项解释：

```python
sys.executable, 'dreamerv3/main.py'
```

用当前 Python 运行 DreamerV3 的主训练脚本。

```python
'--logdir', f'{LOGDIR}/atari_pong_gpu_1k'
```

把日志和 checkpoint 保存到：

```text
/kaggle/working/logdir/atari_pong_gpu_1k
```

这个名字表示这是 Atari/Pong GPU 1000-step 小规模训练。

```python
'--configs', 'atari'
```

使用 DreamerV3 内置的 Atari 配置。

```python
'--task', 'atari_pong'
```

指定任务是 Atari Pong。

```python
'--run.steps', str(SMOKE_STEPS)
```

训练步数来自 Cell 2 的 `SMOKE_STEPS`。当前是 1000。

```python
'--run.envs', '1'
```

只启动 1 个环境。这样慢一点，但更省资源，也更容易排查问题。

```python
'--batch_size', '1'
```

每次训练采样的 batch 数量很小，目的是降低显存压力。

```python
'--batch_length', '8'
```

每条训练序列长度是 8。之前 100-step 测试时默认长度太大，replay 里连续数据不够，所以这里调小。

```python
'--report_length', '8'
```

生成报告时使用的序列长度也调小，避免短跑时 replay 数据不足。

```python
'--replay_context', '0'
```

不额外要求前置上下文帧，进一步降低短跑时对连续数据长度的要求。

```python
'--jax.platform', 'cuda'
```

明确要求 JAX 使用 NVIDIA GPU。

```python
'--jax.compute_dtype', 'float32'
```

使用 float32 计算。虽然 bfloat16 可能更省显存，但在 smoke test 阶段 float32 更直接，减少 dtype 兼容问题。

```python
'--jax.prealloc', 'False'
```

告诉 JAX 不要一开始就预占大部分 GPU 显存。Kaggle 共享/限制环境里这样更稳。

如果这个 cell 成功，通常会看到类似输出：

```text
JAX devices ... cuda
Initializing parameters...
Done initializing!
Compiling train and report...
Done compiling!
Start training loop
```

成功后，Kaggle 输出目录里应该有：

```text
/kaggle/working/logdir/atari_pong_gpu_1k/config.yaml
/kaggle/working/logdir/atari_pong_gpu_1k/ckpt/...
```

## Cell 13: 输出训练结果摘要和曲线

这格原来是 TensorBoard。TensorBoard 是一个网页服务，适合人工交互查看曲线，但在 Kaggle batch run 里可能一直运行，导致 notebook 不容易自动结束。

所以现在最后一格改成普通 Python 分析代码：读取训练日志文件，打印最后一次 loss，并画静态 loss/score 曲线。它运行完会正常结束。

它读取的文件是：

```text
/kaggle/working/logdir/atari_pong_gpu_1k/metrics.jsonl
/kaggle/working/logdir/atari_pong_gpu_1k/scores.jsonl
```

`metrics.jsonl` 里通常有训练 loss、fps、replay ratio 等指标。

`scores.jsonl` 里通常有 episode score。但 1000 steps 很短，Pong 不一定已经完成一整局，所以 score 可能暂时为空。

核心逻辑可以理解为：

```python
run_dir = Path(LOGDIR) / 'atari_pong_gpu_1k'
metrics_path = run_dir / 'metrics.jsonl'
scores_path = run_dir / 'scores.jsonl'
```

先确定本次运行的日志目录。

然后读取 JSONL 文件：

```python
metrics = load_jsonl(metrics_path)
scores = load_jsonl(scores_path)
```

JSONL 的意思是：每一行都是一个 JSON 对象。DreamerV3 每次写日志时，会往文件里追加一行。

最后一格会打印类似：

```text
Latest metric step: ...
Latest train losses:
  train/loss/con: ...
  train/loss/dyn: ...
  train/loss/image: ...
  train/loss/policy: ...
  train/loss/rep: ...
  train/loss/repval: ...
  train/loss/rew: ...
  train/loss/value: ...
```

这些 loss 的含义大致是：

- `train/loss/image`：世界模型重建画面的误差。
- `train/loss/dyn`：动力学模型预测下一个隐状态的误差。
- `train/loss/rep`：表示学习相关损失。
- `train/loss/rew`：奖励预测误差。
- `train/loss/value`：价值函数损失。
- `train/loss/policy`：策略优化相关损失。
- `train/loss/con`：continue/head 相关损失，和 episode 是否继续有关。
- `train/loss/repval`：representation value 相关损失。

注意：这些 loss 在强化学习里不是越低就马上代表智能体越强。当前 1000 steps 很短，主要看训练流程是否稳定、loss 是否能正常记录。

## 当前推荐运行方式

当前 notebook 的设置对应：

```text
RUN_GPU_ATARI_SMOKE = True
SMOKE_STEPS = 1000
logdir = /kaggle/working/logdir/atari_pong_gpu_1k
task = atari_pong
batch_size = 1
batch_length = 8
report_length = 8
replay_context = 0
jax.platform = cuda
jax.compute_dtype = float32
jax.prealloc = False
```

这一步的目标是验证 1000 steps 是否稳定，不是训练出高分。

## 如果运行失败，优先看什么

优先复制最后 30-100 行，尤其是包含这些关键词的部分：

```text
Traceback
AssertionError
RuntimeError
ValueError
CUDA_ERROR
out of memory
CalledProcessError
```

不要只复制这些 CUDA/XLA 噪声：

```text
Unable to register cuFFT factory
Unable to register cuDNN factory
Unable to register cuBLAS factory
computation placer already registered
```

这些通常不是最终失败原因，要继续往下看真正的 traceback。

## 这份 notebook 的学习主线

你可以按这个思路理解整个流程：

1. 先设置开关和目录。
2. 打印机器环境，确认是不是有 GPU。
3. 下载 DreamerV3 源码。
4. 安装依赖。
5. 安装 Atari ROM。
6. 用新 Python 进程检查包能不能 import。
7. 用新 Python 进程检查 JAX 能不能看到 GPU。
8. 检查 DreamerV3 自己的 Atari/Pong wrapper 能不能创建环境。
9. 暂时跳过 Crafter。
10. 暂时跳过 CPU debug。
11. 正式跑 GPU Atari/Pong 1000-step 小训练。
12. 读取 metrics/scores，输出 loss 并画静态曲线。

也就是：先检查工具，再检查环境，再检查 GPU，最后才进入训练。
