# 多轮对话检索增强生成

模式识别小组大作业。系统按这条链路工作：

文档预处理与索引构建 → Query 构造 → Top-K 检索 → 上下文构造 → 大模型回答

生成模型是本机的 `models/Qwen2.5-3B`。检索向量模型是 `models/all-MiniLM-L6-v2`，第一次建索引或检索时如果目录是空的，会自动下载到这里。三个数据集用同一套代码，靠命令行参数切换。

每一步怎么做，写在 `docs/` 下面对应的目录里：

| 步骤 | 说明 |
|---|---|
| 文档预处理与索引构建 | [数据集](docs/文档预处理与索引构建/数据集.md)、[索引](docs/文档预处理与索引构建/索引.md) |
| Query 构造 | [查询构造](docs/Query%20构造/查询构造.md) |
| Top-K 检索 | [检索](docs/Top-K%20检索/检索.md) |
| 上下文构造 | [上下文构造](docs/上下文构造/上下文构造.md) |
| 大模型回答 | [回答](docs/大模型回答/回答.md) |

可运行的命令和参数在 [docs/命令手册.md](docs/命令手册.md)。

## 现在做到哪

已完成：

- 三个数据集的对话和检索语料都已下载
- 上面五步的基线都能跑通。查询默认是 `question`，`history` 和 `rewrite` 可以用参数切换
- 检索结果里的 `is_gold` 标出这一条是不是标注段落

还没有：

- 把 `is_gold` 汇总成 Recall@K、F1 的评测脚本
- 自己的 Query Rewrite 模型（`rewrite` 目前用 QReCC 自带的金标改写）
- Reranker、消融实验和论文正文

## 目录

| 路径 | 用途 |
|---|---|
| `src/` | 数据加载、切分、索引、查询、检索匹配、生成 |
| `scripts/` | 下载、建索引、跑基线 |
| `docs/` | 分步说明和命令手册 |
| `configs/baseline.yaml` | 默认超参备忘。脚本读的是命令行参数，不读这个文件 |
| `data/` | 数据集和索引，只留本机 |
| `models/` | 模型权重，只留本机 |
| `results/` | 每次运行写出的 JSONL |
| `paper/` | 以后放 CVPR 论文 |

数据集长什么样、索引现在覆盖多少，见 [数据集](docs/文档预处理与索引构建/数据集.md)。QReCC 官方没有 validation，TopiOCQA 官方没有独立 test。划分怎么选，写在命令手册里。

## 怎么跑

在仓库根目录：

```bash
conda activate rag
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=4   # nvidia-smi 里选一张空闲卡

python scripts/run_baseline.py --dataset multidoc2dial --split validation --max_examples 5
```

结果写到 `results/`。QReCC、TopiOCQA 的命令和全部参数见 [docs/命令手册.md](docs/命令手册.md)。

## 环境

每人在自己电脑上建一个名为 `rag` 的 conda 环境，不提交到仓库。

```bash
conda create -n rag python=3.10 -y
conda activate rag
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

实验室机器的驱动是 CUDA 12.2，要用 cu121 的 PyTorch。默认 `pip install torch` 常会装成 CUDA 13，GPU 会不可用。

确认：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

新开终端后先 `conda activate rag`。下载 Hugging Face 文件时加上面的 `HF_ENDPOINT` 和 `HF_HUB_DISABLE_XET`。Qwen 和 MiniLM 不需要 token；以后若改用要同意协议的模型，再把 `.env.example` 复制为 `.env` 并填写 `HF_TOKEN`。

## 协作

每人从自己的分支改，不直接在 `main` 上开发。

```bash
git checkout main
git pull
git checkout -b feat/你的功能名
```

改完后提交并推送，再到 GitHub 开 Pull Request。提交前看 `git status`：`data/`、`models/`、`.env` 和权重文件不要加入提交。
