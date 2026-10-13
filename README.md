# 多轮对话检索增强生成

模式识别小组大作业。基础系统是：文档切分与建索引 → 构造查询 → Top-K 检索 → 拼上下文 → 大模型回答。改进方向、生成模型和评测规模尚未确定。

## 目录

| 路径 | 用途 |
|---|---|
| `src/` | 检索、查询构造、回答与评测代码 |
| `scripts/` | 下载数据、建索引、跑实验的命令 |
| `configs/` | 实验配置 |
| `paper/` | CVPR LaTeX 论文 |
| `results/` | 小体量结果表。大文件不要提交 |
| `data/` | 数据集，仅放本机，已被忽略 |
| `models/` | 模型权重，仅放本机，已被忽略 |

## 环境

每人在自己的电脑上从零建一个名为 `rag` 的 conda 环境。环境留在本机，不提交到仓库。

### 1. 安装 Miniconda

打开 <https://docs.anaconda.com/miniconda/>，下载 Windows 64-bit 安装包并安装。安装结束后关掉原来的 PowerShell，重新打开一个，执行：

```powershell
conda init powershell
```

再关掉并重新打开 PowerShell，执行 `conda --version`。能看到版本号就可以继续。

### 2. 创建环境并安装依赖

在仓库根目录执行：

```powershell
conda create -n rag python=3.10 -y
conda activate rag
pip install -r requirements.txt
```

提示符前面出现 `(rag)` 表示当前终端已经进入这个环境。用下面这条命令确认依赖可用：

```powershell
python -c "import datasets, faiss, sentence_transformers; print('ok')"
```

输出 `ok` 即安装完成。

### 3. 以后每次使用

新开一个终端后，环境默认是关闭的。先进入仓库目录，再执行：

```powershell
conda activate rag
```

看到 `(rag)` 后再运行项目里的代码。

生成模型和向量模型都下载到本机 `models/`，不使用 API 密钥。Qwen、BGE 可以直接下。若以后改用需要同意协议的模型（例如 Llama），再把 `.env.example` 复制为 `.env`，填入 `HF_TOKEN`。

## 协作

1. 仓库所有者在 GitHub 的 Settings → Collaborators 里按用户名邀请组员，对方接受后才有推送权限。
2. 每人从自己的分支改，不直接在 `main` 上开发。

```powershell
git checkout main
git pull
git checkout -b feat/你的功能名
```

改完后：

```powershell
git add .
git commit -m "说明这次改了什么"
git push -u origin feat/你的功能名
```

然后在 GitHub 上开 Pull Request，合并进 `main`。每天开始前先切回 `main` 并 `git pull`。

提交前看一下 `git status`。出现 `data/`、`models/`、`.env` 或权重文件时，不要加入提交。
