# src/pipeline/io.py
from __future__ import annotations
import os, base64
from functools import lru_cache
from pathlib import Path
import yaml, fsspec
from dotenv import load_dotenv

def find_project_root(
    start: str | Path | None = None,
    markers=("pyproject.toml", ".git", "config"),
) -> Path:
    """
    Find project root by walking up from `start` (default: CWD).
    Returns the first directory that contains any of the `markers`.
    """
    if start is None:
        # in scripts __file__ exists; in notebooks use CWD
        start = Path.cwd()
    else:
        start = Path(start).resolve()

    for parent in [start, *start.parents]:
        for m in markers:
            if (parent / m).exists():
                return parent
    return start  # fallback: stay where you are

# optional: allow override via env var
ROOT = Path(find_project_root())

# ---------- 1) Azure 存储配置 ----------
path_env = f"{ROOT}/.env"
load_dotenv(dotenv_path=path_env, override=False)  # 读取 .env 到环境变量（若没有 .env 也不报错）

os.environ.pop("AZURE_STORAGE_CONNECTION_STRING", None)
os.environ.pop("AZURE_STORAGE_SAS_TOKEN", None)
ACC = (os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or "").strip()
KEY = (os.getenv("AZURE_STORAGE_ACCOUNT_KEY") or "").strip()
if not ACC or not KEY:
    raise RuntimeError("请在 .env 或环境变量中设置 AZURE_STORAGE_ACCOUNT_NAME / AZURE_STORAGE_ACCOUNT_KEY")
# 提前校验 Key（能早发现粘贴/截断问题）
base64.b64decode(KEY, validate=True)

storage_options = {"account_name": ACC, "account_key": KEY}

@lru_cache(maxsize=1)
def get_fs():
    return fsspec.filesystem("az", **storage_options)
fs = get_fs()

# 创建 Azure 路径对应的目录
def ensure_dir_az(path: str) -> None:
    # 兼容 'az://container/...' 或 'container/...'
    p = path[5:] if isinstance(path, str) and path.startswith("az://") else path
    fs.makedirs(p, exist_ok=True)

# ---------- 2) 配置文件加载 ----------
path_cfg = f"{ROOT}/config/config.yaml"
def load_cfg(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
cfg = load_cfg(path_cfg)


# ---------- 3) 创建本地路径 ----------
def ensure_dir_local(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

