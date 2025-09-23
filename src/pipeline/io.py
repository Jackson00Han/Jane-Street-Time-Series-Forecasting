# src/pipeline/io.py
from __future__ import annotations
import os, base64
from functools import lru_cache
from pathlib import Path
import yaml, fsspec
from dotenv import load_dotenv

# ---------- 1) 仓库根：向上寻找 pyproject.toml ----------
def _repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(8):  # 往上最多找 8 层
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    # 兜底：如果没找到，就退回到 src/../../
    return start.resolve().parents[2]

ROOT = _repo_root(Path(__file__).parent)

# ---------- 2) 环境与配置 ----------
# 先加载仓库根的 .env（若存在）
load_dotenv(ROOT / ".env", override=True)

# 允许通过环境变量覆盖配置位置（例如 JS_CONFIG=/path/to/data.yaml）
CFG_PATH = Path(os.getenv("JS_CONFIG") or (ROOT / "config" / "data.yaml"))
if not CFG_PATH.exists():
    raise FileNotFoundError(f"配置文件不存在: {CFG_PATH}")

with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg: dict = yaml.safe_load(f)

# ---------- 3) Azure 凭据（保持你的“只用账号+密钥”的策略） ----------
# 避免误用连接串/SAS（清掉可能残留的变量）
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
    storage_options = {"account_name": ACC, "account_key": KEY}
    return fsspec.filesystem("az", **storage_options)

fs = get_fs()  # 与旧代码兼容：仍然暴露 fs 变量

# ---------- 4) 路径助手 ----------
def P(kind: str, subpath: str = "") -> str:
    """
    kind: 'az' | 'np' | 'local'
      - az:   az://container/prefix/exp_root[/sub]
      - np:   container/prefix/exp_root[/sub]   (不带协议，供 numpy.save 等)
      - local: <cfg.local.root>/<exp_root>[/sub]
    """
    container = str(cfg["blob"]["container"]).strip("/")
    prefix    = str(cfg["blob"]["prefix"]).strip("/")
    version   = str(cfg["exp_root"]).strip("/")
    sub       = str(subpath).strip("/")

    if kind == "az":
        base = f"az://{container}" + (f"/{prefix}" if prefix else "") + f"/{version}"
        return f"{base}/{sub}" if sub else base
    if kind == "np":
        base = f"{container}" + (f"/{prefix}" if prefix else "") + f"/{version}"
        return f"{base}/{sub}" if sub else base
    if kind == "local":
        base = (Path(cfg["local"]["root"]) / version).as_posix()
        return f"{base}/{sub}" if sub else base
    raise ValueError("kind must be 'az', 'np', or 'local'")

# ---------- 5) 目录确保存在 ----------
def ensure_dir_local(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_dir_az(path: str) -> None:
    # 兼容 'az://container/...' 或 'container/...'
    p = path[5:] if isinstance(path, str) and path.startswith("az://") else path
    get_fs().makedirs(p, exist_ok=True)
