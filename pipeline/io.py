# pipeline/io.py  — minimal & stable (AccountName+Key -> connection_string)
from __future__ import annotations
import os, base64
from pathlib import Path
import yaml, fsspec
from dotenv import load_dotenv

# 项目根目录（io.py 位于 pipeline/ 下）
ROOT = Path(__file__).resolve().parents[1]

# 读取仓库根 .env（覆盖同名旧变量）
load_dotenv(ROOT / ".env", override=True)

# 读取配置
with open(ROOT / "config" / "data.yaml", "r", encoding="utf-8") as f:
    cfg: dict = yaml.safe_load(f)

os.environ.pop("AZURE_STORAGE_CONNECTION_STRING", None)
os.environ.pop("AZURE_STORAGE_SAS_TOKEN", None)

# 仅使用 AccountName + AccountKey
ACC = (os.getenv("AZURE_STORAGE_ACCOUNT_NAME") or "").strip()
KEY = (os.getenv("AZURE_STORAGE_ACCOUNT_KEY") or "").strip()
if not ACC or not KEY:
    raise RuntimeError("请在 .env 设置 AZURE_STORAGE_ACCOUNT_NAME / AZURE_STORAGE_ACCOUNT_KEY")

# 提前校验 Key（能早发现粘贴/截断问题）
base64.b64decode(KEY, validate=True)

storage_options = {"account_name": ACC, "account_key": KEY}
fs = fsspec.filesystem("az", **storage_options)

# 路径助手
def P(kind: str, subpath: str = "") -> str:
    container  = str(cfg["blob"]["container"]).strip("/")
    prefix     = str(cfg["blob"]["prefix"]).strip("/")
    version    = str(cfg["exp_root"]).strip("/")
    sub = str(subpath).strip("/")

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


# 目录确保存在（本地）
def ensure_dir_local(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

# 目录确保存在（Azure Blob，经由 fsspec/adlfs）
def ensure_dir_az(path: str) -> None:
    # 兼容传入 'az://container/...' 或 'container/...'
    p = path[5:] if isinstance(path, str) and path.startswith("az://") else path
    fs.makedirs(p, exist_ok=True)
