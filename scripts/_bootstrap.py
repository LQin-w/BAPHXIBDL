from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable


def bootstrap() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def run_cli(main_fn: Callable[[], None]) -> None:
    try:
        main_fn()
    except KeyboardInterrupt:
        print("执行已中断。", file=sys.stderr)
        raise SystemExit(130)
    except FileNotFoundError as exc:
        print(f"文件不存在: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except RuntimeError as exc:
        message = str(exc)
        if "torch.cuda.is_available()" in message or "请求设备" in message:
            print(
                f"设备错误: {message}\n如需临时改用 CPU，请追加参数 --set runtime.device=cpu",
                file=sys.stderr,
            )
            raise SystemExit(1)
        raise
