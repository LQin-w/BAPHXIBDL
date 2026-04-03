from __future__ import annotations

import argparse

from _bootstrap import bootstrap


def main() -> None:
    bootstrap()
    from rhpe_boneage.training.runner import train_main

    parser = argparse.ArgumentParser(description="训练 RHPE 手骨骨龄预测模型")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="覆盖配置，格式 key=value")
    args = parser.parse_args()
    train_main(config_path=args.config, overrides=args.overrides)


if __name__ == "__main__":
    main()
