from __future__ import annotations

import argparse

from _bootstrap import bootstrap


def main() -> None:
    bootstrap()
    from rhpe_boneage.training.runner import evaluate_main

    parser = argparse.ArgumentParser(description="在测试集上评估或导出预测")
    parser.add_argument("--checkpoint", required=True, help="待评估 checkpoint")
    parser.add_argument("--config", default=None, help="可选配置文件，会覆盖 checkpoint 内配置")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="覆盖配置，格式 key=value")
    args = parser.parse_args()
    evaluate_main(
        checkpoint_path=args.checkpoint,
        split="test",
        config_path=args.config,
        overrides=args.overrides,
    )


if __name__ == "__main__":
    main()
