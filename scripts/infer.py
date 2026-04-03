from __future__ import annotations

import argparse

from _bootstrap import bootstrap, run_cli


def main() -> None:
    bootstrap()
    from rhpe_boneage.training.runner import evaluate_main

    parser = argparse.ArgumentParser(description="对任意 RHPE 风格数据执行推理")
    parser.add_argument("--checkpoint", required=True, help="待推理 checkpoint")
    parser.add_argument("--config", default=None, help="可选配置文件，会覆盖 checkpoint 内配置")
    parser.add_argument("--image-dir", default=None, help="图像目录")
    parser.add_argument("--csv-path", default=None, help="CSV 路径")
    parser.add_argument("--roi-json-path", default=None, help="ROI json 路径")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="覆盖配置，格式 key=value")
    args = parser.parse_args()

    manual_split = None
    if args.image_dir and args.csv_path and args.roi_json_path:
        manual_split = {
            "split": "test",
            "image_dir": args.image_dir,
            "csv_path": args.csv_path,
            "roi_json_path": args.roi_json_path,
        }

    evaluate_main(
        checkpoint_path=args.checkpoint,
        split="test",
        config_path=args.config,
        overrides=args.overrides,
        manual_split=manual_split,
    )


if __name__ == "__main__":
    run_cli(main)
