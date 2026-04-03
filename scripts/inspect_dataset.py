from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap


def main() -> None:
    bootstrap()
    from rhpe_boneage.data.discovery import build_dataset_index

    parser = argparse.ArgumentParser(description="检查 RHPE 数据目录与严格映射")
    parser.add_argument("--dataset-root", default="dataset", help="数据根目录")
    parser.add_argument("--verify-images", action="store_true", help="是否实际校验图像可读性")
    args = parser.parse_args()

    payload = build_dataset_index(args.dataset_root, verify_images=args.verify_images)
    print(json.dumps(payload["reports"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
