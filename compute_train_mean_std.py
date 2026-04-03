from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# 常见图像扩展名。脚本会递归扫描这些文件。
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 RHPE_train 灰度图像的 mean/std")
    parser.add_argument(
        "--image-dir",
        default="dataset/RHPE_train",
        help="训练图像目录，默认使用 dataset/RHPE_train",
    )
    parser.add_argument(
        "--output",
        default="train_mean_std.json",
        help="统计结果保存路径，默认写入根目录下的 train_mean_std.json",
    )
    return parser.parse_args()


def iter_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"给定路径不是目录: {image_dir}")
    return sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_grayscale_array(path: Path) -> np.ndarray:
    # 统一按灰度单通道读取，并转成 [0, 1] 浮点值，方便后续直接得到可用于归一化的 mean/std。
    with Image.open(path) as image:
        grayscale = image.convert("L")
        array = np.asarray(grayscale, dtype=np.float64) / 255.0
    if array.ndim != 2:
        raise ValueError(f"图像不是二维灰度图: {path}")
    return array


def compute_mean_std(image_paths: list[Path]) -> dict[str, object]:
    total_files = len(image_paths)
    valid_files = 0
    skipped_files = 0
    pixel_count = 0
    pixel_sum = 0.0
    pixel_sum_sq = 0.0
    skipped_details: list[dict[str, str]] = []

    for image_path in image_paths:
        try:
            array = load_grayscale_array(image_path)
        except Exception as exc:  # pragma: no cover - 真实坏图保护
            skipped_files += 1
            warning = f"警告: 跳过损坏或不可读图像 -> {image_path} | 原因: {exc}"
            print(warning, file=sys.stderr)
            skipped_details.append({"path": str(image_path), "error": str(exc)})
            continue

        valid_files += 1
        pixel_count += int(array.size)
        pixel_sum += float(array.sum(dtype=np.float64))
        pixel_sum_sq += float(np.square(array, dtype=np.float64).sum(dtype=np.float64))

    if valid_files == 0 or pixel_count == 0:
        raise RuntimeError("没有可用于统计的有效图像，请检查 dataset/RHPE_train 是否存在可读图片。")

    mean = pixel_sum / pixel_count
    variance = max(pixel_sum_sq / pixel_count - mean * mean, 0.0)
    std = math.sqrt(variance)

    return {
        "total_image_files": total_files,
        "used_image_files": valid_files,
        "skipped_image_files": skipped_files,
        "pixel_count": pixel_count,
        "mean": mean,
        "std": std,
        "mean_255": mean * 255.0,
        "std_255": std * 255.0,
        "skipped_details": skipped_details,
    }


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_path = Path(args.output)

    image_paths = iter_image_paths(image_dir)
    result = compute_mean_std(image_paths)
    result["image_dir"] = str(image_dir.resolve())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(f"训练图像目录: {image_dir.resolve()}")
    print(f"递归找到图像数量: {result['total_image_files']}")
    print(f"参与统计图像数量: {result['used_image_files']}")
    print(f"跳过图像数量: {result['skipped_image_files']}")
    print(f"总像素数量: {result['pixel_count']}")
    print(f"mean (0-1): {result['mean']:.8f}")
    print(f"std  (0-1): {result['std']:.8f}")
    print(f"mean (0-255): {result['mean_255']:.4f}")
    print(f"std  (0-255): {result['std_255']:.4f}")
    print(f"结果已保存到: {output_path.resolve()}")


if __name__ == "__main__":
    main()
