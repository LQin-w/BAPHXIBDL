# RHPE 手骨骨龄预测工程框架

这是一个围绕 RHPE 数据集构建的工程化骨龄回归项目。当前主工程位于 `src/rhpe_boneage/`，通过 `scripts/*.py` 提供训练、验证、测试、推理、数据检查、Optuna 搜索和 Tk UI。仓库同时保留了 `SIMBA/` 与 `Bonet-master/` 两份参考实现，但当前主流程不会直接 import 或执行这些旧代码。

## 当前仓库的真实定位

- 任务：基于 RHPE 手部 X 光图像预测骨龄
- 默认训练目标：`relative bone age = Boneage - Chronological`
- 默认运行声明：`experiment.mode=enhanced`
- 默认模型：`ensemble`，即 `resnet18 + efficientnet_b0`
- 默认分支：`global_local`
- 默认 metadata 模式：`mlp`
- 默认图像归一化：`auto_train_stats`
- 当前 test 集只有 `ID / Male / Chronological`，没有 `Boneage` 真值

这意味着：

- `experiment.mode` 主要用于实验声明和一致性告警，不会偷偷切换模型。
- 实际行为由 `model.*`、`data.*`、`training.*` 配置决定。
- 当前项目是“受 SIMBA 和 BoNet 启发的工程实现”，不是两篇论文原始代码的逐行复现。

## 与论文的对应关系

本仓库对应论文文件 `基于深度学习的X光片手骨骨龄预测.docx`。论文中的方法名称可概括为：

```text
基于 SIMBA 与 BoNet 继承创新的多模态骨龄评估方法
```

论文主张的核心链路与代码实现关系如下：

- 继承 BoNet：使用 ROI bbox、keypoints、全局 heatmap、局部 patch、局部 heatmap 和 ROI 几何信息。
- 继承 SIMBA：使用 `Male`、`Chronological` 作为元信息，并默认预测 `Boneage - Chronological` 的相对骨龄。
- 本文增强：使用 `resnet18 + efficientnet_b0` 双骨干、Heatmap Guidance、CBAM、全局/局部分支协同和 metadata 融合。
- 工程落地：所有训练、评估、绘图和结果追溯都在 `src/rhpe_boneage/` 与 `scripts/` 中实现，输出以每个 `outputs/*/` 目录内的 JSON/CSV/PNG 为准。

论文正文报告的主实验结果来自当前仓库保留的：

```text
outputs/150-8-no2_train_20260408_110554
```

该目录的 `best_metrics.json` 显示：best epoch 为 74，验证集 `MAE=7.4158`、`MAD=5.8906`、`R2=0.9446`。由于 RHPE test CSV 不含 `Boneage`，论文和本仓库都以验证集作为本地可计算指标的最终评估依据。

## 目录结构

```text
configs/                   当前主工程配置
scripts/                   训练、评估、推理、UI、数据检查入口
src/rhpe_boneage/          主工程代码
dataset/                   RHPE 数据与标注
outputs/                   训练/评估产物
single_infer/              单例推理示例数据
SIMBA/                     官方/参考 SIMBA 代码快照
Bonet-master/              官方/参考 BoNet 代码快照
requirements.txt           Python 依赖
```

主工程内部结构：

```text
src/rhpe_boneage/
  config.py                YAML + CLI override 合并
  data/                    数据发现、严格索引、数据集与增强
  models/                  backbone、CBAM、本地分支、多模态融合
  training/                训练循环、loss、指标、checkpoint、评估
  utils/                   日志、设备探测、JSON/绘图工具
```

## 数据集布局

当前仓库已经包含 RHPE 数据目录，主流程默认从 `dataset/` 自动发现：

```text
dataset/
  RHPE_train/
  RHPE_val/
  RHPE_test/
  RHPE_Annotations/
    RHPE_Boneage_train.csv
    RHPE_Boneage_val.csv
    RHPE_Gender_Chronological_test.csv
    RHPE_anatomical_ROIs_train.json
    RHPE_anatomical_ROIs_val.json
    RHPE_anatomical_ROIs_test.json
    Readme.txt
```

当前本地数据检查结果：

- `train`: 5491 张图像，5491 条严格匹配记录
- `val`: 713 张图像，713 条严格匹配记录
- `test`: 79 张图像，79 条严格匹配记录
- 当前 `scripts/inspect_dataset.py --dataset-root dataset` 结果为 0 缺图、0 缺 CSV、0 缺 ROI、0 重复 ID

论文中使用的数据范围与当前 `dataset_summary.json` 对齐：

| split | 样本数 | Chronological 范围 | Boneage 范围 | 说明 |
| --- | ---: | ---: | ---: | --- |
| train | 5491 | 10-242 | 12-228 | 用于训练 |
| val | 713 | 11-265 | 12-228 | 用于验证和本地最终指标 |
| test | 79 | 29-201 | 无 | 只导出预测，不计算误差 |

字段约定：

- train / val CSV：`ID, Male, Boneage, Chronological`
- test CSV：`ID, Male, Chronological`
- ROI JSON：COCO 风格，包含 `bbox`、`keypoints`

`dataset/RHPE_Annotations/Readme.txt` 还记录了数据集发布后的官方清理说明；当前仓库中的 split 计数已经与这份说明一致。

## 环境安装

代码使用了 Python 3.10+ 的语法和 `torch>=2.1` 的接口，建议环境至少满足：

- Python 3.10+
- PyTorch 2.1+
- torchvision 0.16+

最小安装方式：

```bash
python -m pip install -r requirements.txt
```

依赖见 `requirements.txt`：

- `torch`, `torchvision`, `torchaudio`
- `numpy`, `pandas`, `scikit-learn`
- `Pillow`, `opencv-python`, `albumentations`
- `matplotlib`, `optuna`, `tqdm`, `PyYAML`

论文记录的实验环境为：

- CPU：12th Gen Intel Core i7-12700H
- 内存：16GB
- GPU：NVIDIA GeForce RTX 3060 Laptop
- 系统：Ubuntu 22.04 WSL2
- Python：3.12.12
- PyTorch：2.10.0+cu130
- torchvision：0.25.0+cu130
- CUDA：13.0

`requirements.txt` 写的是最低工程依赖；论文结果应以每个输出目录中的 `runtime.json` 和 `effective_params.json` 作为复现实验记录。

## 快速开始

先检查数据：

```bash
python scripts/inspect_dataset.py --dataset-root dataset
python scripts/inspect_dataset.py --dataset-root dataset --verify-images
```

默认训练：

```bash
python scripts/train.py --config configs/default.yaml
```

按论文主实验结构复跑：

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --set experiment.name=150-8-no2 \
  --set runtime.allow_cpu_fallback=false \
  --set training.epochs=150 \
  --set training.compile=true \
  --set training.early_stopping_patience=16 \
  --set training.early_stopping_min_delta=0.01
```

也可以使用已整理的 overlay：

```bash
python scripts/train.py --config configs/ablation_innovation_00_full.yaml
```

启动图形界面：

```bash
python scripts/train_ui.py
```

用 UI 的当前参数直接启动训练而不打开窗口：

```bash
python scripts/train_ui.py --auto-run --config configs/default.yaml
```

续训：

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --set training.resume_checkpoint=outputs/xxx/model/last_checkpoint.pt
```

验证 / 测试：

```bash
python scripts/validate.py --checkpoint outputs/xxx/model/best_model.pt
python scripts/test.py --checkpoint outputs/xxx/model/best_model.pt
```

推理：

```bash
python scripts/infer.py --checkpoint outputs/xxx/model/best_model.pt
```

对任意 RHPE 风格目录推理：

```bash
python scripts/infer.py \
  --checkpoint outputs/xxx/model/best_model.pt \
  --image-dir path/to/images \
  --csv-path path/to/annotations.csv \
  --roi-json-path path/to/rois.json
```

仓库内置了一个单例推理样例：

```bash
python scripts/infer.py \
  --checkpoint outputs/xxx/model/best_model.pt \
  --image-dir single_infer/images \
  --csv-path single_infer/meta.csv \
  --roi-json-path single_infer/roi.json
```

Optuna 搜索：

```bash
python scripts/tune.py --config configs/default.yaml
```

## 配置系统

训练、UI 和 Optuna 搜索的配置加载顺序为：

```text
configs/default.yaml
-> 你传入的 --config
-> 命令行 --set key=value 覆盖
```

验证、测试和推理会先读取 checkpoint 内保存的配置，再应用你额外传入的配置：

```text
configs/default.yaml
-> checkpoint 内的 config
-> 可选 --config
-> 命令行 --set key=value 覆盖
```

这意味着：

- `configs/default.yaml` 是完整配置。
- `configs/quality.yaml`、`configs/speed.yaml` 和 `configs/ablation_innovation_*.yaml` 只是 overlay，不是完整独立配置。
- `configs/simba_like.yaml` 与 `configs/Bonet-master_like.yaml` 是更接近论文思路的整套预设。

当前配置文件含义：

- `configs/default.yaml`：当前推荐默认工程配置
- `configs/simba_like.yaml`：SIMBA 风格基线，`relative` 目标 + `simba_multiplier`
- `configs/Bonet-master_like.yaml`：BoNet 风格基线，`direct` 目标 + `global_only` + gender metadata
- `configs/quality.yaml`：在默认配置上增强正则和训练强度
- `configs/speed.yaml`：在默认配置上降低分辨率并偏向更快训练
- `configs/ablation_innovation_00_full.yaml`：150 epoch 完整创新配置
- `configs/ablation_innovation_01_wo_efficientnet_b0.yaml`：移除 EfficientNet 分支，仅保留 ResNet
- `configs/ablation_innovation_02_wo_heatmap_guidance.yaml`：关闭全局 heatmap guidance
- `configs/ablation_innovation_03_wo_cbam.yaml`：关闭全局与局部 CBAM
- `configs/ablation_innovation_04_wo_local_branch.yaml`：关闭局部分支，使用 `global_only`

常见运行方式：

```bash
python scripts/train.py --config configs/simba_like.yaml
python scripts/train.py --config configs/Bonet-master_like.yaml
python scripts/train.py --config configs/quality.yaml
python scripts/train.py --config configs/speed.yaml
python scripts/train.py --config configs/ablation_innovation_00_full.yaml
```

默认目标的计算方式：

```text
relative_age = Boneage - Chronological
predicted_relative_age = model(...)
final_boneage = Chronological + predicted_relative_age
```

如果你改成：

- `model.target_mode=direct`：直接回归骨龄
- `model.relative_target_direction=chronological_minus_boneage`：相对年龄方向反转

## 论文方法与代码实现

当前默认主流程接入 7 类输入，和论文“输入模态及作用”表一致：

| 输入 | 来源 | 代码中的作用 |
| --- | --- | --- |
| 灰度全局图像 | X 光图像 + bbox 裁剪 | 全局视觉分支输入 |
| 全局 ROI heatmap | keypoints 高斯响应 | Heatmap Guidance 区域引导 |
| 局部 patch | keypoints 周围裁切 | 局部分支细粒度纹理输入 |
| 局部 heatmap | 同位置 heatmap 裁切 | 局部分支关键区域提示 |
| ROI 几何向量 | bbox + keypoints 归一化拼接 | ROI 几何编码输入 |
| `Male` | CSV 元信息 | 性别先验 |
| `Chronological` | CSV 元信息 | 实际年龄先验，且参与最终骨龄恢复 |

关键代码映射：

- `src/rhpe_boneage/data/discovery.py`：自动发现 train/val/test 图像目录、CSV 和 ROI JSON，并按 ID 做严格对齐。
- `src/rhpe_boneage/data/dataset.py`：读取灰度图、bbox 裁剪、生成 heatmap、提取局部 patch/heatmap、构建 ROI 几何向量。
- `src/rhpe_boneage/models/backbones.py`：封装 `torchvision` 的 ResNet 与 EfficientNet；1 通道灰度图会复制成 3 通道以兼容标准 backbone。
- `src/rhpe_boneage/models/multimodal.py`：实现 Heatmap Guidance、MetadataEncoder、FusionHead、单骨干模型和双骨干均值集成。
- `src/rhpe_boneage/models/local_branch.py`：实现局部 patch 编码、metadata 参与的局部注意力池化和 ROI 几何编码。
- `src/rhpe_boneage/models/cbam.py`：实现通道注意力 + 空间注意力的 CBAM。
- `src/rhpe_boneage/training/engine.py`：实现相对骨龄目标、最终骨龄恢复、MAE/MAD 计算和预测 CSV 输出。
- `src/rhpe_boneage/utils/plots.py`：生成论文中使用的曲线、散点图、残差图、误差直方图和 `best_metrics.json`。

模型实际结构：

1. 全局分支：`BackboneEncoder -> HeatmapGuidance -> CBAM -> pooling -> projection`
2. 局部分支：`PatchEncoder -> AttentionPool -> ROI Geometry Encoder -> local fusion`
3. 元信息分支：支持 `mlp`、`simba_multiplier`、`simba_hybrid`，主实验使用 `mlp`
4. 融合头：metadata 先生成 gate 调制视觉特征，再拼接 `visual + gated_visual + metadata`，通过 MLP 回归
5. 双骨干：`resnet18` 和 `efficientnet_b0` 各自完整前向，最后对两个预测值取平均

因此，论文里的“多模态融合”在代码中具体落到局部注意力池化、ROI 几何编码、metadata gate 和 MLP 融合头；当前实现没有单独的 Transformer 或通用 cross-attention 模块。

实现细节：

- `global_crop_mode=bbox` 时，会基于 ROI bbox 加 margin 做全局裁剪。
- 局部分支在没有有效关键点时，会回退到 bbox 中心 patch。
- `target_mode=relative` 时，训练目标是 `Boneage - Chronological`；预测后再加回 `Chronological` 得到最终骨龄。
- 目标值和 `Chronological` 都会用训练集统计量做标量标准化，checkpoint 内会保存 normalizer。

## 图像归一化与数据检查

当前默认行为：

- `data.normalization.source=auto_train_stats`
- 首次运行时自动统计 train 集灰度图 `mean/std`
- 默认缓存到 `dataset/train_mean_std.json`
- 实际使用的归一化信息会写入每次实验目录的 `image_normalization.json`

如果你想手动指定：

```yaml
data:
  normalization:
    source: manual
    mean: 0.42
    std: 0.21
```

数据索引是严格匹配的：

```text
ID -> image_path -> csv_row -> roi_annotation
```

检查项包括：

- 缺失图像
- 缺失 CSV 记录
- 缺失 ROI 标注
- 重复 CSV ID
- 重复图像 ID
- 重复 ROI ID
- 图像不可读
- 图像名与标准化 ID 不匹配

其中重复 ROI 会直接报错，避免把不可信标注带进训练。

## 输出目录

每次运行都会创建独立目录，命名规则为：

```text
outputs/{experiment.name}_{purpose}_{YYYYMMDD_HHMMSS}
```

其中 `purpose` 可能是：

- `train`
- `val`
- `test`
- `optuna`
- `trial`

训练目录的典型结构：

```text
outputs/exp_xxx/
  config.yaml
  config.json
  run_config.json
  effective_params.json
  config_summary.json
  runtime.json
  dataset_report.json
  dataset_summary.json
  dataloader.json
  image_normalization.json
  history.csv
  val_predictions.csv
  val_metrics.json
  test_predictions.csv
  test_metrics.json
  metrics.json
  best_metrics.json
  metrics_summary.csv
  run.log
  model/
    best_model.pt
    last_checkpoint.pt
  plots/
    curves.png
    loss_curve.png
    mae_curve.png
    mad_curve.png
    val_scatter.png
    val_residual.png
    error_histogram_val.png
    ...
```

说明：

- `config_summary.json`：从当前配置推导出的简明结构摘要
- `effective_params.json`：实际生效的关键参数、运行设备与 DataLoader 参数
- `metrics_summary.csv`：每个 epoch 的历史 + 额外一行 `best_model`
- `best_metrics.json`：最佳 checkpoint 的摘要
- `metrics.json`：训练入口下与 `best_metrics.json` 同步；评估入口下则是当前 split 的指标
- `run.log`：完整训练/评估日志

## test 集没有真值时会发生什么

当前 RHPE test CSV 不包含 `Boneage`，所以：

- 可以输出预测结果
- 不能计算 test `loss / MAE / MAD`
- 不能生成 `test_scatter.png`、`test_residual.png`、`error_histogram_test.png`

系统会自动退化为输出：

- `test_predictions.csv`
- `plots/test_prediction_histogram.png`
- `test_prediction_summary.json`
- `test_report_note.txt`

## UI 能力

`scripts/train_ui.py` 是一个基于 Tk 的训练界面，当前支持：

- 读取并编辑配置
- 中英文界面切换
- 选择 resume checkpoint
- 保存修改后的 YAML
- 启动训练
- 通过 `TrainingControl` 请求优雅停止

UI 里展示的是“可见字段”，默认会把主实验参数暴露出来，隐藏一部分工程/调试项。

## 参考代码的边界

仓库中的：

- `SIMBA/`
- `Bonet-master/`

保留的是论文参考代码快照。它们仍然是旧的、以 Horovod 为中心的独立训练脚本，不属于当前主工程运行链路。当前工程只是吸收了其中的部分思路，例如：

- `SIMBA`：relative target、identity markers、metadata multiplier/hybrid
- `BoNet`：ROI、keypoints、heatmap、局部信息利用

更准确的表述应该是：

- 当前项目受 `SIMBA` 启发，但不是 `SIMBA` 原始代码复现
- 当前项目受 `BoNet` 启发，但不是 `BoNet` 原始代码复现

## 仓库内已保留的示例输出

下面这些结果来自当前仓库里已经存在的本地输出目录，适合用来对照 README 与产物格式；它们不是论文级 benchmark 声明。指标来自各目录的 `best_metrics.json`，单位与 RHPE 骨龄标签一致。

| 输出目录 | 说明 | best epoch | Val MAE | Val MAD | Val R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `outputs/150-8-no2_train_20260408_110554` | 150 epoch 完整实验保留结果 | 74 | 7.4158 | 5.8906 | 0.9446 |
| `outputs/90-8-no2_train_20260414_080940` | 90 epoch 完整实验保留结果 | 64 | 7.7652 | 6.3047 | 0.9399 |
| `outputs/ablation_innovation_01_wo_efficientnet_b0_train_20260422_201325` | 消融：仅 ResNet，移除 EfficientNet | 3 | 10.4572 | 8.1162 | 0.8944 |
| `outputs/ablation_innovation_02_wo_heatmap_guidance_train_20260423_074747` | 消融：关闭 heatmap guidance | 118 | 7.7747 | 6.2461 | 0.9384 |
| `outputs/ablation_innovation_03_wo_cbam_train_20260424_074631` | 消融：关闭 CBAM | 27 | 9.8041 | 7.8008 | 0.9061 |
| `outputs/ablation_innovation_04_wo_local_branch_train_20260424_104008` | 消融：关闭局部分支 | 134 | 7.5760 | 6.2539 | 0.9436 |
| `outputs/90-8-no1-SIMBA_like_train_20260410_190034` | SIMBA 风格基线 | 86 | 9.7403 | 7.8633 | 0.9069 |
| `outputs/90-8-no1-Bonet-master_like/90-8-no1-Bonet-master_like_train_20260410_171042` | BoNet 风格基线保留结果 | 79 | 9.7740 | 7.0000 | 0.8732 |
| `outputs/merged_90-8-no1-Bonet-master_like_train_20260410` | 历史合并产物 | 79 | 9.7740 | 7.0000 | 0.8732 |

论文中的主结果对应 `150-8-no2` 这一行，即 `MAE=7.42`、`MAD=5.89`、`R2=0.9446`。论文中与 SIMBA 的参考性比较可对应 `90-8-no1-SIMBA_like_train_20260410_190034`，其验证集 `MAD=7.8633`。

注意：

- 当前仓库没有保留 `ablation_innovation_00_full` 对应命名的输出目录；完整实验可参考 `150-8-no2` 与 `90-8-no2` 两个保留结果。
- `merged_*` 目录是历史保留产物，当前主工程代码里没有对应的自动合并脚本入口。
- 如果你要报告实验结果，应优先以每次运行目录中的 `best_metrics.json`、`metrics_summary.csv`、`run_config.json` 为准。

## 已知限制

- 当前 test split 没有骨龄真值，只能做预测导出，不能做最终 test 指标评估。
- 当前主工程默认面向 RHPE 风格目录和字段命名，不是通用任意骨龄数据集适配器。
- `configs/Bonet-master_like.yaml` 与 `configs/simba_like.yaml` 是“论文风格基线”，不是对原论文网络和训练流程的严格复现。
- 参考目录 `SIMBA/` 和 `Bonet-master/` 依赖的旧环境与当前 `requirements.txt` 不完全一致，不建议与主工程混用。
- `torch.compile` 是工程优化项，不是方法核心；是否真正启用需要查看具体运行目录的 `effective_params.json`。
- 论文中的“多源信息对齐/融合”在当前代码中对应 metadata gate、局部注意力池化和 MLP 融合，不应表述为独立 Transformer/cross-attention 架构。

## 建议的论文/报告表述

如果你要基于这个仓库写论文或实验报告，推荐写法：

- 方法：一个融合 SIMBA 与 BoNet 思想的工程化多模态骨龄回归框架
- 主实验：以 `configs/default.yaml` 或你自己的 overlay 为主
- 消融轴：`ensemble_mode`、`heatmap_guidance.enabled`、`cbam.enabled`、`branch_mode`、`target_mode`、`metadata.mode`
- 复现实验依据：`config.yaml`、`run_config.json`、`effective_params.json`、`best_metrics.json`

这样可以避免把“实验声明模式”误写成“模型自动切换”。
