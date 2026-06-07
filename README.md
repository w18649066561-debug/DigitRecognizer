# Digit Recognizer 高分 CNN 方案

本仓库是 Kaggle [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)
竞赛的一个高分训练脚本，使用 TensorFlow/Keras 构建卷积神经网络。

脚本包含数据增强、验证集划分、学习率自动调整、早停、模型 checkpoint、
多模型集成和测试时增强，可以直接读取 Kaggle 标准 CSV 数据并生成可提交的
`submission.csv` 文件。

## 项目结构

```text
DigitRecognizer/
├── train_high_score.py
├── README.md
├── .gitignore
├── train/
│   └── train.csv
└── test/
    └── test.csv
```

数据文件体积较大，已经通过 `.gitignore` 排除，不会提交到 GitHub。请从
Kaggle 下载数据后放到对应目录。

## 环境准备

建议使用 Python 3.10 或更高版本。

安装依赖：

```powershell
py -m pip install numpy pandas scikit-learn tensorflow
```

当前脚本在 Windows 原生环境可以直接运行。需要注意的是，TensorFlow 2.11
及之后的版本在 Windows 原生环境默认使用 CPU。如果想加速训练，建议使用
WSL2 或其他支持 GPU 的深度学习环境。

## 数据放置

请将 Kaggle 数据放在以下路径：

```text
train/train.csv
test/test.csv
```

其中：

- `train.csv` 需要包含 `label` 和 `pixel0` 到 `pixel783`
- `test.csv` 需要包含 `pixel0` 到 `pixel783`

## 快速检查

如果只是想确认代码流程能跑通，可以使用小样本快速检查：

```powershell
py .\train_high_score.py --quick-check --epochs 1 --models 1 --tta 1 --output test\submission_quick_check.csv
```

这个命令只用于检查训练、预测和保存文件流程，不适合作为 Kaggle 提交结果。

## 默认训练

运行：

```powershell
py .\train_high_score.py
```

默认配置：

- 训练 3 个 CNN 模型
- 每个模型最多训练 30 轮
- batch size 为 128
- 使用 10% 分层验证集
- 使用 4 轮测试时增强

默认提交文件会保存到：

```text
test/submission_high_score.csv
```

## 冲分训练

如果机器性能允许，可以增加模型数量、训练轮数和测试时增强次数：

```powershell
py .\train_high_score.py --models 5 --epochs 40 --tta 8
```

更激进的配置：

```powershell
py .\train_high_score.py --models 10 --epochs 45 --tta 12
```

CPU 训练会比较慢，建议先用默认命令跑出一个稳定结果，再根据时间继续加大
集成规模。

## 输出格式

脚本会生成 Kaggle 要求的提交文件：

```text
ImageId,Label
1,2
2,0
3,9
...
```

上传 `test/submission_high_score.csv` 到 Kaggle 即可查看分数。

## 脚本特点

- 使用卷积神经网络识别 28x28 灰度手写数字
- 使用随机旋转、平移、缩放增强训练集
- 使用 Batch Normalization 和 Dropout 提高泛化能力
- 使用 ReduceLROnPlateau 自动降低学习率
- 使用 EarlyStopping 自动恢复最佳验证集权重
- 支持多模型集成，降低单模型随机性
- 支持测试时增强，提高提交稳定性

## Git 说明

仓库只提交代码、README 和配置文件。以下内容不会提交：

- `train/*.csv`
- `test/*.csv`
- `models/`
- 历史数据和本地材料
- 训练产生的模型文件和提交文件
