# GPT-2 预训练演示说明

## 项目概述

`gpt2_pretraining.py` 是一个基于 NumPy 实现的 GPT-2 预训练演示文件，旨在展示 GPT-2 模型的预训练过程，包括模型架构、前向传播、损失计算、梯度下降和参数更新等核心步骤。

## 功能特点

- **NumPy 实现**：使用 NumPy 提供高效的数值计算
- **Wandb 集成**：支持使用 Weights & Biases 记录训练过程
- **完整的预训练流程**：包含模型初始化、数据准备、训练和测试的完整流程
- **教育价值**：代码结构清晰，注释详细，适合学习和教学

## 安装和运行

### 依赖项

- **必要依赖**：
  - NumPy：提供高效的数值计算
- **可选依赖**：
  - Weights & Biases：用于训练过程可视化

### 运行方法

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 直接运行 Python 文件：

```bash
python gpt2_pretraining.py
```

3. 代码会：
   - 使用 NumPy 实现进行计算
   - 如果安装了 Weights & Biases，记录训练过程
   - 如果未安装 Weights & Biases，跳过日志记录

## 代码结构

### 核心组件

1. **模型架构**：
   - 词嵌入（Token Embedding）
   - 位置嵌入（Positional Embedding）
   - Transformer 块（包含多头注意力和前馈网络）
   - 层归一化（Layer Normalization）
   - 输出层（Projection to Vocabulary）

2. **训练流程**：
   - 数据准备：创建训练和测试数据
   - 前向传播：计算模型输出
   - 损失计算：使用交叉熵损失
   - 梯度计算：使用有限差分法
   - 参数更新：使用梯度下降

3. **辅助功能**：
   - 分词器：将文本转换为 token ID
   - 模型初始化：创建小型 GPT 模型
   - 文本生成：使用训练后的模型生成文本

### 文件结构

```
gpt2_pretraining.py
├── 导入
├── 模型组件
│   ├── 激活函数（gelu）
│   ├── 损失函数（cross_entropy_loss）
│   ├── 模型组件（attention, mha, transformer_block, gpt2）
│   ├── 梯度计算（compute_gradients）
│   └── 参数更新（update_parameters）
├── 辅助函数
│   ├── create_vocab_and_tokenizer：创建词汇表和分词器
│   └── prepare_data：准备训练和测试数据
└── 主函数
    ├── 模型初始化
    ├── 数据准备
    ├── 训练循环
    └── 模型测试
```

## 训练数据和测试数据

### 训练数据

包含 5 个简单的英语句子，用于训练模型：

- "the quick brown fox jumps over the lazy dog"
- "the cat chased the mouse into the house"
- "a quick brown fox ran on the street"
- "the lazy dog saw a cat and ran away"
- "the fox jumps over the dog and cat"

### 测试数据

包含 3 个测试提示，用于评估模型生成能力：

- "the quick brown"
- "the cat chased"
- "a lazy dog"

## 模型配置

为了演示目的，使用了小型模型配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| 词汇表大小 | 1000 | 模型的词汇表大小 |
| 嵌入维度 | 64 | 词嵌入和位置嵌入的维度 |
| 层数 | 2 | Transformer 层数 |
| 注意力头数 | 2 | 注意力机制的头数 |
| 上下文长度 | 10 | 模型能处理的最大序列长度 |

## 运行结果分析

### 训练过程

训练过程会输出每个 epoch 的损失值，示例输出：

```
Starting pretraining...
Epoch 10/100, Loss: 2.2579
Epoch 20/100, Loss: 2.2579
...
Epoch 100/100, Loss: 2.2579
```

### 文本生成

训练完成后，模型会使用测试提示生成文本，示例输出：

```
Testing the trained model...

Test input: 'the quick brown' -> [0, 1, 2]
Generated text: the quick brown the the the the the

Test input: 'the cat chased' -> [0, 10, 18]
Generated text: the cat chased the the the the the

Test input: 'a lazy dog' -> [8, 6, 7]
Generated text: a lazy dog the the the the the
```

## 技术亮点

1. **模块化结构**：将模型组件拆分为独立函数，便于理解和维护
2. **完整的训练流程**：展示了从数据准备到模型测试的完整过程
3. **教育导向**：代码注释详细，结构清晰，适合学习和教学
4. **简化实现**：使用有限差分法计算梯度，避免了复杂的反向传播实现
5. **高效计算**：使用 NumPy 实现，提供高效的数值计算

## 限制和注意事项

1. **模型大小**：为了演示目的，使用了非常小的模型，生成能力有限
2. **训练数据**：训练数据量少，模型泛化能力有限
3. **梯度计算**：使用有限差分法计算梯度，效率低于解析梯度
4. **采样方法**：只实现了贪婪解码，没有实现其他采样方法（如 top-k、top-p 等）

## 教育价值

这个实现展示了以下核心概念：

- Transformer 架构的基本组件和工作原理
- 自回归语言建模的训练目标和方法
- 梯度下降优化过程的基本原理
- 文本生成的基本流程
- 模型训练和评估的完整流程

## 扩展建议

1. **增加训练数据**：使用更大的语料库提高模型性能
2. **使用真实的分词器**：如 BPE 或 WordPiece 分词器
3. **实现更高效的梯度计算**：使用自动微分库或手动实现反向传播
4. **添加更多采样方法**：实现 top-k、top-p 等采样方法
5. **使用 GPU 加速**：通过 CUDA 或其他 GPU 库加速计算

## 总结

`gpt2_pretraining.py` 是一个教育性质的项目，通过简化的 NumPy 实现展示了 GPT-2 模型的预训练过程。它不依赖于特定的深度学习框架，实现了完整的训练流程，适合作为学习和理解 GPT 模型的入门材料。
