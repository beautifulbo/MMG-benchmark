# WandB 监控功能使用指南

## ✅ 已完成的功能

已为MMG-benchmark添加完整的 **Weights & Biases (WandB)** 监控支持，可以实时查看训练和评估的各项指标。

---

## 🚀 快速开始

### 1. 安装WandB（如果尚未安装）

```bash
pip install wandb
```

### 2. 登录WandB

```bash
wandb login
```

首次登录会提示输入API key，可以在 [wandb.ai/authorize](https://wandb.ai/authorize) 获取。

---

## 📊 监控内容

### 训练时自动记录的指标

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **训练损失** | `train/loss_total` | 总损失值 |
| **学习率** | `train/learning_rate` | 当前epoch的学习率 |
| **训练时间** | `train/epoch_time_sec` | 每个epoch耗时 |
| **验证指标** | `eval/valid_recall@k`, `valid_ndcg@k` 等 | 验证集上的各项推荐指标 |
| **测试指标** | `eval/test_recall@k`, `test_ndcg@k` 等 | 测试集上的各项推荐指标 |
| **最佳模型** | `best/valid_*`, `best/test_*` | 最终的最佳结果汇总 |

### 额外功能

- 📈 **实时曲线图**：自动绘制训练损失、验证/测试指标的曲线
- 🔍 **梯度监控**：自动监控模型梯度和参数分布
- 💾 **模型保存**：自动将最佳模型上传到WandB Artifacts
- ⚙️ **超参数记录**：自动记录所有配置参数

---

## 🎯 使用方法

### 方法1：默认启用（推荐）

**训练时：**
```bash
cd MMG-benchmark

# 训练CRLMMNAR模型（默认开启wandb监控）
python train.py --model CRLMMNAR --dataset baby --missing_modal 1 --missing_ratio 0.666
```

**评估时：**
```bash
# 评估时需要手动指定开启wandb
python eval.py --model CRLMMNAR --dataset baby \
    --checkpoint saved/CRLMMNAR-(999,)/best_model.pth \
    --gpu_id 0 \
    --wandb_enabled True
```

### 方法2：通过config_dict控制

在代码中调用时：

```python
config_dict = {
    'gpu_id': '0',
    'missing_modal': 1,
    'missing_ratio': 0.666,
    
    # WandB 配置
    'wandb_enabled': True,           # 启用/禁用wandb
    'wandb_project': 'My-Project',   # 项目名称（可选）
}

quick_start(model='CRLMMNAR', dataset='baby', config_dict=config_dict)
```

### 方法3：完全禁用WandB

```bash
# 训练时不使用wandb
python train.py --model CRLMMNAR --dataset baby --missing_modal 1 --missing_ratio 0.666
# 然后修改train.py中的默认值为False，或传入config_dict:
# config_dict = {'wandb_enabled': False}
```

---

## 🖥️ WandB Dashboard 查看

训练启动后，终端会显示：

```
[WandB] ✓ Monitoring initialized!
       Project: MMG-Benchmark
       Run name: CRLMMNAR_baby_miss0.666_all
       Track at: https://wandb.ai/<your-username>/MMG-Benchmark/runs/<run-id>
```

点击链接即可打开 **WandB Dashboard**，可以查看：

1. **📉 Charts面板** - 所有指标的实时曲线图
2. **🔍 System面板** - GPU利用率、内存占用等系统信息
3. **⚙️ Config面板** - 所有超参数配置
4. **📦 Artifacts面板** - 保存的模型文件

---

## 📁 新增/修改的文件

### 新增文件
- [`utils/wandb_monitor.py`](utils/wandb_monitor.py) - WandB监控工具类

### 修改文件
- [`train.py`](train.py) - 集成训练过程监控
- [`eval.py`](eval.py) - 集成评估过程监控

---

## ⚙️ 高级配置

### 自定义项目名称和运行名称

```python
config_dict = {
    # ...其他配置...
    
    'wandb_project': 'CRL-MMNAR-Experiments',  # 自定义项目名
}
```

运行名称会自动生成，格式为：
```
{Model}_{Dataset}_miss{MissingRatio}_{ModalityType}
```
例如：`CRLMMNAR_baby_miss0.666_all`

### 在代码中手动创建monitor

```python
from utils.wandb_monitor import WandBMonitor, init_wandb_monitor

# 方式1：快速初始化
monitor = init_wandb_monitor(config, project_name="MyProject")

# 方式2：完全自定义
monitor = WandBMonitor(
    config=config,
    project_name="My-Custom-Project",
    run_name="experiment-v1",  # 自定义run名称
    enabled=True
)

# 手动记录自定义指标
monitor.log_train_metrics(epoch=0, train_loss=0.5)
monitor.log_eval_metrics(epoch=0, valid_result={'recall@10': 0.123})
monitor.finish()
```

---

## 🐛 常见问题

### Q1: WandB未安装怎么办？
A: 系统会自动检测并禁用wandb，不会影响正常训练。安装命令：`pip install wandb`

### Q2: 如何离线模式运行？
A: 设置环境变量：
```bash
export WANDB_MODE=offline
```
数据会保存在本地，稍后可以用 `wandb sync` 上传。

### Q3: 如何停止监控？
A: 在config_dict中设置 `'wandb_enabled': False`

### Q4: 多次运行如何区分？
A: 每次运行会自动生成不同的Run ID，可以在WandB Dashboard中按Run分组查看。

### Q5: 如何导出数据？
A: 在WandB Dashboard中点击 "Export" 按钮，可导出CSV/JSON格式数据。

---

## 📝 示例输出

训练时的终端输出示例：

```
[WandB] ✓ Monitoring initialized!
       Project: MMG-Benchmark
       Run name: CRLMMNAR_baby_miss0.666_all
       Track at: https://wandb.ai/user/MMG-Benchmark/runs/abc123

epoch 0 training [time: 12.34s, train loss: 0.8234]
[WandB] Logging metrics for epoch 0...

epoch 0 evaluating [time: 2.34s, valid_score: 0.0523]
valid result: 
    recall@10: 0.0891    ndcg@10: 0.0445
test result: 
    recall@10: 0.0912    ndcg@10: 0.0456
██ CRLMMNAR--Best validation results updated!!!
Best model saved to: saved/CRLMMNAR-(999,)/best_model.pth
[WandB] Model checkpoint saved as artifact: best

...（后续epochs）

█████████████ BEST ████████████████
Parameters: seed=(999,),
Valid: recall@10: 0.1523    ndcg@10: 0.0789
Test: recall@10: 0.1601    ndcg@10: 0.0823
[WandB] Monitoring finished
```

---

## 🎨 可视化效果预览

在WandB Dashboard中可以看到：

1. **Training Loss Curve** - 训练损失下降曲线
2. **Validation/Test Metrics** - Recall@K, NDCG@K等指标的对比曲线  
3. **Learning Rate Schedule** - 学习率变化曲线
4. **Gradient Distributions** - 各层梯度直方图
5. **Parameter Statistics** - 参数统计信息

---

## 🔗 相关资源

- [WandB官方文档](https://docs.wandb.ai/)
- [PyTorch + WandB教程](https://docs.wandb.ai/guides/integrations/pytorch)
- [MMG-benchmark GitHub](https://github.com/example/MMG-benchmark) （假设）
