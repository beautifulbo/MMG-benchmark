# MMG-Benchmark 缺失模态功能测试指南

## 环境准备

### 方案 1：使用 uv（推荐）

```bash
# 1. 安装 uv（如果尚未安装）
# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 或使用 pip:
pip install uv

# 2. 进入项目目录
cd d:\Paper\mutimodal\mode_omission\DGMRec\MMG-benchmark

# 3. 使用 uv 创建虚拟环境
uv venv --python 3.9

# 4. 激活虚拟环境
# Windows PowerShell:
.venv\Scripts\activate

# 5. 安装依赖
uv pip install -r requirements.txt

# 6. 验证安装
uv python --version
```

### 方案 2：使用 conda（备选）

```bash
# 1. 创建 conda 环境
conda create -n mmg-benchmark python=3.9 -y

# 2. 激活环境
conda activate mmg-benchmark

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 功能测试

### 测试 1: Text-Only 缺失模态生成

```bash
# 激活虚拟环境后执行
python -m data.preprocessing.missing_modality --dataset baby --modality text --missing_ratio 0.666
```

**预期输出**:
```
[Missing Modality] Mode: TEXT-ONLY
  - Total missing items: <数字>
  - Items with missing TEXT: <数字>
  - Items with missing IMAGE: 0
  - Items with missing ALL: 0

[Saved] baby/missing_items_0.666_TEXT.npy
```

**验证命令**:
```bash
python -c "
import numpy as np
data = np.load('data/baby/missing_items_0.666_TEXT.npy', allow_pickle=True).item()
print('=== Text-Only Missing Modality Data ===')
print(f'Text-missing items count: {len(data[\"t\"])}')
print(f'Image-missing items count: {len(data[\"v\"])}')
print(f'All-missing items count: {len(data[\"all\"])}')
assert len(data['v']) == 0, 'Image-missing should be empty'
assert len(data['all']) == 0, 'All-missing should be empty'
print('[PASS] Text-only modality test passed!')
"
```

### 测试 2: Image-Only 缺失模态生成

```bash
python -m data.preprocessing.missing_modality --dataset baby --modality image --missing_ratio 0.666
```

**预期输出**:
```
[Missing Modality] Mode: IMAGE-ONLY
  - Total missing items: <数字>
  - Items with missing TEXT: 0
  - Items with missing IMAGE: <数字>
  - Items with missing ALL: 0

[Saved] baby/missing_items_0.666_IMAGE.npy
```

**验证命令**:
```bash
python -c "
import numpy as np
data = np.load('data/baby/missing_items_0.666_IMAGE.npy', allow_pickle=True).item()
print('=== Image-Only Missing Modality Data ===')
print(f'Text-missing items count: {len(data[\"t\"])}')
print(f'Image-missing items count: {len(data[\"v\"])}')
print(f'All-missing items count: {len(data[\"all\"])}')
assert len(data['t']) == 0, 'Text-missing should be empty'
assert len(data['all']) == 0, 'All-missing should be empty'
print('[PASS] Image-only modality test passed!')
"
```

### 测试 3: All 缺失模态生成（向后兼容）

```bash
python -m data.preprocessing.missing_modality --dataset baby --modality all --missing_ratio 0.666
```

**预期输出**:
```
[Missing Modality] Mode: ALL (mixed distribution)
  - Total missing items: <数字>
  - Items with missing TEXT only: <数字>
  - Items with missing IMAGE only: <数字>
  - Items with missing ALL modalities: <数字>

[Saved] baby/missing_items_0.666.npy
```

**验证命令**:
```bash
python -c "
import numpy as np
data = np.load('data/baby/missing_items_0.666.npy', allow_pickle=True).item()
print('=== All Missing Modality Data ===')
print(f'Text-only missing: {len(data[\"t\"])}')
print(f'Image-only missing: {len(data[\"v\"])}')
print(f'All-modalities missing: {len(data[\"all\"])}')
assert len(data['all']) > 0, 'All-missing should not be empty in mixed mode'
print('[PASS] All-modality test passed!')
"
```

### 测试 4: 默认行为（不指定 --modality）

```bash
python -m data.preprocessing.missing_modality --dataset baby --missing_ratio 0.666
```

**预期**: 行为与 `--modality all` 相同，生成 `missing_items_0.666.npy`（无后缀）

---

## 集成测试：模型训练验证

### 测试 5: 使用 Text-Only 数据训练

```bash
python train.py \
    --model DGMRec \
    --dataset baby \
    --gpu_id 0 \
    --missing_modal 1 \
    --missing_ratio 0.666 \
    --missing_modality_type text \
    --epochs 2 \
    --stopping_step 1
```

**预期输出**:
```
[DGMRec] Loaded missing items from: missing_items_0.666_TEXT.npy
[DGMRec] Missing modality type: text
  - Items missing TEXT: <数字>
  - Items missing IMAGE: 0
  - Complete items (no missing): <数字>
```

### 测试 6: 使用 Image-Only 数据训练

```bash
python train.py \
    --model DGMRec \
    --dataset baby \
    --gpu_id 0 \
    --missing_modal 1 \
    --missing_ratio 0.666 \
    --missing_modality_type image \
    --epochs 2 \
    --stopping_step 1
```

### 测试 7: 使用 All 数据训练（回归测试）

```bash
python train.py \
    --model DGMRec \
    --dataset baby \
    --gpu_id 0 \
    --missing_modal 1 \
    --missing_ratio 0.666 \
    --epochs 2 \
    --stopping_step 1
```

---

## 边界条件测试

### 测试 8: missing_ratio = 0.0（无缺失）

```bash
python -m data.preprocessing.missing_modality --dataset baby --modality text --missing_ratio 0.0
```

**预期**: 所有缺失项数组为空，无错误

### 测试 9: missing_ratio = 1.0（全部缺失）

```bash
python -m data.preprocessing.missing_modality --dataset baby --modality image --missing_ratio 1.0
```

**预期**: 所有 items 都被标记为缺失图像模态

### 测试 10: 无效参数处理

```bash
# 无效的 modality 值
python -m data.preprocessing.missing_modality --dataset baby --modality invalid
```

**预期**: argparse 显示错误提示并退出

---

## 向后兼容性测试

### 测试 11: 加载旧格式文件

如果你之前已经生成了 `missing_items_0.666.npy`（旧格式），可以测试：

```bash
python train.py \
    --model DGMRec \
    --dataset baby \
    --gpu_id 0 \
    --missing_modal 1 \
    --missing_ratio 0.666 \
    --epochs 1
```

**预期**: 正确加载旧格式文件，训练正常进行（因为默认 `missing_modality_type='all'`）

---

## 快速测试脚本

创建 `test_missing_modality.ps1`（Windows PowerShell）：

```powershell
# 激活虚拟环境
.venv\Scripts\Activate.ps1

$ErrorActionPreference = "Stop"
$passCount = 0
$totalTests = 4

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "MMG-Benchmark Missing Modality Feature Test" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

try {
    # Test 1: Text-only
    Write-Host "[Test 1/4] Generating TEXT-only missing modality..." -ForegroundColor Yellow
    python -m data.preprocessing.missing_modality --dataset baby --modality text --missing_ratio 0.666
    $passCount++
    Write-Host "[PASS] Test 1/4 completed`n" -ForegroundColor Green

    # Test 2: Image-only
    Write-Host "[Test 2/4] Generating IMAGE-only missing modality..." -ForegroundColor Yellow
    python -m data.preprocessing.missing_modality --dataset baby --modality image --missing_ratio 0.666
    $passCount++
    Write-Host "[PASS] Test 2/4 completed`n" -ForegroundColor Green

    # Test 3: All modalities
    Write-Host "[Test 3/4] Generating ALL missing modalities..." -ForegroundColor Yellow
    python -m data.preprocessing.missing_modality --dataset baby --modality all --missing_ratio 0.666
    $passCount++
    Write-Host "[PASS] Test 3/4 completed`n" -ForegroundColor Green

    # Test 4: Default behavior
    Write-Host "[Test 4/4] Testing default behavior (backward compatibility)..." -ForegroundColor Yellow
    python -m data.preprocessing.missing_modality --dataset baby --missing_ratio 0.666
    $passCount++
    Write-Host "[PASS] Test 4/4 completed`n" -ForegroundColor Green

    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "All Tests Passed! ($passCount/$totalTests)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
}
catch {
    Write-Host "`n[ERROR] $_" -ForegroundColor Red
    Write-Host "`nFailed at test $passCount of $totalTests" -ForegroundColor Red
    exit 1
}
```

运行脚本：
```powershell
.\test_missing_modality.ps1
```

---

## 常见问题排查

### 问题 1: ModuleNotFoundError

**错误**: `ModuleNotFoundError: No module named 'torch'`

**解决**:
```bash
uv pip install torch torchvision torch-geometric
```

### 问题 2: 文件未找到

**错误**: `FileNotFoundError: Missing modality file not found`

**原因**: 尚未生成对应的缺失模态数据文件

**解决**: 先运行预处理脚本生成数据文件

### 问题 3: CUDA/GPU 相关错误

**错误**: CUDA out of memory / GPU not available

**解决**:
- 使用 CPU 训练：将 `--gpu_id` 改为 `-1` 或移除 GPU 参数
- 减小 batch_size：在配置中设置较小的 `train_batch_size`

---

## 测试结果记录

| 测试项 | 状态 | 备注 |
|--------|------|------|
| Text-Only 生成 | ⬜ 待测试 | |
| Image-Only 生成 | ⬜ 待测试 | |
| All-Modalities 生成 | ⬜ 待测试 | |
| 默认行为兼容性 | ⬜ 待测试 | |
| Text-Only 训练集成 | ⬜ 待测试 | |
| Image-Only 训练集成 | ⬜ 待测试 | |
| All-Modalities 训练 | ⬜ 待测试 | |
| 旧格式文件加载 | ⬜ 待测试 | |
| 边界条件 (ratio=0) | ⬜ 待测试 | |
| 边界条件 (ratio=1) | ⬜ 待测试 | |

---

## 下一步

完成所有测试后：

✅ 如果所有测试通过 → **功能实现成功！**
❌ 如果有测试失败 → 检查错误信息并根据本指南排查问题
