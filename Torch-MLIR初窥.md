# Torch-MLIR

Code: [https://github.com/llvm/torch-mlir](https://github.com/llvm/torch-mlir)

<img src="assets/pytorch-mlir-arch.png" height=400/> <img src="assets/torch-mlir-arch.png" height=400/>

## 构建torch-mlir环境

- 直接安装快照

```bash
conda create -n torch-mlir python=3.11 -y
conda activate torch-mlir
python -m pip install --upgrade pip

pip install --pre torch-mlir torchvision \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

- 或编译安装

编译失败多次，暂时放弃

## torch-mlir测试

### 张量矩阵乘

- python测试代码

```python
import torch
from torch_mlir import fx
from torch_mlir.compiler_utils import run_pipeline_with_repro_report

# 定义一个简单的PyTorch模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

# 创建模型实例
model = SimpleModel()

# 生成Torch dialect IR
module = fx.export_and_import(model, torch.randn(3, 4), torch.randn(4, 3))

# 打印生成的IR
print(module)

# 转换为Linalg dialect
run_pipeline_with_repro_report(
    module,
    (
        "builtin.module("
        "func.func(torch-simplify-shape-calculations),"
        "func.func(torch-decompose-complex-ops),"
        "torch-backend-to-linalg-on-tensors-backend-pipeline)"
    ),
    "Lowering Torch IR to Linalg dialect",
    enable_ir_printing=False)

# 打印转换的IR
print(module)
```

- 输出

```
module {
  func.func @main(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[3,3],f32> {
    %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,3],f32> -> !torch.vtensor<[3,3],f32>
    return %0 : !torch.vtensor<[3,3],f32>
  }
}

module {
  func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<3x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<3x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<3x4xf32>, tensor<4x3xf32>) outs(%1 : tensor<3x3xf32>) -> tensor<3x3xf32>
    return %2 : tensor<3x3xf32>
  }
}
```

***
⭐ I like your Star!
🔙 [Go Back](README.md)
