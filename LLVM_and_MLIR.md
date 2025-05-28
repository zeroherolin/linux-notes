# LLVM & MLIR

Clang Webpage: [https://clang.llvm.org/](https://clang.llvm.org/) \
MLIR Webpage: [https://mlir.llvm.org/](https://mlir.llvm.org/)

## 构建MLIR环境

```bash
conda create -n mlir python=3.12
conda activate mlir

git clone https://github.com/llvm/llvm-project.git

cd llvm-project

python -m pip install --upgrade pip
python -m pip install -r mlir/python/requirements.txt

mkdir build && cd build

cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=`which python3`

ninja

sudo nano ~/.bashrc
## 添加
export LLVM_BUILD_DIR=~/llvm-project/build
export PATH=$LLVM_BUILD_DIR/bin:$PATH
export PYTHONPATH=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
```

通过pybind是为以后在python中调用做准备

## MLIR编译

- 需要安装Clang编译器

```bash
sudo apt install clang
```

- 查看支持的pass

Passes [https://mlir.llvm.org/docs/Passes/](https://mlir.llvm.org/docs/Passes/)

- mlir源码

simple_complex.mlir

```
module {
    // 声明外部函数
    llvm.func @printf(!llvm.ptr, ...) -> i32
    llvm.func @atof(!llvm.ptr) -> f64

    // 定义输出格式字符串
    llvm.mlir.global internal constant @fmt_str("%f %f\n\00")

    func.func @main( // int main(int argc, char* argv[])
        %argc: i32, %argv: !llvm.ptr
    ) -> i32 {
        // 定义索引常量
        %c1 = arith.constant 1 : i64
        %c2 = arith.constant 2 : i64
        %c3 = arith.constant 3 : i64
        %c4 = arith.constant 4 : i64

        // 获取argv元素地址（char**的索引访问）
        %argv1_ptr = llvm.getelementptr %argv[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %argv2_ptr = llvm.getelementptr %argv[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %argv3_ptr = llvm.getelementptr %argv[%c3] : (!llvm.ptr, i64) -> !llvm.ptr, i64
        %argv4_ptr = llvm.getelementptr %argv[%c4] : (!llvm.ptr, i64) -> !llvm.ptr, i64

        // 加载实际字符串指针（char*）
        %argv1 = llvm.load %argv1_ptr : !llvm.ptr -> !llvm.ptr
        %argv2 = llvm.load %argv2_ptr : !llvm.ptr -> !llvm.ptr
        %argv3 = llvm.load %argv3_ptr : !llvm.ptr -> !llvm.ptr
        %argv4 = llvm.load %argv4_ptr : !llvm.ptr -> !llvm.ptr

        // 转换字符串为浮点数
        %re1_f64 = llvm.call @atof(%argv1) : (!llvm.ptr) -> f64
        %im1_f64 = llvm.call @atof(%argv2) : (!llvm.ptr) -> f64
        %re2_f64 = llvm.call @atof(%argv3) : (!llvm.ptr) -> f64
        %im2_f64 = llvm.call @atof(%argv4) : (!llvm.ptr) -> f64

        // 转换为f32
        %re1 = arith.truncf %re1_f64 : f64 to f32
        %im1 = arith.truncf %im1_f64 : f64 to f32
        %re2 = arith.truncf %re2_f64 : f64 to f32
        %im2 = arith.truncf %im2_f64 : f64 to f32

        // 创建复数
        %cplx1 = complex.create %re1, %im1 : complex<f32>
        %cplx2 = complex.create %re2, %im2 : complex<f32>

        // 复数加法
        %result = complex.add %cplx1, %cplx2 : complex<f32>

        // 提取实部和虚部
        %real = complex.re %result : complex<f32>
        %imag = complex.im %result : complex<f32>

        // 转换为f64：printf的%f格式必须64位双精度浮点数
        %real64 = arith.extf %real : f32 to f64
        %imag64 = arith.extf %imag : f32 to f64

        // 调用printf输出结果
        %fmt = llvm.mlir.addressof @fmt_str : !llvm.ptr
        llvm.call @printf(%fmt, %real64, %imag64) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32

        // 返回0
        %zero = arith.constant 0 : i32
        func.return %zero : i32
    }
}
```

- 执行Lowering Pass

```bash
mlir-opt simple_complex.mlir \
    --convert-complex-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    -o simple_complex_llvm.mlir
```

- 生成llvm ir

```bash
mlir-translate --mlir-to-llvmir simple_complex_llvm.mlir -o simple_complex.ll
```

- 编译生成可执行文件

```bash
clang -O3 simple_complex.ll -o simple_complex -Wno-override-module -mllvm -opaque-pointers
# -opaque-pointers选项启用不透明指针模式：LLVM 15开始默认启用opaque-pointers，Clang 14仅部分支持或不支持此特性
```

- 完整脚本（含验证）

run_simple_complex.sh

```bash
#!/bin/bash

# 编译阶段
build() {
    mlir-opt simple_complex.mlir \
        --convert-complex-to-llvm \
        --convert-arith-to-llvm \
        --convert-func-to-llvm \
        -o simple_complex_llvm.mlir

    mlir-translate --mlir-to-llvmir simple_complex_llvm.mlir -o simple_complex.ll

    clang -O3 simple_complex.ll -o simple_complex -Wno-override-module -mllvm -opaque-pointers
}

# 执行阶段
run() {
    local re1=$1
    local im1=$2
    local re2=$3
    local im2=$4

    ./simple_complex "$re1" "$im1" "$re2" "$im2"
}

# 主流程
build && run "$1" "$2" "$3" "$4"
```

输出

> $ ./run_simple_complex.sh 1.5 2.3 3.7 4.1 \
5.200000 6.400000

## MLIR to C

- mlir源码

vector_add.mlir

```
func.func @vector_add(%arg0: memref<10xf32>, %arg1: memref<10xf32>, %arg2: memref<10xf32>) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c10 step %c1 {
    %a = memref.load %arg0[%i] : memref<10xf32>
    %b = memref.load %arg1[%i] : memref<10xf32>
    %sum = arith.addf %a, %b : f32
    memref.store %sum, %arg2[%i] : memref<10xf32>
  }
  return
}
```

- Pass到emitc方言

```bash
mlir-opt vector_add.mlir \
  --convert-arith-to-emitc \
  --convert-func-to-emitc \
  --convert-math-to-emitc \
  --convert-memref-to-emitc \
  --convert-scf-to-emitc \
  -o vector_add_emitc.mlir
```

这里pass有问题，手动调整如下

vector_add_emitc.mlir

```
emitc.func @vector_add(
    %arg0: !emitc.array<10xf32>, %arg1: !emitc.array<10xf32>, %arg2: !emitc.array<10xf32>
) {
    %c0 = "emitc.constant"() {value = 0 : index} : () -> index
    %c10 = "emitc.constant"() {value = 10 : index} : () -> index
    %c1 = "emitc.constant"() {value = 1 : index} : () -> index
    emitc.for %i = %c0 to %c10 step %c1 {
        %a_lv = emitc.subscript %arg0[%i] : (!emitc.array<10xf32>, index) -> !emitc.lvalue<f32>
        %b_lv = emitc.subscript %arg1[%i] : (!emitc.array<10xf32>, index) -> !emitc.lvalue<f32>
        %c_lv = emitc.subscript %arg2[%i] : (!emitc.array<10xf32>, index) -> !emitc.lvalue<f32>
        %a = emitc.load %a_lv : !emitc.lvalue<f32>
        %b = emitc.load %b_lv : !emitc.lvalue<f32>
        %sum = emitc.add %a, %b : (f32, f32) -> f32
        "emitc.assign"(%c_lv, %sum) : (!emitc.lvalue<f32>, f32) -> ()
    }
    emitc.return
}
```

- 编译到C

vector_add.cpp

```cpp
void vector_add(float v1[10], float v2[10], float v3[10]) {
  size_t v4 = 0;
  size_t v5 = 10;
  size_t v6 = 1;
  for (size_t v7 = v4; v7 < v5; v7 += v6) {
    float v8 = v1[v7];
    float v9 = v2[v7];
    float v10 = v8 + v9;
    v3[v7] = v10;
  }
  return;
}
```

- 测试

```cpp
#include <stdio.h>

void vector_add(float v1[10], float v2[10], float v3[10]);

int main() {
  float vector1[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  float vector2[10] = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  float result[10];

  vector_add(vector1, vector2, result);

  printf("Result:\n");
  for (int i = 0; i < 10; i++) {
    printf("%.2f ", result[i]);
  }
  printf("\n");
  
  return 0;
}
```

输出

> Result: \
11.00 11.00 11.00 11.00 11.00 11.00 11.00 11.00 11.00 11.00 


***
⭐ I like your Star!
🔙 [Go Back](README.md)
