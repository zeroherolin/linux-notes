# Allo: Accelerator Design Language

<img src="assets/allo-logo.png" width=150/>

Paper: [Allo: A Programming Model for Composable Accelerator Design] [https://dl.acm.org/doi/10.1145/3656401](https://dl.acm.org/doi/10.1145/3656401) \
Code: [https://github.com/cornell-zhang/allo](https://github.com/cornell-zhang/allo) \
Documentation: [https://cornell-zhang.github.io/allo/](https://cornell-zhang.github.io/allo/)

## 已更新到LLVM19

### 编译构建环境

```bash
conda create -n allo python=3.12 -y
conda activate allo

cd ~
git clone https://github.com/cornell-zhang/allo.git
cd allo
git checkout 9c7de52

cd ..
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout fbec1c2

cp ~/allo/externals/llvm_patch ~/llvm-project
git apply llvm_patch

python3 -m pip install --upgrade pip
python3 -m pip install numpy PyYAML dataclasses pybind11>=2.9.0

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

cd ~/anaconda3/envs/allo/lib
sudo rm -f libstdc++.so.6
sudo ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 libstdc++.so.6

sudo nano ~/.bashrc
## 添加
export LLVM_BUILD_DIR=~/llvm-project/build
export PATH=$LLVM_BUILD_DIR/bin:$PATH
conda activate allo

source ~/.bashrc

cd ~/allo
python3 -m pip install -v -e .
```

### 或下载Docker镜像构建

```bash
docker pull chhzh123/allo:llvm-19.x-py3.12
docker run --rm -it chhzh123/allo:llvm-19.x-py3.12
(docker) $ git clone https://github.com/cornell-zhang/allo.git && cd allo
(docker) $ python3 -m pip install -v -e .
```

### 或自建dockerfile

- build镜像

```bash
docker pull ubuntu:latest
docker build --network=host -f demo.dockerfile -t myallo:1.0.0 .
```

demo.dockerfile

```dockerfile
FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt -y install sudo

USER root

# install essentials
RUN sudo apt update && \
    apt -y install git wget vim gdb make software-properties-common libssl-dev ninja-build

# set git proxy
# RUN git config --global http.proxy http://127.0.0.1:7890 && \
#     git config --global https.proxy https://127.0.0.1:7890

# install gcc-9
RUN sudo apt -y install build-essential && \
    sudo apt update && \
    sudo add-apt-repository ppa:ubuntu-toolchain-r/ppa -y && \
    sudo apt update && \
    sudo apt -y install gcc-9 g++-9 && \
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 \
                             --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    sudo update-alternatives --config gcc

# install cmake
WORKDIR /root/
RUN cd /root/ && wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9.tar.gz && \
    tar -xzvf cmake-3.27.9.tar.gz && \
    cd cmake-3.27.9 && \
    ./bootstrap && \
    make -j`nproc`
ENV PATH="${PATH}:/root/cmake-3.27.9/bin"

# install conda env
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh && \
    bash /root/miniconda.sh -b -p /root/miniconda && \
    /root/miniconda/bin/conda init bash
ENV PATH="${PATH}:/root/miniconda/bin"
RUN conda create -n allo python=3.12 -y
SHELL ["/root/miniconda/bin/conda", "run", "-n", "allo", "/bin/bash", "-c"]
RUN echo "conda activate allo" >> ~/.bashrc

# download allo
RUN cd /root && \
    git clone https://github.com/cornell-zhang/allo.git && \
    cd /root/allo && \
    git checkout 9c7de52

# download llvm-project
RUN cd /root && \
    git clone https://github.com/llvm/llvm-project.git || \
    while ! git clone https://github.com/llvm/llvm-project.git; \
        do sleep 5 && echo "clone error...retry..."; \
    done

# install llvm
RUN cd /root/llvm-project && \
    git checkout fbec1c2 && \
    cp /root/allo/externals/llvm_patch /root/llvm-project && \
    git apply llvm_patch
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy PyYAML dataclasses pybind11>=2.9.0
RUN cd /root/llvm-project && \
    mkdir build && cd build && \
    cmake -G Ninja ../llvm \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_BUILD_EXAMPLES=ON \
        -DLLVM_TARGETS_TO_BUILD="X86" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_INSTALL_UTILS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE=`which python3`
RUN cd /root/llvm-project/build && ninja

# fix libstdc++.so.6
RUN cd /root/miniconda/envs/allo/lib && \
    rm -f libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 libstdc++.so.6

# set path
ENV LLVM_BUILD_DIR="/root/llvm-project/build"
ENV PATH="${PATH}:/root/llvm-project/build/bin"

# unset git proxy
# RUN git config --global --unset http.proxy && \
#     git config --global --unset https.proxy

# install allo
RUN cd /root/allo && \
    python3 -m pip install -v -e .
```

- 创建实例并运行

```bash
docker run --network=host --rm -it myallo:1.0.0
```

### 测试

执行`python -m pytest tests`之前先注释vitis（cmake）环境，有冲突

```bash
cd ~/allo
python -m pytest tests
```

### 示例：GEMM

```python
import allo
from allo.ir.types import float32

M, N, K = 1024, 1024, 1024


def gemm(A: float32[M, K], B: float32[K, N]) -> float32[M, N]:
    C: float32[M, N] = 0.0
    for i, j in allo.grid(M, N):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C


s = allo.customize(gemm)

s.reorder("k", "j")
print(s.module)

s.buffer_at(s.C, axis="i")
print(s.module)

s.pipeline("j")
print(s.module)

mod = s.build(target="vhls", mode="csyn", project="gemm.prj")
```

## 整数输出稳态脉动阵列

<img src="assets/systolic_array_pes.png" width=400/>

- 文献版本（已弃用）

```python
import allo
from allo.ir.types import int8, int16

M, N, K = 4, 4, 4


# Algorithm specification
def gemm(A: int8[M, K], B: int8[K, N]) -> int16[M, N]:
    C: int16[M, N] = 0
    for i, j in allo.grid(M, N, name="PE"):
        for k in allo.reduction(K):
            C[i, j] += A[i, k] * B[k, j]
    return C


# Schedule construction
s = allo.customize(gemm)
buf_A = s.buffer_at(s.A, "j")  # p0
buf_B = s.buffer_at(s.B, "j")  # p1
pe = s.unfold("PE", axis=[0, 1])  # p2
s.partition(s.C, dim=[0, 1])  # p3
s.partition(s.A, dim=0)  # p4
s.partition(s.B, dim=1)  # p5
s.relay(buf_A, pe, axis=1, depth=M + 1)  # p6
s.relay(buf_B, pe, axis=0, depth=N + 1)  # p7
```

- 当前版本

```python
import allo
from allo.ir.types import int8, int16, index

M, N, K = 4, 4, 4


def kernel(
        A_in: int8[K],
        B_in: int8[K],
        A_out: int8[K],
        B_out: int8[K],
        C: int16[M, N],
        i: index,
        j: index,
):
    for k in range(K):
        a: int8 = A_in[k]
        b: int8 = B_in[k]
        C[i, j] += a * b
        A_out[k] = a
        B_out[k] = b


def systolic_array(A: int8[M, K], B: int8[K, N], C: int16[M, N]):
    A_fifo: int8[M, N + 1, K]
    B_fifo: int8[N, M + 1, K]

    for k in range(K, name="data_load"):
        for m in range(M):
            A_fifo[m, 0, k] = A[m, k]
        for n in range(N):
            B_fifo[n, 0, k] = B[k, n]
    for i, j in allo.grid(M, N, name="PE"):
        kernel(
            A_fifo[i, j], B_fifo[j, i], A_fifo[i, j + 1], B_fifo[j, i + 1], C, i, j
        )
    A_drain: int8[M]
    B_drain: int8[N]
    for k in range(K, name="data_drain"):
        for m in range(M):
            A_drain[m] = A_fifo[m, N, k]
        for n in range(N):
            B_drain[n] = B_fifo[n, M, k]


s = allo.customize(systolic_array)
s.partition(s.C, dim=0)  # required, otherwise it will fail dataflow checking
s.partition(s.A, dim=1)
s.partition(s.B, dim=2)
pe = s.unfold("PE", [0, 1])  # specify which are spatial loops
s.to(s.A_fifo, pe, axis=1, depth=M + 1)
s.to(s.B_fifo, pe, axis=0, depth=N + 1)

print(s.module)
mod = s.build(target="vivado_hls", mode="hw", project="systolic_array.prj")
```

## Allo原语

```python
# 将循环i拆分为一个两级嵌套循环，其中v作为内部循环的边界
s.split(i,v)

# 将同一嵌套循环中的多个子循环l融合为一个
s.fuse(*l)

# 在同一嵌套循环中切换子循环l的顺序
s.reorder(*l)

# 将操作Op1的循环i合并到操作Op2中相应的循环级别
s.compute_at(Op1,Op2,i)

# 通过因子v展开循环i
s.unroll(i,v)

# 将循环i展开为硬件实例
s.unfold(i)

# 以具有目标启动间隔v的流水线方式调度循环i
s.pipeline(i,v)

# 在循环i处创建一个中间缓冲区，用于存储数组A的结果
s.buffer_at(A,i)

# 创建一个存储数组值A的缓冲区，其中的值会在循环i中重复使用
s.reuse_at(A,i)

# 以因子v对数组A的维度d进行循环/块分区
s.partition(A,d,v)

# 以因子v将数组A的维度i压缩成字
s.pack(A,i,v)

# 使用深度v的FIFO将阵列A连接到目标Dst
s.relay(A,Dst,v)
```

***
⭐ I like your Star!
🔙 [Go Back](README.md)
