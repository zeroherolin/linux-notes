# Raspberry Pi 5 & Arm64

## 镜像和烧录器下载

[https://cn.ubuntu.com/download/raspberry-pi](https://cn.ubuntu.com/download/raspberry-pi) \
[https://www.raspberrypi.com/software/](https://www.raspberrypi.com/software/)

## Arm64安装Docker

```bash
wget https://download.docker.com/linux/static/stable/aarch64/docker-27.3.1.tgz
wget https://github.com/docker/compose/releases/download/v2.31.0/docker-compose-linux-aarch64

tar -xvf docker-27.3.1.tgz

sudo cp -p docker/* /usr/bin

sudo cp docker-compose-linux-aarch64 /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

sudo nano /etc/systemd/system/docker.service
## 添加如下
```

docker.service

```
[Unit]
Description=Docker Application Container Engine
Documentation=https://docs.docker.com
After=network-online.target firewalld.service
Wants=network-online.target

[Service]
Type=notify
# the default is not to use systemd for cgroups because the delegate issues still
# exists and systemd currently does not support the cgroup feature set required
# for containers run by docker
ExecStart=/usr/bin/dockerd
ExecReload=/bin/kill -s HUP $MAINPID
# Having non-zero Limit*s causes performance problems due to accounting overhead
# in the kernel. We recommend using cgroups to do container-local accounting.
LimitNOFILE=infinity
LimitNPROC=infinity
LimitCORE=infinity
# Uncomment TasksMax if your systemd version supports it.
# Only systemd 226 and above support this version.
#TasksMax=infinity
TimeoutStartSec=0
# set delegate yes so that systemd does not reset the cgroups of docker containers
Delegate=yes
# kill only the docker process, not all processes in the cgroup
KillMode=process
# restart the docker process if it exits prematurely
Restart=on-failure
StartLimitBurst=3
StartLimitInterval=60s

[Install]
WantedBy=multi-user.target
```

```bash
sudo chmod +x /etc/systemd/system/docker.service

sudo groupadd docker
cat /etc/group | grep docker
sudo usermod -aG docker [用户名]

sudo systemctl daemon-reload
sudo systemctl start docker
sudo systemctl enable docker.service

sudo reboot
```

## Arm64编译XDMA驱动

- 编辑Makefile

**注意linux源码目录**（部分系统没有安装源码，最好选择有源码的系统镜像）

```makefile
SHELL = /bin/bash
#
# optional makefile parameters:
# - DEBUG=<0|1>,        enable verbose debug print-out in the driver
# - config_bar_num=,        xdma pci config bar number
# - xvc_bar_num=,        xvc pci bar #
# - xvc_bar_offset=,        xvc register base offset
#
ifneq ($(xvc_bar_num),)
        XVC_FLAGS += -D__XVC_BAR_NUM__=$(xvc_bar_num)
endif

ifneq ($(xvc_bar_offset),)
        XVC_FLAGS += -D__XVC_BAR_OFFSET__=$(xvc_bar_offset)
endif

$(warning XVC_FLAGS: $(XVC_FLAGS).)

topdir := $(shell cd $(src)/.. && pwd)

TARGET_MODULE:=xdma

EXTRA_CFLAGS := -I$(topdir)/include $(XVC_FLAGS)

ifeq ($(DEBUG),1)
        EXTRA_CFLAGS += -D__LIBXDMA_DEBUG__
endif
ifneq ($(config_bar_num),)
        EXTRA_CFLAGS += -DXDMA_CONFIG_BAR_NUM=$(config_bar_num)
endif

#EXTRA_CFLAGS += -DINTERNAL_TESTING

$(TARGET_MODULE)-objs := libxdma.o xdma_cdev.o cdev_ctrl.o cdev_events.o cdev_sgdma.o cdev_xvc.o cdev_bypass.o xdma_mod.o xdma_thread.o
obj-m := $(TARGET_MODULE).o
BUILDSYSTEM_DIR:=/lib/modules/6.6.62+rpt-rpi-2712/build # linux 源码目录
PWD:=$(shell pwd)

all :
        $(MAKE) -C $(BUILDSYSTEM_DIR) M=$(PWD) modules

clean:
        $(MAKE) -C $(BUILDSYSTEM_DIR) M=$(PWD) clean
        @/bin/rm -f *.ko modules.order *.mod.c *.o *.o.ur-safe .*.o.cmd
```

- 本地编译（成功）

```bash
cd xdma_driver_arm64/xdma
make
```

- 交叉编译（有报错，暂时弃用）

安装交叉编译工具链

```bash
cd ~
wget https://snapshots.linaro.org/gnu-toolchain/11.3-2022.06-1/aarch64-linux-gnu/gcc-linaro-11.3.1-2022.06-x86_64_aarch64-linux-gnu.tar.xz
tar -xvf gcc-linaro-11.3.1-2022.06-x86_64_aarch64-linux-gnu.tar.xz
sudo nano ~/.bashrc
## 添加
export PATH=~/gcc-linaro-11.3.1-2022.06-x86_64_aarch64-linux-gnu/bin:$PATH

source ~/.bashrc
```

下载内核源码进行编译

```bash
git clone https://github.com/raspberrypi/linux.git
cd linux
git checkout -b rpi-6.6.y
ARCH=arm64 KERNEL=kernel_2712 make bcm2712_defconfig

cd ../../../xdma
make
```

***
⭐ I like your Star!
🔙 [Go Back](README.md)
