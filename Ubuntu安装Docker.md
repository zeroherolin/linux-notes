# Ubuntu安装Docker

## 安装一些依赖

```bash
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
```

## 添加docker官方GPG密钥

```bash
sudo -i
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/trusted.gpg.d/docker-ce.gpg
exit
```

## 验证公钥

```bash
sudo apt-key fingerprint 0EBFCD88
```

## 添加Docker阿里稳定版软件源

```bash
sudo add-apt-repository "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
```

## 安装默认最新版
```bash
sudo apt install docker-ce docker-ce-cli containerd.io
```

## 设置代理（Clash端口7890）

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
cd /etc/systemd/system/docker.service.d
sudo touch http-proxy.conf
sudo nano http-proxy.conf
## 添加
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7890"
Environment="HTTPS_PROXY=http://127.0.0.1:7890"
```

## 添加用户到docker用户组

```bash
sudo groupadd docker
cat /etc/group | grep docker
sudo usermod -aG docker [用户名]
```

## 重启服务

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker

sudo reboot
```

***
⭐ I like your Star!
🔙 [Go Back](README.md)
