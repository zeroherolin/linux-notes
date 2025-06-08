# Ollama使用

Webpage: [https://ollama.com/download](https://ollama.com/download)

## 安装

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

> Installing ollama to /usr/local \
Downloading Linux amd64 bundle \
######################################################################## 100.0%##O=#  # \
Creating ollama user... \
Adding ollama user to render group... \
Adding ollama user to video group... \
Adding current user to ollama group... \
Creating ollama systemd service... \
Enabling and starting ollama service... \
Created symlink /etc/systemd/system/default.target.wants/ollama.service → /etc/systemd/system/ollama.service. \
NVIDIA GPU installed.

## 配置

- 编辑ollama.service

```bash
sudo nano /etc/systemd/system/ollama.service
```

添加如下内容：

```
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0:11434"

[Install]
WantedBy=default.target
```

- 重启服务

```bash
systemctl daemon-reload
systemctl restart ollama
```

## 访问测试

[http://192.168.3.88:11434/](http://192.168.3.88:11434/)

> Ollama is running

## Ollama命令

```bash
ollama serve     # 启动ollama
ollama create    # 从模型文件创建模型
ollama show      # 显示模型信息
ollama run       # 运行模型
ollama pull      # 从注册仓库中拉取模型
ollama push      # 将模型推送到注册仓库
ollama list      # 列出已下载模型
ollama cp        # 复制模型
ollama rm        # 删除模型
ollama help      # 获取有关任何命令的帮助信息
```

## 示例

模型库查看：[https://ollama.com/library](https://ollama.com/library)

- Run

```bash
ollama pull llama3:8b
ollama list
ollama run llama3:8b
```

- Chat via API

```bash
curl http://192.168.3.88:11434/api/chat -d '{"model": "llama3:8b", "messages": [{ "role": "user", "content": "你好啊"}]}'

curl http://192.168.3.88:11434/api/chat -d '{"model": "llama3:8b", "messages": [{"role": "system", "content": "你是一个乐于助人的AI助手。"}, {"role": "user", "content": "你好啊"}], "stream": false}'

curl http://192.168.3.88:11434/api/chat -d '{"model": "wangshenzhi/llama3-8b-chinese-chat-ollama-q4:latest", "messages": [{"role": "system", "content": "你是奥特曼。"}, {"role": "user", "content": "你好，介绍一下你自己。"}], "stream": false}'
```

## 停止

```bash
sudo systemctl stop ollama.service
```

## 卸载

```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm /etc/systemd/system/ollama.service

sudo rm $(which ollama)

sudo rm -r /usr/share/ollama
sudo userdel ollama
sudo groupdel ollama
```

***
⭐ I like your Star!
🔙 [Go Back](README.md)
