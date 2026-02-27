# AI VoiceSageX 内部python服务

## 开发PYTHON版本
3.11.12

## 服务启动
```bash
python server.py
# or
python server.py -c [config_path]
```

## API
见ApiFox-通晓ai-后端py模块

## 构建指令
```bash

cd docker

# 构建项目基础python环境
docker compose -f docker-compose-base.yaml build

# 构建项目python_api
docker compose build

```