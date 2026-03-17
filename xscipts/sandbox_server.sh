HOST="0.0.0.0"  # 监听所有网络接口，允许跨机器访问
PORT=8000

apt -y insyall podman
apt install -y kmod
podman run -it -p $PORT:$PORT vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609

# 打印配置信息
echo "=========================================="
echo "  启动 Sandbox API 服务"
echo "=========================================="
echo "监听地址: ${HOST}:${PORT}"
echo ""
echo "本机 IP 地址："
hostname -I | awk '{print $1}'
echo ""
echo ""
echo "服务启动后可通过以下地址访问："
echo "  本地: http://localhost:${PORT}/run_code"
echo "  远程: http://$(hostname -I | awk '{print $1}'):${PORT}/run_code"
echo "=========================================="
echo ""