# GPU 实时
watch -n 1.0 nvidia-smi

# CPU/RAM
htop          # 看 python 进程的 RES
free -h       # 看可用内存/交换
vmstat 1

# 若崩了，立刻查看是否被 OOM-killer 杀掉
sudo journalctl -k -n 200 | egrep -i "out of memory|killed process|oom-killer"
