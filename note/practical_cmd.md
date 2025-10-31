# Practical Cmd: Realtime Monitoring & OOM Debug

## GPU (real-time)
watch -d -n 1.0 nvidia-smi
# optional: per-GPU/ per-process telemetry
nvidia-smi dmon -s pucvmet     # device-level (power/util/clk/mem/enc/dec/temp)
nvidia-smi pmon -c 1           # process-level GPU usage snapshot

## CPU / RAM
htop                            # add VIRT/RES/SHR columns; sort by RES
free -h                         # quick available memory / swap
vmstat 1                        # r=run queue; si/so=swap-in/out (should be ~0)

## If it crashed, check whether OOM-killer terminated it
# systemd hosts:
sudo journalctl -k --since "-10 min" | egrep -i "out of memory|killed process|oom-killer"
# non-systemd / containers:
dmesg -T | egrep -i "out of memory|killed process|oom-killer" | tail -n 200

## List and terminate any leftover ipykernel / Jupyter processes
pgrep -af "ipykernel|jupyter-(notebook|lab)"
pkill -TERM -f "ipykernel|jupyter-(notebook|lab)"   # graceful
sleep 2
pgrep -af "ipykernel|jupyter-(notebook|lab)" || echo "clean"
pkill -KILL -f "ipykernel|jupyter-(notebook|lab)"   # last resort

## Optional sanity checks (limits / cgroups)
ulimit -a
cat /proc/meminfo | egrep "MemAvailable|Swap"
[ -f /sys/fs/cgroup/memory.max ] && cat /sys/fs/cgroup/memory.max
