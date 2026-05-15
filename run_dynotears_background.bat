@echo off
REM 后台运行 DYNOTEARS 因果发现算法
REM 输出将保存到日志文件

echo Starting DYNOTEARS training in background...
echo Log file: dynotears_training.log

python run_dynotears_dag.py --line xin1 --epochs 300 --threshold 0.03 > dynotears_training.log 2>&1

echo Training completed. Check dynotears_training.log for details.
pause
