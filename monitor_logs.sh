#!/bin/bash
# Monitor VoxCPM logs in real-time
echo "监控 VoxCPM 日志（当你点击生成按钮时看这里）..."
echo "================================================================"
tail -f /tmp/voxcpm_app.log
