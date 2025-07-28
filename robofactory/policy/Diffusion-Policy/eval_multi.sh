#!/bin/bash

# 确保任务名称作为参数传入
if [ -z "$1" ]; then
    echo "Usage: $0 <task_name> <data_num> <checkpoint_num>"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Usage: $0 <task_name> <data_num> <checkpoint_num>"
    exit 1
fi
if [ -z "$3" ]; then
    echo "Usage: $0 <task_name> <data_num> <checkpoint_num>"
    exit 1
fi

CONFIG_NAME="$1"
DATA_NUM="$2"
CHECKPOINT_NUM="$3"
DEBUG_MODE="$4"
TASK_NAME="$5"
# 生成带时间戳的日志文件
LOG_FILE="eval_results_${TASK_NAME}_${DATA_NUM}_${CHECKPOINT_NUM}_$(date +"%Y%m%d_%H%M%S").log"

echo "Evaluating task: $TASK_NAME"
echo "Evaluating task: $TASK_NAME"  >> "$LOG_FILE"
TOTAL=0
SUCCESS=0
SUCCESS_RATE=0

# 如果 DEBUG_MODE 是 "0" 或 "False"，则启用 --quiet 参数
QUIET_FLAG=""
if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
    QUIET_FLAG="--quiet"
fi

for SEED in {1000..1099}
do
    echo "Running evaluation with seed $SEED for task $CONFIG_NAME..."
    OUTPUT=""
    if [[ "$DEBUG_MODE" == "0" || "$DEBUG_MODE" == "false" ]]; then
        OUTPUT=$(python ./policy/Diffusion-Policy/eval_multi_dp.py \
                --config="$CONFIG_NAME" \
                --data-num=$DATA_NUM \
                --checkpoint-num=$CHECKPOINT_NUM \
                --render-mode="sensors" \
                -o="rgb" \
                -b="cpu" \
                -n 1 \
                -s $SEED \
                $QUIET_FLAG)
    else
        OUTPUT=$(python ./policy/Diffusion-Policy/eval_multi_dp.py \
                --config="$CONFIG_NAME" \
                --data-num=$DATA_NUM \
                --checkpoint-num=$CHECKPOINT_NUM \
                --render-mode="sensors" \
                -o="rgb" \
                -b="cpu" \
                -n 1 \
                -s $SEED \
                $QUIET_FLAG)
    fi
    echo "$OUTPUT"
    LAST_LINE=$(echo "$OUTPUT" | tail -n 1)  # 获取输出的最后一行
    FINE=0
    # 当output为success时，表示任务成功
    if [[ $LAST_LINE == *"success"* ]]; then
        FINE=1
        SUCCESS=$((SUCCESS + 1))
    fi
    TOTAL=$((TOTAL + 1))
    SUCCESS_RATE=$(echo "scale=4; $SUCCESS / $TOTAL * 100" | bc)
    echo "$SEED, $FINE, $SUCCESS_RATE%" >> "$LOG_FILE"
    echo "Seed $SEED done. Success Rate: $SUCCESS_RATE%"
done
echo "Total: $TOTAL, Success: $SUCCESS, Success Rate: $SUCCESS_RATE%" >> "$LOG_FILE"

echo "Evaluation completed. Results saved in $LOG_FILE."
