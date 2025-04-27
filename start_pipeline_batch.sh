#!/bin/bash

# Arrays of different goals, input and prediction window sizes
GOALS=("Goal" "Shot") #array of prediction goals (list here PIs to predict)
INPUTWINDOWS=(180 120 60 36 12) #array of number of intervalls, example: 180 = 15 minutes
PREDICTIONWINDOWS=(12 36 60 120 180)   #array of number of intervalls, example: 36 = 3 minutes
CONFIG_PATH=("training_config.yml")

for GOAL in "${GOALS[@]}"; do  
    OUTPUT="output/${GOAL}"         ##define here the directory to store the results of prediction experiments
    if [ ! -d "$OUTPUT" ]; then   # Create the OUTPUT directory if it doesn't exist    
        mkdir "$OUTPUT"
        echo "Created directory: $OUTPUT"
    fi
    for IW in "${INPUTWINDOWS[@]}"; do
        for PW in "${PREDICTIONWINDOWS[@]}"; do
            TIMESTRING=$(date +"%m_%d_%Hh_%Mm_%Ss")
            echo "Started Job: ${IW}_${PW}_classification_${GOAL}_${TIMESTRING}"
            (python3 run.py "$CONFIG_PATH" --lookback "$IW" --outlook "$PW" --pred_goal "$GOAL" --outputpath "$OUTPUT" --layerCnt 2)
            echo "Ended Job: ${IW}_${PW}_classification_${GOAL}_${TIMESTRING}"
        done        
    done
done

wait
echo "All Jobs terminated"
