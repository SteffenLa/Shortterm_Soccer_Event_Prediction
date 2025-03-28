#!/bin/bash

# Liste der Ziele, Lookbacks und Outlooks definieren
GOALS=("Goal") #array of prediction goals (list here PIs to predict)
INPUTWINDOWS=(180) #array of number of intervalls, example: 180 = 15 minutes
PREDICTIONWINDOWS=(36)   #array of number of intervalls, example: 36 = 3 minutes
MLMODEL=("applicationscenario_isGoal3minOL")
CONFIG_PATH=("training_config.yml")

# Schleifen über Ziele, Lookbacks und Outlooks
for GOAL in "${GOALS[@]}"; do
  OUTPUT="output/applicationscenario"
    for IW in "${INPUTWINDOWS[@]}"; do
        for PW in "${PREDICTIONWINDOWS[@]}"; do
            TIMESTRING=$(date +"_%m_%d_%Hh_%Mm_%Ss")
            echo "Started Job: ${IW}_${PW}_classification_${GOAL}_${MLMODEL}_${TIMESTRING}"
            (python3 run.py "$CONFIG_PATH" --lookback "$IW" --outlook "$PW" --pred_goal "$GOAL" --model "$MLMODEL" --outputpath "$OUTPUT" --layerCnt 2 --piselection "") &
            echo "Ended Job: ${IW}_${PW}_classification_${GOAL}_${MLMODEL}_${TIMESTRING}"
        done        
    done
done

# Warten, bis alle Jobs beendet sind
wait
echo "All Jobs terminated"
