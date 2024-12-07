#!/bin/bash
#
python3 main.py --dataset car-hacking -ct binary --model DCNN --label 1 --retrain

python3 main.py --dataset in-vehicle_chevrolet -ct binary --model DCNN --label 1 --retrain

python3 main.py --dataset road -ct binary --model DCNN --label 2 --retrain

python3 main.py --dataset car-hacking -ct binary --model LSTM --label 1 --retrain

python3 main.py --dataset in-vehicle_kia -ct binary --model LSTM --label 1 --retrain

python3 main.py --dataset road -ct binary --model LSTM --label 1 --retrain

python3 main.py --dataset road -ct multiclass --model LSTM --retrain
