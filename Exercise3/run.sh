killall -9 rcssserver
sleep 5
python main.py --experiment "exp3" --lr 0.00001 

killall -9 rcssserver
sleep 20
killall -9 rcssserver
sleep 20

python main.py --experiment "exp4" --lr 0.000025

killall -9 rcssserver
sleep 20
killall -9 rcssserver
sleep 20

python main.py --experiment "exp5" --lr 0.00005 --n_workers 8