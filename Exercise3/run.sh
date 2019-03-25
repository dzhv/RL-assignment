killall -9 rcssserver
sleep 5
python main.py --experiment "exp1" --lr 0.0001 

killall -9 rcssserver
sleep 20
killall -9 rcssserver
sleep 20

python main.py --experiment "exp2" --lr 0.00005 