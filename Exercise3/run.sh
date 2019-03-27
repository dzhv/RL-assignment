killall -9 rcssserver
sleep 5

python main.py --experiment "exp7" --lr 0.000025 --discountFactor 0.96

sleep 5
rm -r log
killall -9 rcssserver
sleep 10
killall -9 rcssserver
sleep 3

python main.py --experiment "exp8" --lr 0.000025 --discountFactor 0.96 --n_workers 8 --numEpisodes 25000

rm -r log