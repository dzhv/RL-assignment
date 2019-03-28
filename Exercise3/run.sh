killall -9 rcssserver
sleep 5

python main.py --experiment "exp9" --lr 0.000025 --discountFactor 0.99 --n_workers 6 --num_layers 3 --hidden_size 25

sleep 5
rm -r log
killall -9 rcssserver
sleep 10
killall -9 rcssserver
sleep 3

python main.py --experiment "exp10" --lr 0.000025 --discountFactor 0.99 --n_workers 6 --num_layers 3 --hidden_size 40

sleep 5
rm -r log
killall -9 rcssserver
sleep 10
killall -9 rcssserver
sleep 3

python main.py --experiment "exp11" --lr 0.000015 --discountFactor 0.99 --n_workers 6 --num_layers 3 --hidden_size 40