killall -9 rcssserver
sleep 3
QLearning/QLearningAgent.sh -l 0.1 -d 0.97 -c 0.0006 -e "exp6"
killall -9 rcssserver
sleep 10
QLearning/QLearningAgent.sh -l 0.1 -d 0.95 -c 0.0006 -e "exp7"
killall -9 rcssserver
sleep 10
QLearning/QLearningAgent.sh -l 0.1 -d 0.96 -c 0.0005 -e "exp8"
killall -9 rcssserver
