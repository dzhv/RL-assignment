killall -9 rcssserver
sleep 3
QLearning/QLearningAgent.sh -l 0.1 -d 0.95 -c 0.0006 -e "exp9"
killall -9 rcssserver
sleep 10
