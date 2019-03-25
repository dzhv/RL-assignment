#!/bin/bash

# Number of defense agents must be added by one to take into account of goalkeeper
# Cannot run an environment where defending agents exist but none are playing
# goalkeeper

episodes=5000

while getopts l:d:c:e: option
do
case "${option}"
in
l) LR=${OPTARG};;
d) DF=${OPTARG};;
c) DC=${OPTARG};;
e) EXP=$OPTARG;;
esac
done


./../../../bin/HFO --defense-agents=2 --offense-agents=1 --offense-on-ball 11 --trials $episodes --headless --deterministic --discrete=True --frames-per-trial 2000 --untouched-time 2000 &
sleep 5
./DiscreteHFO/Initiator.py --numTrials=$episodes --numPlayingDefenseNPCs=1 --numAgents=1 >/dev/null 2>&1 &
echo "Environment Initialized"
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 5
./QLearning/QLearningBase.py --numOpponents=1 --numEpisodes=$episodes --learningRate $LR --discountFactor $DF --decayConstant $DC --experiment $EXP &
echo "Attacker Controller Initialized"

sleep 5
./DiscreteHFO/Goalkeeper.py --numEpisodes=$episodes >/dev/null 2>&1 &
echo "Goalkeeper Initialized"

sleep 5
./DiscreteHFO/DiscretizedDefendingPlayer.py --numEpisodes=$episodes --id=1 >/dev/null 2>&1 &
echo "Defending Player Initialized"

sleep 5
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
