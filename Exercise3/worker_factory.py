import network_factory
import torch.multiprocessing as mp
import Worker
from Policy import Policy
from SharedAdam import SharedAdam
from Environment import HFOEnv
import policy_worker
from logger import Logger

def create_workers(config, logger):
    counter = mp.Value('i', 0)
    
    learning_network = network_factory.create_network()
    target_network = network_factory.create_network()
    learning_network.load_state_dict(target_network.state_dict())    

    optimizer = SharedAdam(learning_network.parameters(), lr=config["learning_rate"])
    optimizer.share_memory()

    workers = []
    for idx in range(0, config["n_workers"]):
        networks = {
            "learning": learning_network,
            "target": target_network 
        }

        # environment = create_environment(idx)
        policy = Policy(epsilon=config["startingEpsilons"][idx], 
            numUpdates=config["numPolicyUpdates"], minEpsilon=config["minEpsilons"][idx], logger=logger)
        trainingArgs = (idx, networks, optimizer, counter, policy, config, logger)
        p = mp.Process(target=Worker.train, args=trainingArgs)

        logger.log("Starting process: {0}".format(idx))

        p.start()        

        logger.log("Process started: {0}".format(idx))
        workers.append(p)
        logger.log("Worker Appended: {0}".format(idx))    

    logger.log("Creating the greedy worker")    
    p = create_greedy_worker(networks, counter, config)
    p.start()
    workers.append(p)
    logger.log("Greedy worker started and appended")
    
    return workers, target_network

def create_greedy_worker(networks, counter, config):
    logger = Logger("output_{0}_greedy.out".format(config['experiment']))
    environment = HFOEnv(port=6321, seed=86868686, numOpponents=1)
    environment.connectToServer()

    w_args = (100000, networks, environment, Policy(logger=logger), logger, counter)
    return mp.Process(target=policy_worker.run, args=w_args)