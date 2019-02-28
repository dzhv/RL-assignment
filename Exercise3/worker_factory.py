import network_factory
import torch.multiprocessing as mp
import Worker
from Policy import Policy
from SharedAdam import SharedAdam
from Environment import HFOEnv

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
        policy = Policy(epsilon=config["epsilons"][idx])
        trainingArgs = (idx, networks, optimizer, counter, policy, config, logger)
        p = mp.Process(target=Worker.train, args=trainingArgs)

        logger.log("Starting process: {0}".format(idx))

        p.start()        

        logger.log("Process started: {0}".format(idx))
        workers.append(p)
        logger.log("Worker Appended: {0}".format(idx))
    
    return workers