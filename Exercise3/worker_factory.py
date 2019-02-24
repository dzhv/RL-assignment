import network_factory
import torch.multiprocessing as mp
import Worker
from Policy import Policy
from SharedAdam import SharedAdam
from Environment import HFOEnv

def create_environment(idx):
    port = 6000 + idx
    rnd_seed = 11111 * idx + 111
    environment = HFOEnv(port=port, seed=rnd_seed, numOpponents=1)
    environment.connectToServer()
    return environment

def create_workers(config):
    counter = mp.Value('i', 0)
    # lock = mp.Lock()
    
    learning_network = network_factory.create_network()
    target_network = network_factory.create_network()
    learning_network.load_state_dict(target_network.state_dict())

    optimizer = SharedAdam(learning_network.parameters(), lr=1e-4)
    optimizer.share_memory()

    workers = []
    for idx in range(0, config["n_workers"]):
        # worker_network = network_factory.create_network()
        networks = {
            # "worker": worker_network
            "learning": learning_network,
            "target": target_network 
        }

        environment = create_environment(idx)
        policy = Policy(epsilon=config["epsilons"][idx])
        trainingArgs = (idx, networks, optimizer, counter, environment, policy, config)
        p = mp.Process(target=Worker.train, args=trainingArgs)
        p.start()
        workers.append(p)
    
    return workers