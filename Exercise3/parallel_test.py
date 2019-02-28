import torch.multiprocessing as mp

def start():
    
    workers = []
    for idx in range(0, 3):
        print("\nIN LOOP, IDX: {0}\n".format(idx))

        
        trainingArgs = (idx, 0)
        p = mp.Process(target=train, args=trainingArgs)

        print("\nSTARTING PROCESS: {0}\n".format(idx))

        p.start()        

        print("\nPROCESS STARTED: {0}\n".format(idx))
        workers.append(p)
        print("\nWorker Appended: {0}\n".format(idx))
    
    return workers

def train(id, _):
    i = 0
    while True:
        if i % 100000 == 0:
            print("training id: {0}".format(id))

for worker in start():
    print("Joining worker")
    w.join()