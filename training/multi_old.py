from multiprocessing import Process, Pipe

num_pools = 20
def standalone(conn,visualize=True,ID=0):
    addrs = ['http://127.0.0.1:500' + str(i) for i in range(num_pools)]
    from opensim_rl import OpenSimEnv
    e = OpenSimEnv(visualize=False,remote_base = addrs[ID],skipcount=1)

    while True:
        msg = conn.recv()

        # messages should be tuples,
        # msg[0] should be string

        if msg[0] == 'reset':
            obs = e.reset()
            conn.send(obs)
        elif msg[0] == 'step':
            four = e.step(msg[1])
            conn.send(four)

        else:
            e.exit()
            conn.close()
            del e
            return

class ei:
    def __init__(self,visualize=True,ID=0):
        self.pc, self.cc = Pipe()
        self.p = Process(target = standalone, args=(self.cc, visualize,(ID % num_pools) ), daemon=True)
        self.p.start()

        self.occupied = False

    def reset(self):
        self.pc.send(('reset',))
        return self.pc.recv()

    def step(self,actions):
        self.pc.send(('step',actions,))
        return self.pc.recv()

    def __del__(self):
        self.pc.send(('exit',))
        print('(ei)waiting for join...')
        self.p.join()

class eipool:
    def __init__(self,n=1,showfirst=True):
        import threading as th
        self.pool = [ei(visualize=(True if i==0 and showfirst else False), ID=i) for i in range(n)]
        self.lock = th.Lock()

    def acq_env(self):
        self.lock.acquire()
        for e in self.pool:
            if e.occupied == False:
                e.occupied = True # occupied
                self.lock.release()
                return e # return the envinstance

        self.lock.release()
        return False # no available ei

    def rel_env(self,ei):
        self.lock.acquire()
        for e in self.pool:
            if e == ei:
                e.occupied = False # freed
        self.lock.release()

    def num_free(self):
        return sum([0 if e.occupied else 1 for e in self.pool])

    def num_total(self):
        return len(self.pool)

    def all_free(self):
        return self.num_free()==self.num_total()

    def __del__(self):
        for e in self.pool:
            del e

if __name__ == '__main__':
    from opensim_rl import OpenSimEnv
    grande = OpenSimEnv(visualize=False,remote_base = 'http://127.0.0.1:5001',skipcount=2)

    ep = eipool(5)

    def run():
        env = ep.acq_env()
        if env ==False:
            print('shi!!!!')
            return

        observation = env.reset()
        for i in range(500):
            observation, reward, done, info = env.step(grande.action_space.sample())
            # print(observation)
            if done:
                break;

        ep.rel_env(env)

    def para():
        import threading as th
        ts = [th.Thread(target=run,daemon=True) for i in range(4)]
        for i in ts:
            i.start()
        for i in ts:
            i.join()

    para()
    para()
    del ep
