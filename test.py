
# number of points to randomly sample
K = 2

# Algorithm 4
# tries to connect new point q to current tree T
# fails is obstacle is in the way
def Extend(T, q):
    raise NotImplementedError

# Algorithm 3
def Random_Config(R_T_a, q_goal, P_goal, P_outside):
    raise NotImplementedError

# Algorithm 2
def Connect(T, q):
    raise NotImplementedError

# Algorithm 5
def Swap(T_a, T_b):
    raise NotImplementedError


def Path(T_a, T_b):
    raise NotImplementedError

# Algorithm 1
# perform ARRT-Connect
def ARRT_Connect(start, goal):
    global K
    # init start tree
    T_a = [start]
    # init goal tree
    T_b = [goal]
    
    for k in K:
        q_random = Random_Config(R_T_a, q_goal, P_goal, P_outside)
        if Extend(T_a, q_random) is not None:
            # not trapped
            q_new = None #??
            if Connect(T_b, q_new) is not None:
                # reached each other?
                return Path(T_a, T_b)
            
        Swap(T_a, T_b)

    # failure
    return None
