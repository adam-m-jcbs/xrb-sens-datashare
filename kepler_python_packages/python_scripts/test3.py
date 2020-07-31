

import numpy as np
from pylab import plot

def f():
    n = 5000
    m = 9
    x = np.arange(n)
    # sets = np.random.choice(x,(n,m),replace=(True, False))
    # sets = np.random.randint(0,n,(n,m))
    sets = [np.random.choice(x,(m,)) for j in x]

    done = []
    do = [int(np.random.randint(n))]
    v = np.ndarray((n,2), dtype='i8')
    k = 0
    while len(do) > 0:
        i = do.pop()
        # print('processing {:d}'.format(i)) 
        done += [i]
        for j in sets[i]:
            if not ((j in done) or (j in do)):
                do += [j]
                # print('adding {:d}'.format(j))

        v[k,:] = len(do),len(done)                
        # print('do: {:4d}, done: {:4d}'.format(*v[k]))
        k += 1
    v = v[:k]
    print(v.shape)

    plot([0,1],[0,1], color='k')
    plot(v[:,1]/len(v),v[:,0]/len(v))
    return v
        
