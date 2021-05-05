
from scipy.stats import gamma
import numpy as np
from matplotlib import pyplot as plt

x = np.arange(0, 10, 0.001)
for enl,color in zip([1,4,10],['b','r','g']):
    rv = gamma(enl, scale =1./enl)
    y = rv.pdf(x)
    avg = (x*y).sum()/y.sum()
    med = x[y.cumsum()>=(y.sum()/2.)][0]
    print(enl, avg, med)
    plt.plot(x, rv.pdf(x), label='enl=%d'%enl, color=color)
    plt.axvline(med, linestyle='--', color=color, label='median enl=%d'%enl)
    
plt.axvline(1., linestyle=':', label='mean')

plt.xlim((0,3))
plt.legend()
plt.xlabel('speckled/noise-free')
plt.ylabel('pdf')
plt.savefig('gamma_enl.png')
