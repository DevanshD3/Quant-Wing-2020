import numpy
import time
from numba import autojit

#numba makes the process faster
@autojit
def binomial_tree_call_put (n, t, s0, sigma, r, k, call = True ,arr_out=False):
    #init
    dt = t/n
    u = numpy.exp(sigma*(dt**(0.5)))
    d = 1/u
    p = (numpy.exp(r*dt)-d) / (u-d)

    # price tree
    price_tree = numpy.zeros([n+1,n+1])
    
    for i in range(n+1):
        for j in range(n+1):
            price_tree[j,i] = s0*(d**j)*(u**(i-j))



    # option value
    option = numpy.zeros([n+1,n+1])
    if call:
        option[: , n] = numpy.maximum(numpy.zeros(n+1), price_tree[:, n]-k)
    else:
        option[: , n] = numpy.maximum(numpy.zeros(n+1), k-price_tree[:, n])
    
    # calculate option price at t = 0
    for i in numpy.arange(n-1 ,-1,-1 ):
        for j in numpy.arange(0, i+1):
            option[j,i] = numpy.exp(-r*dt)*(p*option[j, i+1]+(1-p)*option[j+1, i+1])


    # Return
    if arr_out:
        return [option[0, 0], price_tree, option]
    else:
        return option[0,0]

# checking the desired output  
print(binomial_tree_call_put(n=50,t=1,s0=100,sigma=0.1,r=0.05,k=100,call=True,arr_out=False))





