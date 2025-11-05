 # Monte Carlo method to calculate ln2 to 2dp
import math
import random
import matplotlib.pyplot as plt

N = 10000
print (" Parameter : N = ", str (N ))
# Array of iterations.
iterations = []
# Array of results.
results = []

count_in = 0
for i in range (N ):
    # random.random returns a random double in range 0-1.
    x = random.random() + 1
    y = random.random()
   
    
    if y <= (1/x):
        outcome = 1
    else:
        outcome = 0
    
    count_in += outcome
    
    
    fraction_in = count_in /( i +1)

    # Store the results into the array.
    results.append(fraction_in )
    # Store iteration into the array.
    iterations.append(i +1)

    
    


fig = plt . figure ()
plt.plot( iterations , results , "k-", label =" numerical ln2 ")


plt.grid ( True )
plt.legend()
plt.ylabel(" Result [ -] ")
plt.xlabel(" Iteration [ -] ")
print("Approximation of ln2 to 2dp is " + str(round(fraction_in,2)))
plt.show()

