import random, numpy, math, copy, matplotlib.pyplot as plt
numberOfcity = 500
cities = [random.sample(range(50), 2) for x in range(numberOfcity)]
tour = random.sample(range(numberOfcity),numberOfcity)
for temperature in numpy.logspace(0,5,num=100000)[::-1]:
	[i,j] = sorted(random.sample(range(numberOfcity),2))
	newTour =  tour[:i] + tour[j:j+1] +  tour[i+1:j] + tour[i:i+1] + tour[j+1:]
	if math.exp( ( sum([ math.sqrt(sum([(cities[tour[(k+1) % numberOfcity]][d] - cities[tour[k % numberOfcity]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]]) - sum([ math.sqrt(sum([(cities[newTour[(k+1) % numberOfcity]][d] - cities[newTour[k % numberOfcity]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]])) / temperature) > random.random():
		tour = copy.copy(newTour)
plt.plot([cities[tour[i % numberOfcity]][0] for i in range(numberOfcity+1)], [cities[tour[i % numberOfcity]][1] for i in range(numberOfcity+1)], 'xb-');
plt.show()