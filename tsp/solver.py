#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from random import shuffle
import random
import time
from greedy_numpy import solve_tsp 
#from tsp import SimulatedAnnealing
#from tsp import GeneticAlgorithm


Point = namedtuple("Point", ['x', 'y'])

def length2(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calculate_obj (solution, points):
	obj = length2(points[solution[-1]], points[solution[0]])
	for index in range(0, len(solution)-1):
		obj += length2(points[solution[index]], points[solution[index+1]])
	return obj

def distL2(x1,y1, x2,y2):
    """Compute the L2-norm (Euclidean) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    xdiff = x2 - x1
    ydiff = y2 - y1
    return (math.sqrt(xdiff*xdiff + ydiff*ydiff) + .5)

def mk_matrix(coord, dist):
    """Compute a distance matrix for a set of points.

    Uses function 'dist' to calculate distance between
    any two points.  Parameters:
    -coord -- list of tuples with coordinates of all points, [(x1,y1),...,(xn,yn)]
    -dist -- distance function
    """
    n = len(coord)
    D = {}      # dictionary to hold n times n matrix
    for i in range(n-1):
        for j in range(i+1,n):
            (x1,y1) = coord[i]
            (x2,y2) = coord[j]
            D[i,j] = dist(x1, y1, x2, y2)
            D[j,i] = D[i,j]
    return n,D

def mk_closest(D, n):
    """Compute a sorted list of the distances for each of the nodes.

    For each node, the entry is in the form [(d1,i1), (d2,i2), ...]
    where each tuple is a pair (distance,node).
    """
    C = []
    for i in range(n):
        dlist = [(D[i,j], j) for j in range(n) if j != i]
        dlist.sort()
        C.append(dlist)
    return C

def length(tour, D):
    """Calculate the length of a tour according to distance matrix 'D'."""
    z = D[tour[-1], tour[0]]    # edge from last to first city of the tour
    for i in range(1,len(tour)):
        z += D[tour[i], tour[i-1]]      # add length of edge from city i-1 to i
    return z

def randtour(n):
    """Construct a random tour of size 'n'."""
    sol = range(n)      # set solution equal to [0,1,...,n-1]
    random.shuffle(sol) # place it in a random order
    return sol


def nearest(last, unvisited, D):
    """Return the index of the node which is closest to 'last'."""
    near = unvisited[0]
    min_dist = D[last, near]
    for i in unvisited[1:]:
        if D[last,i] < min_dist:
            near = i
            min_dist = D[last, near]
    return near


def nearest_neighbor(n, i, D):
    """Return tour starting from city 'i', using the Nearest Neighbor.

    Uses the Nearest Neighbor heuristic to construct a solution:
    - start visiting city i
    - while there are unvisited cities, follow to the closest one
    - return to city i
    """
    unvisited = range(n)
    unvisited.remove(i)
    last = i
    tour = [i]
    while unvisited != []:
        next = nearest(last, unvisited, D)
        tour.append(next)
        unvisited.remove(next)
        last = next
    return tour

def exchange_cost(tour, i, j, D):
    """Calculate the cost of exchanging two arcs in a tour.

    Determine the variation in the tour length if
    arcs (i,i+1) and (j,j+1) are removed,
    and replaced by (i,j) and (i+1,j+1)
    (note the exception for the last arc).

    Parameters:
    -t -- a tour
    -i -- position of the first arc
    -j>i -- position of the second arc
    """
    n = len(tour)
    a,b = tour[i],tour[(i+1)%n]
    c,d = tour[j],tour[(j+1)%n]
    return (D[a,c] + D[b,d]) - (D[a,b] + D[c,d])


def exchange(tour, tinv, i, j):
    """Exchange arcs (i,i+1) and (j,j+1) with (i,j) and (i+1,j+1).

    For the given tour 't', remove the arcs (i,i+1) and (j,j+1) and
    insert (i,j) and (i+1,j+1).

    This is done by inverting the sublist of cities between i and j.
    """
    n = len(tour)
    if i>j:
        i,j = j,i
    assert i>=0 and i<j-1 and j<n
    path = tour[i+1:j+1]
    path.reverse()
    tour[i+1:j+1] = path
    for k in range(i+1,j+1):
        tinv[tour[k]] = k


def improve(tour, z, D, C):
    """Try to improve tour 't' by exchanging arcs; return improved tour length.

    If possible, make a series of local improvements on the solution 'tour',
    using a breadth first strategy, until reaching a local optimum.
    """
    n = len(tour)
    tinv = [0 for i in tour]
    for k in range(n):
        tinv[tour[k]] = k  # position of each city in 't'
    for i in range(n):
        a,b = tour[i],tour[(i+1)%n]
        dist_ab = D[a,b]
        improved = False
        for dist_ac,c in C[a]:
            if dist_ac >= dist_ab:
                break
            j = tinv[c]
            d = tour[(j+1)%n]
            dist_cd = D[c,d]
            dist_bd = D[b,d]
            delta = (dist_ac + dist_bd) - (dist_ab + dist_cd)
            if delta < 0:       # exchange decreases length
                exchange(tour, tinv, i, j)
                z += delta
                improved = True
                break
        if improved:
            continue
        for dist_bd,d in C[b]:
            if dist_bd >= dist_ab:
                break
            j = tinv[d]-1
            if j==-1:
                j=n-1
            c = tour[j]
            dist_cd = D[c,d]
            dist_ac = D[a,c]
            delta = (dist_ac + dist_bd) - (dist_ab + dist_cd)
            if delta < 0:       # exchange decreases length
                exchange(tour, tinv, i, j)
                z += delta
                break
    return z


def localsearch(tour, z, D, C=None):
    """Obtain a local optimum starting from solution t; return solution length.

    Parameters:
      tour -- initial tour
      z -- length of the initial tour
      D -- distance matrix
    """
    n = len(tour)
    if C == None:
        C = mk_closest(D, n)     # create a sorted list of distances to each node
    while 1:
        newz = improve(tour, z, D, C)
        if newz < z:
            z = newz
        else:
            break
    return z


def multistart_localsearch(k, n, D, report=None):
    """Do k iterations of local search, starting from random solutions.

    Parameters:
    -k -- number of iterations
    -D -- distance matrix
    -report -- if not None, call it to print verbose output

    Returns best solution and its cost.
    """
    C = mk_closest(D, n) # create a sorted list of distances to each node
    bestt=None
    bestz=None
    for i in range(0,k):
        tour = randtour(n)
        z = length(tour, D)
        z = localsearch(tour, z, D, C)
        if z < bestz or bestz == None:
            bestz = z
            bestt = list(tour)
            if report:
                report(z, tour)

    return bestt, bestz

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    
    if nodeCount > 1000:
        points = []
        for i in range(1, nodeCount+1):
            line = lines[i]
            parts = line.split()
            points.append(Point(float(parts[0]), float(parts[1])))
        if nodeCount < 3000:
            edges = [[0 for x in range (nodeCount)] for y in range (nodeCount)]
            for i in range (nodeCount -1):
                for j in range(i+1,nodeCount):
                    edge = length2(points[i], points[j])
                    edges[i][j] = edge
                    edges[j][i] = edge
            solution = solve_tsp(edges)
        else:
            init_solution = range(nodeCount)
            #print len(init_solution)
            init_obj = calculate_obj(init_solution, points)
            #print 'init_obj', init_obj
            solution = []
            
            for i in range (300): 
                shuffle_sol = [x for x in init_solution]                
                shuffle(shuffle_sol)
                #print 'len shuffle_sol', len(shuffle_sol)
                shuffle_obj = calculate_obj(shuffle_sol, points)
                #print 'shuffle_obj', shuffle_obj
                if shuffle_obj < init_obj:
                    print 'Better solution found!'
                    solution = [x for x in shuffle_sol]
                    init_obj = shuffle_obj
                    init_solution = [x for x in shuffle_sol]
                else: 
                    solution = [x for x in init_solution]

        #print 'len of solution', len(solution)
        obj = calculate_obj(solution, points)
        # prepare the solution in the specified output format
        output_data = '%.2f' % obj + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, solution))

        return output_data
    
    
    #small list
    coord = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        coord.append((float(parts[0]), float(parts[1])))
    
    #print coord
    n, D = mk_matrix(coord, distL2)
    
    from time import clock
    init = clock()
    def report_sol(obj, s=""):
        print "cpu:%g\tobj:%g\ttour:%s" % \
              (clock(), obj, s)

    print "*** travelling salesman problem ***"
    print

    # random construction
    #print "random construction + local search:"
    tour = randtour(n)     # create a random tour
    z = length(tour, D)     # calculate its length
    #print "random:", tour, z, '  -->  ',   
    z = localsearch(tour, z, D)      # local search starting from the random tour
    #print tour, z
    #print

    # greedy construction
    print "greedy construction with nearest neighbor + local search:"
    for i in range(n):
        tour = nearest_neighbor(n, i, D)     # create a greedy tour, visiting city 'i' first
        z = length(tour, D)
        #print "nneigh:", tour, z, '  -->  ',
        z = localsearch(tour, z, D)
        #print tour, z
    print

    # multi-start local search
    print "random start local search:"
    niter = 300
    tour,z = multistart_localsearch(niter, n, D, None)
    #assert z == length(tour, D)
    print "best found solution (%d iterations): z = %g" % (niter, z)
    #print tour

    obj = length(tour, D)
    solution = tour
    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

