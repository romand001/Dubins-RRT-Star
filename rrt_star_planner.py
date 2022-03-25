"""
Assignment #2 Template file
"""
import random
import math
import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees* (RRT)
for the problem setup given by the RRT_DUBINS_PROMLEM class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file rrt_star_planner.py. Your
   implementation can be tested by running RRT_DUBINS_PROBLEM.PY (check the 
   main function).
2. Read all class and function documentation in RRT_DUBINS_PROBLEM carefully.
   There are plenty of helper function in the class to ease implementation.
3. Your solution must meet all the conditions specificed below.
4. Below are some do's and don'ts for this problem as well.

Conditions
-------------------
There are some conditions to be satisfied for an acceptable solution.
These may or may not be verified by the marking script.

1. The solution loop must not run for more that a certain number of random iterations
   (Specified by a class member called MAX_ITER). This is mainly a safety
   measure to avoid time-out-related issues and will be set generously.
2. The planning function must return a list of nodes that represent a collision-free path
   from start node to the goal node. The path states (path_x, path_y, path_yaw)
   specified by each node must define a Dubins-style path and traverse from node i-1 -> node i.
   (READ the documentation for the node class to understand the terminology)
3. The returned path should have the start node at index 0 and goal node at index -1,
   while the parent node for node i from the list should be node i-1 from the list, ie,
   the path should be a valid list of nodes.
   (READ the documentation of the node to understand the terminology)
4. The node locations must not lie outside the map boundaries specified by
   RRT_DUBINS_PROBLEM.map_area.

DO(s) and DONT(s)
-------------------
1. Do not rename the file rrt_star_planner.py for submission.
2. Do not change change the PLANNING function signature.
3. Do not import anything other than what is already imported in this file.
4. You can write more function in this file in order to reduce code repitition
   but these function can only be used inside the PLANNING function.
   (since only the planning function will be imported)
"""

def angle_difference(current, target):
    '''
    compute the shortest angle difference between two angles
    '''
    return (target - current + 3 * np.pi) % (2*np.pi) - np.pi

def nearest(node_coords, rand_state, yaw_weight=1.0):
    ''' 
    Find the index of nearest node to the random state
    '''
    node_coords = np.array(node_coords)
    rand_state = np.array(rand_state).reshape((1,3))
    diff = node_coords - rand_state # difference between all node coords and new node
    diff[:,2] = yaw_weight * angle_difference(node_coords[:,2], rand_state[:,2]) # difference between yaw angles
    diff = np.sum(diff**2, axis=1) # sum of squared differences
    min_idx = np.argmin(diff) # index of minimum difference

    return min_idx

def random_config(x_lim, y_lim, goal, exploit_prob):
    ''' 
    Sample a random configuration
    '''
    # select goal node with probability of exploit_prob
    if np.random.rand() < exploit_prob:
        # exploit goal
        rand_state = [goal.x, goal.y, goal.yaw]
    else:
        # explore random state
        rand_state = [
            np.random.uniform(low=x_lim[0], high=x_lim[1]),
            np.random.uniform(low=y_lim[0], high=y_lim[1]),
            np.random.uniform(low=0, high=2*np.pi)
        ]

    return rand_state

def neighbourhood_rad(card_V, gamma):
    '''
    Compute the neighbourhood radius for the RRT* algorithm.
    INPUTS:
        card_V: cardinality of the vertez set (number of nodes)
        gamma: user-defined parameter, with optimality condition defined in Theorem 38 in RRT* paper
    '''
    d = 3 # dimension of the state space (SE2, so d=3)

    return gamma * (np.log(card_V) / card_V)**(1/d)

def incremental_cost(node):
    '''
    Compute the incremental cost of a node (cost from parent to this node)
    '''
    return node.cost - node.parent.cost

def update_costs_recursive(node_idx, node_list, child_dict, incremental_costs):
    '''
    Recursively update the costs of all children of a node
    '''

    node_list[node_idx].cost = node_list[node_idx].parent.cost + incremental_costs[node_idx]

    for child_idx in child_dict[node_idx]:
        update_costs_recursive(child_idx, node_list, child_dict, incremental_costs)

def rrt_star_planner(rrt_dubins, display_map=False):
    """
        Execute RRT* planning using Dubins-style paths. Make sure to populate the node_list.

        Inputs
        -------------
        rrt_dubins  - (RRT_DUBINS_PROBLEM) Class conatining the planning
                      problem specification
        display_map - (boolean) flag for animation on or off (OPTIONAL)

        Outputs
        --------------
        (list of nodes) This must be a valid list of connected nodes that form
                        a path from start to goal node

        NOTE: In order for rrt_dubins.draw_graph function to work properly, it is important
        to populate rrt_dubins.nodes_list with all valid RRT nodes.
    """

    # maintain child and parent relationships in dictionaries for faster lookup
    child_dict = {0: []}
    parent_dict = {0: None}

    # maintain incremental cost of each node (cost from parent to this node), same indexing as node_list
    # this is used to propagate improved costs to children after rewiring
    incremental_costs = [0]

    # store node coordinates here for faster distance checking
    node_coords = [[rrt_dubins.node_list[0].x, rrt_dubins.node_list[0].y, rrt_dubins.node_list[0].yaw]]

    d = 3 # dimension of the state space (SE2, so d=3)
    zeta_d = 4/3 * np.pi # volume of the unit d-ball in the d-dimensional Euclidean space
    mu_X_free = 0.8 * (rrt_dubins.x_lim[1] - rrt_dubins.x_lim[0]) \
        * (rrt_dubins.y_lim[1] - rrt_dubins.y_lim[0]) * 2 * np.pi # volume of the obstacle-free space, assume free area is 80% of total area
    gamma_FOS = 1.0 # factor of safety so that gamma > expression from Theorem 38
    gamma = gamma_FOS * 2 * (1 + 1/d)**(1/d) * (mu_X_free/zeta_d)**(1/d) # optimality condition from Theorem 38

    yaw_weight = 4.0 # weight for yaw in distance metric

    exploit_prob = 0.2 # probability of sampling goal itself
    goal_dist = 0.5 # acceptable distance from goal

    # LOOP for max iterations
    i = 0
    while i < rrt_dubins.max_iter:
        i += 1

        ''' GET RANDOM STATE '''

        # sample random state
        rand_state = random_config(rrt_dubins.x_lim, rrt_dubins.y_lim, rrt_dubins.goal, exploit_prob)

        ''' CREATE NEW NODE AT RANDOM STATE (IF POSSIBLE), AND CONNECT IT TO NEAREST NODE '''

        # Find an existing node nearest to the random vehicle state
        nearest_idx = nearest(node_coords, rand_state, yaw_weight)
        nearest_node = rrt_dubins.node_list[nearest_idx]

        # create temporary node at random state, for collision checking
        new_node = rrt_dubins.propogate(nearest_node, rrt_dubins.Node(*rand_state))

        # Check if the path between nearest node and random state has obstacle collision
        if not rrt_dubins.check_collision(new_node):
            # it has a collision, so continue to next iteration
            continue

        # add new node and take care of data structures accordingly
        rrt_dubins.node_list.append(new_node) # add new node to list
        new_idx = len(rrt_dubins.node_list) - 1 # index of new node
        node_coords.append([new_node.x, new_node.y, new_node.yaw]) # add node coordinates for distance evaluation
        child_dict[new_idx] = [] # initialize child list for new node
        incremental_costs.append(incremental_cost(new_node)) # add incremental cost to list

        # deal with child and parent relationships after adding the new node
        child_dict.setdefault(nearest_idx, []).append(new_idx) # add new node as child of nearest node
        parent_dict[new_idx] = nearest_idx # add nearest node as parent of new node

        ''' GET NEIGHBOURHOOD '''

        # get squared neighbourhood radius
        sqr_rad = neighbourhood_rad(len(rrt_dubins.node_list), gamma)**2
        # print(f'Neighbourhood radius: {np.sqrt(sqr_rad)}m')

        # get neighbouring nodes within the neighbourhood radius
        # produces list of tuples: (index in rrt_dubins.node_list, neighbour)
        # we are storing the index for each neighbour, so we can easily access it in main node list
        neighbourhood = [(i, node) for (i, node) in enumerate(rrt_dubins.node_list) if 
            node.parent is not None and i != new_idx and # don't choose root node or new node
            (rand_state[0] - node.x)**2 + (rand_state[1] - node.y)**2 + 
                (yaw_weight * angle_difference(node.yaw, rand_state[2]))**2 < sqr_rad]

        ''' RECONNECT NEW NODE TO BEST NEIGHBOUR '''

        # find best neighbour (the one that will result in the least cost)
        # initialize with connection to nearest node
        best_cost = rrt_dubins.calc_new_cost(nearest_node, new_node)
        best_neighbour = nearest_node
        best_neighbour_idx = nearest_idx
        for neighbour_idx, neighbour in neighbourhood:
            # check for collision free path from neighbour to new node
            temp_new_node = rrt_dubins.propogate(neighbour, new_node) # temporary, for collision checking and cost calculation
            if temp_new_node.cost < best_cost and rrt_dubins.check_collision(temp_new_node):
                # if it's better cost and collision free, update best neighbour
                best_cost = temp_new_node.cost
                best_neighbour = neighbour
                best_neighbour_idx = neighbour_idx

        # connect new node as child of best neighbour and replace in node list
        new_node = rrt_dubins.propogate(best_neighbour, new_node)
        rrt_dubins.node_list[-1] = new_node
        incremental_costs[-1] = incremental_cost(new_node) # update incremental cost from best neighbour

        # deal with child and parent relationships after reconnecting new node to the best neighbour instead of nearest node
        child_dict[nearest_idx].pop() # remove new node from children list of nearest node
        child_dict.setdefault(best_neighbour_idx, []).append(new_idx) # add new node to children list of best neighbour
        parent_dict[new_idx] = best_neighbour_idx # set parent of new node to be the best neighbour

        ''' REWIRE (CONNECT NEIGHBOURS THROUGH NEW NODE IF IT'S BETTER) '''

        # rewire the tree within neighbourhood, checking if the new node is a better parent for any of its neighbours
        for neighbour_idx, neighbour in neighbourhood:
            # check for collision free path from new node to neighbour, and if the cost is better than the current
            temp_neighbour = rrt_dubins.propogate(new_node, neighbour) # temporary, for collision checking and cost calculation
            if temp_neighbour.cost < neighbour.cost and rrt_dubins.check_collision(temp_neighbour):
                
                # rewire the neighbour by replacing it in the node list
                rrt_dubins.node_list[neighbour_idx] = temp_neighbour
                # neighbour's children must have their parent set to this new node object (temp_neighbour)
                for child_idx in child_dict[neighbour_idx]:
                    rrt_dubins.node_list[child_idx].parent = temp_neighbour

                incremental_costs[neighbour_idx] = incremental_cost(temp_neighbour) # update incremental cost from new node

                # deal with child and parent relationships after rewiring
                old_parent_idx = parent_dict[neighbour_idx] # get index of neighbour's old parent
                child_dict[old_parent_idx].remove(neighbour_idx) # remove rewired neighbour from children list of its old parent
                child_dict.setdefault(new_idx, []).append(neighbour_idx) # add rewired neighbour to children list of new node
                parent_dict[neighbour_idx] = new_idx # set parent of rewired neighbour to be new node

                # recursively update costs of children starting from rewired neighbour
                update_costs_recursive(neighbour_idx, rrt_dubins.node_list, child_dict, incremental_costs)

        # Draw current view of the map
        # PRESS ESCAPE TO EXIT
        if display_map:
            rrt_dubins.draw_graph()

        # Check if new_node is close to goal
        if rrt_dubins.calc_dist_to_goal(new_node.x, new_node.y) <= goal_dist:
            # print("Iters:", i, ", number of nodes:", len(rrt_dubins.node_list))
            break

    if i == rrt_dubins.max_iter:
        print('reached max iterations')
        return None
    else:
        path = [rrt_dubins.node_list[-1]]
        while path[0].parent is not None:
            path.insert(0, path[0].parent)
        return path
