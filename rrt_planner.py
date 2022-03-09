"""
Assignment #2 Template file
"""
import random
import math
import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees (RRT)
for the problem setup given by the RRT_DUBINS_PROMLEM class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file rrt_planner.py. Your implementation
   can be tested by running RRT_DUBINS_PROBLEM.PY (check the main function).
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
1. Do not rename the file rrt_planner.py for submission.
2. Do not change change the PLANNING function signature.
3. Do not import anything other than what is already imported in this file.
4. You can write more function in this file in order to reduce code repitition
   but these function can only be used inside the PLANNING function.
   (since only the planning function will be imported)
"""

def rrt_planner(rrt_dubins, display_map=False):
    """
        Execute RRT planning using Dubins-style paths. Make sure to populate the node_list.

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

    exploit_prob = 0.2 # probability of sampling goal itself
    goal_dist = 0.5 # acceptable distance from goal
    step_size = 4 * np.pi * rrt_dubins.curvature # max path length for each step
    sqr_step = step_size**2 # squared step size

    # LOOP for max iterations
    i = 0
    while i < rrt_dubins.max_iter:
        i += 1

        # select goal node with probability of exploit_prob
        if np.random.rand() < exploit_prob:
            # exploit goal
            rand_state = [rrt_dubins.goal.x, rrt_dubins.goal.y, rrt_dubins.goal.yaw]
        else:
            # explore random state
            rand_state = [
                np.random.uniform(low=rrt_dubins.x_lim[0], high=rrt_dubins.x_lim[1]),
                np.random.uniform(low=rrt_dubins.y_lim[0], high=rrt_dubins.y_lim[1]),
                np.random.uniform(low=0, high=2*np.pi)
            ]

        # Find an existing node nearest to the random vehicle state
        shortest_dist = float('inf')
        nearest_node = None
        for node in rrt_dubins.node_list:
            # omit sqrt because it's slow
            dist = (node.x - rand_state[0])**2 + (node.y - rand_state[1])**2

            if dist < shortest_dist:
                nearest_node = node
                shortest_dist = dist

        # create temporary node at random state, for computing the path prior to truncating
        temp_new_node = rrt_dubins.propogate(nearest_node, rrt_dubins.Node(*rand_state))

        # check if path from nearest node to temporary node is shorter than the step size
        if len(temp_new_node.path_x) < step_size/0.1:
            # if so, then it is the new node that we add
            new_node = temp_new_node
        else:
            # otherwise, create new node at truncated point in path
            trunc_index = int(step_size/0.1)-1
            new_node = rrt_dubins.Node(
                temp_new_node.path_x[trunc_index],
                temp_new_node.path_y[trunc_index],
                temp_new_node.path_yaw[trunc_index]
            )
            new_node.path_x = temp_new_node.path_x[:trunc_index+1]
            new_node.path_y = temp_new_node.path_y[:trunc_index+1]
            new_node.path_yaw = temp_new_node.path_yaw[:trunc_index+1]
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + step_size

        # Check if the path between nearest node and random state has obstacle collision
        # Add the node to nodes_list if it is valid
        if rrt_dubins.check_collision(new_node):
            rrt_dubins.node_list.append(new_node) # Storing all valid nodes

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