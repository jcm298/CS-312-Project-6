#!/usr/bin/python3
import random

from TSPSubproblem import TSPSubproblem
from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    pass
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

from TSPClasses import *
import heapq  # priority queue


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setup_with_scenario(self, scenario):
        self._scenario = scenario

    # Entry point for the default solver which finds a valid random tour.
    # Returns results dictionary for GUI that contains three ints (cost of solution,
    # time spent to find solution, number of permutations tried during search); the
    # solution found; and three null values for fields not used for this algorithm
    def default_random_tour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        num_cities = len(cities)
        found_tour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not found_tour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(num_cities)
            route = []
            # Now build the route using the random permutation
            for i in range(num_cities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                found_tour = True
        end_time = time.time()
        results['cost'] = bssf.cost if found_tour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['solution'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    # This is the entry point for the greedy solver; used to find initial B&B BSSF.
    # Returns a results dictionary for tne GUI that contains the cost of best solution,
    # time spent to find best solution, total number of solution paths tried, the best
    # solution found, and three null values for fields not used for this algorithm
    def greedy(self, time_allowance=60.0):
        # TOTAL TIME COMPLEXITY: O(n^3), but I believe it's closer to O(n^2) in the average case.
        # Triple-nested loops.
        # TOTAL SPACE COMPLEXITY: O(n). Storing the cities and keeping track of which
        # cities have been visited both take arrays of size n to store.
        results = {}
        count = 0
        bssf = None
        cities = self._scenario.getCities()  # takes O(n) space
        num_cities = len(cities)
        found_tour = False
        start_time = time.time()

        # The exact math gets tricky, but the average case loops no more than n times for large n
        while not found_tour and time.time() - start_time < time_allowance:
            # everything before the for-loops runs in constant time
            count += 1
            visited = np.zeros(num_cities)  # takes O(n) space
            first_city = np.random.randint(0, num_cities)
            route = [cities[first_city]]
            visited[first_city] = True
            next_city = None
            for i in range(num_cities - 1):  # loops n times
                next_city = None
                next_path_length = np.inf
                for j in range(num_cities):  # loops n times
                    if j != route[i]._index and not visited[j]:
                        if route[i].costTo(cities[j]) < next_path_length:
                            next_path_length = route[i].costTo(cities[j])
                            next_city = cities[j]
                if next_city is None:  # This path is invalid; force restart
                    break
                else:
                    route.append(next_city)
                    visited[next_city._index] = True
            bssf = TSPSolution(route)
            if bssf.cost < np.inf and next_city is not None:  # Then path is valid
                found_tour = True

        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['solution'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    # This is the entry point for the branch-and-bound algorithm.
    # Returns results dictionary for GUI that contains the cost of best solution, time spent to
    # find best solution, total number solutions found during search (does not include the initial
    # BSSF), the best solution found, the max queue size, total number of states created, and number
    # of pruned states.
    def branchAndBound(self, time_allowance=60.0):
        # TOTAL WORST CASE TIME COMPLEXITY: O(n^2*n!). Generating substates takes O(n^2)
        # time, and that loops up to n! times, but with good pruning and initial BSSF value it's
        # probably closer to n^2 * exponential time.
        # TOTAL WORST CASE SPACE COMPLEXITY: O(n^2*n!). Each substate takes O(n^2) space,
        # and the priority queue stores up to n! substates. However, this is much higher than
        # reality since most substates are pruned rather than being inserted into the queue. The
        # actual space complexity seems to be closer to exponential, rather than worse than
        # factorial.
        results = {}
        solution_count = 0
        max_heap_size = 0
        total_states = 0
        pruned_states = 0
        greedy_sol = self.greedy()  # Runs in O(n^3) time
        bssf = greedy_sol['solution']
        bssf_cost = greedy_sol['cost']
        cities = self._scenario.getCities()
        num_cities = len(cities)
        start_time = time.time()

        # initialize cost matrix (et al.). [from][to] = [row][col]
        # runs in O(n^2) time; not a limiting step
        cost_matrix = np.full((num_cities, num_cities), np.inf)
        for i in range(num_cities):
            for j in range(num_cities):
                cost_matrix[i][j] = cities[i].costTo(cities[j])
        city_availability = np.full(num_cities, True)
        city_order = []
        first_city = cities[0]

        parent_problem = TSPSubproblem(cost_matrix, 0, city_availability, city_order, first_city)
        total_states += 1

        # subproblems is kept as an array of heaps; each index keeps track of
        # the subproblems on a given level.
        subproblems = [[parent_problem]]
        curr_heap_size = 1
        max_heap_size = 1

        # This while loop can loop up to n! times, but it's closer to exponential in the average
        # case
        while curr_heap_size > 0 and time.time() - start_time < time_allowance:
            for level in range(len(subproblems)):
                if len(subproblems[level]) == 0:
                    continue
                if len(subproblems) < level + 2:  # then subproblems[level + 1] DNE yet
                    subproblems.append([])
                next_problem = heapq.heappop(subproblems[level])  # heappop runs in O(nlogn)
                curr_heap_size -= 1

                if next_problem.lower_bound < bssf_cost:
                    # Generating subproblems takes O(n) time since it only needs to search
                    # through the array of size n storing the cities left to travel to
                    city_availability = next_problem.city_availability
                    new_problems = []
                    for i in range(len(city_availability)):
                        if city_availability[i]:
                            total_states += 1
                            new_problem = TSPSubproblem(next_problem.cost_matrix,
                                                        next_problem.lower_bound,
                                                        city_availability,
                                                        next_problem.city_order, cities[i])
                            new_problems.append(new_problem)
                    # Analyzing new subproblems takes O(n^2)
                    for problem in new_problems:  # loops up to n times
                        # checking is_complete_solution takes O(n) time
                        if problem.is_complete_solution() and problem.lower_bound < bssf_cost:
                            solution_count += 1
                            bssf = TSPSolution(problem.city_order)
                            bssf_cost = bssf.cost
                        elif problem.lower_bound < bssf_cost:
                            heapq.heappush(subproblems[level + 1], problem)
                            # heappush runs in O(logn)
                            # The heap stores up to n! states, each of which take O(n^2) space,
                            # but with good pruning it becomes closer to 2^n states
                            curr_heap_size += 1
                            if curr_heap_size > max_heap_size:
                                max_heap_size = curr_heap_size
                        else:
                            pruned_states += 1
                else:
                    pruned_states += 1

        end_time = time.time()
        results['cost'] = bssf_cost
        results['time'] = end_time - start_time
        results['count'] = solution_count
        results['solution'] = bssf
        results['max'] = max_heap_size
        results['total'] = total_states
        results['pruned'] = pruned_states
        return results

    # This is the entry point for the group project algorithm: A* ALGORITHM.
    # Returns a results dictionary for GUI that contains the cost of the best solution,
    # time spent to find best solution, total number of solutions found during search, the best
    # solution found, and three null values not used for this implementation.
    def fancy(self, time_allowance=60.0):
        pass
