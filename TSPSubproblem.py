from copy import deepcopy
import numpy as np

# Overall, each Subproblem takes up O(n^2) space
class TSPSubproblem:

    def __init__(self, parent_cost_matrix, parent_lower_bound, city_availability,
                 city_order, next_city):
        self.cost_matrix = deepcopy(parent_cost_matrix)  # O(n^2) space
        self.lower_bound = deepcopy(parent_lower_bound)  # O(1) space
        self.city_availability = deepcopy(city_availability)  # O(n) space
        self.city_order = deepcopy(city_order)  # O(n) space
        self.latest_city = deepcopy(next_city)  # O(1) space

        self.city_availability[next_city._index] = False
        self.city_order.append(next_city)
        self.update()

    # Uses the cost matrix and knowledge of the previous and next city indexes to update the cost
    # matrix for this subproblem and find a new lower bound
    def update(self):
        prev_city = None
        if len(self.city_order) > 1:
            prev_city = self.city_order[-2]
        next_city = self.city_order[-1]

        matrix_dimension = len(self.cost_matrix)
        # if this isn't the first city, update so prev_city "from" row is inf and next_city "to"
        # column is inf
        # Also add the distance traveled to lower_bound
        if prev_city is not None:
            self.lower_bound += self.cost_matrix[prev_city._index][next_city._index]
            for i in range(matrix_dimension):
                self.cost_matrix[prev_city._index][i] = np.inf
                self.cost_matrix[i][next_city._index] = np.inf

        # update so all rows contain a 0 (or all inf)
        for i in range(matrix_dimension):
            row_min = np.inf
            for j in range(matrix_dimension):
                if self.cost_matrix[i][j] < row_min:
                    row_min = self.cost_matrix[i][j]
            if row_min != 0 and row_min != np.inf:
                self.lower_bound += row_min
                for j in range(matrix_dimension):
                    self.cost_matrix[i][j] -= row_min

        # update so all columns contain a 0 (or all inf)
        for j in range(matrix_dimension):
            col_min = np.inf
            for i in range(matrix_dimension):
                if self.cost_matrix[i][j] < col_min:
                    col_min = self.cost_matrix[i][j]
            if col_min != 0 and col_min != np.inf:
                self.lower_bound += col_min
                for i in range(matrix_dimension):
                    self.cost_matrix[i][j] -= col_min

    # O(n) time; needs to search up to the entirety of city_availability
    def is_complete_solution(self):
        for city_is_available in self.city_availability:
            if city_is_available:
                return False
        return True

    def __lt__(self, other):
        if self.lower_bound < other.lower_bound:
            return True
        return False
