

import cplex
from cplex.exceptions import CplexError
import math
import sys

# ---------------------------
# Data Structures and Helpers
# ---------------------------

class Customer:
    def __init__(self, number, x, y, demand):
        self.number = number
        self.x = x
        self.y = y
        self.demand = demand

def read_instance(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Remove any empty lines or lines with only whitespace
    lines = [line.strip() for line in lines if line.strip()]
    
    # Read header
    header = lines[0].strip().split()
    num_customers = int(header[0])
    best_known_solution = float(header[1])
    
    # Vehicle capacity
    capacity = int(lines[1])
    
    # Depot coordinates
    depot_info = lines[2].strip().split()
    depot_x = int(depot_info[0])
    depot_y = int(depot_info[1])
    depot = Customer(0, depot_x, depot_y, 0)
    
    # Read customers
    customers = []
    for line in lines[3:3 + num_customers]:
        parts = line.strip().split()
        cust_number = int(parts[0])
        x = int(parts[1])
        y = int(parts[2])
        demand = int(parts[3])
        customers.append(Customer(cust_number, x, y, demand))
    
    return num_customers, capacity, depot, customers

def distance(c1, c2):
    return math.hypot(c1.x - c2.x, c1.y - c2.y)

# ---------------------------
# Column Generation Algorithm
# ---------------------------

def solve_cvrp_column_generation(filename):
    # Read instance data
    num_customers, capacity, depot, customers = read_instance(filename)
    customer_indices = [c.number for c in customers]
    
    # Initialize Master Problem
    master_prob = cplex.Cplex()
    master_prob.objective.set_sense(master_prob.objective.sense.minimize)
    
    # Initial feasible solution: Each customer is served individually
    routes = []
    ub_cost = 0
    for c in customers:
        route = [depot.number, c.number, depot.number]
        cost = distance(depot, c) * 2
        routes.append({'route': route, 'cost': cost, 'customers': [c.number], 'demand': c.demand})
        ub_cost += cost
    print(ub_cost)

    # Add variables (columns) to Master Problem
    var_names = []
    for idx, r in enumerate(routes):
        var_name = f"r_{idx}"
        var_names.append(var_name)
        master_prob.variables.add(obj=[r['cost']],
                                  lb=[0],
                                  ub=[1], #[ub_cost] ? FIXME
                                  types=['C'],
                                  names=[var_name])
    
    # Add constraints: Each customer is visited exactly once
    rows = []
    for c in customers:
        col = []
        coef = []
        for idx, r in enumerate(routes):
            if c.number in r['customers']:
                col.append(idx)
                coef.append(1)
        rows.append([col, coef])
    
    master_prob.linear_constraints.add(lin_expr=rows,
                                       senses=["E"] * num_customers,
                                       rhs=[1] * num_customers)
    
    # Column Generation Loop
    iteration = 0
    while True:
        iteration += 1
        print(f"Iteration {iteration}")
        
        # Solve Master Problem
        master_prob.solve()
        
        if master_prob.solution.get_status() != 1:
            print("Master problem not optimal.")
            break
        
        # Get dual values
        duals = master_prob.solution.get_dual_values()
        pi = {customers[i].number: duals[i] for i in range(num_customers)}
        
        # Solve Pricing Problem (SPPRC)
        new_route = pricing_problem(customers, depot, capacity, pi)
        
        # Check if new route has negative reduced cost
        if new_route is None or new_route['reduced_cost'] >= -1e-6:
            print("No more improving routes. Algorithm converged.")
            break
        
        # Add new variable (column) to Master Problem
        col = [0] * len(routes)
        col.append(1)  # New variable
        
        # Update variable names
        var_name = f"r_{len(routes)}"
        var_names.append(var_name)
        master_prob.variables.add(obj=[new_route['cost']],
                                  lb=[0],
                                  ub=[1],
                                  types=['C'],
                                  names=[var_name])
        
        # Update routes
        routes.append(new_route)
        
        # Update constraints
        for idx, c in enumerate(customers):
            if c.number in new_route['customers']:
                master_prob.linear_constraints.set_coefficients(idx, len(routes) - 1, 1)
            else:
                master_prob.linear_constraints.set_coefficients(idx, len(routes) - 1, 0)
    
    # Retrieve final solution
    print("Final solution:")
    solution_values = master_prob.solution.get_values()
    total_cost = master_prob.solution.get_objective_value()
    print(f"Total cost: {total_cost}")
    for idx, val in enumerate(solution_values):
        if val > 1e-6:
            print(f"Route {idx}: {routes[idx]['route']} with cost {routes[idx]['cost']} and value {val}")
    
def pricing_problem(customers, depot, capacity, pi):
    """
    Solve the Pricing Problem to find a new route with negative reduced cost.
    This is a SPPRC, which is NP-hard, but for simplicity, we can use a heuristic.
    """
    # For simplicity, we use a simple heuristic: consider a route visiting a subset of customers
    # not yet covered by low dual prices.
    # In practice, more sophisticated methods like labeling algorithms are used.
    
    # Build a graph with reduced costs
    nodes = [depot] + customers
    num_nodes = len(nodes)
    reduced_costs = [[0]*num_nodes for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                cost = distance(nodes[i], nodes[j])
                if nodes[j].number != 0:
                    reduced_costs[i][j] = cost - pi[nodes[j].number]
                else:
                    reduced_costs[i][j] = cost
    
    # Heuristic: Build a route by selecting customers with negative reduced costs
    route = [depot.number]
    total_demand = 0
    total_cost = 0
    customers_in_route = []
    remaining_customers = customers.copy()
    
    while remaining_customers:
        min_reduced_cost = float('inf')
        next_customer = None
        for c in remaining_customers:
            if total_demand + c.demand > capacity:
                continue
            rc = reduced_costs[route[-1]][c.number]
            if rc < min_reduced_cost:
                min_reduced_cost = rc
                next_customer = c
        if next_customer is None:
            break
        route.append(next_customer.number)
        total_demand += next_customer.demand
        total_cost += distance(nodes[route[-2]], next_customer)
        customers_in_route.append(next_customer.number)
        remaining_customers.remove(next_customer)
    
    if route[-1] != depot.number:
        route.append(depot.number)
        total_cost += distance(nodes[route[-2]], depot)
    
    # Compute reduced cost of the route
    reduced_cost = total_cost - sum(pi[c] for c in customers_in_route)
    
    if reduced_cost < -1e-6:
        new_route = {
            'route': route,
            'cost': total_cost,
            'customers': customers_in_route,
            'demand': total_demand,
            'reduced_cost': reduced_cost
        }
        return new_route
    else:
        return None

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python cvrp_column_generation.py <instance_file>")
    #     sys.exit(1)
    
    instance_file = './Instances/example_c75.txt' # sys.argv[1]
    solve_cvrp_column_generation(instance_file)
