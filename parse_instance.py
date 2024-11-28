import requests

def parse_vrp_instance(url):
    # Fetch the file from the URL
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Error fetching file. Status Code:", response.status_code)
        return None

    # Read the file content
    lines = response.text.strip().splitlines()

    # Initialize storage variables
    num_customers = 0
    best_known_solution = 0
    vehicle_capacity = 0
    depot_coords = ()
    customers = []
    solutions = []
    vehicle_tours = []
    service_time = []
    tour_time_limit = None

    # Process each line based on the provided format
    line_idx = 0

    # Parse the first line: number of customers and best known solution
    first_line = lines[line_idx].split(',')
    num_customers = int(first_line[0].strip())
    best_known_solution = int(first_line[1].strip())
    line_idx += 1

    # Parse the second line: vehicle capacity
    vehicle_capacity = int(lines[line_idx].strip())
    line_idx += 1

    # Parse the third line: depot coordinates (xdepot, ydepot)
    depot_coords = tuple(map(int, lines[line_idx].strip().split()))
    line_idx += 1

    # Parse customer data
    for _ in range(num_customers):
        customer_data = list(map(int, lines[line_idx].strip().split()))
        customer_number, x, y, demand = customer_data
        customers.append({
            'customer_number': customer_number,
            'x': x,
            'y': y,
            'demand': demand
        })
        line_idx += 1

    # Check if solutions exist in the file (optional part)
    if line_idx < len(lines):
        try:
            # Parse the number of vehicle tours, service times, and tour time limit
            solutions_data = list(map(int, lines[line_idx].strip().split()))
            num_vehicle_tours = solutions_data[0]
            service_time = solutions_data[1:num_customers + 1]  # For each customer
            tour_time_limit = solutions_data[-1]
            line_idx += 1

            # Parse each tour
            for _ in range(num_vehicle_tours):
                tour_data = list(map(int, lines[line_idx].strip().split()))
                num_customers_in_tour = tour_data[0]
                tour_customers = tour_data[1:num_customers_in_tour + 1]
                vehicle_tours.append(tour_customers)
                line_idx += 1

            solutions = {
                'num_vehicle_tours': num_vehicle_tours,
                'service_time': service_time,
                'tour_time_limit': tour_time_limit,
                'vehicle_tours': vehicle_tours
            }

        except ValueError:
            # If there is no solution part or improperly formatted solution part
            pass

    # Store the parsed data in a dictionary
    vrp_data = {
        'num_customers': num_customers,
        'best_known_solution': best_known_solution,
        'vehicle_capacity': vehicle_capacity,
        'depot_coords': depot_coords,
        'customers': customers,
        'solutions': solutions if solutions else None
    }

    return vrp_data


# Example usage
url = "http://example.com/vrp-instance.txt"  # Replace with actual URL
vrp_data = parse_vrp_instance(url)

# Print parsed data for verification
if vrp_data:
    print("Number of customers:", vrp_data['num_customers'])
    print("Best known solution:", vrp_data['best_known_solution'])
    print("Vehicle capacity:", vrp_data['vehicle_capacity'])
    print("Depot coordinates:", vrp_data['depot_coords'])
    print("Customer data:", vrp_data['customers'])
    if vrp_data['solutions']:
        print("Solution data:", vrp_data['solutions'])
