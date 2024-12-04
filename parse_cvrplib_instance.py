import sys
import os

def modify_vrp_instance(input_file_path, output_file_path):
    """
    Modifies the node/customer numbering in NODE_COORD_SECTION and DEMAND_SECTION
    to start from 0 instead of 1.

    Parameters:
    - input_file_path: Path to the original VRP instance file.
    - output_file_path: Path to save the modified VRP instance file.
    """
    try:
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading the file '{input_file_path}': {e}")
        sys.exit(1)

    # Initialize variables to keep track of the current section
    current_section = None
    counter = 0  # Counter for renumbering

    # Define the sections we are interested in
    target_sections = {'NODE_COORD_SECTION', 'DEMAND_SECTION'}

    # Iterate over each line and modify as needed
    for i, line in enumerate(lines):
        stripped_line = line.strip()

        # Check if the line starts a new section
        if stripped_line in target_sections:
            current_section = stripped_line
            counter = 0  # Reset counter for the new section
            continue  # Move to the next line

        # Check if the line starts a different section
        elif any(stripped_line.startswith(section) for section in {'NAME', 'COMMENT', 'TYPE', 'DIMENSION', 'EDGE_WEIGHT_TYPE', 'CAPACITY', 'DEPOT_SECTION', 'EOF'}):
            current_section = None  # Exit any target section
            continue  # Move to the next line

        # If we're in a target section, modify the numbering
        if current_section in target_sections:
            parts = stripped_line.split()
            if len(parts) < 2:
                # Not enough parts to modify; skip or handle as needed
                continue
            try:
                # Replace the first part with the new numbering
                new_number = str(counter)
                first_space_idx = line.find(parts[0])
                parts[0] = new_number
                # Reconstruct the line preserving original spacing
                # Find the index where the first space occurs

                if first_space_idx != -1:
                    # Replace from the start of the line up to the first space
                    # with the new number, keeping the original spacing
                    # after the number
                    lines[i] = f"{parts[0]} {' '.join(line.split()[1:])}\n"
                else:
                    # If no space found, just join the parts
                    lines[i] = ' '.join(parts) + '\n'
                counter += 1
            except Exception as e:
                print(f"Error processing line {i+1}: {e}")
                continue

    # Write the modified lines to the output file
    try:
        with open(output_file_path, 'w') as file:
            file.writelines(lines)
        print(f"Modified VRP instance saved to '{output_file_path}'.")
    except Exception as e:
        print(f"Error writing to the file '{output_file_path}': {e}")
        sys.exit(1)

    

if __name__ == "__main__":
    input_file_path = './Instances/P-n19-k2.txt' #input("Input File Path: ")
    output_file_path = './Instances/Adj-P-n19-k2.txt' #input("Output File Path: ")
    modify_vrp_instance(input_file_path, output_file_path)

