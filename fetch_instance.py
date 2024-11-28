import os
import requests

def save_vrp_instance_to_file(url, filename):
    # Fetch the file from the URL
    response = requests.get(url)

    # Check for successful response
    if response.status_code != 200:
        print(f"Error fetching file. Status Code: {response.status_code}")
        return None

    # Create the directory /Instances/ if it doesn't exist
    if not os.path.exists('Instances'):
        os.makedirs('Instances')

    # File path where the content will be saved
    file_path = os.path.join('Instances', filename)

    # Save the content of the URL to a file
    with open(file_path, 'w') as file:
        file.write(response.text)

    print(f"File saved to: {file_path}")
    return file_path

# Example usage
url = "http://mistic.heig-vd.ch/taillard/problemes.dir/vrp.dir/christofides.dir/c75.txt"  # Replace with actual URL
filename = "example_c75.txt"  # Desired file name
save_vrp_instance_to_file(url, filename)
