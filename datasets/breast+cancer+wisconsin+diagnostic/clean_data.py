import csv

def remove_first_column(input_file, output_file):
    try:
        # Open the input CSV file for reading
        with open(input_file, 'r') as infile:
            csv_reader = csv.reader(infile)
            
            # Read all rows and remove the first column
            modified_rows = [row[1:] for row in csv_reader]
        
        # Open the output CSV file for writing
        with open(output_file, 'w', newline='') as outfile:
            csv_writer = csv.writer(outfile)
            
            # Write the modified rows to the output file
            csv_writer.writerows(modified_rows)
        
        print(f"First column removed successfully. Modified file saved as '{output_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# Replace 'input.csv' with the path to your input file
# Replace 'output.csv' with the desired path for the output file
input_csv = 'wdbc.data'
output_csv = 'wdbc_cleaned.data'
remove_first_column(input_csv, output_csv)