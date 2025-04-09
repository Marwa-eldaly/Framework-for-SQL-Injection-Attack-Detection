
import pandas as pd
import urllib.parse
import re
import sys

# Check if the user provided a file path as an argument
if len(sys.argv) < 2:
    print("Usage: python decode_ids.py <path_to_csv_file>")
    sys.exit(1)

# Get the file path from the command line arguments
file_path = sys.argv[1]

# Load the CSV file
df = pd.read_csv(file_path, header=None)

# Assuming the column name is 'data' as a placeholder
column_name = df.columns[0]

# Function to extract 'id' value and URL decode it
def extract_and_decode(value):
    match = re.search(r'id=([^&]*)', value)
    if match:
        return urllib.parse.unquote(match.group(1))
    return None

# Extract and URL decode 'id' values
df['decoded_id'] = df[column_name].apply(extract_and_decode)

# Remove duplicate 'decoded_id' values
df = df.drop_duplicates(subset=['decoded_id'])

# Save only the decoded 'id' values to a new CSV file
#output_path = 'Decoded_Ids_Only.csv'
output_path = 'New_' + file_path
df[['decoded_id']].to_csv(output_path, index=False, header=['decoded_id'])

