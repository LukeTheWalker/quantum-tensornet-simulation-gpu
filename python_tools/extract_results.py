import sqlite3
import numpy as np
import sys
# Connect to the SQLite database
conn = sqlite3.connect('db-v2.sqlite')

# Create a cursor object
cursor = conn.cursor()

# Define the query to retrieve the first input and output vector for program_id 1
query = """
SELECT input_vector, output_vector
FROM experiments
WHERE program_id = 10 AND id = ?
LIMIT 1;
"""

# Execute the query
cursor.execute(query, (sys.argv[1],))

# Fetch the result
result = cursor.fetchone()

# Check if a result was found
if result:
    input_vector_blob = result[0]
    output_vector_blob = result[1]

    # Convert the BLOBs back to numpy arrays
    input_vector = np.frombuffer(input_vector_blob, dtype=np.complex128)
    output_vector = np.frombuffer(output_vector_blob, dtype=np.complex128)

    

    np.set_printoptions(linewidth=200, threshold = 1025, formatter={'all': lambda x: "{:g}".format(x)})
    # Print the vectors
    print("Input Vector:\n", input_vector)
    print("Output Vector:\n", output_vector)
    # print size
    print("Input Vector size:", input_vector.size)
    print("Output Vector size:", output_vector.size)
else:
    print("No data found for program_id 1")

# Close the connection
conn.close()
