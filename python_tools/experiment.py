import numpy as np
import sqlite3
import struct
import sys

def get_results():
    conn = sqlite3.connect('data/db_ours.sqlite')

    # Create a cursor object
    cursor = conn.cursor()

    # Define the query to retrieve the first input and output vector for program_id 1
    query = """
    SELECT input_vector, output_vector
    FROM experiments
    WHERE program_id = ? AND id = ?
    LIMIT 1;
    """

    query_time = """
    SELECT contraction_cpu_time_us
    FROM programs
    WHERE id = ?
    LIMIT 1;
    """
    # Execute the query
    cursor.execute(query, (sys.argv[1], sys.argv[2]))

    # Fetch the result
    result = cursor.fetchone()

    # Check if a result was found
    if result:
        input_vector_blob = result[0]
        output_vector_blob = result[1]

        # Convert the BLOBs back to numpy arrays
        input_vector = np.frombuffer(input_vector_blob, dtype=np.complex128)
        output_vector = np.frombuffer(output_vector_blob, dtype=np.complex128)
    else:
        print(f"No data found for {sys.argv[1]} and experiment_id {sys.argv[2]}.")

    # Execute the query
    cursor.execute(query_time, (sys.argv[1],))

    # Fetch the result
    result = cursor.fetchone()

    # Check if a result was found
    if result:
        reference_cpu = result[0]
    else:
        print(f"No data found for {sys.argv[1]}.")

    # Close the connection
    conn.close()

    return input_vector, output_vector, reference_cpu


# there is a txt file with the data of the gates
# read the file
with open("build/unitary.txt") as f:
    a = f.readlines()

a = [x.strip('\n') for x in a]
a = "".join(a)
a = a.split(", ")[:-1]

# do the above in a more pythonic way
a = [tuple(map(float, x.strip("()").split(","))) for x in a]

print("Size of a:", len(a))

# create a vector of (1,0) (0,0) * 15
# v = [(.25, 0), (.25, 0),(.25, 0),(.25, 0),(.25, 0),(.25, 0),(.25, 0),(.25, 0),(.25, 0),(.25, 0),(.25, 0),(.25, 0),(.25, 0),(.25, 0),(.25, 0),(.25, 0),]
# v = [(0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (1,0)]
# v = [(1,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)]
# v = [(0.5,0), (0.5,0), (0.5,0), (0.5,0)]

# transform to numpy array of complex numbers
a = np.array([complex(x[0], x[1]) for x in a])
# v = np.array([complex(x[0], x[1]) for x in v])

v_in, v_out, ref_time = get_results()

print("v_in size:", v_in.size)

print("input")
print(v_in)

# reshape to 2D array
a = a.reshape(v_in.size, v_in.size)

print("----------------------")
print("Reference")
print(v_out)

# perform matrix multiplication
result = np.dot(v_in, a)

print("----------------------")

print("Nostro")
print(result)

# print the result
# print(result)
# for i in range(16):
#     print(f"{result[i].real}", end=" ")
# print()
# np.set_printoptions(linewidth=200, formatter={'all': lambda x: "{:g}".format(x)})
# print(result)

# check the norm of the result
print(np.linalg.norm(result - v_out))

print(f"Reference CPU time: {ref_time/1000} ms")
