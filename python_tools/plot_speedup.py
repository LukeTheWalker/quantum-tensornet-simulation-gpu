import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_folder = "../data"

# Connect to the database
conn = sqlite3.connect(f'{data_folder}/db_ours.sqlite')

# Load table into pandas dataframe
df_ref = pd.read_sql_query("SELECT * FROM programs", conn)

# Close the connection
conn.close()

# read files from csv file
df_val = pd.read_csv('../build/times.csv', header=None)

# assign column names
df_val.columns = ['time', 'id']

# print sum of all times
print(f"total experiment time_ {df_val['time'].sum()}")

# merge the two dataframes
df = pd.merge(df_val, df_ref, on='id')

# calculate the speedup
df['speedup'] = (df['contraction_cpu_time_us'] / 1000) / df['time']

# the filename is circuit_{qubits}_{depths}.qasm, so we can extract the number of qubits and depths and create new columns
df['qubits'] = df['filename'].str.extract(r'_(\d+)_').astype(int)
df['depth'] = df['filename'].str.extract(r'_(\d+).qasm').astype(int)

plt.figure(figsize=(10,6))

# for each number of qubits, plot the speedup in function of increasing depth
for qubits in df['qubits'].unique():
    print(qubits)
    df_qubits = df[df['qubits'] == qubits]
    print(df_qubits)
    plt.plot(df_qubits['depth'], df_qubits['speedup'], label=f'{qubits} qubits')

plt.title('Speedup vs Depth for different number of qubits')
plt.xlabel('Depth')
plt.ylabel('Speedup')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.grid()
plt.savefig(f'{data_folder}/speedup.png')

# new figure
plt.figure(figsize=(10,6))

# make the mean of the speedup for each number of qubits on the df subset of relevant columns
df_mean = df[['qubits', 'speedup']].groupby('qubits').mean().reset_index()
plt.plot(df_mean['qubits'], df_mean['speedup'], 'o-')
plt.title('Mean Speedup vs Number of qubits')
plt.xlabel('Number of qubits')
plt.ylabel('Mean Speedup')
plt.grid()
plt.savefig(f'{data_folder}/mean_speedup.png')

