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
df_val = pd.read_csv('../data/times_double_old.csv', header=None)
df_val_new = pd.read_csv('../data/times_double_new.csv', header=None)

# assign column names
df_val.columns = ['time', 'id']
df_val_new.columns = ['time', 'id']

# print sum of all times
print(f"total experiment time_ {df_val['time'].sum()}")
print(f"total experiment time_ {df_val_new['time'].sum()}")

# merge the two dataframes
df = pd.merge(df_val, df_ref, on='id')
df_new = pd.merge(df_val_new, df_ref, on='id')

# calculate the speedup
df['speedup'] = (df['contraction_cpu_time_us'] / 1000) / df['time']
df_new['speedup'] = (df_new['contraction_cpu_time_us'] / 1000) / df_new['time']

# the filename is circuit_{qubits}_{depths}.qasm, so we can extract the number of qubits and depths and create new columns
df['qubits'] = df['filename'].str.extract(r'_(\d+)_').astype(int)
df['depth'] = df['filename'].str.extract(r'_(\d+).qasm').astype(int)

df_new['qubits'] = df_new['filename'].str.extract(r'_(\d+)_').astype(int)
df_new['depth'] = df_new['filename'].str.extract(r'_(\d+).qasm').astype(int)

qubit_set =  df['qubits'].unique()
qubit_set.sort()
qubit_set = qubit_set[::-1]

colors = { 
    5 : 'tab:olive',
    6 : 'tab:blue', 
    7 : 'tab:orange',
    8 : 'tab:green',
    9 : 'tab:red',
    10 : 'tab:purple'
}

plt.figure(figsize=(10,6))

# for each number of qubits, plot the speedup in function of increasing depth
for qubits in qubit_set[:3]:
    print(qubits)
    df_qubits = df[df['qubits'] == qubits]
    df_qubits_new = df_new[df_new['qubits'] == qubits]
    print(df_qubits)
    plt.plot(df_qubits['depth'], df_qubits['speedup'], label=f'{qubits} qubits', color=colors[qubits])
    plt.plot(df_qubits_new['depth'], df_qubits_new['speedup'], label=f'{qubits} qubits new', color=colors[qubits], linestyle='--')

plt.title('Speedup vs Depth for different number of qubits')
plt.xlabel('Depth')
plt.ylabel('Speedup')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.grid()
plt.savefig(f'{data_folder}/speedup_high_qbits_count.png')
# plt.savefig(f'{data_folder}/speedup_high_qbits_count.svg', transparent=True, bbox_inches='tight')
plt.clf()

plt.figure(figsize=(10,6))

# for each number of qubits, plot the speedup in function of increasing depth
for qubits in qubit_set[2:]:
    print(qubits)
    df_qubits = df[df['qubits'] == qubits]
    df_qubits_new = df_new[df_new['qubits'] == qubits]
    print(df_qubits)
    plt.plot(df_qubits['depth'], df_qubits['speedup'], label=f'{qubits} qubits', color=colors[qubits])
    plt.plot(df_qubits_new['depth'], df_qubits_new['speedup'], label=f'{qubits} qubits new', color=colors[qubits], linestyle='--')

plt.title('Speedup vs Depth for different number of qubits')
plt.xlabel('Depth')
plt.ylabel('Speedup')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.grid()
plt.savefig(f'{data_folder}/speedup_low_qbits_count.png')
# plt.savefig(f'{data_folder}/speedup_low_qbits_count.svg', transparent=True, bbox_inches='tight')
plt.clf()

# new figure
plt.figure(figsize=(10,6))

# make the mean of the speedup for each number of qubits on the df subset of relevant columns
df_mean = df[['qubits', 'speedup']].groupby('qubits').mean().reset_index()
df_mean_new = df_new[['qubits', 'speedup']].groupby('qubits').mean().reset_index()
plt.plot(df_mean['qubits'], df_mean['speedup'], 'o-')
plt.plot(df_mean_new['qubits'], df_mean_new['speedup'], 'o--')
plt.title('Mean Speedup vs Number of qubits')
plt.xlabel('Number of qubits')
plt.ylabel('Mean Speedup')
plt.grid()
plt.savefig(f'{data_folder}/mean_speedup.png')
# plt.savefig(f'{data_folder}/mean_speedup.svg', transparent=True, bbox_inches='tight')

