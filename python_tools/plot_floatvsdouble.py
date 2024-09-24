import csv
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import numpy as np
from scipy.stats import linregress

conn = sqlite3.connect(f'../data/db_ours.sqlite')

# Load table into pandas dataframe
df_ref = pd.read_sql_query("SELECT * FROM programs", conn)

# Close the connection
conn.close()

data_folder = "../data"

# read files from csv file
df_float = pd.read_csv(f'{data_folder}/times_float.csv', header=None)
df_double = pd.read_csv(f'{data_folder}/times_double.csv', header=None)

# merge all the dataframes
df_float.columns = ['time', 'id']
df_double.columns = ['time', 'id']

df_float['type'] = 'float'
df_double['type'] = 'double'

df = pd.merge(df_float, df_ref, on='id')
df = pd.merge(df, df_double, on='id')

# calculate the speedup
df['speedup'] =  df['time_y'] / df['time_x']

# the filename is circuit_{qubits}_{depths}.qasm, so we can extract the number of qubits and depths and create new columns
df['qubits'] = df['filename'].str.extract(r'_(\d+)_').astype(int)
df['depth'] = df['filename'].str.extract(r'_(\d+).qasm').astype(int)

plt.figure(figsize=(10,6))

qubit_set = df['qubits'].unique()
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

# for each number of qubits, plot the speedup in function of increasing depth
for qubits in qubit_set:
    df_qubits = df[df['qubits'] == qubits]
    plt.plot(df_qubits['depth'], df_qubits['speedup'], label=f'{qubits} qubits', color=colors[qubits])

    # calculate the regression line
    slope, intercept, r_value, p_value, std_err = linregress(df_qubits['depth'], df_qubits['speedup'])
    regression_line = intercept + slope * df_qubits['depth']
    # plt.plot(df_qubits['depth'], regression_line, '--', color='gray')

plt.title('Speedup vs Depth for different number of qubits')
plt.xlabel('Depth')
plt.ylabel('Speedup')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.grid()
# plt.savefig(f'../data/speedup_float_vs_double.png')
plt.savefig(f'../data/speedup_float_vs_double.svg', transparent=True, bbox_inches='tight')

# calculate the average speedup for each qubit
avg_speedup = df.groupby('qubits')['speedup'].mean()
print(avg_speedup)