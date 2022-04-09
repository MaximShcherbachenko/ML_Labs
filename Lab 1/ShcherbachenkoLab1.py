import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # debug 5th task

# 1. Load the data and store it in dataframe
path = 'Vehicle_Sales.csv'
df = pd.read_csv(path)

# 2. Amount of rows and columns
print("Rows = {}\n".format(len(df)))
print("Columns = {}\n".format(len(df.columns)))

# 3. Top K + 7 and bottom 5K - 3 rows, where K = 2
print("Top 9 rows:")
print(df.head(9))
print("\nBottom 12 rows:")
print(df.tail(12))

# 4. Data type of each column
print(df.dtypes)

# 5. Convert Total Sales data types to float
for idx in range(len(df)):
    df['Total Sales New'][idx] = df['Total Sales New'][idx][1:]

for idx in range(len(df)):
    df['Total Sales Used'][idx] = df['Total Sales Used'][idx][1:]

df[['Total Sales New', 'Total Sales Used']] = df[['Total Sales New', 'Total Sales Used']].astype("float")
print(df[['Total Sales New', 'Total Sales Used']].dtypes)

# 6. Create new fields
df['Total All'] = df[['New', 'Used']].sum(axis=1)


df['Total Sales All'] = df[['Total Sales New', 'Total Sales Used']].sum(axis=1)


df_test = (df[['New', 'Used']].diff(periods=1, axis=1))
df_test.replace(np.nan, 0, inplace=True)
df_adder = pd.DataFrame(columns=['Difference of New and Used'])

for idx in range(len(df_test)):
    df_adder.loc[idx] = max(df_test.values.tolist()[idx])

df['Difference of New and Used'] = df_adder['Difference of New and Used']

# 7. Replace columns
new_columns = ['Year', 'Month', 'Total Sales All', 'Total Sales New', 'Total Sales Used', 'Total All', 'New',
               'Used', 'Difference of New and Used']
df = df[new_columns]

# 8. Requests_1
print(df[['Year', 'Month']][df['New'] < df['Used']])
print(df[['Year', 'Month']][df['Total Sales All'] == df['Total Sales All'].min()])
print(df[['Year', 'Month']][df['Used'] == df['Used'].max()])

# 9. Requests_2
print(df.groupby(['Year']).agg({'Total All': 'sum'}))


df_req = (df[df['Month'] == 'OCT'])[['Month', 'Total Sales Used']].groupby(['Month']).mean()
df_req.rename(columns={'Total Sales Used': 'Mean'}, inplace=True)
print(df_req)

# 10. Bar plot
plt.bar(df['Month'].unique(), df['New'][df['Year'] == 2007].tolist(), color='purple', edgecolor='black')
plt.xlabel("Month")
plt.ylabel("New")
plt.title("Sales new per month (2007)")
plt.show()

# 11. Pie plot
df_plot = df[['Year', 'Total All']].groupby(['Year'], as_index=False).sum()
plt.pie(df_plot['Total All'], labels=df_plot['Year'], autopct='%1.2f%%')
plt.title("Total sales per year")
plt.show()

# 12. Two plots
plt.plot(df['Total Sales New'].values, label='New')
plt.plot(df['Total Sales Used'].values, label='Used')
plt.title("Sales income from old and new cars")
plt.legend()
plt.show()
