import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import savefig

df=pd.read_csv("C:\\Users\\kareddy sindhuja\\Downloads\\Financial_Management_Dataset.csv")
print("DF rows and columns=",df.shape)
#convert and extract date components
df['Date']=pd.to_datetime(df['Date'])
df['Year']=df['Date'].dt.year
df['Month']=df['Date'].dt.month
df['Month_Year']= df['Date'].dt.to_period('M').astype(str)
sns.set(style="whitegrid")
print(df.head(4))
# what are the monthly trends in spending and income across different departments
monthly_dept=df.groupby(['Month_Year','Department','Transaction Type'])['Amount'].sum().reset_index()
print("1.Monthly trends in spending and income across departments")
print(monthly_dept.head())
plt.figure(figsize=(14,6))
sns.lineplot(data=monthly_dept,x='Month_Year',y='Amount',hue='Department',style='Transaction Type',marker='o')
plt.xticks(rotation=45)
plt.title('Monthly Spending and Income by Department')
plt.tight_layout()
plt.show()
# 2.Departments overshooting budgets by category
category_dept=df.groupby(['Department','Category'])['Amount'].sum().reset_index()
print("\n 2.Department_wise spending by category(to spot budget overshoots)")
print(category_dept.sort_values(by='Amount',ascending=False).head(10))
# 3.Average monthly income expenditure per account
monthly_account=df[df['Transaction Type']=='Debit'].groupby(['Month_Year','Account Name'])['Amount'].mean().reset_index()
print("\n 3.Average monthly debit per account")
print(monthly_account.head())
# 4.Net Cash Flow Per Month
cash_flow=df.pivot_table(index='Month_Year',columns='Transaction Type',values='Amount',aggfunc='sum').fillna(0)
cash_flow['Net Cash Flow']=cash_flow['Credit']-cash_flow['Debit']
print("\n 4.Net Cash Flow Per Month")
print(cash_flow.head())
cash_flow[['Credit','Debit','Net Cash Flow']].plot(kind='line',figsize=(14,6),marker='o')
plt.title('Monthly Net Cash Flow')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# 5.TOP categories for inflow and outflow
TOP_categories=df.groupby(['Category','Transaction Type'])['Amount'].sum().reset_index()
print("\n5.Top categories for inflow and outflow")
print(TOP_categories.sort_values(by='Amount',ascending=False).head(10))
# 6.Large Transaction detection
high_transaction=df[df['Amount']>df['Amount'].mean()+2*df['Amount'].std()]
print("\n6.Transaction significantly above average(possible liquidity risk")
print(high_transaction[['Transaction ID','Date','Amount','Department','Category']].head())
# 7. Spending Trend Per Category
category_trend=df[df['Transaction Type']=='Debit'].groupby(['Month_Year','Category'])['Amount'].sum().reset_index()
print("\n7.Spending Trend Per Category")
print(category_trend.head())
sns.lineplot(data=category_trend,x='Month_Year',y='Amount',hue='Category',marker='o')
plt.xticks(rotation=45)
plt.title("Spending Trend Per Category")
plt.tight_layout()
plt.show()
# 8.Top approvers by volume and value
approver_stats=(df.groupby('Approved By')['Amount'].agg(['count','sum']).reset_index().
rename(columns={'count':'Transaction Count','sum':'Total Amount'}))
print("\n8.Top approvers by Transaction count and value")
print(approver_stats.sort_values(by='Total Amount',ascending=False).head())
# 9.Non_Operational Analysis
Non_Operational=df[df['Category'].isin(['Travel','Maintenance','Ad_hoc Expense'])]
non_op_summary=Non_Operational.groupby('Category')['Amount'].sum().reset_index()
print("\n9.Non_Operational Analysis")
print(non_op_summary.head())
# 10.unusual department patterns
transaction_count=df.groupby(['Department','Transaction Type'])['Amount'].agg(['count','mean','sum']).reset_index()
print("\n10.Unusual Department Analysis")
print(transaction_count.sort_values(by='sum',ascending=False).head(10))
# 11.Outliers in Transaction Amount
outliers = df[df['Amount'] > df['Amount'].mean() + 2 * df['Amount'].std()]
print("\n11. Outlier Transactions (Above 2 Std Dev)")
print(outliers[['Transaction ID', 'Date', 'Amount', 'Department', 'Category']].head())

#12.Forecasting next month's spend (simple last-month extrapolation)
last_month=df[df['Date'].dt.month==df['Date'].max().month]
fore_cast=last_month.groupby('Category')['Amount'].mean().reset_index()
print("\n12.Forecasting of next month's spending by category")
print(fore_cast)
#13.Transaction type trend over time
ttype_trend=df.groupby(['Month_Year','Transaction Type'])['Amount'].sum().reset_index()
print("\n13.Transaction type trend over time")
print(ttype_trend.head())
sns.lineplot(data=ttype_trend,x='Month_Year',y='Amount',hue='Transaction Type',marker='o')
plt.title("Transaction type trend over time")
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
#14.Approver patterns in high-value transactions
high_value_approvals=df[df['Amount']>4000].groupby('Approved By')['Amount'].count().reset_index().sort_values(by='Amount',ascending=False)
print("\n14. High-value transaction approvals by person")
print(high_value_approvals.head())
#15. Distribution of transaction type by category
ttype_dist = df.groupby(['Category', 'Transaction Type'])['Amount'].sum().unstack().fillna(0)
print("\n15. Transaction Type Distribution by Category")
print(ttype_dist)

