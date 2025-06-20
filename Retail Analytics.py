import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Load dataset
file_path = 'C:\\Users\\kareddy sindhuja\\Downloads\\Superstore.csv'
df=pd.read_csv(file_path,encoding='ISO-8859-1')
print(df.head(6))
#preprocessing
df['Order Date']=pd.to_datetime(df['Order Date'],dayfirst=True,errors='coerce')
df['Ship Date']=pd.to_datetime(df['Ship Date'],dayfirst=True,errors='coerce')
df['Delivery Time']=(df['Ship Date']-df['Order Date']).dt.days
df['Year']=df['Order Date'].dt.year
df['Month']=df['Order Date'].dt.month
sns.set(style="whitegrid")
# 1. Profit margins by category and sub-category
category_profit=df.groupby(['Category','Sub-Category'])[['Sales','Profit']].sum().reset_index()
print("Profit margins by category and sub-category")
print(category_profit)
'''plt.figure(figsize=(12,6))
sns.barplot(data=category_profit,x='Sub-Category',y='Profit',hue='Category')
plt.title('profit by Category and Sub-Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()'''
# 2. Sales and profit by state
state_profit=df.groupby('State')[['Sales','Profit']].sum().reset_index()
print("\n2.Sales and profit by state")
print(state_profit.sort_values(by='Sales',ascending=False).head())
'''plt.figure(figsize=(12,6))
top_states=(state_profit.sort_values(by='Sales',ascending=False).head(10))
sns.barplot(data=top_states,x='State',y='Profit')
plt.title('top 10 states by sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()'''
# 3.Discount impact on profit
discount_impact =df.groupby('Discount')[['Sales','Profit']].mean().reset_index()
print("\n3.Discount impact on profit")
print(discount_impact)
'''plt.figure(figsize=(10,5))
sns.lineplot(data=discount_impact,x='Discount',y='Profit',label='Profit')
sns.lineplot(data=discount_impact,x='Discount',y='Sales',label='Sales')
plt.title(" Impact of Discount on Sales and profit")
plt.tight_layout()
plt.show()'''
# 4.Loss_making Products
product_loss=df.groupby('Product Name')[['Sales','Profit']].sum().reset_index()
print("\n4.Top 10 Loss_making products")
print(product_loss.sort_values(by='Profit',ascending=False).head(10))
# 5. Average delivery time by ship mode
delivery_by_mode=df.groupby('Ship Mode')['Delivery Time'].mean().reset_index()
print("\n5.Average delivery time by shipping mode")
print(delivery_by_mode)
'''plt.figure(figsize=(8,5))
sns.barplot(data=delivery_by_mode,x='Ship Mode',y='Delivery Time')
plt.title("Average Delivery Time By Shipping Mode")
plt.tight_layout()
plt.show()'''
# 6.Shipping mode frequency and profitability
shipmode_profit=df.groupby('Ship Mode')[['Sales','Profit']].sum().reset_index()
print("\n6.Shipping mode frequency and profitability")
print(shipmode_profit)
#7.Shipping performance by region
shipmode_region=df.groupby(['Region','Ship Mode'])[['Sales','Profit']].sum().reset_index()
print("\n7.Shipping performance by region")
print(shipmode_region)
#8.Average order value by segment
segment_order=df.groupby('Segment').agg({'Sales':'mean','Order ID':'count'}).reset_index()
print("\n8.Average order value by segment")
print(segment_order)
# 9. Top 10 profitable customers
top_customers=(df.groupby(['Customer ID','Customer Name'])[['Profit']].sum()
             .sort_values(by='Profit',ascending=False).head(10).reset_index())
print("\n9.Top 10 profitable customers")
print(top_customers)
# 10.Repeat customers and discount/profit
repeat_customers=df.groupby('Customer ID').agg({'Order ID':'nunique','Discount':'mean','Profit':'sum'}).reset_index()
print("\n10.Repeat customers and discount/profit")
print(repeat_customers.head())
# 11.Monthly sales,profit and discount trends
monthly_trends=df.groupby(['Year','Month'])[['Sales','Profit','Discount']].sum().reset_index()
print("\n11.Monthly trends in Sales,Profit and Discount")
print(monthly_trends.head())
'''plt.figure(figsize=(12,6))
sns.lineplot(data=monthly_trends,x='Month',y='Sales',hue='Year',marker='d')
plt.title("Monthly trends in Sales,Profit and Discount")
plt.tight_layout()
plt.show()'''
# 12.Peak sales Months
monthly_peak=monthly_trends.sort_values(by='Sales',ascending=False).head(10)
print("\n12.Peak sales months")
print(monthly_peak)
# 13.Sub_Category sales per unit
value_per_unit = df.groupby('Sub-Category').agg({'Sales': 'sum', 'Quantity': 'sum'})
value_per_unit['Sales per Unit'] = value_per_unit['Sales'] / value_per_unit['Quantity']
value_per_unit = value_per_unit.reset_index()
print("\n13. Sub-Category Sales per Unit")
print(value_per_unit)
# 14. Product combinations (market basket)
market_basket =df.groupby('Order ID')['Product Name'].apply(lambda x:list(x)).reset_index()
print("\n14.Sample Product Combinations")
print(market_basket.head())
# 15. Predicting profit using regression
X = df[['Sales', 'Quantity', 'Discount']]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression().fit(X_train, y_train)
model_score = model.score(X_test, y_test)
print(f"\n15. Predictive Model R² Score for Profit Estimation: {model_score:.4f}")




