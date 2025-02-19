import pandas as pd
import numpy as np

sales_data = pd.DataFrame({
    'product_id': [101, 102, 103, 104, 105, 106, 107, 108],
    'category': ['Электроника', 'Электроника', 'Одежда', 'Одежда', 'Книги', 'Книги', 'Игрушки', 'Игрушки'],
    'price': [15000, 12000, np.nan, 2500, 500, np.nan, 800, 1000],
    'quantity_sold': [10, 15, 20, 50, 100, 80, 30, 25],
    'revenue': [150000, 180000, np.nan, np.nan, 50000, np.nan, np.nan, np.nan]
})
sales_data['price'] = sales_data.groupby('category')['price'].transform(lambda x: x.fillna(x.mean()))
sales_data['revenue'] = sales_data['price'] * sales_data['quantity_sold']
max_revenue_category = sales_data.groupby('category')['revenue'].sum().idxmax()

transactions_data = pd.DataFrame({
    'user_id': [1, 2, 3, 1, 4, 5, 2, 3, 4, 1, 5, 2],
    'amount': [500, 2000, 1500, 800, 5000, 750, 3000, 1200, 3500, 200, 250, 900],
    'transaction_type': ['debit', 'credit', 'debit', 'debit', 'credit', 'debit', 'credit', 'debit', 'credit', 'debit', 'debit', 'credit'],
    'timestamp': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03', '2024-01-04', '2024-01-04', '2024-01-05', '2024-01-05', '2024-01-06'])
})
amount_95p = transactions_data['amount'].quantile(0.95)
high_avg_users = transactions_data.groupby('user_id')['amount'].mean()[lambda x: x > amount_95p]
amount_90p = transactions_data['amount'].quantile(0.90)
high_tx_users = transactions_data.groupby(['user_id', transactions_data['timestamp'].dt.date])['amount'].apply(lambda x: (x > amount_90p).sum()).reset_index()
high_tx_users = high_tx_users[high_tx_users['amount'] >= 3]['user_id'].unique()
most_active_day = transactions_data['timestamp'].dt.date.value_counts().idxmax()

orders_data = pd.DataFrame({
    'customer_id': [101, 102, 103, 101, 104, 105, 102, 103, 106, 107, 101, 108],
    'order_date': pd.to_datetime(['2023-05-01', '2023-06-01', '2023-07-01', '2024-01-01', '2024-02-15', '2024-02-20', '2023-08-10', '2024-03-05', '2024-03-15', '2024-04-01', '2023-09-20', '2023-10-10']),
    'order_amount': [15000, 20000, 10000, 5000, 7000, 8000, 25000, 12000, 18000, 20000, 16000, 9000]
})
last_year = orders_data[orders_data['order_date'].dt.year == 2023]['customer_id'].unique()
this_year = orders_data[orders_data['order_date'].dt.year == 2024]['customer_id'].unique()
lost_customers = set(last_year) - set(this_year)
single_order_customers = (orders_data['customer_id'].value_counts() == 1).mean() * 100
active_customers = orders_data['customer_id'].value_counts()[lambda x: x > 3].index
times_between_orders = orders_data.groupby('customer_id')['order_date'].diff().dropna().median()

movies_data = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5, 1, 2, 3, 6, 7, 8, 9, 10, 1, 3, 5, 7, 9],
    'movie_id': [101, 102, 103, 104, 105, 101, 102, 103, 104, 105, 106, 107, 108, 106, 107, 108, 109, 110],
    'rating': [5, 4, 3, 5, 2, 4, 3, 5, 4, 5, 3, 2, 4, 5, 4, 3, 2, 1],
    'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12', '2024-01-13', '2024-01-14', '2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18'])
})
top_movies = movies_data.groupby('movie_id').filter(lambda x: len(x) > 50).groupby('movie_id')['rating'].mean().nlargest(5)
rating_per_day = movies_data.groupby('user_id').apply(lambda x: x['rating'].count() / x['timestamp'].nunique()).idxmax()

resumes_data = pd.DataFrame({
    'resume_id': range(1, 11),
    'name': ['Иван', 'Мария', 'Алексей', 'Сергей', 'Ольга', 'Николай', 'Анна', 'Павел', 'Юлия', 'Максим'],
    'email': ['ivan@mail.com', 'maria@mail.com', 'alex@mail.com', 'sergey@mail.com', 'olga@mail.com', 'nikolay@mail.com', 'anna@mail.com', 'pavel@mail.com', 'yulia@mail.com', 'max@mail.com'],
    'phone': ['+79110001111', '+79110002222', '+79110003333', '+79110004444', '+79110005555', '+79110006666', '+79110007777', '+79110008888', '+79110009999', '+79110000000'],
    'experience_years': [2, 5, 1, 10, 7, 3, 0, 8, 6, 12],
    'skills': ['Python, SQL', 'Java, Spring', 'HTML, CSS, JavaScript', 'C++, Linux', 'Go, Docker', 'Python, Machine Learning', 'Senior Developer, Management', 'JavaScript, React', 'C#, .NET', 'Data Science, AI']
})
resumes_data.drop_duplicates(subset=['email', 'phone'], inplace=True)
senior_anomalies = resumes_data[(resumes_data['experience_years'] < 1) & resumes_data['skills'].str.contains('Senior Developer')]
popular_skills = resumes_data[resumes_data['experience_years'] > 5]['skills'].str.split(', ').explode().value_counts().head(10)
