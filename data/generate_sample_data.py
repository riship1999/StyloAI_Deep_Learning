import pandas as pd
import numpy as np
from faker import Faker
import os

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()

def generate_sample_data(n_users=1000, n_items=500, n_ratings=10000):
    """Generate sample data for fashion recommender"""
    
    # Create items data
    categories = ['Dress', 'Shirt', 'Pants', 'Shoes', 'Accessories']
    brands = ['Zara', 'H&M', 'Nike', 'Adidas', 'Gucci', 'Prada']
    colors = ['Red', 'Blue', 'Black', 'White', 'Green', 'Yellow']
    
    items = []
    for i in range(n_items):
        items.append({
            'item_id': i,
            'category': np.random.choice(categories),
            'brand': np.random.choice(brands),
            'color': np.random.choice(colors),
            'price': round(np.random.uniform(20, 500), 2),
            'description': fake.text(max_nb_chars=200)
        })
    
    items_df = pd.DataFrame(items)
    
    # Create ratings data
    user_ids = np.random.randint(0, n_users, n_ratings)
    item_ids = np.random.randint(0, n_items, n_ratings)
    ratings = np.random.normal(4, 1, n_ratings).round(1)
    # Clip ratings to be between 1 and 5
    ratings = np.clip(ratings, 1, 5)
    
    ratings_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': pd.date_range(start='2024-01-01', periods=n_ratings, freq='H')
    })
    
    # Create users data
    users = []
    for i in range(n_users):
        users.append({
            'user_id': i,
            'age': np.random.randint(18, 70),
            'gender': np.random.choice(['M', 'F']),
            'style_preference': np.random.choice(['Casual', 'Formal', 'Sporty', 'Vintage']),
            'signup_date': fake.date_between(start_date='-2y', end_date='today')
        })
    
    users_df = pd.DataFrame(users)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save dataframes
    items_df.to_csv('data/items.csv', index=False)
    ratings_df.to_csv('data/ratings.csv', index=False)
    users_df.to_csv('data/users.csv', index=False)
    
    print("Sample data generated successfully!")
    print(f"Items shape: {items_df.shape}")
    print(f"Ratings shape: {ratings_df.shape}")
    print(f"Users shape: {users_df.shape}")

if __name__ == "__main__":
    generate_sample_data()
