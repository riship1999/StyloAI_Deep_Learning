import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Load and prepare the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_content_features(df):
    # Create label encoders for categorical features
    le_brand = LabelEncoder()
    le_category = LabelEncoder()
    le_color = LabelEncoder()
    le_size = LabelEncoder()
    
    # Transform categorical features
    df['brand_encoded'] = le_brand.fit_transform(df['Brand'])
    df['category_encoded'] = le_category.fit_transform(df['Category'])
    df['color_encoded'] = le_color.fit_transform(df['Color'])
    df['size_encoded'] = le_size.fit_transform(df['Size'])
    
    return df

def create_user_item_matrix(df):
    return pd.pivot_table(
        df,
        values='Rating',
        index='User ID',
        columns='Product ID',
        fill_value=0
    )

def calculate_similarity_matrices(df, user_item_matrix):
    # Content-based similarity
    content_features = df[['brand_encoded', 'category_encoded', 'color_encoded', 'size_encoded', 'Price']].values
    content_similarity = cosine_similarity(content_features)
    
    # Collaborative filtering similarity
    user_similarity = cosine_similarity(user_item_matrix)
    
    return content_similarity, user_similarity

class HybridRecommender:
    def __init__(self, df, content_weight=0.5):
        self.df = df
        self.content_weight = content_weight
        self.collaborative_weight = 1 - content_weight
        
        # Prepare data
        self.df = prepare_content_features(self.df)
        self.user_item_matrix = create_user_item_matrix(self.df)
        self.content_similarity, self.user_similarity = calculate_similarity_matrices(
            self.df, self.user_item_matrix
        )
    
    def analyze_diversity(self, recommendations):
        """Analyze the diversity of recommendations"""
        brands = recommendations['Brand'].value_counts()
        categories = recommendations['Category'].value_counts()
        price_range = {
            'min': recommendations['Price'].min(),
            'max': recommendations['Price'].max(),
            'avg': recommendations['Price'].mean()
        }
        
        return {
            'brand_diversity': len(brands),
            'category_diversity': len(categories),
            'price_range': price_range,
            'brand_distribution': brands.to_dict(),
            'category_distribution': categories.to_dict()
        }
    
    def explain_recommendation(self, user_id, product_id):
        """Explain why a product was recommended"""
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        product_data = self.df[self.df['Product ID'] == product_id].iloc[0]
        
        # Get similar users who rated this product highly
        similar_users = self.user_similarity[user_idx].argsort()[-5:][::-1]
        # Convert product_id to string for user_item_matrix lookup
        product_id_str = str(int(product_id))
        similar_users_ratings = self.user_item_matrix.iloc[similar_users].get(product_id_str, pd.Series([0] * len(similar_users)))
        
        # Get content-based similarity
        product_idx = self.df[self.df['Product ID'] == product_id].index[0]
        content_sim = self.content_similarity[product_idx]
        similar_products = self.df.iloc[content_sim.argsort()[-5:][::-1]]
        
        explanation = {
            'collaborative_reason': f"{len(similar_users_ratings[similar_users_ratings > 0])} similar users rated this product",
            'content_reason': f"Similar to products you liked in terms of {product_data['Brand']} brand and {product_data['Category']} category",
            'price_point': f"Price point: ${product_data['Price']} ({'below' if product_data['Price'] < self.df['Price'].mean() else 'above'} average)",
        }
        
        return explanation
    
    def get_recommendations(self, user_id, n_recommendations=5):
        if user_id not in self.user_item_matrix.index:
            return self._get_popular_items(n_recommendations)
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        # Calculate collaborative filtering scores
        cf_scores = self.user_similarity[self.user_item_matrix.index.get_loc(user_id)]
        cf_predictions = np.dot(cf_scores, self.user_item_matrix) / np.sum(np.abs(cf_scores))
        
        # Calculate content-based scores
        content_scores = np.zeros(len(self.df['Product ID'].unique()))
        for item_id in rated_items:
            item_idx = self.df[self.df['Product ID'] == item_id].index[0]
            content_scores += self.content_similarity[item_idx]
        content_scores /= len(rated_items)
        
        # Combine scores
        final_scores = (
            self.collaborative_weight * cf_predictions +
            self.content_weight * content_scores
        )
        
        # Filter out already rated items
        final_scores[rated_items] = -1
        
        # Get top N recommendations
        top_items = final_scores.argsort()[-n_recommendations:][::-1]
        recommendations = self.df[self.df['Product ID'].isin(top_items + 1)][
            ['Product ID', 'Product Name', 'Brand', 'Category', 'Price', 'Color']
        ].drop_duplicates()
        
        return recommendations
    
    def _get_popular_items(self, n_recommendations=5):
        # Return top rated items for new users
        average_ratings = self.df.groupby('Product ID')['Rating'].mean()
        top_items = average_ratings.nlargest(n_recommendations).index
        return self.df[self.df['Product ID'].isin(top_items)][
            ['Product ID', 'Product Name', 'Brand', 'Category', 'Price', 'Color']
        ].drop_duplicates()

# Example usage
if __name__ == "__main__":
    # Load data
    df = load_data('fashion_products.csv')
    
    # Create recommender with equal weights
    recommender = HybridRecommender(df, content_weight=0.5)
    
    # Test with different content weights
    test_weights = [0.3, 0.5, 0.7]
    test_user = 97  # Let's focus on one user for detailed analysis
    
    for weight in test_weights:
        print(f"\n\n=== Testing with content_weight = {weight} ===")
        recommender.content_weight = weight
        recommender.collaborative_weight = 1 - weight
        
        recommendations = recommender.get_recommendations(user_id=test_user, n_recommendations=8)
        print("\nRecommendations:")
        print(recommendations)
        
        # Analyze diversity
        diversity_metrics = recommender.analyze_diversity(recommendations)
        print("\nDiversity Analysis:")
        print(f"Number of unique brands: {diversity_metrics['brand_diversity']}")
        print(f"Number of unique categories: {diversity_metrics['category_diversity']}")
        print(f"Price range: ${diversity_metrics['price_range']['min']} - ${diversity_metrics['price_range']['max']}")
        print(f"Average price: ${diversity_metrics['price_range']['avg']:.2f}")
        
        # Explain first recommendation
        first_rec = recommendations.iloc[0]
        explanation = recommender.explain_recommendation(test_user, first_rec['Product ID'])
        print(f"\nWhy recommend {first_rec['Product Name']} by {first_rec['Brand']}?")
        for key, value in explanation.items():
            print(f"- {value}")
