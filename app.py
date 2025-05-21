import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from hybrid_recommender import HybridRecommender, load_data
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

# Define seasonal preferences globally
seasonal_preferences = {
    'Spring': {'colors': ['Yellow', 'White', 'Pink'], 'categories': ['Dress', 'T-shirt']},
    'Summer': {'colors': ['White', 'Blue', 'Yellow'], 'categories': ['T-shirt', 'Shoes']},
    'Fall': {'colors': ['Brown', 'Black', 'Red'], 'categories': ['Sweater', 'Jeans']},
    'Winter': {'colors': ['Black', 'Grey', 'Blue'], 'categories': ['Sweater', 'Jeans', 'Shoes']}
}

# Function to calculate user preferences
def calculate_user_preferences(df, user_id):
    """Calculate user's preferences based on their purchase history"""
    user_data = df[df['User ID'] == user_id]
    
    # Brand preferences
    brand_prefs = user_data.groupby('Brand')['Rating'].agg(['mean', 'count']).round(2)
    brand_prefs = brand_prefs.sort_values('mean', ascending=False)
    
    # Category preferences
    category_prefs = user_data.groupby('Category')['Rating'].agg(['mean', 'count']).round(2)
    category_prefs = category_prefs.sort_values('mean', ascending=False)
    
    # Price sensitivity
    price_correlation = user_data['Price'].corr(user_data['Rating'])
    avg_price = user_data['Price'].mean()
    price_range = {
        'min': user_data['Price'].min(),
        'max': user_data['Price'].max(),
        'avg': avg_price,
        'preferred_range': f"${int(avg_price * 0.8)} - ${int(avg_price * 1.2)}"
    }
    
    return {
        'brand_preferences': brand_prefs,
        'category_preferences': category_prefs,
        'price_sensitivity': price_correlation,
        'price_range': price_range
    }

# Function to create preference charts
def create_preference_charts(preferences):
    """Create visualizations for user preferences"""
    # Brand preferences chart
    brand_fig = go.Figure()
    brand_fig.add_trace(go.Bar(
        x=preferences['brand_preferences'].index,
        y=preferences['brand_preferences']['mean'],
        name='Average Rating',
        marker_color='#ff4b4b'
    ))
    brand_fig.add_trace(go.Bar(
        x=preferences['brand_preferences'].index,
        y=preferences['brand_preferences']['count'],
        name='Purchase Count',
        marker_color='#1f77b4',
        yaxis='y2'
    ))
    brand_fig.update_layout(
        title='Brand Preferences',
        yaxis=dict(title='Average Rating'),
        yaxis2=dict(title='Purchase Count', overlaying='y', side='right'),
        barmode='group',
        height=400
    )
    
    # Category preferences chart
    category_fig = go.Figure()
    category_fig.add_trace(go.Bar(
        x=preferences['category_preferences'].index,
        y=preferences['category_preferences']['mean'],
        name='Average Rating',
        marker_color='#ff4b4b'
    ))
    category_fig.add_trace(go.Bar(
        x=preferences['category_preferences'].index,
        y=preferences['category_preferences']['count'],
        name='Purchase Count',
        marker_color='#1f77b4',
        yaxis='y2'
    ))
    category_fig.update_layout(
        title='Category Preferences',
        yaxis=dict(title='Average Rating'),
        yaxis2=dict(title='Purchase Count', overlaying='y', side='right'),
        barmode='group',
        height=400
    )
    
    return brand_fig, category_fig

# Function to analyze style profile
def analyze_style_profile(df, user_id):
    """Analyze user's style profile based on their purchase patterns"""
    user_data = df[df['User ID'] == user_id]
    
    # Style profile based on category combinations
    style_combinations = []
    for _, group in user_data.groupby('User ID'):
        categories = sorted(group['Category'].unique())
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                style_combinations.append(f"{categories[i]} + {categories[j]}")
    
    # Calculate favorite color combinations
    colors = user_data['Color'].value_counts()
    
    # Price tier preference
    price_mean = df['Price'].mean()
    price_std = df['Price'].std()
    
    price_tiers = pd.cut(
        user_data['Price'],
        bins=[0, price_mean - price_std, price_mean, price_mean + price_std, float('inf')],
        labels=['Budget', 'Value', 'Premium', 'Luxury']
    ).value_counts()
    
    return {
        'style_combinations': style_combinations,
        'color_preferences': colors,
        'price_tiers': price_tiers
    }

# Function to create style charts
def create_style_charts(style_profile):
    """Create visualizations for style profile"""
    # Color preferences
    color_fig = px.pie(
        values=style_profile['color_preferences'].values,
        names=style_profile['color_preferences'].index,
        title='Color Preferences',
        hole=0.4
    )
    color_fig.update_layout(height=400)
    
    # Price tier preferences
    price_tier_fig = go.Figure()
    price_tier_fig.add_trace(go.Bar(
        x=style_profile['price_tiers'].index,
        y=style_profile['price_tiers'].values,
        marker_color=['#90EE90', '#87CEEB', '#FFB6C1', '#DDA0DD']
    ))
    price_tier_fig.update_layout(
        title='Price Tier Preferences',
        xaxis_title='Price Tier',
        yaxis_title='Number of Purchases',
        height=400
    )
    
    return color_fig, price_tier_fig

# Function to generate style insights
def generate_style_insights(style_profile, preferences):
    """Generate natural language insights about user's style"""
    insights = []
    
    # Color insights
    top_colors = style_profile['color_preferences'].head(2)
    color_insight = f"Prefers {top_colors.index[0].lower()} and {top_colors.index[1].lower()} colored items"
    insights.append(color_insight)
    
    # Price tier insights
    dominant_tier = style_profile['price_tiers'].idxmax()
    price_insight = f"Typically shops in the {dominant_tier.lower()} price tier"
    insights.append(price_insight)
    
    # Brand loyalty
    top_brand = preferences['brand_preferences'].iloc[0]
    brand_loyalty = "High" if top_brand['count'] > 3 else "Moderate" if top_brand['count'] > 1 else "Low"
    brand_insight = f"{brand_loyalty} brand loyalty, with preference for {top_brand.name}"
    insights.append(brand_insight)
    
    # Category mix
    category_mix = preferences['category_preferences'].head(2)
    category_insight = f"Most interested in {category_mix.index[0]} and {category_mix.index[1]}"
    insights.append(category_insight)
    
    return insights

# Function to generate outfit combinations
def generate_outfit_combinations(style_profile, preferences):
    """Generate outfit combinations based on user preferences"""
    outfits = []
    
    # Get top categories
    top_categories = preferences['category_preferences'].head(4)
    top_colors = style_profile['color_preferences'].head(3)
    
    # Basic outfit combinations
    basic_combinations = [
        ('Men\'s Fashion', ['T-shirt', 'Jeans', 'Shoes']),
        ('Women\'s Fashion', ['Dress', 'Shoes']),
        ('Women\'s Fashion', ['Sweater', 'Jeans', 'Shoes']),
        ('Kids\' Fashion', ['T-shirt', 'Jeans', 'Shoes'])
    ]
    
    # Generate outfits based on user's preferred categories
    for category in top_categories.index:
        base_category = category.split()[0]  # Get Men's/Women's/Kids'
        for combo in basic_combinations:
            if combo[0] == base_category:
                outfit = {
                    'name': f"{top_colors.index[0]} {combo[1][0]} + {top_colors.index[1]} {combo[1][1]}",
                    'items': combo[1],
                    'style': 'Casual' if 'T-shirt' in combo[1] else 'Smart',
                    'occasion': 'Everyday' if 'T-shirt' in combo[1] else 'Semi-formal'
                }
                outfits.append(outfit)
    
    return outfits

# Function to analyze seasonal trends
def analyze_seasonal_trends(df, user_id):
    """Analyze seasonal trends in user's purchases"""
    user_data = df[df['User ID'] == user_id]
    
    # Analyze color and category matches per season
    seasonal_scores = {}
    for season, prefs in seasonal_preferences.items():
        color_match = user_data['Color'].isin(prefs['colors']).mean()
        category_match = user_data['Category'].isin(prefs['categories']).mean()
        seasonal_scores[season] = (color_match + category_match) / 2
    
    return {
        'seasonal_scores': seasonal_scores,
        'preferred_seasons': sorted(seasonal_scores.items(), key=lambda x: x[1], reverse=True)
    }

# Function to create seasonal chart
def create_seasonal_chart(seasonal_analysis):
    """Create radar chart for seasonal preferences"""
    seasons = list(seasonal_analysis['seasonal_scores'].keys())
    scores = list(seasonal_analysis['seasonal_scores'].values())
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],  # Complete the circle
        theta=seasons + [seasons[0]],  # Complete the circle
        fill='toself',
        name='Season Preference',
        line_color='#ff4b4b'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title='Seasonal Style Preference',
        height=400
    )
    
    return fig

# Function to create outfit cards
def create_outfit_cards(outfits):
    """Create visual cards for outfit combinations"""
    outfit_html = ""
    for i, outfit in enumerate(outfits):
        outfit_html += f"""
        <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h4>{outfit['name']}</h4>
            <p><strong>Items:</strong> {' + '.join(outfit['items'])}</p>
            <p><strong>Style:</strong> {outfit['style']} | <strong>Occasion:</strong> {outfit['occasion']}</p>
        </div>
        """
    return outfit_html

# Function to predict style trends
def predict_style_trends(df, user_id, style_profile):
    """Predict upcoming style trends based on user preferences"""
    user_data = df[df['User ID'] == user_id]
    
    # Analyze current preferences
    current_trends = {
        'colors': style_profile['color_preferences'].head(3).index.tolist(),
        'categories': user_data['Category'].value_counts().head(3).index.tolist(),
        'price_tier': style_profile['price_tiers'].idxmax()
    }
    
    # Define trend predictions
    trend_predictions = {
        'emerging_colors': {
            'Spring': ['Sage Green', 'Lavender', 'Coral'],
            'Summer': ['Ocean Blue', 'Sunny Yellow', 'Mint'],
            'Fall': ['Rust Orange', 'Deep Purple', 'Forest Green'],
            'Winter': ['Ice Blue', 'Burgundy', 'Charcoal']
        },
        'style_movements': {
            'Sustainable': ['Eco-friendly materials', 'Timeless designs', 'Quality over quantity'],
            'Smart Casual': ['Versatile pieces', 'Mix of formal and casual', 'Layered looks'],
            'Tech-Wear': ['Functional fabrics', 'Modern silhouettes', 'Adaptable designs'],
            'Vintage Revival': ['Classic patterns', 'Retro influences', 'Traditional craftsmanship']
        }
    }
    
    # Generate personalized trend insights
    current_season = get_current_season()
    next_season = get_next_season(current_season)
    
    return {
        'current_trends': current_trends,
        'upcoming_colors': trend_predictions['emerging_colors'][next_season],
        'style_movements': trend_predictions['style_movements'],
        'current_season': current_season,
        'next_season': next_season
    }

# Function to get current season
def get_current_season():
    """Get current season based on month"""
    month = datetime.now().month
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

# Function to get next season
def get_next_season(current_season):
    """Get next season"""
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    current_idx = seasons.index(current_season)
    next_idx = (current_idx + 1) % 4
    return seasons[next_idx]

# Function to generate advanced outfits
def generate_advanced_outfits(style_profile, preferences):
    """Generate more advanced outfit combinations"""
    outfits = []
    
    # Get user preferences
    top_categories = preferences['category_preferences'].head(5)
    top_colors = style_profile['color_preferences'].head(4)
    price_tier = style_profile['price_tiers'].idxmax()
    
    # Advanced outfit templates
    outfit_templates = {
        'Business': {
            'items': ['Blazer', 'Shirt', 'Trousers', 'Formal Shoes'],
            'occasions': ['Office', 'Meeting', 'Interview'],
            'style_notes': ['Professional', 'Polished', 'Confident']
        },
        'Smart Casual': {
            'items': ['Sweater', 'Shirt', 'Chinos', 'Loafers'],
            'occasions': ['Dinner', 'Date', 'Casual Friday'],
            'style_notes': ['Sophisticated', 'Relaxed', 'Versatile']
        },
        'Weekend': {
            'items': ['T-shirt', 'Jeans', 'Sneakers', 'Jacket'],
            'occasions': ['Shopping', 'Casual Outing', 'Travel'],
            'style_notes': ['Comfortable', 'Stylish', 'Easy-going']
        },
        'Evening': {
            'items': ['Dress', 'Heels', 'Clutch'],
            'occasions': ['Party', 'Event', 'Dinner'],
            'style_notes': ['Elegant', 'Glamorous', 'Sophisticated']
        }
    }
    
    # Generate outfits based on user preferences
    for style, template in outfit_templates.items():
        # Create base outfit
        outfit = {
            'name': f"{style} Ensemble",
            'items': template['items'],
            'colors': [top_colors.index[i % len(top_colors)] for i in range(len(template['items']))],
            'occasions': template['occasions'],
            'style_notes': template['style_notes'],
            'price_tier': price_tier
        }
        outfits.append(outfit)
        
        # Create variation
        variation = outfit.copy()
        variation['name'] = f"Alternative {style} Look"
        variation['colors'] = [top_colors.index[i % len(top_colors)] for i in range(1, len(template['items']) + 1)]
        outfits.append(variation)
    
    return outfits

# Function to create advanced outfit cards
def create_advanced_outfit_cards(outfits):
    """Create enhanced visual cards for outfit combinations"""
    outfit_html = ""
    for i, outfit in enumerate(outfits):
        color_chips = ' '.join([f'<span style="display: inline-block; width: 20px; height: 20px; margin-right: 5px; background-color: {get_color_hex(color)}; border-radius: 50%;"></span>'
                              for color in outfit['colors']])
        
        outfit_html += f"""
        <div style='background-color: #1E1E1E; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <h4>{outfit['name']}</h4>
                <div style='display: flex; align-items: center;'>
                    {color_chips}
                </div>
            </div>
            <p style='margin: 1rem 0;'><strong>Items:</strong> {' + '.join(outfit['items'])}</p>
            <p><strong>Perfect for:</strong> {', '.join(outfit['occasions'])}</p>
            <div style='background-color: #2E2E2E; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;'>
                <p style='margin: 0;'><strong>Style Notes:</strong> {' • '.join(outfit['style_notes'])}</p>
            </div>
            <div style='margin-top: 0.5rem;'>
                <span class='badge' style='background-color: #2E2E2E; padding: 0.25rem 0.5rem; border-radius: 15px;'>
                    {outfit['price_tier']} Range
                </span>
            </div>
        </div>
        """
    return outfit_html

# Function to get color hex
def get_color_hex(color):
    """Convert color name to hex code"""
    color_map = {
        'Black': '#000000',
        'White': '#FFFFFF',
        'Red': '#FF0000',
        'Blue': '#0000FF',
        'Green': '#008000',
        'Yellow': '#FFFF00',
        'Purple': '#800080',
        'Pink': '#FFC0CB',
        'Orange': '#FFA500',
        'Brown': '#A52A2A',
        'Grey': '#808080'
    }
    return color_map.get(color, '#000000')

# Function to suggest accessories
def suggest_accessories(style_profile, outfit_type):
    """Suggest accessories based on outfit type and style preferences"""
    accessory_suggestions = {
        'Business': {
            'Essential': ['Classic Watch', 'Leather Belt', 'Structured Bag'],
            'Optional': ['Silk Scarf', 'Tie Clip', 'Cufflinks'],
            'Statement': ['Designer Watch', 'Premium Briefcase', 'Pocket Square']
        },
        'Smart Casual': {
            'Essential': ['Leather Strap Watch', 'Canvas Belt', 'Crossbody Bag'],
            'Optional': ['Sunglasses', 'Statement Necklace', 'Leather Bracelet'],
            'Statement': ['Designer Scarf', 'Unique Jewelry', 'Premium Tote']
        },
        'Weekend': {
            'Essential': ['Sports Watch', 'Casual Belt', 'Backpack'],
            'Optional': ['Baseball Cap', 'Beaded Bracelet', 'Bandana'],
            'Statement': ['Designer Sneakers', 'Luxury Sunglasses', 'Statement Bag']
        },
        'Evening': {
            'Essential': ['Clutch', 'Statement Earrings', 'Elegant Watch'],
            'Optional': ['Cocktail Ring', 'Evening Scarf', 'Hair Accessories'],
            'Statement': ['Designer Clutch', 'Fine Jewelry', 'Hair Ornaments']
        }
    }
    
    price_tier = style_profile['price_tiers'].idxmax()
    tier_mapping = {
        'Budget': 'Essential',
        'Value': 'Optional',
        'Premium': 'Statement',
        'Luxury': 'Statement'
    }
    
    suggested_tier = tier_mapping[price_tier]
    return {
        'primary': accessory_suggestions[outfit_type][suggested_tier],
        'alternative': accessory_suggestions[outfit_type]['Optional']
    }

# Function to get color combinations
def get_color_combinations(primary_color):
    """Get harmonious color combinations"""
    color_harmony = {
        'Black': {
            'Monochrome': ['White', 'Grey'],
            'Complementary': ['Gold', 'Silver'],
            'Accent': ['Red', 'Yellow']
        },
        'White': {
            'Monochrome': ['Black', 'Grey'],
            'Complementary': ['Navy', 'Brown'],
            'Accent': ['Red', 'Blue']
        },
        'Blue': {
            'Monochrome': ['Light Blue', 'Navy'],
            'Complementary': ['Orange', 'Yellow'],
            'Accent': ['White', 'Grey']
        },
        'Red': {
            'Monochrome': ['Pink', 'Burgundy'],
            'Complementary': ['Green', 'Blue'],
            'Accent': ['White', 'Black']
        },
        'Green': {
            'Monochrome': ['Lime', 'Forest Green'],
            'Complementary': ['Red', 'Purple'],
            'Accent': ['White', 'Brown']
        }
    }
    return color_harmony.get(primary_color, {
        'Monochrome': ['White', 'Black'],
        'Complementary': ['Blue', 'Brown'],
        'Accent': ['Red', 'Gold']
    })

# Function to create style movement analysis
def create_style_movement_analysis(preferences, style_profile):
    """Create detailed style movement analysis"""
    movements = {
        'Minimalist': {
            'attributes': ['Clean lines', 'Neutral colors', 'Quality basics'],
            'indicators': ['Monochrome preference', 'Basic essentials', 'Premium quality'],
            'score': 0
        },
        'Avant-Garde': {
            'attributes': ['Unique pieces', 'Bold colors', 'Experimental cuts'],
            'indicators': ['Statement pieces', 'Color mixing', 'Modern silhouettes'],
            'score': 0
        },
        'Classic': {
            'attributes': ['Timeless pieces', 'Traditional patterns', 'Refined style'],
            'indicators': ['Traditional items', 'Neutral palette', 'Quality focus'],
            'score': 0
        },
        'Bohemian': {
            'attributes': ['Natural fabrics', 'Layered looks', 'Artistic elements'],
            'indicators': ['Pattern mixing', 'Comfortable fits', 'Artistic details'],
            'score': 0
        },
        'Street Style': {
            'attributes': ['Urban elements', 'Casual comfort', 'Bold statements'],
            'indicators': ['Casual pieces', 'Sporty elements', 'Statement accessories'],
            'score': 0
        },
        'Eco-Conscious': {
            'attributes': ['Sustainable materials', 'Timeless design', 'Ethical choices'],
            'indicators': ['Quality over quantity', 'Natural materials', 'Versatile pieces'],
            'score': 0
        }
    }
    
    # Calculate movement scores based on preferences
    for movement in movements:
        score = 0
        # Color preferences alignment
        if movement in ['Minimalist', 'Classic'] and len(style_profile['color_preferences']) < 5:
            score += 0.3
        elif movement in ['Avant-Garde', 'Bohemian'] and len(style_profile['color_preferences']) > 5:
            score += 0.3
            
        # Price tier alignment
        price_tier = style_profile['price_tiers'].idxmax()
        if movement in ['Minimalist', 'Classic', 'Eco-Conscious'] and price_tier in ['Premium', 'Luxury']:
            score += 0.3
        elif movement in ['Street Style', 'Bohemian'] and price_tier in ['Value', 'Budget']:
            score += 0.3
            
        # Category variety alignment
        if movement in ['Avant-Garde', 'Bohemian'] and len(style_profile['style_combinations']) > 3:
            score += 0.4
        elif movement in ['Minimalist', 'Classic'] and len(style_profile['style_combinations']) < 4:
            score += 0.4
            
        movements[movement]['score'] = min(score, 1.0)
    
    return movements

# Function to get advanced style movements
def get_advanced_style_movements():
    """Get expanded list of style movements with detailed attributes"""
    return {
        'Minimalist': {
            'attributes': ['Clean lines', 'Neutral colors', 'Quality basics'],
            'signature_pieces': ['White shirt', 'Black pants', 'Leather tote'],
            'styling_tips': ['Focus on fit', 'Choose quality fabrics', 'Minimal accessories'],
            'color_palette': ['Black', 'White', 'Grey', 'Navy']
        },
        'Avant-Garde': {
            'attributes': ['Unique pieces', 'Bold colors', 'Experimental cuts'],
            'signature_pieces': ['Statement coat', 'Architectural shoes', 'Dramatic accessories'],
            'styling_tips': ['Mix textures', 'Play with proportions', 'Layer creatively'],
            'color_palette': ['Black', 'White', 'Red', 'Metallic']
        },
        'Classic': {
            'attributes': ['Timeless pieces', 'Traditional patterns', 'Refined style'],
            'signature_pieces': ['Blazer', 'Oxford shirt', 'Pencil skirt'],
            'styling_tips': ['Invest in basics', 'Choose timeless cuts', 'Add subtle details'],
            'color_palette': ['Navy', 'White', 'Beige', 'Burgundy']
        },
        'Bohemian': {
            'attributes': ['Natural fabrics', 'Layered looks', 'Artistic elements'],
            'signature_pieces': ['Maxi dress', 'Embroidered top', 'Fringe bag'],
            'styling_tips': ['Layer different lengths', 'Mix patterns', 'Add natural accessories'],
            'color_palette': ['Earth tones', 'Rust', 'Cream', 'Turquoise']
        },
        'Street Style': {
            'attributes': ['Urban elements', 'Casual comfort', 'Bold statements'],
            'signature_pieces': ['Graphic tee', 'Sneakers', 'Bomber jacket'],
            'styling_tips': ['Mix high-low pieces', 'Add statement accessories', 'Layer strategically'],
            'color_palette': ['Black', 'White', 'Neon accents', 'Primary colors']
        },
        'Eco-Conscious': {
            'attributes': ['Sustainable materials', 'Timeless design', 'Ethical choices'],
            'signature_pieces': ['Organic cotton tee', 'Recycled denim', 'Vegan accessories'],
            'styling_tips': ['Choose quality over quantity', 'Opt for natural materials', 'Support ethical brands'],
            'color_palette': ['Natural tones', 'Sage', 'Sand', 'Ocean blue']
        },
        'Preppy': {
            'attributes': ['Polished casual', 'Traditional elements', 'Structured pieces'],
            'signature_pieces': ['Polo shirt', 'Chinos', 'Loafers'],
            'styling_tips': ['Coordinate colors', 'Add classic accessories', 'Balance casual and formal'],
            'color_palette': ['Navy', 'Pink', 'Kelly green', 'White']
        },
        'Romantic': {
            'attributes': ['Feminine details', 'Soft fabrics', 'Delicate elements'],
            'signature_pieces': ['Floral dress', 'Lace top', 'Ballet flats'],
            'styling_tips': ['Layer delicate pieces', 'Add feminine touches', 'Choose soft colors'],
            'color_palette': ['Blush', 'Lavender', 'Soft blue', 'Cream']
        },
        'Athleisure': {
            'attributes': ['Sports-inspired', 'Comfortable', 'Technical fabrics'],
            'signature_pieces': ['Premium leggings', 'Designer sneakers', 'Tech jacket'],
            'styling_tips': ['Mix athletic and fashion pieces', 'Focus on fit', 'Add luxe touches'],
            'color_palette': ['Black', 'Grey', 'White', 'Neon accents']
        },
        'Edgy': {
            'attributes': ['Bold elements', 'Dark colors', 'Statement pieces'],
            'signature_pieces': ['Leather jacket', 'Ripped jeans', 'Combat boots'],
            'styling_tips': ['Mix textures', 'Add hardware details', 'Create contrast'],
            'color_palette': ['Black', 'Grey', 'Deep red', 'Metallic']
        }
    }

# Function to create style movement radar chart
def create_style_movement_radar(movement_scores):
    """Create radar chart for style movement analysis"""
    categories = list(movement_scores.keys())
    values = list(movement_scores.values())
    
    # Complete the circular shape
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#FF4B4B'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title='Style Movement Profile',
        height=500
    )
    
    return fig

# Function to get outfit-specific styling recommendations
def get_outfit_specific_styling(style_movement, occasion):
    """Get detailed styling recommendations for specific occasions"""
    styling_guide = {
        'Minimalist': {
            'Work': {
                'Key Pieces': ['Tailored blazer', 'Silk shirt', 'Straight-leg pants'],
                'Colors': ['Navy', 'White', 'Grey'],
                'Accessories': ['Watch', 'Belt', 'Professional Bag']
            },
            'Casual': {
                'Key Pieces': ['White t-shirt', 'High-waisted jeans', 'Leather sneakers'],
                'Colors': ['White', 'Black', 'Beige'],
                'Accessories': ['Canvas tote', 'Delicate necklace', 'Sunglasses']
            },
            'Evening': {
                'Key Pieces': ['Black dress', 'Structured clutch', 'Heeled sandals'],
                'Colors': ['Black', 'White', 'Metallic'],
                'Accessories': ['Statement ring', 'Minimal bracelet', 'Simple earrings']
            }
        },
        'Classic': {
            'Work': {
                'Key Pieces': ['Pencil skirt', 'Button-down shirt', 'Pumps'],
                'Colors': ['Navy', 'White', 'Burgundy'],
                'Accessories': ['Structured bag', 'Pearl earrings', 'Classic watch']
            },
            'Casual': {
                'Key Pieces': ['Polo shirt', 'Chinos', 'Loafers'],
                'Colors': ['Khaki', 'Navy', 'White'],
                'Accessories': ['Leather belt', 'Simple bracelet', 'Canvas watch']
            },
            'Evening': {
                'Key Pieces': ['Little black dress', 'Silk scarf', 'Classic pumps'],
                'Colors': ['Black', 'Navy', 'Red'],
                'Accessories': ['Pearl necklace', 'Diamond studs', 'Evening bag']
            }
        }
        # Add more movements and occasions as needed
    }
    
    return styling_guide.get(style_movement, {}).get(occasion, {})

# Function to get outfit completion suggestions
def get_outfit_completion_suggestions(user_purchases, style_profile):
    """Suggest items to complete existing outfits"""
    existing_categories = user_purchases['Category'].unique()
    missing_categories = {
        'Formal': ['Suit', 'Dress Shirt', 'Dress Shoes', 'Tie'],
        'Casual': ['T-shirt', 'Jeans', 'Sneakers', 'Sweater'],
        'Athletic': ['Sports Top', 'Sports Bottom', 'Athletic Shoes', 'Jacket'],
        'Evening': ['Dress', 'Heels', 'Clutch', 'Accessories']
    }
    
    completion_suggestions = {}
    for style, items in missing_categories.items():
        missing_items = [item for item in items if item not in existing_categories]
        if missing_items:
            completion_suggestions[style] = missing_items
    
    return completion_suggestions

# Function to get occasion-based recommendations
def get_occasion_recommendations(style_profile, occasion):
    """Get recommendations for specific occasions"""
    occasion_styles = {
        'Work': {
            'essential': ['Blazer', 'Dress Shirt', 'Formal Pants'],
            'colors': ['Navy', 'Black', 'Grey'],
            'accessories': ['Watch', 'Belt', 'Professional Bag']
        },
        'Weekend': {
            'essential': ['T-shirt', 'Jeans', 'Sneakers'],
            'colors': ['White', 'Blue', 'Beige'],
            'accessories': ['Sunglasses', 'Casual Watch', 'Backpack']
        },
        'Special Event': {
            'essential': ['Suit', 'Dress', 'Formal Shoes'],
            'colors': ['Black', 'Navy', 'Burgundy'],
            'accessories': ['Tie', 'Clutch', 'Statement Jewelry']
        },
        'Vacation': {
            'essential': ['Swimwear', 'Resort Wear', 'Sandals'],
            'colors': ['White', 'Blue', 'Yellow'],
            'accessories': ['Hat', 'Beach Bag', 'Sunglasses']
        }
    }
    
    return occasion_styles[occasion]

# Page config
st.set_page_config(
    page_title="Fashion Recommender System",
    page_icon="👕",
    layout="wide"
)

# Set dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .css-1d391kg {
        background-color: #1E1E1E;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E1E1E;
    }
    .stTabs [data-baseweb="tab"] {
        color: #FAFAFA;
    }
    div[data-testid="stMarkdownContainer"] {
        color: #FAFAFA;
    }
    .card {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #FAFAFA;
    }
    .style-card {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #FAFAFA;
    }
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #1E1E1E;
        border: 1px solid #2E2E2E;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommender' not in st.session_state:
    df = load_data('fashion_products.csv')
    st.session_state.recommender = HybridRecommender(df)
    st.session_state.df = df

# Title and description
st.title("👕 Fashion Recommendation System")
st.markdown("""
This intelligent system combines collaborative and content-based filtering to provide personalized fashion recommendations.
Adjust the settings in the sidebar to see how different parameters affect the recommendations.
""")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # User selection with search
    st.subheader("User Selection")
    user_search = st.text_input("Search User ID", "")
    unique_users = sorted(st.session_state.df['User ID'].unique())
    filtered_users = [user for user in unique_users if str(user).startswith(user_search)] if user_search else unique_users
    selected_user = st.selectbox("Select User ID", filtered_users)
    
    st.markdown("---")
    
    # Algorithm settings
    st.subheader("Algorithm Settings")
    content_weight = st.slider(
        "Content-Based Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Adjust the balance between content-based and collaborative filtering"
    )
    
    n_recommendations = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=15,
        value=8,
        step=1
    )
    
    st.markdown("---")
    
    # Filters
    st.subheader("Filters")
    price_range = st.slider(
        "Price Range ($)",
        min_value=int(st.session_state.df['Price'].min()),
        max_value=int(st.session_state.df['Price'].max()),
        value=(int(st.session_state.df['Price'].min()), int(st.session_state.df['Price'].max()))
    )
    
    selected_categories = st.multiselect(
        "Categories",
        options=sorted(st.session_state.df['Category'].unique()),
        default=[]
    )
    
    selected_brands = st.multiselect(
        "Brands",
        options=sorted(st.session_state.df['Brand'].unique()),
        default=[]
    )

# Update recommender weights
st.session_state.recommender.content_weight = content_weight
st.session_state.recommender.collaborative_weight = 1 - content_weight

# Get recommendations
recommendations = st.session_state.recommender.get_recommendations(
    user_id=selected_user,
    n_recommendations=n_recommendations
)

# Apply filters
if selected_categories:
    recommendations = recommendations[recommendations['Category'].isin(selected_categories)]
if selected_brands:
    recommendations = recommendations[recommendations['Brand'].isin(selected_brands)]
recommendations = recommendations[
    (recommendations['Price'] >= price_range[0]) &
    (recommendations['Price'] <= price_range[1])
]

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Recommended Products")
    
    if recommendations.empty:
        st.warning("No products match your current filters. Try adjusting the filters in the sidebar.")
    else:
        # Display recommendations in a grid
        for i in range(0, len(recommendations), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(recommendations):
                    item = recommendations.iloc[i + j]
                    with cols[j]:
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h3>{item['Product Name']} by {item['Brand']}</h3>
                                <p>Category: {item['Category']}</p>
                                <p>Price: ${item['Price']}</p>
                            </div>
                            """, unsafe_allow_html=True)

with col2:
    st.subheader("Analysis Dashboard")
    
    if not recommendations.empty:
        # Get diversity metrics
        diversity_metrics = st.session_state.recommender.analyze_diversity(recommendations)
        
        # Metrics cards
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="metric-card">
                <h4>Brand Diversity</h4>
                <h2>{} brands</h2>
            </div>
            """.format(diversity_metrics['brand_diversity']), unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div class="metric-card">
                <h4>Category Diversity</h4>
                <h2>{} categories</h2>
            </div>
            """.format(diversity_metrics['category_diversity']), unsafe_allow_html=True)
        
        # Price distribution
        price_fig = go.Figure()
        price_fig.add_trace(go.Box(
            y=recommendations['Price'],
            name="Price Distribution",
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        price_fig.update_layout(
            title="Price Distribution",
            showlegend=False,
            height=300
        )
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Brand distribution
        brands = pd.Series(diversity_metrics['brand_distribution'])
        brand_fig = px.pie(
            values=brands.values,
            names=brands.index,
            title="Brand Distribution"
        )
        brand_fig.update_layout(height=300)
        st.plotly_chart(brand_fig, use_container_width=True)

# User's previous purchases
st.markdown("---")
st.subheader("User Profile Analysis")

# Get user's purchase history
user_purchases = st.session_state.df[
    st.session_state.df['User ID'] == selected_user
][['Product Name', 'Brand', 'Category', 'Price', 'Rating']]

if not user_purchases.empty:
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Purchase History", "Price vs Rating", "Similar Users"])
    
    with tab1:
        st.dataframe(
            user_purchases,
            use_container_width=True
        )
    
    with tab2:
        fig = px.scatter(
            user_purchases,
            x='Price',
            y='Rating',
            color='Category',
            hover_data=['Brand', 'Category'],
            title="Previous Purchases - Price vs Rating"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Create user similarity network
        user_idx = st.session_state.recommender.user_item_matrix.index.get_loc(selected_user)
        similar_users = st.session_state.recommender.user_similarity[user_idx]
        
        # Get top 5 similar users
        top_similar = pd.Series(similar_users, index=st.session_state.recommender.user_item_matrix.index)
        top_similar = top_similar.nlargest(6)[1:]  # Exclude self
        
        # Create network graph
        G = nx.Graph()
        G.add_node(selected_user, size=20, color='red')
        
        # Add edges to similar users
        for user, similarity in top_similar.items():
            G.add_node(user, size=15, color='blue')
            G.add_edge(selected_user, user, weight=similarity)
        
        # Create network visualization using plotly
        edge_x = []
        edge_y = []
        pos = nx.spring_layout(G)
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_colors.append(G.nodes[node]['color'])
            node_sizes.append(G.nodes[node]['size'])
            node_text.append(f'User {node}')
        
        # Create the network plot
        network_fig = go.Figure()
        
        # Add edges
        network_fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        network_fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="bottom center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line_width=2
            )
        ))
        
        network_fig.update_layout(
            title="User Similarity Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
        
        st.plotly_chart(network_fig, use_container_width=True)

# Advanced user analysis
if not user_purchases.empty:
    st.markdown("---")
    st.subheader("Advanced User Analysis")
    
    # Calculate user preferences
    preferences = calculate_user_preferences(st.session_state.df, selected_user)
    
    # Create preference visualizations
    brand_fig, category_fig = create_preference_charts(preferences)
    
    # Display price sensitivity insights
    price_sensitivity = preferences['price_sensitivity']
    price_range = preferences['price_range']
    
    # Create columns for metrics
    col_price1, col_price2, col_price3 = st.columns(3)
    
    with col_price1:
        st.metric(
            "Price Sensitivity",
            f"{price_sensitivity:.2f}",
            help="Correlation between price and rating. Positive values indicate preference for higher-priced items."
        )
    
    with col_price2:
        st.metric(
            "Average Spend",
            f"${price_range['avg']:.2f}"
        )
    
    with col_price3:
        st.metric(
            "Preferred Price Range",
            price_range['preferred_range']
        )
    
    # Display preference charts
    tab_prefs1, tab_prefs2 = st.tabs(["Brand Analysis", "Category Analysis"])
    
    with tab_prefs1:
        st.plotly_chart(brand_fig, use_container_width=True)
        
        # Add brand preference table
        st.markdown("### Detailed Brand Analysis")
        brand_analysis = preferences['brand_preferences'].copy()
        brand_analysis['mean'] = brand_analysis['mean'].round(2)
        brand_analysis['count'] = brand_analysis['count'].round(0)
        st.dataframe(
            brand_analysis,
            use_container_width=True
        )
    
    with tab_prefs2:
        st.plotly_chart(category_fig, use_container_width=True)
        
        # Add category preference table
        st.markdown("### Detailed Category Analysis")
        category_analysis = preferences['category_preferences'].copy()
        category_analysis['mean'] = category_analysis['mean'].round(2)
        category_analysis['count'] = category_analysis['count'].round(0)
        st.dataframe(
            category_analysis,
            use_container_width=True
        )
    
    # Add recommendation explanation
    st.markdown("---")
    st.subheader("Recommendation Insights")
    
    # Calculate recommendation relevance
    if not recommendations.empty:
        brand_match = recommendations['Brand'].isin(preferences['brand_preferences'].index[0:2]).mean() * 100
        category_match = recommendations['Category'].isin(preferences['category_preferences'].index[0:2]).mean() * 100
        price_match = (
            (recommendations['Price'] >= price_range['avg'] * 0.8) &
            (recommendations['Price'] <= price_range['avg'] * 1.2)
        ).mean() * 100
        
        # Display relevance metrics
        col_rel1, col_rel2, col_rel3 = st.columns(3)
        
        with col_rel1:
            st.metric(
                "Brand Relevance",
                f"{brand_match:.1f}%",
                help="Percentage of recommendations matching top brand preferences"
            )
        
        with col_rel2:
            st.metric(
                "Category Relevance",
                f"{category_match:.1f}%",
                help="Percentage of recommendations matching top category preferences"
            )
        
        with col_rel3:
            st.metric(
                "Price Range Match",
                f"{price_match:.1f}%",
                help="Percentage of recommendations within preferred price range"
            )

# Style profile analysis
if not user_purchases.empty:
    st.markdown("---")
    st.subheader("Style Profile Analysis")
    
    # Calculate style profile
    style_profile = analyze_style_profile(st.session_state.df, selected_user)
    
    # Create style visualizations
    color_fig, price_tier_fig = create_style_charts(style_profile)
    
    # Generate style insights
    style_insights = generate_style_insights(style_profile, preferences)
    
    # Display style profile summary
    st.markdown("### Style Insights")
    for insight in style_insights:
        st.markdown(f"- {insight}")
    
    # Display style visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(color_fig, use_container_width=True)
    
    with col2:
        st.plotly_chart(price_tier_fig, use_container_width=True)
    
    # Show style combinations
    if style_profile['style_combinations']:
        st.markdown("### Common Category Combinations")
        combinations_text = " • ".join(style_profile['style_combinations'])
        st.markdown(f"""
        <div style='background-color: #2E2E2E; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;'>
            {combinations_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Add personalized recommendations based on style profile
    st.markdown("### Personalized Style Recommendations")
    
    # Filter recommendations based on style profile
    if not recommendations.empty:
        top_colors = style_profile['color_preferences'].head(2)
        style_matched_recs = recommendations[
            (recommendations['Color'].isin(top_colors.index)) |
            (recommendations['Category'].isin(preferences['category_preferences'].head(2).index))
        ]
        
        if not style_matched_recs.empty:
            st.markdown("Based on your style profile, these items might particularly interest you:")
            
            for _, item in style_matched_recs.head(3).iterrows():
                st.markdown(f"""
                <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h4>{item['Product Name']} by {item['Brand']}</h4>
                    <p>Category: {item['Category']} • Color: {item['Color']} • Price: ${item['Price']}</p>
                    <p style='color: #666;'>Matches your preference for {item['Color'].lower()} items and {item['Category'].lower()}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Add style tips
        st.markdown("### Style Tips")
        st.markdown("""
        <div style='background-color: #2E2E2E; padding: 1rem; border-radius: 5px; margin-top: 1rem;'>
            <h4>💡 Personalized Tips</h4>
            <ul>
                <li>Try mixing your favorite categories for a cohesive look</li>
                <li>Use accessories to transition outfits between occasions</li>
                <li>Consider weather-appropriate layering options</li>
                <li>Experiment with color combinations within your preferred palette</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Seasonal analysis and outfit suggestions
if not user_purchases.empty:
    st.markdown("---")
    st.subheader("Seasonal Analysis & Outfit Suggestions")
    
    # Calculate seasonal trends
    seasonal_analysis = analyze_seasonal_trends(st.session_state.df, selected_user)
    
    # Create seasonal radar chart
    seasonal_fig = create_seasonal_chart(seasonal_analysis)
    
    # Generate outfit combinations
    outfits = generate_outfit_combinations(style_profile, preferences)
    
    # Display seasonal analysis and outfits in tabs
    tab1, tab2 = st.tabs(["Seasonal Analysis", "Outfit Suggestions"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(seasonal_fig, use_container_width=True)
        
        with col2:
            st.markdown("### Seasonal Style Insights")
            for season, score in seasonal_analysis['preferred_seasons']:
                st.markdown(f"""
                <div style='background-color: {('#2E2E2E' if score > 0.5 else '#1E1E1E')}; 
                            padding: 0.5rem; border-radius: 5px; margin: 0.5rem 0;'>
                    <strong>{season}:</strong> {score:.0%} style match
                </div>
                """, unsafe_allow_html=True)
        
        # Add seasonal recommendations
        st.markdown("### Seasonal Recommendations")
        top_season = seasonal_analysis['preferred_seasons'][0][0]
        st.markdown(f"""
        <div style='background-color: #2E2E2E; padding: 1rem; border-radius: 5px; margin-top: 0.5rem;'>
            <h4>🌟 {top_season} Style Tips</h4>
            <ul>
                <li>Your style aligns well with {top_season} fashion trends</li>
                <li>Consider adding {', '.join(seasonal_preferences[top_season]['colors'][:2])} pieces to your wardrobe</li>
                <li>Focus on {', '.join(seasonal_preferences[top_season]['categories'])} for the {top_season} season</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Personalized Outfit Combinations")
        st.markdown("Based on your style preferences and purchase history, here are some outfit suggestions:")
        
        # Display outfit combinations
        st.markdown(create_outfit_cards(outfits), unsafe_allow_html=True)
        
        # Add style tips
        st.markdown("### Mix & Match Tips")
        st.markdown("""
        <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 5px; margin-top: 1rem;'>
            <h4>👔 Style Guidelines</h4>
            <ul>
                <li>Mix casual and smart pieces for versatile looks</li>
                <li>Use accessories to transition outfits between occasions</li>
                <li>Consider weather-appropriate layering options</li>
                <li>Experiment with color combinations within your preferred palette</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Advanced outfit generator
if not user_purchases.empty:
    st.markdown("---")
    st.subheader("Advanced Outfit Generator")
    
    # Generate advanced outfits
    advanced_outfits = generate_advanced_outfits(style_profile, preferences)
    
    # Display advanced outfit combinations
    st.markdown(create_advanced_outfit_cards(advanced_outfits), unsafe_allow_html=True)
    
    # Add advanced style tips
    st.markdown("### Advanced Style Guidelines")
    st.markdown("""
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;'>
        <div style='background-color: #2E2E2E; padding: 1rem; border-radius: 5px;'>
            <h5>Color Theory</h5>
            <ul>
                <li>Use the 60-30-10 color rule</li>
                <li>Pair neutrals with statement pieces</li>
                <li>Consider color temperature</li>
            </ul>
        </div>
        <div style='background-color: #2E2E2E; padding: 1rem; border-radius: 5px;'>
            <h5>Proportion Rules</h5>
            <ul>
                <li>Balance loose with fitted items</li>
                <li>Consider your body proportions</li>
                <li>Use the rule of thirds</li>
            </ul>
        </div>
        <div style='background-color: #2E2E2E; padding: 1rem; border-radius: 5px;'>
            <h5>Texture Mixing</h5>
            <ul>
                <li>Combine different fabric weights</li>
                <li>Mix smooth with textured pieces</li>
                <li>Layer contrasting materials</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Style trend predictions
if not user_purchases.empty:
    st.markdown("---")
    st.subheader("Style Trend Predictions")
    
    # Generate trend predictions
    trends = predict_style_trends(st.session_state.df, selected_user, style_profile)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upcoming Style Trends")
        st.markdown(f"""
        <div style='background-color: #2E2E2E; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h4>🔮 {trends['next_season']} Trend Forecast</h4>
            <p>Based on your style preferences and emerging trends:</p>
            <ul>
                <li>Trending Colors: {', '.join(trends['upcoming_colors'])}</li>
                <li>Current Favorites: {', '.join(trends['current_trends']['colors'][:2])}</li>
                <li>Suggested Categories: {', '.join(trends['current_trends']['categories'][:2])}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Style Movement Match")
        for movement, attributes in trends['style_movements'].items():
            st.markdown(f"""
            <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 5px; margin-bottom: 0.5rem;'>
                <h5>{movement}</h5>
                <p style='font-size: 0.9em; color: #666;'>{' • '.join(attributes[:2])}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Add trend adaptation suggestions
    st.markdown("### How to Adapt Your Style")
    st.markdown("""
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;'>
        <div style='background-color: #2E2E2E; padding: 1rem; border-radius: 5px;'>
            <h5>Incorporate New Colors</h5>
            <ul>
                <li>Start with accessories</li>
                <li>Try color blocking</li>
                <li>Mix with your current palette</li>
            </ul>
        </div>
        <div style='background-color: #2E2E2E; padding: 1rem; border-radius: 5px;'>
            <h5>Update Your Basics</h5>
            <ul>
                <li>Refresh core pieces</li>
                <li>Choose versatile items</li>
                <li>Invest in quality</li>
            </ul>
        </div>
        <div style='background-color: #2E2E2E; padding: 1rem; border-radius: 5px;'>
            <h5>Experiment with Trends</h5>
            <ul>
                <li>Try one trend at a time</li>
                <li>Mix with classic pieces</li>
                <li>Focus on wearability</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Update metrics with trend indicators
if not user_purchases.empty:
    col_trend1, col_trend2, col_trend3 = st.columns(3)

    with col_trend1:
        st.metric(
            "Style Evolution",
            f"{len(trends['current_trends']['categories'])} Categories",
            "Expanding",
            help="Your style variety is growing"
        )

    with col_trend2:
        st.metric(
            "Trend Alignment",
            f"{trends['next_season']}",
            "Upcoming",
            help="How well your style aligns with upcoming trends"
        )

    with col_trend3:
        st.metric(
            "Style Movement",
            "Smart Casual",
            "Trending",
            help="Current dominant style movement in your preferences"
        )

# Advanced style analysis
if not user_purchases.empty:
    st.markdown("---")
    st.subheader("Advanced Style Analysis")
    
    # Create tabs for different analyses
    trend_tab1, trend_tab2, trend_tab3 = st.tabs(["Style Movements", "Color Harmony", "Accessory Guide"])
    
    with trend_tab1:
        st.markdown("### Style Movement Analysis")
        
        # Create style movement analysis
        movement_analysis = create_style_movement_analysis(preferences, style_profile)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create radar chart
            movement_scores = {name: details['score'] for name, details in movement_analysis.items()}
            radar_fig = create_style_movement_radar(movement_scores)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            # Show top style movements with expanded details
            st.markdown("### Your Style DNA")
            movements = get_advanced_style_movements()
            top_movements = sorted(movement_analysis.items(), key=lambda x: x[1]['score'], reverse=True)[:3]
            
            for movement, details in top_movements:
                movement_info = movements[movement]
                st.markdown(f"""
                <div style='background-color: #1E1E1E; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
                    <h4>{movement} ({details['score']:.0%})</h4>
                    <div style='margin-top: 0.5rem;'>
                        <h5>Signature Pieces</h5>
                        <p style='color: #666;'>{', '.join(movement_info['signature_pieces'])}</p>
                        <h5>Color Palette</h5>
                        <div style='display: flex; gap: 0.5rem; margin-top: 0.5rem;'>
                            {' '.join([f'<span style="display: inline-block; width: 25px; height: 25px; background-color: {get_color_hex(color)}; border-radius: 50%;"></span>'
                                     for color in movement_info['color_palette']])}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Add outfit-specific styling recommendations
        st.markdown("### Occasion-Specific Styling")
        
        # Get top style movement
        top_movement = top_movements[0][0]
        
        # Create tabs for different occasions
        occasion_tab1, occasion_tab2, occasion_tab3 = st.tabs(["Work", "Casual", "Evening"])
        
        occasions = ['Work', 'Casual', 'Evening']
        tabs = [occasion_tab1, occasion_tab2, occasion_tab3]
        
        for tab, occasion in zip(tabs, occasions):
            with tab:
                styling = get_outfit_specific_styling(top_movement, occasion)
                if styling:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div style='background-color: #1E1E1E; padding: 1.5rem; border-radius: 10px;'>
                            <h4>Key Pieces</h4>
                            <ul style='margin: 0; padding-left: 1.2rem;'>
                                {' '.join([f'<li>{piece}</li>' for piece in styling['Key Pieces']])}
                            </ul>
                            <h4 style='margin-top: 1rem;'>Accessories</h4>
                            <ul style='margin: 0; padding-left: 1.2rem;'>
                                {' '.join([f'<li>{acc}</li>' for acc in styling['Accessories']])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style='background-color: #1E1E1E; padding: 1.5rem; border-radius: 10px;'>
                            <h4>Colors</h4>
                            <div style='display: flex; gap: 0.5rem; margin-bottom: 1rem;'>
                                {' '.join([f'<span style="display: inline-block; width: 30px; height: 30px; background-color: {get_color_hex(color)}; border-radius: 50%;" title="{color}"></span>'
                                         for color in styling['Colors']])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    with trend_tab2:
        st.markdown("### Color Harmony Guide")
        
        # Get user's top colors
        top_colors = style_profile['color_preferences'].head(3)
        
        for color in top_colors.index:
            combinations = get_color_combinations(color)
            st.markdown(f"""
            <div style='background-color: #1E1E1E; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
                <h4>Combinations with {color}</h4>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;'>
                    <div>
                        <h5>Monochrome</h5>
                        <div style='display: flex; gap: 0.5rem;'>
                            {' '.join([f'<span style="display: inline-block; width: 30px; height: 30px; background-color: {get_color_hex(c)}; border-radius: 50%;"></span>'
                                     for c in combinations['Monochrome']])}
                        </div>
                        <p style='font-size: 0.9em; color: #666;'>{', '.join(combinations['Monochrome'])}</p>
                    </div>
                    <div>
                        <h5>Complementary</h5>
                        <div style='display: flex; gap: 0.5rem;'>
                            {' '.join([f'<span style="display: inline-block; width: 30px; height: 30px; background-color: {get_color_hex(c)}; border-radius: 50%;"></span>'
                                     for c in combinations['Complementary']])}
                        </div>
                        <p style='font-size: 0.9em; color: #666;'>{', '.join(combinations['Complementary'])}</p>
                    </div>
                    <div>
                        <h5>Accent Colors</h5>
                        <div style='display: flex; gap: 0.5rem;'>
                            {' '.join([f'<span style="display: inline-block; width: 30px; height: 30px; background-color: {get_color_hex(c)}; border-radius: 50%;"></span>'
                                     for c in combinations['Accent']])}
                        </div>
                        <p style='font-size: 0.9em; color: #666;'>{', '.join(combinations['Accent'])}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with trend_tab3:
        st.markdown("### Accessory Recommendations")
        
        # Get accessory suggestions for each outfit type
        outfit_types = ['Business', 'Smart Casual', 'Weekend', 'Evening']
        
        for outfit_type in outfit_types:
            accessories = suggest_accessories(style_profile, outfit_type)
            st.markdown(f"""
            <div style='background-color: #1E1E1E; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
                <h4>{outfit_type} Accessories</h4>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;'>
                    <div>
                        <h5>Must-Have</h5>
                        <ul style='margin: 0; padding-left: 1.2rem;'>
                            {' '.join([f'<li>{item}</li>' for item in accessories['primary']])}
                        </ul>
                    </div>
                    <div>
                        <h5>Optional Additions</h5>
                        <ul style='margin: 0; padding-left: 1.2rem;'>
                            {' '.join([f'<li>{item}</li>' for item in accessories['alternative']])}
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add accessory styling tips
        st.markdown("""
        <div style='background-color: #2E2E2E; padding: 1.5rem; border-radius: 10px; margin-top: 1rem;'>
            <h4>🎯 Accessory Styling Tips</h4>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;'>
                <div>
                    <h5>Layering</h5>
                    <ul>
                        <li>Mix different necklace lengths</li>
                        <li>Combine bracelet styles</li>
                        <li>Stack rings thoughtfully</li>
                    </ul>
                </div>
                <div>
                    <h5>Balance</h5>
                    <ul>
                        <li>One statement piece at a time</li>
                        <li>Match metals consistently</li>
                        <li>Consider outfit complexity</li>
                    </ul>
                </div>
                <div>
                    <h5>Occasion</h5>
                    <ul>
                        <li>Adapt to dress code</li>
                        <li>Consider activity level</li>
                        <li>Match formality level</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Update metrics with style movement indicators
if not user_purchases.empty:
    col_style1, col_style2, col_style3, col_style4 = st.columns(4)
    
    with col_style1:
        top_movement = max(movement_analysis.items(), key=lambda x: x[1]['score'])
        st.metric(
            "Primary Style",
            top_movement[0],
            f"{top_movement[1]['score']:.0%}",
            help="Your dominant style movement"
        )
    
    with col_style2:
        second_movement = sorted(movement_analysis.items(), key=lambda x: x[1]['score'], reverse=True)[1]
        st.metric(
            "Secondary Style",
            second_movement[0],
            f"{second_movement[1]['score']:.0%}",
            help="Your secondary style influence"
        )
    
    with col_style3:
        style_versatility = len([m for m in movement_analysis.values() if m['score'] > 0.3])
        st.metric(
            "Style Versatility",
            f"{style_versatility}/10",
            "Diverse" if style_versatility > 5 else "Focused",
            help="Number of significant style influences"
        )
    
    with col_style4:
        style_consistency = max(m['score'] for m in movement_analysis.values())
        st.metric(
            "Style Consistency",
            f"{style_consistency:.0%}",
            "Strong" if style_consistency > 0.7 else "Evolving",
            help="Strength of your primary style preference"
        )

# Add occasion-based recommendations feature
if not user_purchases.empty:
    st.markdown("---")
    st.subheader("Occasion-Based Style Guide")
    
    # Add occasion selector
    occasion = st.selectbox(
        "Select the occasion",
        ["Work", "Weekend", "Special Event", "Vacation"]
    )
    
    occasion_recs = get_occasion_recommendations(style_profile, occasion)
    st.markdown(f"""
    <div style='background-color: #1E1E1E; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;'>
        <h4>✨ {occasion} Style Guide</h4>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;'>
            <div>
                <h5>Essential Items</h5>
                <ul>
                    {' '.join([f'<li>{item}</li>' for item in occasion_recs['essential']])}
                </ul>
            </div>
            <div>
                <h5>Recommended Colors</h5>
                <div style='display: flex; gap: 0.5rem; margin: 0.5rem 0;'>
                    {' '.join([f'<span style="display: inline-block; width: 30px; height: 30px; background-color: {get_color_hex(color)}; border-radius: 50%;" title="{color}"></span>'
                             for color in occasion_recs['colors']])}
                </div>
            </div>
            <div>
                <h5>Must-Have Accessories</h5>
                <ul>
                    {' '.join([f'<li>{item}</li>' for item in occasion_recs['accessories']])}
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add outfit completion suggestions
if not user_purchases.empty:
    st.markdown("---")
    st.subheader("Complete Your Wardrobe")
    
    completion_suggestions = get_outfit_completion_suggestions(user_purchases, style_profile)
    
    for style, missing_items in completion_suggestions.items():
        st.markdown(f"""
        <div style='background-color: #1E1E1E; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h4>🎯 {style} Wardrobe Essentials</h4>
            <p>Complete your {style.lower()} look with these items:</p>
            <ul>
                {' '.join([f'<li>{item}</li>' for item in missing_items])}
            </ul>
            <p style='font-size: 0.9em; color: #666;'>These suggestions are based on your current wardrobe and style preferences.</p>
        </div>
        """, unsafe_allow_html=True)

# Add footer with stats
st.markdown("---")
if not user_purchases.empty:
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

    with col_stats1:
        st.metric(
            "Total Products",
            len(st.session_state.df['Product ID'].unique()),
            help="Total number of unique products in the system"
        )

    with col_stats2:
        st.metric(
            "Active Users",
            len(st.session_state.df['User ID'].unique()),
            help="Total number of users in the system"
        )

    with col_stats3:
        st.metric(
            "Average Rating",
            f"{st.session_state.df['Rating'].mean():.2f}",
            help="Global average rating across all products"
        )

    with col_stats4:
        st.metric(
            "Total Interactions",
            len(st.session_state.df),
            help="Total number of user-product interactions"
        )

    col_style1, col_style2, col_style3 = st.columns(3)
    
    with col_style1:
        st.metric(
            "Style Variety",
            f"{len(style_profile['style_combinations'])} combinations",
            help="Number of different category combinations in purchases"
        )
    
    with col_style2:
        st.metric(
            "Color Palette",
            f"{len(style_profile['color_preferences'])} colors",
            help="Number of different colors in wardrobe"
        )
    
    with col_style3:
        st.metric(
            "Price Range Spread",
            f"{len(style_profile['price_tiers'])} tiers",
            help="Number of different price tiers in purchases"
        )
