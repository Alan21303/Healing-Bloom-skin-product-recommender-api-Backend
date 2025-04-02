import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

current_dir = os.path.dirname(__file__)
df2 = pd.read_csv(os.path.join(current_dir, 'final.csv'))
makeup = pd.read_csv(os.path.join(current_dir, 'makeup_final.csv'))
entries = len(df2)
LABELS = list(df2.label.unique())

# Features
features = [
    'normal', 'dry', 'oily', 'combination', 'acne', 'sensitive',
    'fine lines', 'wrinkles', 'redness', 'dull', 'pore',
    'pigmentation', 'blackheads', 'whiteheads', 'blemishes',
    'dark circles', 'eye bags', 'dark spots'
]

# Utility functions
def search_concern(target, i):
    return target in df2.iloc[i]['concern']

def name2index(name):
    return df2[df2["name"] == name].index.tolist()[0]

def index2prod(index):
    return df2.iloc[index]

def wrap(info_arr):
    return {
        'brand': info_arr[0],
        'name': info_arr[1],
        'price': info_arr[2],
        'url': info_arr[3],
        'img': info_arr[4],
        'skin type': info_arr[5],
        'concern': str(info_arr[6]).split(',')
    }

def wrap_makeup(info_arr):
    return {
        'brand': info_arr[0],
        'name': info_arr[1],
        'price': info_arr[2],
        'url': info_arr[3],
        'img': info_arr[4],
        'skin type': info_arr[5],
        'skin tone': info_arr[6]
    }

# Initialize one-hot encodings
one_hot_encodings = np.zeros([entries, len(features)])

# Skin type encoding
for i in range(entries):
    sk_type = df2.iloc[i]['skin type']
    if sk_type == 'all':
        one_hot_encodings[i][0:5] = 1
    else:
        for j in range(5):
            if features[j] == sk_type:
                one_hot_encodings[i][j] = 1

# Concern encoding
for i in range(entries):
    for j in range(5, len(features)):
        if features[j] in df2.iloc[i]['concern']:
            one_hot_encodings[i][j] = 1

def recs_cs(vector=None, name=None, label=None, count=5):
    """Get top recommendations using cosine similarity"""
    products = []
    
    if name:
        print(f"\n[REC CS] Getting recommendations similar to: {name}")
        idx = name2index(name)
        fv = one_hot_encodings[idx]
    elif vector:
        print(f"\n[REC CS] Using feature vector: {vector}")
        fv = vector
    
    # Calculate cosine similarity
    cs_values = cosine_similarity(np.array([fv, ]), one_hot_encodings)
    df2['cs'] = cs_values[0]
    
    # Filter by label if specified
    dff = df2[df2['label'] == label] if label else df2
    if name: 
        dff = dff[dff['name'] != name]
    
    # Get top recommendations
    recommendations = dff.sort_values('cs', ascending=False).head(count)
    print(f"[REC CS] Top {count} recommendations for {label or 'all'}:")
    print(recommendations[['name', 'brand', 'cs']].to_string(index=False))
    
    # Format results
    data = recommendations[['brand', 'name', 'price', 'url', 'img', 'skin type', 'concern']].to_dict('split')['data']
    return [wrap(element) for element in data]

def recs_essentials(vector=None, name=None):
    """Generate essential product recommendations"""
    print(f"\n[REC ENGINE] {'Starting essentials recommendation for vector' if vector else 'Finding similar products to'} {vector or name}")
    
    response = {}
    for label in LABELS:
        print(f"\nProcessing category: {label.upper()}")
        r = recs_cs(vector, name, label)
        print(f"Found {len(r)} products in {label}")
        response[label] = r
    return response

def makeup_recommendation(skin_tone, skin_type):
    """Generate makeup recommendations"""
    print(f"\n[MAKEUP ENGINE] Starting recommendations for {skin_tone} skin tone and {skin_type} skin type")
    
    result = []
    dff = pd.DataFrame()
    
    # Foundation recommendations
    print("\nSearching for foundation...")
    foundation = makeup[(makeup['skin tone'] == skin_tone) & 
                       (makeup['skin type'] == skin_type) & 
                       (makeup['label'] == 'foundation')].head(2)
    print(f"Found {len(foundation)} foundations")
    
    # Concealer recommendations
    print("\nSearching for concealer...")
    concealer = makeup[(makeup['skin tone'] == skin_tone) & 
                      (makeup['skin type'] == skin_type) & 
                      (makeup['label'] == 'concealer')].head(2)
    print(f"Found {len(concealer)} concealers")
    
    # Primer recommendations
    print("\nSearching for primer...")
    primer = makeup[(makeup['skin tone'] == skin_tone) & 
                   (makeup['skin type'] == skin_type) & 
                   (makeup['label'] == 'primer')].head(2)
    print(f"Found {len(primer)} primers")
    
    # Combine results
    dff = pd.concat([foundation, concealer, primer])
    print(f"\nTotal products before shuffle: {len(dff)}")
    
    # Shuffle and process results
    dff = dff.sample(frac=1)
    print(f"Final products after shuffle: {len(dff)}")
    
    data = dff[['brand', 'name', 'price', 'url', 'img', 'skin type', 'skin tone']].to_dict('split')['data']
    
    # Format results
    for idx, element in enumerate(data, 1):
        print(f"\nProduct #{idx}:")
        print(f"Brand: {element[0]}")
        print(f"Name: {element[1]}")
        print(f"Price: {element[2]}")
        result.append(wrap_makeup(element))
    
    return result