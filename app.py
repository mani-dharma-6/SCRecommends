#!/usr/bin/env python
# coding: utf-8

# Importing the libraries
import pandas as pd
import numpy as np

# read the data into a dataframe
df = pd.read_csv('sc_products_2.csv')

# creating a copy of df to avoid repeat reading from the csv
df1 = df.copy()
df1.head()

# Display the data types of the DataFrame after conversion
df1.info()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Select relevant features for content-based filtering
features = ['desc', 'name','url']

# Combine the selected features into a single string for each item
df1['combined_features'] = df1.apply(lambda row: ' '.join(row[features].astype(str)), axis=1)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the TF-IDF vectorizer on the combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(df1['combined_features'])

# Compute similarity scores using cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print("the cosine similarity score is")
print(cosine_sim)

def get_recommendations(product_id, cosine_sim=cosine_sim, data=df1):
    # Convert the product_id to lowercase
    product_id = product_id.lower()

    # Get the lowercase version of the product names
    lowercase_product = data['name'].str.lower()

    # Get the indices of the products that contain the product_id
    indices = lowercase_product[lowercase_product.str.contains(product_id)].index.tolist()

    # If no match is found, return an empty list
    if not indices:
        return "The product is not in the cart, could you please retry"

    # Get the pairwise similarity scores of all products with those products
    sim_scores = [(i, cosine_sim[idx][i]) for idx in indices for i in range(len(cosine_sim[idx]))]

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Initialize an empty DataFrame to store the recommended products
    recommend_products = pd.DataFrame(columns=['Name', 'Url'])

    # Iterate over the product indices
    for idx in product_indices:
        # Get the product name and URL
        product = data['name'].iloc[idx]
        product_url = data['url'].iloc[idx]

        # If the product is not already in the recommended products list, add it
        if product not in recommend_products['Name'].values:
            new_row = pd.DataFrame({'Name': [product], 'Url': [product_url]})
            recommend_products = pd.concat([recommend_products, new_row], ignore_index=True)

        # If we have found 10 unique products, stop looking
        if len(recommend_products) == 10:
            break

    return recommend_products

import streamlit as st
st.title("SCRecommends")
st.subheader("Banking AI Recommendation System")
with st.sidebar:    
    st.image('sc-lock-up-english-grey-rgb.png', width=200) 
    Final_product=st.text_input('Enter your query for a recommendation')
    recommended_products = get_recommendations(Final_product)

    

# Assuming Final_product and recommended_products are defined earlier in your code
if Final_product != '':
    st.write("Your recommended products for your search:", Final_product)

    # Display the dataframe with custom column configuration
    st.dataframe(recommended_products, 
                 column_config={
                    "Name": "Name",
                    "Url": st.column_config.LinkColumn("Url")
                },
                hide_index=True)
