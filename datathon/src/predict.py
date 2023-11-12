import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

def open_image(filename):
    # Check if the image file exists
    if os.path.isfile(filename):
        # Load the image and display it
        image = plt.imread(filename)
        plt.imshow(image)
        plt.show()
    else:
        print(f"Image file {filename} does not exist.")


if __name__ == '__main__':

    df = pd.read_csv('datathon/dataset/outfit_data_clean.csv')

    df['des_color_specification_esp'] = pd.Categorical(df['des_color_specification_esp']).codes
    df['des_agrup_color_eng'] = pd.Categorical(df['des_agrup_color_eng']).codes
    df['des_sex'] = pd.Categorical(df['des_sex']).codes
    df['des_age'] = pd.Categorical(df['des_age']).codes
    df['des_fabric'] = pd.Categorical(df['des_fabric']).codes
    df['des_product_category'] = pd.Categorical(df['des_product_category']).codes
    df['des_product_type'] = pd.Categorical(df['des_product_type']).codes


    test_product = "57096007-99"
    all_outfits_present = df.loc[df.cod_modelo_color == test_product, 'cod_outfit']

    test_product_outfit = df.loc[df.cod_modelo_color == test_product].iloc[0].cod_outfit

    test_product_category = df.loc[df.cod_modelo_color == test_product].iloc[0].des_product_category

    test_df = df[df.cod_outfit == test_product_outfit]
    # for outfit in test_df.itertuples():

    #     # Get the filename of the image
    #     product_id = outfit.cod_modelo_color
    #     product = df.loc[df['cod_modelo_color'] == product_id]    

    #     image_filename = product['des_filename'].values[0]
    #     open_image(image_filename)


    grouped_test = test_df.groupby('cod_outfit')
    for c, p in grouped_test:
        x_test = {'cod_outfit': c, 
                  'products': np.array(p[['des_color_specification_esp', 'des_agrup_color_eng', 'des_sex', 'des_age', 'des_fabric', 'des_product_category', 'des_product_type']].values)}

    outfits = []

    grouped = df.groupby('cod_outfit')
    max_score = 0
    max_outfit = 0

    for cod_outfit, products in grouped:
        new_products = np.array(products[['des_color_specification_esp', 'des_agrup_color_eng', 'des_sex', 'des_age', 'des_fabric', 'des_product_category', 'des_product_type']].values)

        len_x = len(x_test['products'])
        len_p = len(new_products)

        if len_x > len_p:
            new_products = np.concatenate((new_products, np.zeros((len_x-len_p, 7))), axis=0)
        elif len_p > len_x:
            #change later
            new_products = new_products[:7]


        sim = cosine_similarity(x_test['products'], new_products)
        sim_score = sim.mean()

        if(sim_score > max_score):
            max_score = sim_score
            max_outfit = cod_outfit

        outfit_data = {
            'cod_outfit': cod_outfit,
            'products':  new_products,
            'sim_score': sim_score,
        }   
        
        
        outfits.append(outfit_data)
    
    outfits_df = df[df.cod_outfit == max_outfit]
    for outfit in outfits_df.itertuples():

        # Get the filename of the image
        product_id = outfit.cod_modelo_color
        product = df.loc[df['cod_modelo_color'] == product_id]    

        if product['des_product_category'].values[0] == test_product_category:
            continue

        image_filename = product['des_filename'].values[0]
        open_image(image_filename)
