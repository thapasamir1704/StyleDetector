import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import os
import numpy as np
import random


def load_model():
    # Define the optimizer with the desired learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    
    # Load the ResNet50 model
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Set 160 layers to be non-trainable
    for layer in conv_base.layers[:160]:
        layer.trainable = False

    # Create the final architecture 
    finalModelCustom = tf.keras.models.Sequential()
    finalModelCustom.add(conv_base)
    finalModelCustom.add(tf.keras.layers.Flatten())
    finalModelCustom.add(tf.keras.layers.Dense(256, activation='relu'))
    finalModelCustom.add(tf.keras.layers.BatchNormalization())
    finalModelCustom.add(tf.keras.layers.Dropout(0.6))
    finalModelCustom.add(tf.keras.layers.BatchNormalization())
    finalModelCustom.add(tf.keras.layers.Dropout(0.6))
    finalModelCustom.add(tf.keras.layers.BatchNormalization())
    finalModelCustom.add(tf.keras.layers.Dropout(0.6))
    finalModelCustom.add(tf.keras.layers.BatchNormalization())
    finalModelCustom.add(tf.keras.layers.Dropout(0.6))
    finalModelCustom.add(tf.keras.layers.BatchNormalization())
    finalModelCustom.add(tf.keras.layers.Dropout(0.6))
    finalModelCustom.add(tf.keras.layers.Dense(4, activation='softmax'))
    
    # Load the pre-trained weights for the model
    finalModelCustom.load_weights('filepath'') #Change this to the filepath for the model
    
    # Compile the model with the desired optimizer
    finalModelCustom.compile(optimizer=optimizer, loss='categorical_crossentropy')
    
    return finalModelCustom


def preprocessing(image_path):
    # Open the image using PIL and convert it to RGB format
    img = Image.open(image_path).convert('RGB') 

    # Resize the image to the desired input shape of the model
    img = img.resize((224, 224))

    # Convert the image to a NumPy array
    img = np.expand_dims(img, axis=0)

    # Preprocess the image using the ResNet50 preprocess_input function
    img = tf.keras.applications.resnet50.preprocess_input(img)  

    return img


# Define lists of keywords for different clothing styles and brands
streetwearTops = ['T-shirts', 'Tees', 'Long Sleeve T-shirts', 'Short Sleeve T-shirts']
streetwearJackets = ['Denim Jacket', 'Puffer Jackets', 'Baggy Jackets', 'Hoodies']
streetwearBottoms = ['Denim pants', 'Ripped Denim Pants', 'Leather jeans', 'Jeans', 'Baggy Jeans', 'Baggy Pants']
streetwearBrands = ['Nike', 'New Balance', 'Stussy', 'Kith', 'Heron Preston', 'Fear of God', 'Ksubi', 'Jordan', 'Converse', 'Supreme', 'Off White', 'Palace', 'Bape', 'Culture Kings', 'Vetements', 'Golf Wang', 'Human Made']

casualTops = ['Shirts', 'Polo Shirts', 'Crewneck shirts', 'V neck shirts', 'Long sleeve shirts']
casualJackets = ['Comfortable Jackets', 'Puffer Jackets']
casualBottoms = ['Comfortable Pants', 'Trousers', 'Cargo pants', 'Slim fit pants', 'smart pants']
casualBrands = ['Uniqlo', 'Stone Island', 'Fred Perry', 'Hackett', 'Ralph Lauren', 'Ellese', 'H&M', 'Dickies', 'Patagonia', 'The North Face', 'Hugo Boss', 'Tommy Hilfiger', 'Guess', 'David Jones', 'Myer', 'The Iconic']

formalTops = ['Business casual Shirts', 'Long sleeve shirts', 'Business shirts']
formalSuits = ['Business suits', 'suits', 'suit jackets']
formalBottoms = ['Business casual pants', 'Business pants']
formalBrands = ['David Jones', 'The Iconic', 'Myer', 'Oxford', 'Tarocash', 'YD', 'Connor', 'Jill Sander']

athleticTops = ['Tank tops', 'Singlets', 'Activewear Tops', 'Hoodies']
athleticJackets = ['Activewear Jackets', 'Performance Jackets', 'Hoodies']
athleticBottoms = ['Activewear bottoms', 'Tracksuit pants', 'Sweatpants', 'Gym pants', 'Fitnesswear pants', 'Fitnesswear shorts', 'Gym shorts', 'Swimming shorts']
athleticBrands = ['Nike', 'Adidas', 'Puma', 'Fila', 'Reebok', 'Under Armour', 'New Balance', 'Asics', 'Athletes foot', 'Rebel sport', 'JD sports', 'Sketchers']


def generate_keywords(classLabel):
    # Initialize empty strings for keywords
    topsKeyword = ''
    bottomsKeyword = ''
    brandsKeyword = ''
    jacketsKeyword = ''
    keywordList = []

    # Choose random keywords based on the class label
    if classLabel == '0':
        topsKeyword = random.choice(athleticTops)
        bottomsKeyword = random.choice(athleticBottoms)
        brandsKeyword = random.choice(athleticBrands)
        jacketsKeyword = random.choice(athleticJackets)
    elif classLabel == '3':
        topsKeyword = random.choice(streetwearTops)
        jacketsKeyword = random.choice(streetwearJackets)
        bottomsKeyword = random.choice(streetwearBottoms)
        brandsKeyword = random.choice(streetwearBrands)
    elif classLabel == '1':
        topsKeyword = random.choice(casualTops)
        jacketsKeyword = random.choice(casualJackets)
        bottomsKeyword = random.choice(casualBottoms)
        brandsKeyword = random.choice(casualBrands)
    elif classLabel == '2':
        topsKeyword = random.choice(formalTops)
        jacketsKeyword = random.choice(formalSuits)
        bottomsKeyword = random.choice(formalBottoms)
        brandsKeyword = random.choice(formalBrands)

    # Append the keywords to a list
    keywordList.append(topsKeyword)
    keywordList.append(bottomsKeyword)
    keywordList.append(brandsKeyword)
    keywordList.append(jacketsKeyword)

    # Remove empty keywords from the list
    for keyword in keywordList:
        if keyword == '':
            keywordList.remove(keyword)

    return keywordList


def search_query(keywords, classLabel, gender):
    # Initialize empty strings for queries
    queryTops = ''
    queryBottoms = ''
    queryJackets = ''
    queryList = []
    
    # Generate search queries based on the class label, gender, and keywords
    if classLabel == '0':
        queryTops = 'Buy Athletic ' + gender + ' ' + keywords[2] + ' ' + keywords[0]
        queryBottoms = 'Buy Athletic ' + gender + ' ' + keywords[2] + ' ' + keywords[1]
        queryJackets = 'Buy Athletic ' + gender + ' ' + keywords[2] + ' ' + keywords[3]
    elif classLabel == '1':
        queryTops = 'Buy Casual ' + gender + ' ' + keywords[2] + ' ' + keywords[0]
        queryBottoms = 'Buy Casual ' + gender + ' ' + keywords[2] + ' ' + keywords[1]
        queryJackets = 'Buy Casual ' + gender + ' ' + keywords[2] + ' ' + keywords[3]
    elif classLabel == '2':
        queryTops = 'Buy Formal ' + gender + ' ' + keywords[2] + ' ' + keywords[0]
        queryBottoms = 'Buy Formal ' + gender + ' ' + keywords[2] + ' ' + keywords[1]
        queryJackets = 'Buy Formal ' + gender + ' ' + keywords[2] + ' ' + keywords[3]
    elif classLabel == '3':
        queryTops = 'Buy Streetwear ' + gender + ' ' + keywords[2] + ' ' + keywords[0]
        queryBottoms = 'Buy Streetwear ' + gender + ' ' + keywords[2] + ' ' + keywords[1]
        queryJackets = 'Buy Streetwear ' + gender + ' ' + keywords[2] + ' ' + keywords[3]

    # Append the queries to a list
    queryList.append(queryTops)
    queryList.append(queryBottoms)
    queryList.append(queryJackets)

    # Remove empty queries from the list
    for query in queryList:
        if query == '':
            queryList.remove(query)
  
    return queryList


def predict_label(classLabel):
    # Assign labels based on the class label
    label = ''
    if classLabel == '0':
        label = 'Athletic'
    elif classLabel == '1':
        label = 'Casual'
    elif classLabel == '2':
        label = 'Formal'
    elif classLabel == '3':
        label = 'Streetwear'
    return label
