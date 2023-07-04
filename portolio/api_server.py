from flask import Flask, render_template, request, jsonify
import csv
import ml_library
from ml_library import load_model
import requests
import os
import io
from PIL import Image
import numpy as np
from bs4 import BeautifulSoup

app = Flask(__name__)
model = ml_library.load_model()

# Set up Google Custom Search API credentials
API_KEY = #put the API key you got from google
SEARCH_ENGINE_ID = # put relevant search engine ID

@app.route('/')
def my_home():
    return render_template('index.html')

@app.route('/<string:page_name>')
def html_page(page_name):
    return render_template(page_name)


def write_to_file(data):
    with open('database.txt', mode='a') as dbase:
        email = data["email"]
        subject = data["subject"]
        message = data["message"]
        file = dbase.write(f'\n{email},{subject},{message}')

def write_to_csv(data):
    with open('database.csv', mode='a') as dbase2:
        email = data["email"]
        subject = data["subject"]
        message = data["message"]
        csv_written = csv.writer(dbase2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_written.writerow([email, subject, message])

#def extract_image_from_page(url):
 #   response = requests.get(url)
  #  if response.status_code == 200:
   #     soup = BeautifulSoup(response.content, 'html.parser')
    #    image_elements = soup.find_all('img')  # Find all image elements on the page
     #   if len(image_elements) >= 1:  # Ensure there is at least one image
      #      image_url = image_elements[0]['src']  # Get the first image URL
       #     return image_url
    #return None


def get_similar_recommendations(query_list):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID
    }
    recommendations = []
    visited_image_urls = set()

    for query in query_list:
        params["q"] = query
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            for item in items:
                image_title = item.get('title')
                description = item.get('snippet')
                page_url = item.get('link')
                if image_title and description and page_url:
                    #image_url = extract_image_from_page(page_url)  # Extract image from the page URL
                    if page_url and page_url not in(visited_image_urls):
                        recommendation = {
                            'image_title': image_title,
                            'description': description,
                            'page_url': page_url
                        }
                        recommendations.append(recommendation)
                        visited_image_urls.add(page_url)
                        break  # Break after finding one recommendation per query

    return recommendations



@app.route ('/submit_gender_form', methods=['POST'])
def submit_gender_form ():
    global gender
    gender= request.form['selectedGender']  
    return render_template('male.html')

@app.route('/submit_image_form', methods=['POST'])
def submit_image_form():
    try:
        # Retrieve the uploaded image from the request
        image = request.files['image']
        # Save the image to a directory
        image_path = os.path.join('/filepath', image.filename) #Change this to filepath to save the files
        image.save(image_path)
        new_image = ml_library.preprocessing(image_path)
        prediction = model.predict(new_image)
        predicted_class = np.argmax(prediction)
        predicted_class = str(predicted_class)
        keywords= ml_library.generate_keywords(predicted_class)
        query = ml_library.search_query(keywords,predicted_class,gender)
        predicted_label = ml_library.predict_label(predicted_class)
        # Perform a search using the labels
        recommendations = get_similar_recommendations(query)
        return render_template('results.html', recommendations=recommendations,predicted_label=predicted_label)
    except:
        return 'Error processing the image'
    
@app.route('/submit_captured_image_form', methods=['POST'])
def submit_captured_image_form():
    try:
        # Retrieve the image data from the request
        image = request.files['image']
        image_filename = 'captured_image.png'
        image_path = os.path.join('filepath', image.filename) #change this to filepath to save the files
        image.save(image_path)
        # Perform image processing and prediction
        new_image = ml_library.preprocessing(image_path)
        prediction = model.predict(new_image)
        predicted_class = np.argmax(prediction)
        predicted_class = str(predicted_class)
        keywords = ml_library.generate_keywords(predicted_class)
        query = ml_library.search_query(keywords, predicted_class, gender)
        predicted_label = ml_library.predict_label(predicted_class)
        # Perform a search using the labels
        recommendations = get_similar_recommendations(query)
        # Render the results template
        return render_template('results.html', recommendations=recommendations,predicted_label=predicted_label)
    except Exception as e:
        return ('Error processing image')



@app.route('/results')
def display_results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run()
