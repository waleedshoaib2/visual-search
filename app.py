import glob
import os
import h5py
import faiss
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing import image

# Constants (Update paths if necessary)
IMAGE_FOLDER = ''
FEATURES_FILE = 'features.hdf5' 
TARGET_SIZE = (224, 224)

# Load MobileNet Model

model = MobileNet(weights='imagenet', include_top=False)

fais_index = faiss.IndexFlatL2(50176) 
with h5py.File(FEATURES_FILE, 'r') as f:
    image_ids = f['image_ids'][:]
    features = f['features'][:]

    print(features.shape)  # Check the shape

    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], 50176) 

    fais_index.add(features)

#  ... (Rest of your code)

# Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads' 

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)  

        file = request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return redirect(url_for('search', filename=file.filename))

    return render_template('index.html')

@app.route('/search/<filename>')
def search(filename):
    user_img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)


    # Feature Extraction
    user_img = image.load_img(user_img_path, target_size=TARGET_SIZE)
    user_img_array = image.img_to_array(user_img)
    user_img_array = np.expand_dims(user_img_array, axis=0)
    user_img_array = preprocess_input(user_img_array)
    user_features = model.predict(user_img_array)
    user_features = np.reshape(user_features, (1, 7*7*1024))

    # Similarity Search
  
    distances, indices = fais_index.search(user_features, k=5)  # Correct way
    print(distances)




    # Get Similar Image Paths
    similar_image_ids = [image_ids[id] for id in indices[0]] 
    similar_image_paths = [os.path.join(IMAGE_FOLDER, f"{image_id.decode('utf-8')}.png") 
                           for image_id in similar_image_ids]

    return render_template('results.html', query_image=filename, results=similar_image_paths)

if __name__ == '__main__':
    app.run(debug=True) 
