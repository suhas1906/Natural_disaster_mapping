import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import re  # Import the regular expression module

app = Flask(__name__)
app.secret_key = "secret_key"

UPLOAD_FOLDER = r"C:\Users\suhas\OneDrive\Desktop\Natural Disaster Managemant\uploads"
TESTING_DATA_PATH = r"C:\Users\suhas\OneDrive\Desktop\Natural Disaster Managemant\testing_data"

# Function to get the label file path
def fetch_label_path(filename, base_path):
    base_name = filename.replace('_post_0.png', '').replace('_post_1.png', '').replace('_post_2.png', '') \
                            .replace('_pre_0.png', '').replace('_pre_1.png', '').replace('_pre_2.png', '')
    label_filename = f"{base_name}.json"
    label_file_path = os.path.join(base_path, 'labels', label_filename)
    return label_file_path

# Function to process label data
def process_label_data(filepath):
    try:
        with open(filepath, 'r') as f:
            label_info = json.load(f)
        return label_info.get('features', {}).get('lng_lat', [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading label file: {filepath} - {e}")
        return None

# Function to derive condition from filename
def derive_condition(name):
    indicators = {
        "No_Damage": "No Damage",
        "Minor_Damage": "Minor Damage",
        "Moderate_Damage": "Moderate Damage",
        "Severe_Damage": "Severe Damage"
    }
    for key, value in indicators.items():
        if name.startswith(key):
            return value
    return None

@app.route('/')
def index():
    return render_template('index.html')

# ... (your existing imports and setup) ...

@app.route('/upload/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'pre_disaster' not in request.files or 'post_disaster' not in request.files:
            flash('Please upload both pre-disaster and post-disaster images.', 'error')
            return redirect(request.url)

        pre_disaster_file = request.files['pre_disaster']
        post_disaster_file = request.files['post_disaster']

        if pre_disaster_file.filename == '' or post_disaster_file.filename == '':
            flash('Please select files to upload.', 'error')
            return redirect(request.url)

        pre_filename = secure_filename(pre_disaster_file.filename)
        post_filename = secure_filename(post_disaster_file.filename)

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        pre_disaster_path = os.path.join(UPLOAD_FOLDER, pre_filename)
        post_disaster_path = os.path.join(UPLOAD_FOLDER, post_filename)
        pre_disaster_file.save(pre_disaster_path)
        post_disaster_file.save(post_disaster_path)

        session['pre_filename'] = pre_filename
        session['post_filename'] = post_filename
        session['pre_disaster_image_path'] = pre_disaster_path
        session['post_disaster_image_path'] = post_disaster_path

        pre_condition = derive_condition(pre_filename)
        post_condition = derive_condition(post_filename)

        result = None
        if pre_condition and post_condition:
            if pre_condition == post_condition:
                result = pre_condition
            else:
                result = f"Pre-Disaster: {pre_condition}, Post-Disaster: {post_condition}"
            session['result'] = result

        # Pass the static URLs to the template
        pre_image_url = url_for('static', filename=f'../uploads/{pre_filename}')
        post_image_url = url_for('static', filename=f'../uploads/{post_filename}')

        return render_template('upload.html',
                               pre_filename=pre_filename,
                               post_filename=post_filename,
                               result=result,
                               pre_image_url=pre_image_url,
                               post_image_url=post_image_url)
    return render_template('upload.html')

# ... (your other routes) ...

@app.route('/generate_heatmap')
def generate_heatmap():
    pre_disaster_path = session.get('pre_disaster_image_path')
    post_disaster_path = session.get('post_disaster_image_path')

    if not pre_disaster_path or not post_disaster_path:
        return jsonify({'heatmap_url': None, 'error': 'Please upload both pre and post-disaster images first.'})

    try:
        img_pre = Image.open(pre_disaster_path).convert('RGB')
        img_post = Image.open(post_disaster_path).convert('RGB')

        if img_pre.size != img_post.size:
            img_post = img_post.resize(img_pre.size)

        np_pre = np.array(img_pre, dtype=np.int32)
        np_post = np.array(img_post, dtype=np.int32)

        diff = np.abs(np_post - np_pre).mean(axis=2)
        normalized_diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff) + 1e-8)
        cmap = plt.cm.viridis
        heatmap_image = (cmap(normalized_diff) * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_image)

        img_stream = BytesIO()
        heatmap_pil.save(img_stream, format='png')
        img_stream.seek(0)
        img_base64 = base64.b64encode(img_stream.read()).decode('utf-8')
        heatmap_url = f'data:image/png;base64,{img_base64}'
        plt.close()

        session['heatmap_url'] = heatmap_url  # Save to session
        return jsonify({'heatmap_url': heatmap_url})

    except FileNotFoundError:
        return jsonify({'heatmap_url': None, 'error': 'Could not open one or both of the uploaded images.'})
    except Exception as e:
        return jsonify({'heatmap_url': None, 'error': str(e)})

@app.route('/heatmap/')
def heatmap():
    return render_template('heatmap.html')

@app.route('/overview/')
def overview():
    return render_template('overview.html')

@app.route('/reports/')
def reports():
    uploaded_files = os.listdir(UPLOAD_FOLDER)
    reports_data = []
    image_pairs = {}
    heatmap_files = {}

    # Regular expression to extract base name from filenames
    base_name_pattern = re.compile(r'^(.*?)(_pre_\d+\.png|_post_\d+\.png|_heatmap\.png)$')

    for filename in uploaded_files:
        match = base_name_pattern.match(filename)
        if match:
            base_name = match.group(1)
            if "_pre_" in filename:
                image_pairs.setdefault(base_name, {})['pre'] = filename
            elif "_post_" in filename:
                image_pairs.setdefault(base_name, {})['post'] = filename
            elif "_heatmap" in filename:
                heatmap_files[base_name] = filename

    for base, images in image_pairs.items():
        report_entry = {'pre_image': None, 'post_image': None, 'heatmap_url': None, 'result': None}
        if 'pre' in images:
            report_entry['pre_image'] = images['pre']
            pre_condition = derive_condition(images['pre'])
            if pre_condition:
                report_entry['result'] = pre_condition
        if 'post' in images:
            report_entry['post_image'] = images['post']
            if not report_entry['result']:
                post_condition = derive_condition(images['post'])
                if post_condition:
                    report_entry['result'] = post_condition
        if base in heatmap_files:
            report_entry['heatmap_url'] = heatmap_files[base]
        reports_data.append(report_entry)

    return render_template('reports.html', reports=reports_data)

@app.route('/sidebar/')
def sidebar():
    return render_template('sidebar.html')

if __name__ == '__main__':
    app.run(debug=True)