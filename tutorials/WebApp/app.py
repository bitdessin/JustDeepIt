import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from justdeepit.models import OD


# Detection model settings
model = None
model = OD('class_label.txt',
           model_arch='fasterrcnn',
           model_weight='fasterrcnn.pth',
           workspace='tmp',
           backend='detectron2')


# Flask
app = Flask(__name__, static_folder='uploads', static_url_path='')
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/', methods=['GET', 'POST'])
def detection_app():
    output_fpath = None
    
    if request.method == 'POST':
        req_file = request.files['file']
        if req_file:
            # upload file
            fname = secure_filename(req_file.filename)
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            req_file.save(fpath)
            
            # detect objects and save the result
            output_fpath = fpath + '.bbox.png'
            pred_output = model.inference(fpath, cpu=1, gpu=0)
            pred_output.draw('bbox', output_fpath, label=True, score=True)
    
    return render_template('index.html', output_fpath=output_fpath)
    

@app.route('/uploads/<fpath>')
def send_image(fpath):
    return send_from_directory(app.config['UPLOAD_FOLDER'], fpath)
