import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug import secure_filename
import scipy
import scipy.misc

import triangulr

Debug = True

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG','JPEG', 'jpeg','PNG',  'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print "here"
            print filename
            points, result = triangulr.getNiceTriangulation( app.config['UPLOAD_FOLDER'] + filename, N = 400 , r = 0 , size = 250, factor_size = 3)
            result_name = 'res_' + filename[:-3] + 'png'
            path_result =  app.config['UPLOAD_FOLDER']
            scipy.misc.imsave(path_result + result_name, result)

            print "--------------"

            return render_template('result.html', filename = result_name)
        else:
            print 'not allowed'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/result/<filename>')
def result(filename):
    print "########"
    print "results: " + filename
    return render_template('translators.html', filename = filename)


if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0')

