from flask import Flask, render_template, request, json
import requests

app = Flask(__name__)

@app.route('/', methods=['GET'])
@app.route('/capture', methods=['GET'])
def capture():
    return render_template('capture.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image = request.files['file']
        color_space = request.form['color_space']
        top_n = request.form['top_n']

        response = requests.post(
            "http://localhost:8000/predict/",
            files={"file": image},
            data={"color_space": color_space, "top_n": top_n}
        )
        
        predictions = json.loads(response.content).get('predictions')
        return render_template('results.html', predictions=json.loads(predictions))

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
