# app.py

from flask import Flask, render_template, request
from main import generate_summary_from_file

app = Flask(__name__)

@app.route('/')
def index():
 return render_template('p1.html')

@app.route('/about')
def about():
        return render_template('about.html')

@app.route('/privacy')
def privacy():
        return render_template('privacy.html')

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    file = request.form['video_url']
    summary = generate_summary_from_file(file)
    if summary:
        return render_template('p1.html', summary=summary)
    else:
        return "Error: Failed to generate summary."

if __name__ == '__main__':
    app.run(debug=True)
