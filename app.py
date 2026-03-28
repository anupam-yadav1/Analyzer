from flask import Flask, render_template, request
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# -------- Extract Text --------
def extract_text(file):
    text = ""
    if file.filename.endswith('.pdf'):
        pdf = PyPDF2.PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()
    elif file.filename.endswith('.docx'):
        text = docx2txt.process(file)
    return text.lower()

# -------- ATS Score --------
def calculate_score(resume, jd):
    cv = CountVectorizer()
    matrix = cv.fit_transform([resume, jd])
    score = cosine_similarity(matrix)[0][1]
    return round(score * 100, 2)

# -------- Route --------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['resume']
        jd = request.form['jd']

        resume_text = extract_text(file)
        score = calculate_score(resume_text, jd.lower())

        return render_template('resume.html', score=score)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)