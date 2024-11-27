from flask import Flask, render_template, request, send_file
from flask_sqlalchemy import SQLAlchemy
import spacy
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx

common_words = set([
    "Matplotlib", "Pandas", "Python", "Django", "Java", "Docker", 
    "Git", "Linux", "SQL", "Tableau", "Machine Learning", "Data Science", 
    "Statistics", "Regression", "AI", "NLP", "Analytics", "Visualization",
    "Development", "Software", "Technologies", "Tools", "Framework", 
    "Skills", "Experience", "Education", "Summary", "Contact", 
    "Visionary", "Tech", "Deep", "Learning", "Techniques"
])

# Flask app and database initialization
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///resumes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model
class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    similarity = db.Column(db.Float)
    resume_text = db.Column(db.Text)

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Extract text from DOCX files
def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
        return ""

def extract_entities(text):
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    doc = nlp(text)
    possible_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON" and len(ent.text.split()) > 1]

    if not possible_names:
        regex_names = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b', text)
        possible_names.extend(regex_names)

    filtered_names = [name for name in possible_names if name.split()[0] not in common_words]

    best_name = "N/A"
    if emails and filtered_names:
        email_usernames = [email.split('@')[0].lower() for email in emails]
        name_scores = {
            name: max(
                cosine_similarity(
                    TfidfVectorizer().fit_transform([name.lower(), username]).toarray()
                )[0, 1]
                for username in email_usernames
            )
            for name in filtered_names
        }
        best_name = max(name_scores, key=name_scores.get) if name_scores else "N/A"

    cleaned_name = " ".join([word for word in best_name.split() if word.lower() not in ["phone", "email"]])
    best_name = cleaned_name
    if emails == []:
        emails.append("N/A")

    return emails, [best_name]

@app.route('/', methods=['GET', 'POST'])
def index():
    global results
    results = []

    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resume_files')

        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        processed_resumes = []
        for resume_file in resume_files:
            resume_path = os.path.join("uploads", resume_file.filename)
            resume_file.save(resume_path)

            if resume_path.endswith(".docx"):
                resume_text = extract_text_from_docx(resume_path)
            else:
                resume_text = ""

            emails, names = extract_entities(resume_text)
            processed_resumes.append((names, emails, resume_text))

        tfidf_vectorizer = TfidfVectorizer()
        job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

        ranked_resumes = []
        for (names, emails, resume_text) in processed_resumes:
            resume_vector = tfidf_vectorizer.transform([resume_text])
            similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100
            ranked_resumes.append((names, emails, similarity))

        ranked_resumes.sort(key=lambda x: x[2], reverse=True)
        results = ranked_resumes

        # Store results in the database
        for names, emails, similarity in ranked_resumes:
            resume_entry = Resume(name=names[0], email=emails[0], similarity=similarity, resume_text="")
            db.session.add(resume_entry)
        db.session.commit()

    return render_template('index.html', results=results)

@app.route('/download_csv')
def download_csv():
    csv_content = "Rank,Name,Email,Similarity\n"
    for rank, (names, emails, similarity) in enumerate(results, start=1):
        name = names[0]
        email = emails[0]
        csv_content += f"{rank},{name},{email},{similarity:.2f}\n"

    csv_filename = "ranked_resumes.csv"
    with open(csv_filename, "w", newline="") as csv_file:
        csv_file.write(csv_content)

    csv_full_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), csv_filename)
    return send_file(csv_full_path, as_attachment=True, download_name="ranked_resumes.csv")

if __name__ == '__main__':
    # Create database tables if not already created
    with app.app_context():
        db.create_all()

    app.run(debug=True)
