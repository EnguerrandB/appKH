import subprocess
import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import pickle

app = Flask(__name__)

# Configuration du dossier de téléchargement
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = './modèle.pkl.gz'  # Chemin vers le modèle
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Charger le modèle lorsque l'application démarre
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

model_data = load_model(MODEL_PATH)

# Vérifier si une extension de fichier est autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Appeler la fonction de comparaison après avoir sauvegardé le fichier
        closest_match = find_closest_file(file_path)
        return f"Le fichier le plus similaire est : {closest_match}", 200

    return 'Invalid file type', 400

def find_closest_file(pdf_path):
    # Appeler la fonction de comparaison pour trouver le fichier le plus similaire
    top_matches = find_closest_pdfs(model_data, pdf_path, top_n=1)
    if top_matches:
        closest_match = top_matches[0]['path']
        return closest_match
    else:
        return "Aucun fichier similaire trouvé"

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Exécuter le script Python et capturer la sortie
        result = subprocess.run(['python3', 'test.py'], capture_output=True, text=True)
        print(result)
        # Envoyer la sortie du script au frontend
        return f"Résultat du script : {result.stdout}", 200
    except Exception as e:
        return f"Erreur : {str(e)}", 500

if __name__ == '__main__':
    # Créer le dossier de téléchargement s'il n'existe pas
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5000)