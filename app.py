import subprocess
import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import pickle

app = Flask(__name__)

# Configuration du dossier de téléchargement
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}



# Vérifier si une extension de fichier est autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')



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