import fitz  # PyMuPDF
import numpy as np
import os
import cv2
import re
import pickle
from flask import Flask, request, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename
import shutil
from pdf2image import convert_from_path
from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Dossiers pour stocker les images et fichiers générés
IMAGE_FOLDER = 'static/images'
PDF_FOLDER = 'static/pdf'
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = './modèle_GedKHV2.pkl'  # Chemin vers le modèle
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['PDF_FOLDER'] = PDF_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Créer les dossiers si nécessaires
for folder in [IMAGE_FOLDER, PDF_FOLDER, UPLOAD_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Charger le modèle lorsque l'application démarre
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

model_data = load_model(MODEL_PATH)

# Fonction pour extraire les caractéristiques ORB d'une image
def extract_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    return descriptors

# Fonction pour calculer le hash perceptuel (pHash) d'une image
def compute_phash(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_image)

# Fonction pour comparer les hashes perceptuels (pHash)
def compare_phash(phash1, phash2):
    # La distance de Hamming entre deux hashes pHash
    return phash1 - phash2

# Fonction pour convertir un PDF en une image composite (en utilisant uniquement la première page)
def pdf_to_composite_image(pdf_path, num_pages=1):
    try:
        images = convert_from_path(pdf_path, dpi=72)[:num_pages]
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        
        composite_image = Image.new('RGB', (max_width, total_height))
        
        y_offset = 0
        for img in images:
            composite_image.paste(img, (0, y_offset))
            y_offset += img.height
        
        composite_image_cv = cv2.cvtColor(np.array(composite_image), cv2.COLOR_RGB2BGR)
        return composite_image_cv
    except Exception as e:
        print(f"Error during PDF conversion {pdf_path}: {e}")
        return None

# Fonction pour extraire le texte de la première page d'un PDF
def extract_text_from_first_page(pdf_path):
    try:
        with fitz.open(pdf_path) as pdf:
            first_page = pdf[0]
            text = first_page.get_text("text")
        return text
    except Exception as e:
        print(f"Error during text extraction from {pdf_path}: {e}")
        return None

# Fonction pour comparer les caractéristiques ORB en utilisant le matching brute-force
def compare_orb_features(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

# Fonction pour comparer le texte en utilisant TF-IDF et la similarité cosinus
def compare_text(text1, text2):
    if text1 and text2:
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0][1]
    return 0

# Fonction pour rechercher des dates dans le texte
def find_dates_in_text(text):
    VALID_MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", 
                    "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

    date_patterns = [
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', 
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b', 
        r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b', 
        r'\b\d{1,2}\s(?:' + '|'.join(VALID_MONTHS) + r')\s\d{4}\b'
    ]

    normalized_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        print(f"Correspondances trouvées pour le motif {pattern} : {matches}")
        normalized_dates += matches

    return normalized_dates


# Fonction pour convertir un PDF en images
def pdf_to_images(pdf_file):
    doc = fitz.open(pdf_file)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        images.append((img, page_num))
    return images, doc


# Fonction pour trouver les fichiers PDF les plus proches dans le modèle
def find_closest_pdfs(model_data, pdf_path, top_n=3):
    composite_image = pdf_to_composite_image(pdf_path, num_pages=1)
    if composite_image is None:
        print(f"Error processing {pdf_path}")
        return []

    new_phash = compute_phash(composite_image)
    new_orb_features = extract_features(composite_image)
    new_text = extract_text_from_first_page(pdf_path)

    results = []

    for entry in model_data:
        model_phash = entry['phash']
        model_orb_features = entry['orb_features']
        model_text = entry['text']

        phash_similarity = 1 - (compare_phash(new_phash, imagehash.hex_to_hash(model_phash)) / len(new_phash.hash.flatten()))
        orb_similarity = compare_orb_features(new_orb_features, model_orb_features) if new_orb_features is not None and model_orb_features is not None else 0
        text_similarity = compare_text(new_text, model_text)

        combined_score = (phash_similarity * 0.5) + (orb_similarity * 0.3) + (text_similarity * 0.2)

        results.append({
            'path': entry['path'],
            'combined_score': combined_score,
            'phash_similarity': phash_similarity,
            'orb_similarity': orb_similarity,
            'text_similarity': text_similarity
        })

    results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
    return results[:top_n]

# Fonction pour trouver le fichier le plus proche
def find_closest_file(pdf_path):
    top_matches = find_closest_pdfs(model_data, pdf_path, top_n=1)
    if top_matches:
        closest_match = top_matches[0]['path']
        return closest_match
    else:
        return None

# Fonction pour extraire le texte du PDF et rechercher les dates
def extract_text_and_find_dates(pdf_file):
    doc = fitz.open(pdf_file)
    full_text = ""
    for page in doc:
        page_text = page.get_text("text")
        full_text += page_text + "\n"
        print(f"Texte extrait de la page {page.number}: {page_text}")

    dates = find_dates_in_text(full_text)
    print(f"Dates trouvées : {dates}")
    return dates

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-file', methods=['POST'])
def process_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', error='Aucun fichier fourni ou sélectionné')
    
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(file_path)
    
    try:
        # Extraire le texte du PDF et trouver les dates
        dates = extract_text_and_find_dates(file_path)
        if not dates:
            dates = "Aucune date trouvée"

        # Trouver le fichier le plus similaire en utilisant le modèle
        closest_match = find_closest_file(file_path)
        suggested_name = file.filename.split('.')[0]
        if closest_match:
            suggested_name = os.path.basename(closest_match).split('.')[0]

        # Si une date est trouvée, on l'ajoute au nom
        if dates and isinstance(dates, list) and len(dates) > 0:
            suggested_name += f"_{dates[0].replace('/', '-')}.pdf"
        else:
            suggested_name += "_modifié.pdf"

        # Convertir le PDF en images
        images, _ = pdf_to_images(file_path)
        
        # Détecter les pages blanches et générer des images
        blank_pages, generated_images = detect_blank_pages_and_generate_images(images)
        
        # Afficher les résultats dans le template HTML (sur la même page)
        return render_template(
            'index.html',
            closest_file={'name': closest_match, 'date': dates[0] if dates else "Aucune date trouvée"},
            pages_and_images=list(zip(blank_pages, generated_images)),
            file_name=file.filename,
            dates=dates,
            blank_pages=blank_pages,
            images=generated_images,
            suggested_name=suggested_name
        )
    
    except Exception as e:
        return render_template('index.html', error=f"Erreur lors du traitement du fichier : {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)