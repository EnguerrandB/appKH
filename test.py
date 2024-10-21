import fitz  # PyMuPDF
import numpy as np
import os
import cv2
import pytesseract
import gzip
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
MODEL_PATH = './modèle.pkl.gz'  # Chemin vers le modèle
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['PDF_FOLDER'] = PDF_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Créer les dossiers si nécessaires
for folder in [IMAGE_FOLDER, PDF_FOLDER, UPLOAD_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Charger le modèle compressé en utilisant gzip
def load_model(model_path):
    with gzip.open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

model_data = load_model(MODEL_PATH)

# Fonction pour extraire les caractéristiques ORB d'une image
def extract_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    return descriptors

def detect_blank_pages_and_generate_images(images, threshold=0.99):
    blank_pages = []
    generated_images = []

    for image, page_num in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(gray_image > 240)
        total_pixels = gray_image.size
        white_ratio = white_pixels / total_pixels

        if white_ratio > threshold:
            blank_pages.append(page_num + 1)
            image_path = os.path.join(app.config['IMAGE_FOLDER'], f"page_{page_num + 1}.png")
            cv2.imwrite(image_path, image)
            generated_images.append(f"page_{page_num + 1}.png")
    
    return blank_pages, generated_images

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

MONTHS_FR_TO_NUM = {
    "janvier": "01", "février": "02", "fevrier": "02", "mars": "03", "avril": "04", "mai": "05", "juin": "06",
    "juillet": "07", "août": "08","aout": "08", "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12", "decembre": "12"
}

VALID_MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", 
                    "juillet", "août", "septembre", "octobre", "novembre", "décembre"]


date_patterns = [
    r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', 
    r'\b\d{1,2}-\d{1,2}-\d{2,4}\b', 
    r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b', 
    r'\b\d{1,2}\s(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s\d{4}\b',
    r',\s*le\s*(\d{1,2})\s+(' + '|'.join(VALID_MONTHS) + r')\s+(\d{4})'
]

def ocr_from_rendered_page(pdf_page):
    """Render a PDF page as an image and perform OCR."""
    zoom = 2  # Adjust this value to increase the resolution (if needed)
    mat = fitz.Matrix(zoom, zoom)
    pix = pdf_page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img)

def extract_text_from_pdf(pdf_file_path):
    """Extract text from PDF using direct extraction and OCR."""
    pdf_document = fitz.open(pdf_file_path)
    direct_text, ocr_text = "", ""

    for page in pdf_document:
        direct_text += page.get_text() + "\n"
        ocr_text += ocr_from_rendered_page(page) + "\n"

    return direct_text.strip().lower(), ocr_text.strip().lower()

def correct_month_ocr_errors(detected_month):
    """Return the detected month if it's valid, or find the closest match."""
    detected_month = detected_month.lower().strip()

    if detected_month in VALID_MONTHS:
        return detected_month

    if len(detected_month) < 3:
        return None

    from difflib import get_close_matches
    closest_match = get_close_matches(detected_month, VALID_MONTHS, n=1, cutoff=0.6)
    
    if closest_match:
        return closest_match[0]
    else:
        return None

def normalize_date(date):
    """Normalize date to DD.MM.YYYY format."""
    if not isinstance(date, str):
        return None

    delimiters = r'[\/\-\.]'  # Ajout des slashs et tirets comme délimiteurs
    if '/' in date or '-' in date or '.' in date:
        # Si la date contient des / ou -, on la sépare et la reformate
        day, month, year = re.split(delimiters, date)
    else:
        # Gérer les dates textuelles comme "22 février 2022"
        match = re.match(r'(\d{1,2})\s(\w+)\s(\d{4})', date)
        if match:
            day, month, year = match.groups()
            month = correct_month_ocr_errors(month)

    # Vérification et ajustement de l'année, du mois et du jour
    if not year.isdigit() or not month:
        return None

    # S'assurer que le jour et l'année sont bien formatés
    day, year = day.zfill(2), year.zfill(4)
    
    # Si le mois est un chiffre, on s'assure qu'il ait 2 chiffres
    if month.isdigit():
        month = month.zfill(2)
    else:
        # Si le mois est textuel, on le convertit en numérique
        month = MONTHS_FR_TO_NUM.get(month.lower(), None)
        if not month:
            return None

    # Retourner la date formatée avec des points
    return f"{day}.{month}.{year}"

def find_dates_in_text(text):
    """Find all date patterns in the text and normalize them to DD.MM.YYYY format."""
    normalized_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        normalized_dates += [normalize_date(date) for date in matches if normalize_date(date)]
    return normalized_dates

def extract_dates_from_pdf(pdf_file_path):
    """Extract all dates found in a PDF file."""
    direct_text, ocr_text = extract_text_from_pdf(pdf_file_path)
    combined_text = direct_text + "\n" + ocr_text

    # Extract dates from the content
    content_dates = find_dates_in_text(combined_text)

    return content_dates

# Fonction pour rechercher des dates dans le texte
def find_dates_in_text(text):
    


    normalized_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        print(f"Correspondances trouvées pour le motif {pattern} : {matches}")
        normalized_dates += matches

    return normalized_dates



def compare_phash(phash1, phash2):
    # La distance de Hamming entre deux hashes pHash
    return phash1 - phash2

def compute_phash(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_image)
    
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

# Fonction pour supprimer les pages sélectionnées
def remove_selected_pages(pdf_file, pages_to_remove):
    doc = fitz.open(pdf_file)
    new_pdf = fitz.open()

    for page_num in range(len(doc)):
        if page_num + 1 not in pages_to_remove:
            new_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
    
    new_pdf_path = os.path.join(app.config['PDF_FOLDER'], 'output.pdf')
    new_pdf.save(new_pdf_path)
    new_pdf.close()
    doc.close()

    return new_pdf_path

@app.route('/remove-pages', methods=['POST'])
def remove_pages():
    pages_to_remove = request.form.getlist('pages')
    file_name = request.form['file_name']
    new_file_name = request.form['new_file_name']

    original_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

    if not pages_to_remove:
        new_pdf_full_path = os.path.join(app.config['PDF_FOLDER'], new_file_name)
        shutil.copy(original_pdf_path, new_pdf_full_path)
        return send_file(new_pdf_full_path, as_attachment=True, download_name=new_file_name)

    pages_to_remove = [int(page) for page in pages_to_remove]
    new_pdf_path = remove_selected_pages(original_pdf_path, pages_to_remove)
    new_pdf_full_path = os.path.join(app.config['PDF_FOLDER'], new_file_name)
    os.rename(new_pdf_path, new_pdf_full_path)
    return send_file(new_pdf_full_path, as_attachment=True, download_name=new_file_name)

@app.route('/static/images/<filename>')
def display_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

def is_numeric_date(date_str):
    # On cherche à savoir si le format est JJ.MM.AAAA
    return bool(re.match(r'\d{1,2}\.\d{1,2}\.\d{4}', date_str))

def convert_french_date_to_numeric(french_date):
    try:
        day, month, year = french_date.split()  # Extrait les parties de la date textuelle
        month_num = MONTHS_FR_TO_NUM[month.lower()]  # Convertit le mois en nombre
        return f"{int(day):02}.{month_num}.{year}"
    except ValueError:
        print(f"Erreur lors de la conversion de la date : {french_date}")
        return french_date
    
def get_good_date(new_date):
    # Vérification du format de la nouvelle date
    if is_numeric_date(new_date):
        # Si la nouvelle date est déjà au format numérique, on s'assure que les jours et mois aient 2 chiffres
        day, month, year = new_date.split(".")
        new_date_numeric = f"{int(day):02}.{int(month):02}.{year}"
    else:
        # La nouvelle date est textuelle et doit être convertie en format numérique
        new_date_numeric = convert_french_date_to_numeric(new_date)
    
    return new_date_numeric

def zip_lists(a, b):
    return zip(a, b)

# Enregistrer le filtre dans Jinja2
app.jinja_env.filters['zip'] = zip_lists

def modify_old_filename_with_date(filename, new_date):
    # Convertir la nouvelle date si nécessaire
    good_date = get_good_date(new_date)
    
    # Modèle pour trouver toutes les dates dans le nom de fichier (formats : JJ.MM.AAAA, JJ-MM-AAAA, JJ/MM/AAAA)
    date_pattern = r'\b(?:\d{1,2}[.]\d{1,2}[.]\d{2,4}|\d{1,2}[-./]\d{4}|\d{4})\b'
    
    # Supprimer toutes les dates présentes dans le nom de fichier
    name_without_dates = re.sub(date_pattern, '', filename).strip()
    
    # Supprimer les espaces superflus causés par la suppression des dates
    name_without_dates = re.sub(r'\s+', ' ', name_without_dates).strip()
    
    # Séparer le nom de fichier et l'extension (s'il y en a)
    name, extension = os.path.splitext(name_without_dates)
    
    # Créer un nouveau nom en ajoutant la date à la fin du nom de fichier
    new_name = f"{name} {good_date}{extension}"
    
    print(new_name)
    return new_name

@app.route('/process-file', methods=['POST'])
def process_file():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', error='Aucun fichier fourni ou sélectionné')
    
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(file_path)

    try:
        # Appeler la nouvelle fonction pour gérer le traitement du fichier
        dates, closest_match = handle_file_processing(file_path)

        if not dates:
            dates = "Aucune date trouvée"
        
        suggested_name = file.filename.split('.')[0]
        
        if closest_match:
            suggested_name = os.path.basename(closest_match).rpartition('.')[0]

        # Normaliser et gérer les dates trouvées
        if dates and isinstance(dates, list) and len(dates) > 0:
            # Normaliser toutes les dates dans la liste
            from collections import Counter

            normalized_dates = [normalize_date(date) for date in dates if normalize_date(date) is not None]
            date_counts = Counter(normalized_dates)
            sorted_dates = [date for date, count in date_counts.most_common()]
            # Afficher les dates triées et regroupées
            print("Dates trouvées (les plus fréquentes en premier) : ", sorted_dates)
            suggested_name = modify_old_filename_with_date(suggested_name, sorted_dates[0]) + ".pdf"
        else:
            suggested_name += "_modifié.pdf"

        # Convertir le PDF en images et détecter les pages blanches
        images, _ = pdf_to_images(file_path)
        blank_pages, generated_images = detect_blank_pages_and_generate_images(images)
        
        # Afficher les résultats dans le template HTML
        return render_template(
            'index.html',
            closest_file={'name': closest_match, 'date': sorted_dates[0] if dates else "Aucune date trouvée"},
            pages_and_images=list(zip(blank_pages, generated_images)),
            file_name=file.filename,
            dates=sorted_dates,
            blank_pages=blank_pages,
            images=generated_images,
            suggested_name=suggested_name
        )
    
    except Exception as e:
        return render_template('index.html', error=f"Erreur lors du traitement du fichier : {str(e)}")

def handle_file_processing(file_path):
    """Traitement du fichier PDF pour extraire les dates et trouver le fichier le plus similaire."""
    # Extraire le texte du PDF et trouver les dates
    dates = extract_dates_from_pdf(file_path)
    
    # Trouver le fichier le plus similaire en utilisant le modèle
    closest_match = find_closest_file(file_path)
    
    return dates, closest_match

if __name__ == '__main__':
    app.run(debug=True)
