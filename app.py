"""
MedScan AI - Chest X-Ray Disease Detection Web Application
Flask backend that serves the trained ResNet18 model for inference.
Detects: COVID-19, Pneumonia, Tuberculosis, and Healthy lungs.
"""

import os
import io
import json
import time
import uuid
import base64
import functools
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file, g

try:
    import pydicom
    DICOM_SUPPORTED = True
except ImportError:
    DICOM_SUPPORTED = False

import jwt
import bcrypt
from flasgger import Swagger

# ============================================================
# APP CONFIGURATION
# ============================================================
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['JWT_SECRET'] = os.environ.get('JWT_SECRET', 'medscan-ai-secret-key-change-in-production-2025')
app.config['JWT_EXPIRATION_HOURS'] = 72  # Token valid for 72 hours
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Swagger / OpenAPI configuration
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: rule.endpoint != 'static' and rule.endpoint != 'index' and rule.endpoint != 'flasgger.static' and rule.endpoint != 'flasgger.apispec_1',
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/api/docs"
}

swagger_template = {
    "info": {
        "title": "MedScan AI — Chest X-Ray Analysis API",
        "description": (
            "REST API for AI-powered chest X-ray disease detection. "
            "Detects COVID-19, Pneumonia, Tuberculosis, and Healthy lungs "
            "using a deep learning model (ResNet18 + optional DenseNet121 ensemble). "
            "Features include Grad-CAM heatmaps, Test-Time Augmentation, "
            "Out-of-Distribution detection, batch processing, and PDF report generation."
        ),
        "version": "2.0.0",
        "contact": {
            "name": "MedScan AI",
        },
        "license": {
            "name": "Educational / Research Use"
        }
    },
    "basePath": "/",
    "schemes": ["http", "https"],
    "tags": [
        {"name": "Authentication", "description": "User registration, login, and profile"},
        {"name": "Prediction", "description": "X-ray analysis and disease detection endpoints"},
        {"name": "Reports", "description": "PDF report generation"},
        {"name": "History", "description": "Scan history management (per-user)"},
        {"name": "Analytics", "description": "Admin analytics dashboard data"},
        {"name": "Information", "description": "Disease info and health status"},
        {"name": "Chatbot", "description": "Medical assistant chatbot"},
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'dcm'}

# Model paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'output', 'chest_xray_model.pth')
DENSENET_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'output', 'chest_xray_densenet121.pth')

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# SQLITE DATABASE FOR HISTORY
# ============================================================
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), 'medscan_history.db')


def get_db():
    """Get a database connection (create tables if needed)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
    return conn


def init_db():
    """Initialize the database schema."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            confidence_pct TEXT NOT NULL,
            all_predictions TEXT,
            inference_time TEXT,
            inference_mode TEXT DEFAULT 'single',
            device_used TEXT,
            is_ood INTEGER DEFAULT 0,
            ood_score REAL,
            patient_name TEXT,
            patient_age TEXT,
            patient_gender TEXT,
            patient_notes TEXT,
            quality_warnings TEXT,
            timestamp TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Add user_id column if upgrading from older schema
    try:
        conn.execute("ALTER TABLE scan_history ADD COLUMN user_id INTEGER REFERENCES users(id)")
    except Exception:
        pass  # Column already exists
    conn.commit()
    conn.close()
    print(f"Database initialized: {DB_PATH}")


# Initialize DB at startup
init_db()


# ============================================================
# JWT AUTHENTICATION HELPERS
# ============================================================

def generate_token(user_id, is_admin=False):
    """Generate a JWT token for a user."""
    payload = {
        'user_id': user_id,
        'is_admin': is_admin,
        'exp': datetime.utcnow() + timedelta(hours=app.config['JWT_EXPIRATION_HOURS']),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, app.config['JWT_SECRET'], algorithm='HS256')


def decode_token(token):
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_current_user():
    """Extract user from Authorization header. Returns user dict or None."""
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None
    token = auth_header[7:]
    payload = decode_token(token)
    if not payload:
        return None
    conn = get_db()
    user = conn.execute("SELECT id, username, email, is_admin, created_at FROM users WHERE id = ?",
                        (payload['user_id'],)).fetchone()
    conn.close()
    if user:
        return dict(user)
    return None


def login_required(f):
    """Decorator that requires a valid JWT token."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required. Please login.'}), 401
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorator that requires admin privileges."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required. Please login.'}), 401
        if not user.get('is_admin'):
            return jsonify({'error': 'Admin access required.'}), 403
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


# ============================================================
# DISEASE INFORMATION DATABASE
# ============================================================
DISEASE_INFO = {
    'COVID-19': {
        'name': 'COVID-19 (SARS-CoV-2 Infection)',
        'severity': 'High',
        'severity_color': '#e74c3c',
        'icon': 'virus',
        'description': 'COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. '
                       'Chest X-rays may show bilateral ground-glass opacities, consolidation, '
                       'and peripheral lung involvement typically in the lower lobes.',
        'symptoms': [
            'Persistent dry cough',
            'Fever and chills',
            'Shortness of breath / difficulty breathing',
            'Fatigue and body aches',
            'Loss of taste or smell',
            'Chest pain or pressure'
        ],
        'recommendations': [
            'Seek immediate medical attention and get a PCR/rapid antigen test',
            'Self-isolate to prevent transmission to others',
            'Monitor oxygen levels with a pulse oximeter (seek ER if SpO2 < 94%)',
            'Stay hydrated and rest; use fever-reducing medication as needed',
            'Follow your healthcare provider\'s treatment plan',
            'Consider antiviral treatment if within the eligibility window'
        ],
        'when_to_seek_emergency': [
            'Oxygen saturation drops below 94%',
            'Severe difficulty breathing or persistent chest pain',
            'Confusion or inability to stay awake',
            'Bluish lips or face (cyanosis)'
        ],
        'treatment_overview': 'Treatment may include antiviral medications (e.g., Paxlovid), '
                              'supplemental oxygen, corticosteroids (dexamethasone), and in severe cases, '
                              'mechanical ventilation. Vaccination remains the best preventive measure.'
    },
    'Healthy': {
        'name': 'Normal / Healthy Lungs',
        'severity': 'None',
        'severity_color': '#27ae60',
        'icon': 'heart-pulse',
        'description': 'The chest X-ray appears normal with no significant abnormalities detected. '
                       'The lung fields are clear, heart size is within normal limits, and no '
                       'pleural effusion or consolidation is observed.',
        'symptoms': [],
        'recommendations': [
            'Continue maintaining a healthy lifestyle',
            'Get regular health check-ups annually',
            'Maintain good respiratory hygiene',
            'Stay up to date with vaccinations (flu, COVID-19, pneumococcal)',
            'Avoid smoking and exposure to secondhand smoke',
            'Exercise regularly to maintain lung health'
        ],
        'when_to_seek_emergency': [],
        'treatment_overview': 'No treatment required. The X-ray findings suggest healthy lung tissue. '
                              'Continue preventive health measures and routine screenings.'
    },
    'Pneumonia': {
        'name': 'Pneumonia (Lung Infection)',
        'severity': 'Moderate to High',
        'severity_color': '#e67e22',
        'icon': 'lungs',
        'description': 'Pneumonia is an infection that inflames the air sacs in one or both lungs. '
                       'Chest X-rays typically show areas of consolidation (white opacities), '
                       'air bronchograms, and may show pleural effusion. It can be caused by '
                       'bacteria, viruses, or fungi.',
        'symptoms': [
            'Cough with phlegm or pus',
            'Fever, sweating, and shaking chills',
            'Shortness of breath during normal activities',
            'Sharp chest pain when breathing or coughing',
            'Fatigue and decreased appetite',
            'Nausea, vomiting, or diarrhea'
        ],
        'recommendations': [
            'Consult a doctor immediately for proper diagnosis and treatment',
            'Complete the full course of prescribed antibiotics (if bacterial)',
            'Get adequate rest and increase fluid intake',
            'Use a humidifier to ease breathing',
            'Take fever-reducing medication as directed',
            'Follow up with a repeat chest X-ray to confirm resolution'
        ],
        'when_to_seek_emergency': [
            'High fever (above 102°F / 39°C) that doesn\'t respond to medication',
            'Severe difficulty breathing or rapid breathing',
            'Coughing up blood',
            'Chest pain that worsens with breathing',
            'Confusion (especially in older adults)'
        ],
        'treatment_overview': 'Treatment depends on the cause: bacterial pneumonia is treated with antibiotics, '
                              'viral pneumonia may require antivirals, and fungal pneumonia requires antifungals. '
                              'Severe cases may need hospitalization with IV antibiotics and oxygen therapy.'
    },
    'Tuberculosis': {
        'name': 'Tuberculosis (TB)',
        'severity': 'High',
        'severity_color': '#e74c3c',
        'icon': 'bacterium',
        'description': 'Tuberculosis is a serious bacterial infection caused by Mycobacterium tuberculosis. '
                       'Chest X-rays may show upper lobe infiltrates, cavitation, hilar lymphadenopathy, '
                       'and pleural effusion. TB primarily affects the lungs but can spread to other organs.',
        'symptoms': [
            'Persistent cough lasting more than 3 weeks',
            'Coughing up blood or sputum (hemoptysis)',
            'Night sweats',
            'Unexplained weight loss',
            'Fever and fatigue',
            'Loss of appetite'
        ],
        'recommendations': [
            'Seek immediate medical evaluation — TB is a notifiable disease',
            'Get a sputum test and TB skin test (Mantoux) or blood test (IGRA)',
            'Begin TB treatment as prescribed (typically 6-9 months of antibiotics)',
            'Take ALL medications exactly as prescribed — DO NOT stop early',
            'Practice respiratory isolation during the infectious period',
            'Notify close contacts for TB screening'
        ],
        'when_to_seek_emergency': [
            'Coughing up large amounts of blood',
            'Severe breathing difficulty',
            'High fever with confusion',
            'Sudden severe chest pain'
        ],
        'treatment_overview': 'Standard TB treatment involves a 6-month regimen: 2 months of isoniazid, '
                              'rifampicin, pyrazinamide, and ethambutol (HRZE), followed by 4 months of '
                              'isoniazid and rifampicin (HR). Directly Observed Therapy (DOT) is recommended. '
                              'Drug-resistant TB may require longer treatment with second-line drugs.'
    }
}


# ============================================================
# MODEL LOADING
# ============================================================
def load_model():
    """Load the trained chest X-ray classification model (ResNet18)."""
    print(f"Loading ResNet18 model from: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']

    # Rebuild model architecture (same as train.py)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print(f"ResNet18 loaded! Classes: {class_names}")
    return model, class_names


def load_densenet_model(class_names):
    """Load the trained DenseNet121 model for ensemble (optional)."""
    if not os.path.exists(DENSENET_MODEL_PATH):
        print(f"DenseNet121 model not found at {DENSENET_MODEL_PATH} — ensemble disabled.")
        print("  Train it with: python train.py --model densenet121")
        return None

    print(f"Loading DenseNet121 model from: {DENSENET_MODEL_PATH}")
    checkpoint = torch.load(DENSENET_MODEL_PATH, map_location=DEVICE, weights_only=False)
    num_classes = checkpoint['num_classes']

    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    print("DenseNet121 loaded! Ensemble mode ENABLED.")
    return model


# Load models at startup
model, class_names = load_model()
densenet_model = load_densenet_model(class_names)
ENSEMBLE_ENABLED = densenet_model is not None

# Map dataset class names to display names
CLASS_MAP = {
    'COVID19': 'COVID-19',
    'NORMAL': 'Healthy',
    'PNEUMONIA': 'Pneumonia',
    'TURBERCULOSIS': 'Tuberculosis'
}

# Image transforms (must match training)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ============================================================
# OUT-OF-DISTRIBUTION (OOD) DETECTION
# ============================================================
def compute_ood_score(image_path):
    """
    Compute Out-of-Distribution score using energy-based detection
    combined with feature-space analysis.
    Returns a dict with OOD metrics and a boolean flag.
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

    # Hook into avgpool to get penultimate features
    features = []
    def hook_fn(module, input, output):
        features.append(output)

    hook = model.avgpool.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)

    hook.remove()

    # 1) Energy score: lower (more negative) = more in-distribution
    temperature = 1.0
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1).item()

    # 2) Feature norm (medical X-rays cluster within a norm range)
    feat_vector = features[0].view(features[0].size(0), -1).squeeze()
    feat_norm = torch.norm(feat_vector).item()

    # 3) Max softmax probability
    probs = torch.nn.functional.softmax(logits, dim=1)
    max_prob = probs.max().item()

    # 4) Prediction entropy (uniform = log(4) ≈ 1.386 = maximum uncertainty)
    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()

    # 5) Image-level heuristics: medical X-rays are mostly grayscale
    img_np = np.array(img.resize((64, 64)))
    r, g, b = img_np[:,:,0].astype(float), img_np[:,:,1].astype(float), img_np[:,:,2].astype(float)
    color_deviation = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))
    is_highly_colorful = color_deviation > 40  # X-rays are grayscale

    # Determine OOD based on multiple indicators
    ood_reasons = []
    ood_score = 0  # accumulate evidence

    if entropy > 1.2:
        ood_reasons.append('High prediction uncertainty (entropy)')
        ood_score += 1
    if max_prob < 0.35:
        ood_reasons.append('Very low classification confidence')
        ood_score += 1
    if energy > -1.5:
        ood_reasons.append('Unusual energy pattern for chest X-ray')
        ood_score += 1
    if is_highly_colorful:
        ood_reasons.append('Image appears highly colorful (not typical for X-ray)')
        ood_score += 1
    if feat_norm < 1.0 or feat_norm > 150.0:
        ood_reasons.append('Feature response outside expected range')
        ood_score += 1

    # Flag as OOD if ≥2 indicators trigger
    is_ood = ood_score >= 2

    return {
        'is_ood': is_ood,
        'ood_score': ood_score,
        'energy': round(energy, 3),
        'feature_norm': round(feat_norm, 3),
        'max_probability': round(max_prob, 4),
        'entropy': round(entropy, 4),
        'color_deviation': round(color_deviation, 2),
        'reasons': ood_reasons,
        'confidence_level': 'in-distribution' if not is_ood else 'out-of-distribution'
    }


# ============================================================
# GRAD-CAM
# ============================================================
def generate_gradcam(image_path, pred_idx):
    """Generate Grad-CAM heatmap for the predicted class."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad_(True)

    # Hook into last conv layer (layer4 for ResNet18)
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    model.eval()
    output = model(img_tensor)
    model.zero_grad()

    # Backward pass for the predicted class
    target = output[0, pred_idx]
    target.backward()

    fh.remove()
    bh.remove()

    # Compute Grad-CAM
    grads = gradients[0].detach().cpu().numpy()[0]   # (C, H, W)
    acts = activations[0].detach().cpu().numpy()[0]   # (C, H, W)
    weights = np.mean(grads, axis=(1, 2))              # (C,)
    cam = np.zeros(acts.shape[1:], dtype=np.float32)   # (H, W)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)  # ReLU
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize to original image size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    cam_np = np.array(cam_img)

    # Create colored heatmap overlay
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    colormap = cm.jet
    heatmap_colored = colormap(cam_np / 255.0)[:, :, :3]  # RGB 0-1
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend with original image
    img_np = np.array(img)
    overlay = (0.55 * img_np + 0.45 * heatmap_colored).astype(np.uint8)

    # Convert to base64
    overlay_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format='PNG')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


# ============================================================
# TEST-TIME AUGMENTATION (TTA)
# ============================================================
tta_transforms = [
    inference_transform,  # Original
    transforms.Compose([  # Horizontal flip
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([  # Slight rotation
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([  # Brightness adjust
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    transforms.Compose([  # Center crop
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
]


# ============================================================
# INFERENCE
# ============================================================
def predict_xray(image_path, use_tta=True):
    """Run inference on a chest X-ray image with optional TTA and ensemble."""
    img = Image.open(image_path).convert('RGB')

    augments = tta_transforms if use_tta else [inference_transform]

    def run_model_inference(m, augments_list):
        """Run a single model with given augmentations."""
        m.eval()
        probs_list = []
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                for t in augments_list:
                    img_tensor = t(img).unsqueeze(0).to(DEVICE)
                    outputs = m(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    probs_list.append(probs[0].cpu().numpy())
        return np.mean(probs_list, axis=0)

    # ResNet18 predictions
    resnet_probs = run_model_inference(model, augments)

    # Ensemble with DenseNet121 if available
    inference_mode = 'single'
    if ENSEMBLE_ENABLED and densenet_model is not None:
        densenet_probs = run_model_inference(densenet_model, augments)
        # Weighted average: ResNet18 (50%) + DenseNet121 (50%)
        probs = 0.5 * resnet_probs + 0.5 * densenet_probs
        inference_mode = 'ensemble'
    else:
        probs = resnet_probs

    pred_idx = probs.argmax()
    pred_class = class_names[pred_idx]
    display_name = CLASS_MAP.get(pred_class, pred_class)
    confidence = float(probs[pred_idx])

    # Build results for all classes
    all_predictions = []
    for i, cls in enumerate(class_names):
        disp = CLASS_MAP.get(cls, cls)
        all_predictions.append({
            'class': disp,
            'probability': float(probs[i]),
            'percentage': f"{float(probs[i]) * 100:.1f}%"
        })

    # Sort by probability descending
    all_predictions.sort(key=lambda x: x['probability'], reverse=True)

    # Get disease info
    disease_data = DISEASE_INFO.get(display_name, {})

    # Generate Grad-CAM heatmap (always from ResNet18 — it has layer4)
    try:
        gradcam_image = generate_gradcam(image_path, pred_idx)
    except Exception:
        gradcam_image = None

    result = {
        'prediction': display_name,
        'confidence': confidence,
        'confidence_pct': f"{confidence * 100:.1f}%",
        'all_predictions': all_predictions,
        'disease_info': disease_data,
        'gradcam_image': gradcam_image,
        'inference_mode': inference_mode,
        'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
        'device_used': str(DEVICE).upper()
    }

    return result


# ============================================================
# DICOM SUPPORT
# ============================================================
def convert_dicom_to_image(dicom_path):
    """Convert a DICOM (.dcm) file to a standard image file (PNG)."""
    if not DICOM_SUPPORTED:
        raise RuntimeError("DICOM support not available. Install pydicom: pip install pydicom")

    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(float)

    # Apply DICOM windowing if available
    if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
        center = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else float(ds.WindowCenter[0])
        width = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else float(ds.WindowWidth[0])
        lower = center - width / 2
        upper = center + width / 2
        pixel_array = np.clip(pixel_array, lower, upper)

    # Normalize to 0-255
    if pixel_array.max() != pixel_array.min():
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0
    pixel_array = pixel_array.astype(np.uint8)

    # Handle PhotometricInterpretation (invert if MONOCHROME1)
    if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array = 255 - pixel_array

    img = Image.fromarray(pixel_array)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Save as PNG
    png_path = dicom_path.rsplit('.', 1)[0] + '.png'
    img.save(png_path)
    return png_path


def preprocess_uploaded_file(filepath):
    """If the file is DICOM, convert it. Otherwise return as-is."""
    ext = filepath.rsplit('.', 1)[-1].lower()
    if ext == 'dcm':
        return convert_dicom_to_image(filepath), True
    return filepath, False


# ============================================================
# ROUTES
# ============================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_xray_image(filepath):
    """Basic quality validation to check if image looks like a chest X-ray."""
    warnings = []
    try:
        img = Image.open(filepath).convert('RGB')
        w, h = img.size

        # Check minimum resolution
        if w < 100 or h < 100:
            return False, "Image resolution is too low (minimum 100x100 pixels).", warnings

        # Check if image is too small
        if w < 200 or h < 200:
            warnings.append("Low resolution image — results may be less accurate.")

        # Check aspect ratio (X-rays are roughly square, max 3:1)
        ratio = max(w, h) / min(w, h)
        if ratio > 3.0:
            warnings.append("Unusual aspect ratio — make sure this is a chest X-ray.")

        # Check if image is roughly grayscale (X-rays are usually grayscale)
        img_np = np.array(img)
        r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
        color_diff = np.mean(np.abs(r.astype(float) - g.astype(float))) + \
                     np.mean(np.abs(g.astype(float) - b.astype(float)))
        if color_diff > 30:
            warnings.append("Image appears to be in color — chest X-rays are typically grayscale. Results may vary.")

        # Check if image is too bright or too dark
        mean_brightness = np.mean(img_np)
        if mean_brightness < 20:
            warnings.append("Image appears very dark — ensure proper X-ray image quality.")
        elif mean_brightness > 240:
            warnings.append("Image appears very bright/overexposed.")

        return True, None, warnings
    except Exception as e:
        return False, f"Could not process image: {str(e)}", warnings


@app.route('/')
def index():
    """Serve the main web application."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Serve the analytics dashboard page."""
    return render_template('dashboard.html')


# ============================================================
# AUTHENTICATION API
# ============================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user account
    Create a new user with username, email, and password.
    The first registered user automatically becomes an admin.
    ---
    tags:
      - Authentication
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - username
            - email
            - password
          properties:
            username:
              type: string
              example: "doctor_smith"
              minLength: 3
            email:
              type: string
              example: "smith@hospital.com"
            password:
              type: string
              example: "securepassword123"
              minLength: 6
    responses:
      201:
        description: User registered successfully
        schema:
          type: object
          properties:
            message:
              type: string
            token:
              type: string
            user:
              type: object
              properties:
                id:
                  type: integer
                username:
                  type: string
                email:
                  type: string
                is_admin:
                  type: boolean
      400:
        description: Validation error or user already exists
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password', '')

    # Validation
    if not username or len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    if not email or '@' not in email:
        return jsonify({'error': 'Valid email is required'}), 400
    if not password or len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    # Hash password
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    conn = get_db()
    try:
        # First user becomes admin automatically
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        is_admin = 1 if user_count == 0 else 0

        cursor = conn.execute(
            "INSERT INTO users (username, email, password_hash, is_admin) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, is_admin)
        )
        conn.commit()
        user_id = cursor.lastrowid
    except sqlite3.IntegrityError as e:
        conn.close()
        if 'username' in str(e):
            return jsonify({'error': 'Username already taken'}), 400
        return jsonify({'error': 'Email already registered'}), 400
    conn.close()

    token = generate_token(user_id, bool(is_admin))
    return jsonify({
        'message': 'Registration successful',
        'token': token,
        'user': {
            'id': user_id,
            'username': username,
            'email': email,
            'is_admin': bool(is_admin)
        }
    }), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login with username/email and password
    Authenticate and receive a JWT token for API access.
    ---
    tags:
      - Authentication
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - login
            - password
          properties:
            login:
              type: string
              example: "doctor_smith"
              description: "Username or email"
            password:
              type: string
              example: "securepassword123"
    responses:
      200:
        description: Login successful
        schema:
          type: object
          properties:
            message:
              type: string
            token:
              type: string
            user:
              type: object
              properties:
                id:
                  type: integer
                username:
                  type: string
                email:
                  type: string
                is_admin:
                  type: boolean
      401:
        description: Invalid credentials
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    login_id = (data.get('login') or '').strip()
    password = data.get('password', '')

    if not login_id or not password:
        return jsonify({'error': 'Username/email and password are required'}), 400

    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?",
        (login_id, login_id.lower())
    ).fetchone()
    conn.close()

    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    user = dict(user)
    if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        return jsonify({'error': 'Invalid credentials'}), 401

    token = generate_token(user['id'], bool(user['is_admin']))
    return jsonify({
        'message': 'Login successful',
        'token': token,
        'user': {
            'id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'is_admin': bool(user['is_admin'])
        }
    })


@app.route('/api/auth/me')
def get_me():
    """Get current user profile
    Returns the authenticated user's profile information.
    Requires a valid JWT token in the Authorization header.
    ---
    tags:
      - Authentication
    parameters:
      - name: Authorization
        in: header
        type: string
        required: true
        description: "Bearer <JWT token>"
    responses:
      200:
        description: User profile
        schema:
          type: object
          properties:
            user:
              type: object
              properties:
                id:
                  type: integer
                username:
                  type: string
                email:
                  type: string
                is_admin:
                  type: boolean
                created_at:
                  type: string
      401:
        description: Not authenticated
    """
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    return jsonify({'user': user})


@app.route('/predict', methods=['POST'])
def predict():
    """Analyze a single chest X-ray image
    Upload a chest X-ray image for AI-powered disease detection.
    Returns prediction with confidence scores, Grad-CAM heatmap, OOD detection, and disease information.
    ---
    tags:
      - Prediction
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Chest X-ray image (JPG, PNG, BMP, TIFF, WebP, or DICOM .dcm)
    responses:
      200:
        description: Successful analysis
        schema:
          type: object
          properties:
            prediction:
              type: string
              example: "COVID-19"
              description: Predicted disease class
            confidence:
              type: number
              example: 0.954
              description: Confidence score (0-1)
            confidence_pct:
              type: string
              example: "95.4%"
            all_predictions:
              type: array
              items:
                type: object
                properties:
                  class:
                    type: string
                  probability:
                    type: number
                  percentage:
                    type: string
            gradcam_image:
              type: string
              description: Base64-encoded Grad-CAM heatmap overlay
            inference_mode:
              type: string
              enum: [single, ensemble]
              description: Whether ensemble was used
            ood_detection:
              type: object
              description: Out-of-distribution detection results
              properties:
                is_ood:
                  type: boolean
                energy:
                  type: number
                reasons:
                  type: array
                  items:
                    type: string
            quality_warnings:
              type: array
              items:
                type: string
      400:
        description: Invalid file or validation error
      500:
        description: Prediction failed
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Convert DICOM if needed
        processed_path, was_dicom = preprocess_uploaded_file(filepath)

        # Validate image quality
        is_valid, error_msg, quality_warnings = validate_xray_image(processed_path)
        if was_dicom:
            quality_warnings.insert(0, "DICOM file detected — converted to PNG for analysis.")
        if not is_valid:
            try: os.remove(filepath)
            except: pass
            if was_dicom:
                try: os.remove(processed_path)
                except: pass
            return jsonify({'error': error_msg}), 400

        # Out-of-Distribution detection
        try:
            ood_result = compute_ood_score(processed_path)
            if ood_result['is_ood']:
                quality_warnings.append(
                    "⚠️ OOD Alert: This image may NOT be a chest X-ray. "
                    "Results may be unreliable."
                )
                for reason in ood_result['reasons']:
                    quality_warnings.append(f"  • {reason}")
        except Exception:
            ood_result = None

        # Run prediction
        start_time = time.time()
        result = predict_xray(processed_path)
        inference_time = time.time() - start_time
        result['inference_time'] = f"{inference_time:.2f}s"
        result['quality_warnings'] = quality_warnings
        if ood_result:
            result['ood_detection'] = ood_result

        # Clean up
        try:
            os.remove(filepath)
            if was_dicom:
                os.remove(processed_path)
        except:
            pass

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Analyze multiple chest X-ray images in batch
    Upload up to 20 chest X-ray images for batch analysis.
    TTA is disabled for batch mode to improve speed.
    ---
    tags:
      - Prediction
    consumes:
      - multipart/form-data
    parameters:
      - name: files
        in: formData
        type: file
        required: true
        description: Multiple chest X-ray images (max 20)
    responses:
      200:
        description: Batch analysis results
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  filename:
                    type: string
                  prediction:
                    type: string
                  confidence_pct:
                    type: string
                  is_ood:
                    type: boolean
                  error:
                    type: string
            total:
              type: integer
      400:
        description: No files or too many files
    """
    files = request.files.getlist('files')
    if not files or len(files) == 0:
        return jsonify({'error': 'No files uploaded'}), 400

    if len(files) > 20:
        return jsonify({'error': 'Maximum 20 files per batch'}), 400

    results = []
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            results.append({
                'filename': file.filename or 'unknown',
                'error': 'Invalid file type',
                'prediction': None
            })
            continue

        try:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # OOD check for batch
            try:
                ood_result = compute_ood_score(filepath)
                ood_flag = ood_result['is_ood']
            except Exception:
                ood_flag = False
                ood_result = None

            start_time = time.time()
            result = predict_xray(filepath, use_tta=False)  # No TTA for batch (speed)
            result['inference_time'] = f"{time.time() - start_time:.2f}s"
            result['filename'] = file.filename
            result['is_ood'] = ood_flag
            if ood_result:
                result['ood_detection'] = ood_result

            try: os.remove(filepath)
            except: pass

            results.append(result)
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
                'prediction': None
            })

    return jsonify({'results': results, 'total': len(results)})


@app.route('/api/health')
def health():
    """Check API health and model status
    Returns server status, loaded models, device info, and supported classes.
    ---
    tags:
      - Information
    responses:
      200:
        description: API health status
        schema:
          type: object
          properties:
            status:
              type: string
              example: "healthy"
            model_loaded:
              type: boolean
            ensemble_enabled:
              type: boolean
            models:
              type: array
              items:
                type: string
              example: ["ResNet18", "DenseNet121"]
            device:
              type: string
              example: "cuda"
            classes:
              type: array
              items:
                type: string
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'ensemble_enabled': ENSEMBLE_ENABLED,
        'models': ['ResNet18'] + (['DenseNet121'] if ENSEMBLE_ENABLED else []),
        'device': str(DEVICE),
        'classes': [CLASS_MAP.get(c, c) for c in class_names]
    })


@app.route('/api/disease-info/<disease>')
def get_disease_info(disease):
    """Get detailed disease information
    Retrieve comprehensive information about a specific disease including
    symptoms, recommendations, emergency guidance, and treatment overview.
    ---
    tags:
      - Information
    parameters:
      - name: disease
        in: path
        type: string
        required: true
        enum: ["COVID-19", "Healthy", "Pneumonia", "Tuberculosis"]
        description: Disease name
    responses:
      200:
        description: Disease information
        schema:
          type: object
          properties:
            name:
              type: string
            severity:
              type: string
            description:
              type: string
            symptoms:
              type: array
              items:
                type: string
            recommendations:
              type: array
              items:
                type: string
            when_to_seek_emergency:
              type: array
              items:
                type: string
            treatment_overview:
              type: string
      404:
        description: Disease not found
    """
    info = DISEASE_INFO.get(disease)
    if info:
        return jsonify(info)
    return jsonify({'error': 'Disease not found'}), 404


# ============================================================
# PDF REPORT GENERATION
# ============================================================
@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate a PDF analysis report
    Generate a professional PDF report from X-ray analysis results.
    Includes patient info, diagnosis, probabilities, symptoms, and recommendations.
    ---
    tags:
      - Reports
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - prediction
            - confidence_pct
            - all_predictions
          properties:
            prediction:
              type: string
              example: "COVID-19"
            confidence_pct:
              type: string
              example: "95.4%"
            all_predictions:
              type: array
              items:
                type: object
            disease_info:
              type: object
            patient_info:
              type: object
              properties:
                name:
                  type: string
                age:
                  type: string
                gender:
                  type: string
                notes:
                  type: string
            timestamp:
              type: string
            device_used:
              type: string
    produces:
      - application/pdf
    responses:
      200:
        description: PDF report file
        schema:
          type: file
      400:
        description: No data provided
      500:
        description: Report generation failed
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
        from reportlab.lib.units import inch, mm
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=30*mm, bottomMargin=20*mm,
                                leftMargin=20*mm, rightMargin=20*mm)

        styles = getSampleStyleSheet()
        elements = []

        # Custom styles
        title_style = ParagraphStyle('Title2', parent=styles['Title'],
            fontSize=22, textColor=colors.HexColor('#1e3a5f'), spaceAfter=4)
        subtitle_style = ParagraphStyle('Subtitle2', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#64748b'), alignment=TA_CENTER, spaceAfter=12)
        heading_style = ParagraphStyle('Heading2Custom', parent=styles['Heading2'],
            fontSize=14, textColor=colors.HexColor('#1e3a5f'), spaceBefore=16, spaceAfter=8)
        body_style = ParagraphStyle('Body2', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#334155'), leading=14)
        small_style = ParagraphStyle('Small2', parent=styles['Normal'],
            fontSize=8, textColor=colors.HexColor('#94a3b8'), leading=10)

        # Header
        elements.append(Paragraph("MedScan AI — Chest X-Ray Analysis Report", title_style))
        elements.append(Paragraph(f"Generated: {data.get('timestamp', 'N/A')} | Device: {data.get('device_used', 'N/A')}", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        elements.append(Spacer(1, 12))

        # Patient info if available
        patient = data.get('patient_info', {})
        if patient and any(patient.values()):
            elements.append(Paragraph("Patient Information", heading_style))
            p_rows = []
            if patient.get('name'): p_rows.append(['Name:', patient['name']])
            if patient.get('age'): p_rows.append(['Age:', patient['age']])
            if patient.get('gender'): p_rows.append(['Gender:', patient['gender']])
            if patient.get('notes'): p_rows.append(['Notes:', patient['notes']])
            if p_rows:
                t = Table(p_rows, colWidths=[80, 400])
                t.setStyle(TableStyle([
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
                    ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1e293b')),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                elements.append(t)
            elements.append(Spacer(1, 8))

        # Diagnosis result
        prediction = data.get('prediction', 'N/A')
        confidence_pct = data.get('confidence_pct', 'N/A')
        severity_map = {'COVID-19': 'High', 'Pneumonia': 'Moderate to High', 'Tuberculosis': 'High', 'Healthy': 'None'}
        severity = severity_map.get(prediction, 'N/A')

        elements.append(Paragraph("Diagnosis Result", heading_style))
        result_data = [
            ['Prediction:', prediction],
            ['Confidence:', confidence_pct],
            ['Severity:', severity],
        ]
        t = Table(result_data, colWidths=[100, 380])
        t.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1e293b')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 8))

        # Probability breakdown
        elements.append(Paragraph("Probability Breakdown", heading_style))
        prob_data = [['Class', 'Probability']]
        for p in data.get('all_predictions', []):
            prob_data.append([p['class'], p['percentage']])
        t = Table(prob_data, colWidths=[240, 240])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 8))

        # Disease info
        disease = data.get('disease_info', {})
        if disease.get('description'):
            elements.append(Paragraph("Clinical Description", heading_style))
            elements.append(Paragraph(disease['description'], body_style))
            elements.append(Spacer(1, 6))

        if disease.get('symptoms'):
            elements.append(Paragraph("Common Symptoms", heading_style))
            for s in disease['symptoms']:
                elements.append(Paragraph(f"  •  {s}", body_style))
            elements.append(Spacer(1, 6))

        if disease.get('recommendations'):
            elements.append(Paragraph("Recommendations", heading_style))
            for r in disease['recommendations']:
                elements.append(Paragraph(f"  •  {r}", body_style))
            elements.append(Spacer(1, 6))

        if disease.get('treatment_overview'):
            elements.append(Paragraph("Treatment Overview", heading_style))
            elements.append(Paragraph(disease['treatment_overview'], body_style))
            elements.append(Spacer(1, 6))

        if disease.get('when_to_seek_emergency') and len(disease['when_to_seek_emergency']) > 0:
            elements.append(Paragraph("⚠ Seek Emergency Care If:", heading_style))
            for e in disease['when_to_seek_emergency']:
                elements.append(Paragraph(f"  •  {e}", body_style))
            elements.append(Spacer(1, 6))

        # Disclaimer
        elements.append(Spacer(1, 16))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        elements.append(Spacer(1, 8))
        disclaimer = ("MEDICAL DISCLAIMER: This AI-generated report is for informational and screening purposes only. "
                       "It is NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare "
                       "provider for accurate diagnosis and treatment. Model accuracy: 91.4%.")
        elements.append(Paragraph(disclaimer, small_style))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(f"© 2026 MedScan AI — Report ID: {uuid.uuid4().hex[:12].upper()}", small_style))

        doc.build(elements)
        buffer.seek(0)

        return send_file(buffer, mimetype='application/pdf',
                         as_attachment=True,
                         download_name=f'MedScan_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')

    except Exception as e:
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500


# ============================================================
# HISTORY API (SQLite)
# ============================================================
@app.route('/api/history', methods=['GET'])
def get_history():
    """Get scan history from database
    Retrieve past X-ray analysis results, ordered by most recent first.
    If authenticated, returns only your scans. Otherwise returns all.
    ---
    tags:
      - History
    parameters:
      - name: Authorization
        in: header
        type: string
        required: false
        description: "Bearer <JWT token> — filter by your scans"
      - name: limit
        in: query
        type: integer
        required: false
        default: 50
        description: Maximum number of records to return
    responses:
      200:
        description: List of scan history entries
        schema:
          type: object
          properties:
            history:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: integer
                  prediction:
                    type: string
                  confidence:
                    type: number
                  confidence_pct:
                    type: string
                  timestamp:
                    type: string
            total:
              type: integer
    """
    limit = request.args.get('limit', 50, type=int)
    user = get_current_user()

    conn = get_db()
    if user:
        rows = conn.execute(
            "SELECT * FROM scan_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user['id'], limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM scan_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()

    history = []
    for row in rows:
        entry = dict(row)
        # Parse JSON fields
        if entry.get('all_predictions'):
            try:
                entry['all_predictions'] = json.loads(entry['all_predictions'])
            except (json.JSONDecodeError, TypeError):
                pass
        if entry.get('quality_warnings'):
            try:
                entry['quality_warnings'] = json.loads(entry['quality_warnings'])
            except (json.JSONDecodeError, TypeError):
                pass
        entry['is_ood'] = bool(entry.get('is_ood', 0))
        history.append(entry)

    return jsonify({'history': history, 'total': len(history)})


@app.route('/api/history', methods=['POST'])
def save_history():
    """Save a scan result to history
    Store an X-ray analysis result in the database for future reference.
    ---
    tags:
      - History
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - prediction
            - confidence
            - confidence_pct
            - timestamp
          properties:
            prediction:
              type: string
            confidence:
              type: number
            confidence_pct:
              type: string
            all_predictions:
              type: array
              items:
                type: object
            inference_time:
              type: string
            inference_mode:
              type: string
            device_used:
              type: string
            is_ood:
              type: boolean
            ood_score:
              type: number
            patient_name:
              type: string
            patient_age:
              type: string
            patient_gender:
              type: string
            patient_notes:
              type: string
            quality_warnings:
              type: array
              items:
                type: string
            timestamp:
              type: string
    responses:
      201:
        description: History entry created
        schema:
          type: object
          properties:
            id:
              type: integer
            message:
              type: string
      400:
        description: Missing required fields
    """
    data = request.get_json()
    if not data or 'prediction' not in data:
        return jsonify({'error': 'Missing prediction data'}), 400

    user = get_current_user()
    user_id = user['id'] if user else None

    conn = get_db()
    cursor = conn.execute("""
        INSERT INTO scan_history
            (user_id, prediction, confidence, confidence_pct, all_predictions,
             inference_time, inference_mode, device_used,
             is_ood, ood_score,
             patient_name, patient_age, patient_gender, patient_notes,
             quality_warnings, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        data.get('prediction', ''),
        data.get('confidence', 0),
        data.get('confidence_pct', ''),
        json.dumps(data.get('all_predictions', [])),
        data.get('inference_time', ''),
        data.get('inference_mode', 'single'),
        data.get('device_used', ''),
        1 if data.get('is_ood') else 0,
        data.get('ood_score', 0),
        data.get('patient_name', ''),
        data.get('patient_age', ''),
        data.get('patient_gender', ''),
        data.get('patient_notes', ''),
        json.dumps(data.get('quality_warnings', [])),
        data.get('timestamp', datetime.now().strftime('%B %d, %Y at %I:%M %p'))
    ))
    conn.commit()
    entry_id = cursor.lastrowid
    conn.close()

    return jsonify({'id': entry_id, 'message': 'Saved to history'}), 201


@app.route('/api/history/<int:entry_id>', methods=['DELETE'])
def delete_history_entry(entry_id):
    """Delete a single history entry
    Remove a specific scan from the history database.
    ---
    tags:
      - History
    parameters:
      - name: entry_id
        in: path
        type: integer
        required: true
        description: ID of the history entry to delete
    responses:
      200:
        description: Entry deleted
      404:
        description: Entry not found
    """
    conn = get_db()
    user = get_current_user()
    if user:
        result = conn.execute("DELETE FROM scan_history WHERE id = ? AND user_id = ?", (entry_id, user['id']))
    else:
        result = conn.execute("DELETE FROM scan_history WHERE id = ?", (entry_id,))
    conn.commit()
    deleted = result.rowcount
    conn.close()

    if deleted:
        return jsonify({'message': 'Entry deleted'})
    return jsonify({'error': 'Entry not found'}), 404


@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear all history
    Delete all scan history entries from the database.
    ---
    tags:
      - History
    responses:
      200:
        description: All history cleared
        schema:
          type: object
          properties:
            message:
              type: string
            deleted:
              type: integer
    """
    conn = get_db()
    user = get_current_user()
    if user:
        result = conn.execute("DELETE FROM scan_history WHERE user_id = ?", (user['id'],))
    else:
        result = conn.execute("DELETE FROM scan_history")
    conn.commit()
    deleted = result.rowcount
    conn.close()
    return jsonify({'message': 'History cleared', 'deleted': deleted})


@app.route('/api/history/export/json')
def export_history_json():
    """Export history as JSON file
    Download all scan history as a JSON file.
    ---
    tags:
      - History
    produces:
      - application/json
    responses:
      200:
        description: JSON file download
    """
    conn = get_db()
    user = get_current_user()
    if user:
        rows = conn.execute("SELECT * FROM scan_history WHERE user_id = ? ORDER BY id DESC", (user['id'],)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM scan_history ORDER BY id DESC").fetchall()
    conn.close()

    history = []
    for row in rows:
        entry = dict(row)
        if entry.get('all_predictions'):
            try:
                entry['all_predictions'] = json.loads(entry['all_predictions'])
            except (json.JSONDecodeError, TypeError):
                pass
        if entry.get('quality_warnings'):
            try:
                entry['quality_warnings'] = json.loads(entry['quality_warnings'])
            except (json.JSONDecodeError, TypeError):
                pass
        history.append(entry)

    buffer = io.BytesIO()
    buffer.write(json.dumps(history, indent=2).encode('utf-8'))
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/json',
        as_attachment=True,
        download_name=f'medscan_history_{datetime.now().strftime("%Y%m%d")}.json'
    )


@app.route('/api/history/export/csv')
def export_history_csv():
    """Export history as CSV file
    Download all scan history as a CSV spreadsheet.
    ---
    tags:
      - History
    produces:
      - text/csv
    responses:
      200:
        description: CSV file download
    """
    import csv

    conn = get_db()
    user = get_current_user()
    if user:
        rows = conn.execute("SELECT * FROM scan_history WHERE user_id = ? ORDER BY id DESC", (user['id'],)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM scan_history ORDER BY id DESC").fetchall()
    conn.close()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    # Header
    writer.writerow([
        'ID', 'Prediction', 'Confidence', 'Confidence %',
        'Inference Time', 'Inference Mode', 'Device',
        'OOD Detected', 'OOD Score',
        'Patient Name', 'Patient Age', 'Patient Gender', 'Patient Notes',
        'Timestamp', 'Created At'
    ])
    for row in rows:
        entry = dict(row)
        writer.writerow([
            entry['id'], entry['prediction'], entry['confidence'],
            entry['confidence_pct'], entry['inference_time'],
            entry['inference_mode'], entry['device_used'],
            'Yes' if entry['is_ood'] else 'No', entry.get('ood_score', ''),
            entry.get('patient_name', ''), entry.get('patient_age', ''),
            entry.get('patient_gender', ''), entry.get('patient_notes', ''),
            entry['timestamp'], entry['created_at']
        ])

    output = io.BytesIO()
    output.write(buffer.getvalue().encode('utf-8'))
    output.seek(0)

    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'medscan_history_{datetime.now().strftime("%Y%m%d")}.csv'
    )


# ============================================================
# ANALYTICS API (Admin Dashboard)
# ============================================================

@app.route('/api/analytics/summary')
def analytics_summary():
    """Get analytics summary data
    Returns total scans, disease distribution, average confidence,
    OOD rate, and other aggregate statistics. Admin-only.
    ---
    tags:
      - Analytics
    parameters:
      - name: Authorization
        in: header
        type: string
        required: true
        description: "Bearer <JWT token> (admin only)"
    responses:
      200:
        description: Analytics summary
        schema:
          type: object
          properties:
            total_scans:
              type: integer
            disease_distribution:
              type: object
            avg_confidence:
              type: number
            ood_rate:
              type: number
            total_users:
              type: integer
            scans_today:
              type: integer
      401:
        description: Not authenticated
      403:
        description: Admin access required
    """
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    if not user.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403

    conn = get_db()

    # Total scans
    total_scans = conn.execute("SELECT COUNT(*) FROM scan_history").fetchone()[0]

    # Disease distribution
    dist_rows = conn.execute(
        "SELECT prediction, COUNT(*) as count FROM scan_history GROUP BY prediction ORDER BY count DESC"
    ).fetchall()
    disease_distribution = {row['prediction']: row['count'] for row in dist_rows}

    # Average confidence
    avg_conf = conn.execute("SELECT AVG(confidence) FROM scan_history").fetchone()[0]
    avg_confidence = round(avg_conf, 4) if avg_conf else 0

    # OOD rate
    ood_count = conn.execute("SELECT COUNT(*) FROM scan_history WHERE is_ood = 1").fetchone()[0]
    ood_rate = round(ood_count / total_scans, 4) if total_scans > 0 else 0

    # Total users
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    # Scans today
    today_str = datetime.now().strftime('%Y-%m-%d')
    scans_today = conn.execute(
        "SELECT COUNT(*) FROM scan_history WHERE DATE(created_at) = ?", (today_str,)
    ).fetchone()[0]

    # Inference mode distribution
    mode_rows = conn.execute(
        "SELECT inference_mode, COUNT(*) as count FROM scan_history GROUP BY inference_mode"
    ).fetchall()
    inference_modes = {row['inference_mode']: row['count'] for row in mode_rows}

    # Average confidence per disease
    avg_per_disease_rows = conn.execute(
        "SELECT prediction, AVG(confidence) as avg_conf FROM scan_history GROUP BY prediction"
    ).fetchall()
    avg_confidence_per_disease = {row['prediction']: round(row['avg_conf'], 4) for row in avg_per_disease_rows}

    conn.close()

    return jsonify({
        'total_scans': total_scans,
        'disease_distribution': disease_distribution,
        'avg_confidence': avg_confidence,
        'ood_rate': ood_rate,
        'ood_count': ood_count,
        'total_users': total_users,
        'scans_today': scans_today,
        'inference_modes': inference_modes,
        'avg_confidence_per_disease': avg_confidence_per_disease,
    })


@app.route('/api/analytics/timeline')
def analytics_timeline():
    """Get scans over time data
    Returns daily scan counts, confidence trends, and disease trends.
    Admin-only.
    ---
    tags:
      - Analytics
    parameters:
      - name: Authorization
        in: header
        type: string
        required: true
        description: "Bearer <JWT token> (admin only)"
      - name: days
        in: query
        type: integer
        required: false
        default: 30
        description: Number of days of history
    responses:
      200:
        description: Timeline data
        schema:
          type: object
          properties:
            daily_scans:
              type: array
              items:
                type: object
                properties:
                  date:
                    type: string
                  count:
                    type: integer
            daily_confidence:
              type: array
              items:
                type: object
                properties:
                  date:
                    type: string
                  avg_confidence:
                    type: number
            disease_trend:
              type: object
      401:
        description: Not authenticated
      403:
        description: Admin access required
    """
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    if not user.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403

    days = request.args.get('days', 30, type=int)
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    conn = get_db()

    # Daily scan counts
    daily_rows = conn.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM scan_history
        WHERE DATE(created_at) >= ?
        GROUP BY DATE(created_at)
        ORDER BY date ASC
    """, (cutoff,)).fetchall()
    daily_scans = [{'date': row['date'], 'count': row['count']} for row in daily_rows]

    # Daily average confidence
    conf_rows = conn.execute("""
        SELECT DATE(created_at) as date, AVG(confidence) as avg_confidence
        FROM scan_history
        WHERE DATE(created_at) >= ?
        GROUP BY DATE(created_at)
        ORDER BY date ASC
    """, (cutoff,)).fetchall()
    daily_confidence = [{'date': row['date'], 'avg_confidence': round(row['avg_confidence'], 4)} for row in conf_rows]

    # Disease trend (daily counts per disease)
    trend_rows = conn.execute("""
        SELECT DATE(created_at) as date, prediction, COUNT(*) as count
        FROM scan_history
        WHERE DATE(created_at) >= ?
        GROUP BY DATE(created_at), prediction
        ORDER BY date ASC
    """, (cutoff,)).fetchall()

    disease_trend = {}
    for row in trend_rows:
        disease = row['prediction']
        if disease not in disease_trend:
            disease_trend[disease] = []
        disease_trend[disease].append({'date': row['date'], 'count': row['count']})

    # Recent scans (last 10)
    recent_rows = conn.execute("""
        SELECT id, prediction, confidence_pct, patient_name, timestamp, is_ood, created_at
        FROM scan_history
        ORDER BY id DESC LIMIT 10
    """).fetchall()
    recent_scans = []
    for row in recent_rows:
        entry = dict(row)
        entry['is_ood'] = bool(entry.get('is_ood', 0))
        recent_scans.append(entry)

    conn.close()

    return jsonify({
        'daily_scans': daily_scans,
        'daily_confidence': daily_confidence,
        'disease_trend': disease_trend,
        'recent_scans': recent_scans,
        'period_days': days,
    })


@app.route('/api/analytics/users')
def analytics_users():
    """Get user statistics
    Returns list of users with their scan counts. Admin-only.
    ---
    tags:
      - Analytics
    parameters:
      - name: Authorization
        in: header
        type: string
        required: true
        description: "Bearer <JWT token> (admin only)"
    responses:
      200:
        description: User statistics
      401:
        description: Not authenticated
      403:
        description: Admin access required
    """
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    if not user.get('is_admin'):
        return jsonify({'error': 'Admin access required'}), 403

    conn = get_db()
    user_rows = conn.execute("""
        SELECT u.id, u.username, u.email, u.is_admin, u.created_at,
               COUNT(s.id) as scan_count
        FROM users u
        LEFT JOIN scan_history s ON s.user_id = u.id
        GROUP BY u.id
        ORDER BY scan_count DESC
    """).fetchall()
    conn.close()

    users = []
    for row in user_rows:
        entry = dict(row)
        entry['is_admin'] = bool(entry.get('is_admin', 0))
        users.append(entry)

    return jsonify({'users': users, 'total': len(users)})


# ============================================================
# CHATBOT
# ============================================================
CHATBOT_KNOWLEDGE = {
    'covid': {
        'keywords': ['covid', 'corona', 'sars', 'covid-19', 'covid19'],
        'responses': [
            {
                'topic': 'COVID-19 Overview',
                'text': 'COVID-19 is caused by the SARS-CoV-2 virus. It primarily affects the respiratory system. '
                        'On chest X-rays, it typically shows bilateral ground-glass opacities and consolidation, '
                        'especially in the lower lobes and peripheral regions.',
                'follow_up': 'Would you like to know about symptoms, treatment, or prevention?'
            }
        ]
    },
    'covid_symptoms': {
        'keywords': ['covid symptom', 'corona symptom', 'covid sign'],
        'responses': [
            {
                'topic': 'COVID-19 Symptoms',
                'text': 'Common COVID-19 symptoms include:\n'
                        '• Fever or chills\n'
                        '• Persistent dry cough\n'
                        '• Shortness of breath\n'
                        '• Fatigue and body aches\n'
                        '• Loss of taste or smell\n'
                        '• Headache and sore throat\n\n'
                        '⚠️ Seek emergency care if you experience difficulty breathing, '
                        'persistent chest pain, confusion, or bluish lips.',
                'follow_up': ''
            }
        ]
    },
    'pneumonia': {
        'keywords': ['pneumonia', 'lung infection'],
        'responses': [
            {
                'topic': 'Pneumonia Information',
                'text': 'Pneumonia is an infection that inflames the air sacs in one or both lungs. '
                        'It can be caused by bacteria, viruses, or fungi.\n\n'
                        'On X-rays, pneumonia appears as areas of white opacity (consolidation), '
                        'air bronchograms, and sometimes pleural effusion.\n\n'
                        'Common symptoms include cough with phlegm, fever, chills, '
                        'and difficulty breathing.',
                'follow_up': 'Would you like to know about treatment options?'
            }
        ]
    },
    'pneumonia_treatment': {
        'keywords': ['pneumonia treat', 'pneumonia medicine', 'pneumonia cure'],
        'responses': [
            {
                'topic': 'Pneumonia Treatment',
                'text': 'Treatment depends on the type and severity:\n\n'
                        '🦠 Bacterial: Antibiotics (amoxicillin, azithromycin)\n'
                        '🦠 Viral: Antivirals + supportive care\n'
                        '🦠 Fungal: Antifungal medications\n\n'
                        'General care includes rest, fluids, fever reducers, '
                        'and a follow-up X-ray to confirm resolution.\n\n'
                        '⚠️ Severe cases may require hospitalization with IV antibiotics and oxygen.',
                'follow_up': ''
            }
        ]
    },
    'tb': {
        'keywords': ['tuberculosis', 'tb ', 'mycobacterium'],
        'responses': [
            {
                'topic': 'Tuberculosis (TB)',
                'text': 'Tuberculosis is a serious bacterial infection caused by Mycobacterium tuberculosis. '
                        'It primarily affects the lungs but can spread to other organs.\n\n'
                        'X-ray findings include upper lobe infiltrates, cavitation, '
                        'hilar lymphadenopathy, and pleural effusion.\n\n'
                        'Key symptoms: persistent cough (>3 weeks), night sweats, '
                        'weight loss, coughing blood, and fatigue.',
                'follow_up': 'Would you like to know about TB treatment?'
            }
        ]
    },
    'tb_treatment': {
        'keywords': ['tb treat', 'tuberculosis treat', 'tb medicine', 'tb cure', 'tb drug'],
        'responses': [
            {
                'topic': 'TB Treatment',
                'text': 'Standard TB treatment is a 6-month regimen:\n\n'
                        '📋 Phase 1 (2 months): Isoniazid + Rifampicin + Pyrazinamide + Ethambutol (HRZE)\n'
                        '📋 Phase 2 (4 months): Isoniazid + Rifampicin (HR)\n\n'
                        '⚠️ CRITICAL: Complete ALL medications as prescribed. '
                        'Stopping early leads to drug-resistant TB which is much harder to treat.\n\n'
                        'Directly Observed Therapy (DOT) is recommended to ensure compliance.',
                'follow_up': ''
            }
        ]
    },
    'healthy': {
        'keywords': ['healthy', 'normal', 'clear', 'no disease', 'good result'],
        'responses': [
            {
                'topic': 'Healthy Lungs',
                'text': 'A normal/healthy chest X-ray shows clear lung fields, '
                        'normal heart size, no consolidation or effusion.\n\n'
                        'To maintain lung health:\n'
                        '• Don\'t smoke and avoid secondhand smoke\n'
                        '• Exercise regularly\n'
                        '• Stay up to date on vaccinations\n'
                        '• Practice good respiratory hygiene\n'
                        '• Get annual health checkups',
                'follow_up': ''
            }
        ]
    },
    'xray_result': {
        'keywords': ['result', 'x-ray', 'xray', 'scan', 'report', 'diagnosis', 'what does'],
        'responses': [
            {
                'topic': 'Understanding Your Results',
                'text': 'Our AI model analyzes chest X-rays for 4 conditions:\n\n'
                        '1️⃣ **COVID-19** — Ground-glass opacities, bilateral involvement\n'
                        '2️⃣ **Pneumonia** — Consolidation, air bronchograms\n'
                        '3️⃣ **Tuberculosis** — Upper lobe cavitation, infiltrates\n'
                        '4️⃣ **Healthy** — Clear lung fields, no abnormalities\n\n'
                        'The confidence percentage shows how certain the AI is. '
                        'Results above 90% are considered high confidence.\n\n'
                        '⚠️ Always consult a doctor for a definitive diagnosis.',
                'follow_up': 'Upload an X-ray above to get your analysis!'
            }
        ]
    },
    'accuracy': {
        'keywords': ['accuracy', 'reliable', 'trust', 'correct', 'wrong', 'mistake'],
        'responses': [
            {
                'topic': 'Model Accuracy',
                'text': 'Our model achieves 91.4% accuracy on the test dataset:\n\n'
                        '• COVID-19: 99% precision, 100% recall (F1: 1.00)\n'
                        '• Pneumonia: 89% precision, 99% recall (F1: 0.94)\n'
                        '• Tuberculosis: 76% precision, 100% recall (F1: 0.86)\n'
                        '• Healthy: 97% precision, 74% recall (F1: 0.84)\n\n'
                        'While these results are strong, this tool is for screening purposes only. '
                        'Always confirm findings with a qualified radiologist or physician.',
                'follow_up': ''
            }
        ]
    },
    'emergency': {
        'keywords': ['emergency', 'urgent', 'danger', 'hospital', 'er ', 'ambulance', 'serious', 'dying'],
        'responses': [
            {
                'topic': '🚨 Emergency Guidance',
                'text': '🚨 If you or someone is experiencing a medical emergency:\n\n'
                        '📞 Call emergency services immediately (911 / 112 / 108)\n\n'
                        'Seek IMMEDIATE medical care for:\n'
                        '• Severe difficulty breathing\n'
                        '• Persistent chest pain or pressure\n'
                        '• Confusion or inability to stay awake\n'
                        '• Bluish lips or face\n'
                        '• Coughing up large amounts of blood\n\n'
                        '⚠️ Do not rely on this AI tool in an emergency. Call for help NOW.',
                'follow_up': ''
            }
        ]
    },
    'prevention': {
        'keywords': ['prevent', 'avoid', 'protect', 'vaccine', 'vaccination', 'mask'],
        'responses': [
            {
                'topic': 'Disease Prevention',
                'text': 'Key respiratory disease prevention measures:\n\n'
                        '💉 Vaccinations: Stay updated on COVID-19, flu, and pneumococcal vaccines\n'
                        '😷 Wear masks in crowded/high-risk settings\n'
                        '🧼 Wash hands frequently with soap for 20+ seconds\n'
                        '🏠 Ensure good ventilation in indoor spaces\n'
                        '🚭 Don\'t smoke — smoking damages lungs severely\n'
                        '🏃 Exercise regularly to strengthen lung capacity\n'
                        '🍎 Eat a balanced, nutrient-rich diet\n'
                        '😴 Get adequate sleep (7-9 hours)',
                'follow_up': ''
            }
        ]
    }
}


def get_chatbot_response(message):
    """Generate a chatbot response based on keyword matching."""
    msg_lower = message.lower().strip()

    # Check greetings
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    if any(msg_lower.startswith(g) or msg_lower == g for g in greetings):
        return {
            'topic': 'Welcome!',
            'text': 'Hello! 👋 I\'m your MedScan AI medical assistant. '
                    'I can help you understand chest X-ray results and provide '
                    'information about respiratory diseases.\n\n'
                    'Try asking me about:\n'
                    '• COVID-19, Pneumonia, or Tuberculosis\n'
                    '• How to read your X-ray results\n'
                    '• Disease prevention tips\n'
                    '• When to seek emergency care',
            'follow_up': ''
        }

    # Check thanks
    if any(w in msg_lower for w in ['thank', 'thanks', 'appreciate']):
        return {
            'topic': '',
            'text': 'You\'re welcome! 😊 Feel free to ask if you have any other questions. '
                    'Remember, always consult a healthcare professional for medical decisions.',
            'follow_up': ''
        }

    # Score each topic by keyword matches
    best_score = 0
    best_response = None

    for topic_key, topic_data in CHATBOT_KNOWLEDGE.items():
        score = 0
        for keyword in topic_data['keywords']:
            if keyword in msg_lower:
                score += len(keyword)  # Longer keyword matches = more specific
        if score > best_score:
            best_score = score
            best_response = topic_data['responses'][0]

    if best_response:
        return best_response

    # Default fallback
    return {
        'topic': '',
        'text': 'I\'m not sure I understand that question. I can help with:\n\n'
                '• **COVID-19** — symptoms, treatment, prevention\n'
                '• **Pneumonia** — causes, signs, treatment\n'
                '• **Tuberculosis** — diagnosis, treatment regimen\n'
                '• **X-ray results** — understanding your scan\n'
                '• **Emergency guidance** — when to seek urgent care\n\n'
                'Try rephrasing your question or pick a topic above!',
        'follow_up': ''
    }


@app.route('/api/chat', methods=['POST'])
def chat():
    """Send a message to the medical assistant chatbot
    Ask questions about respiratory diseases, X-ray results, symptoms, treatment, or prevention.
    ---
    tags:
      - Chatbot
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - message
          properties:
            message:
              type: string
              example: "What are the symptoms of COVID-19?"
    responses:
      200:
        description: Chatbot response
        schema:
          type: object
          properties:
            response:
              type: string
              description: The chatbot's reply text
            topic:
              type: string
              description: Topic heading if applicable
            follow_up:
              type: string
              description: Follow-up suggestion
      400:
        description: No message or empty message
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    message = data['message'].strip()
    if not message:
        return jsonify({'error': 'Empty message'}), 400

    response = get_chatbot_response(message)
    return jsonify({
        'response': response['text'],
        'topic': response.get('topic', ''),
        'follow_up': response.get('follow_up', '')
    })


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  MedScan AI - Chest X-Ray Analysis")
    print("  App:        http://localhost:5000")
    print("  Dashboard:  http://localhost:5000/dashboard")
    print("  API Docs:   http://localhost:5000/api/docs")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
