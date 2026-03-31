/**
 * MedScan AI - Chest X-Ray Disease Detection
 * Frontend Application Logic
 */

// ============================================================
// DARK MODE (runs immediately to prevent flash)
// ============================================================
(function() {
    const saved = localStorage.getItem('medscan-theme');
    if (saved === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
})();

document.addEventListener('DOMContentLoaded', () => {
    // ============================================================
    // AUTHENTICATION
    // ============================================================
    const AUTH_TOKEN_KEY = 'medscan-auth-token';
    const AUTH_USER_KEY = 'medscan-auth-user';

    function getToken() { return localStorage.getItem(AUTH_TOKEN_KEY); }
    function getUser() {
        try { return JSON.parse(localStorage.getItem(AUTH_USER_KEY)); }
        catch { return null; }
    }
    function setAuth(token, user) {
        localStorage.setItem(AUTH_TOKEN_KEY, token);
        localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
        updateAuthUI();
    }
    function clearAuth() {
        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(AUTH_USER_KEY);
        updateAuthUI();
    }
    function authHeaders() {
        const token = getToken();
        if (token) return { 'Authorization': `Bearer ${token}` };
        return {};
    }

    // Auth UI elements
    const loginNavBtn = document.getElementById('loginNavBtn');
    const userDropdown = document.getElementById('userDropdown');
    const userMenuBtn = document.getElementById('userMenuBtn');
    const dropdownMenu = document.getElementById('dropdownMenu');
    const logoutBtn = document.getElementById('logoutBtn');
    const authModal = document.getElementById('authModal');
    const authModalClose = document.getElementById('authModalClose');
    const dashboardLink = document.getElementById('dashboardLink');
    const dropdownDashboard = document.getElementById('dropdownDashboard');

    function updateAuthUI() {
        const user = getUser();
        if (user) {
            loginNavBtn.style.display = 'none';
            userDropdown.style.display = 'flex';
            document.getElementById('usernameDisplay').textContent = user.username;
            document.getElementById('dropdownUserInfo').innerHTML =
                `<span class="dropdown-email">${user.email}</span>` +
                (user.is_admin ? '<span class="admin-badge"><i class="bi bi-shield-check"></i> Admin</span>' : '');
            if (user.is_admin) {
                dashboardLink.style.display = '';
                dropdownDashboard.style.display = '';
            } else {
                dashboardLink.style.display = 'none';
                dropdownDashboard.style.display = 'none';
            }
        } else {
            loginNavBtn.style.display = 'flex';
            userDropdown.style.display = 'none';
            dashboardLink.style.display = 'none';
        }
    }

    // Show/hide dropdown menu
    if (userMenuBtn) {
        userMenuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            dropdownMenu.style.display = dropdownMenu.style.display === 'none' ? 'block' : 'none';
        });
        document.addEventListener('click', () => { dropdownMenu.style.display = 'none'; });
    }

    // Logout
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            clearAuth();
            dropdownMenu.style.display = 'none';
            renderHistory();
        });
    }

    // Open auth modal
    if (loginNavBtn) {
        loginNavBtn.addEventListener('click', () => {
            authModal.style.display = 'flex';
        });
    }
    if (authModalClose) {
        authModalClose.addEventListener('click', () => {
            authModal.style.display = 'none';
        });
    }
    // Close modal on overlay click
    if (authModal) {
        authModal.addEventListener('click', (e) => {
            if (e.target === authModal) authModal.style.display = 'none';
        });
    }

    // Auth tabs
    document.querySelectorAll('.auth-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const panel = tab.dataset.tab;
            document.getElementById('loginPanel').style.display = panel === 'login' ? 'block' : 'none';
            document.getElementById('signupPanel').style.display = panel === 'signup' ? 'block' : 'none';
            // Clear errors
            document.getElementById('loginError').style.display = 'none';
            document.getElementById('signupError').style.display = 'none';
        });
    });

    // Login form
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const loginError = document.getElementById('loginError');
            loginError.style.display = 'none';
            try {
                const res = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        login: document.getElementById('loginId').value.trim(),
                        password: document.getElementById('loginPassword').value
                    })
                });
                const data = await res.json();
                if (res.ok) {
                    setAuth(data.token, data.user);
                    authModal.style.display = 'none';
                    loginForm.reset();
                    renderHistory();
                } else {
                    loginError.textContent = data.error || 'Login failed';
                    loginError.style.display = 'block';
                }
            } catch {
                loginError.textContent = 'Connection error. Please try again.';
                loginError.style.display = 'block';
            }
        });
    }

    // Signup form
    const signupForm = document.getElementById('signupForm');
    if (signupForm) {
        signupForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const signupError = document.getElementById('signupError');
            signupError.style.display = 'none';
            try {
                const res = await fetch('/api/auth/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: document.getElementById('signupUsername').value.trim(),
                        email: document.getElementById('signupEmail').value.trim(),
                        password: document.getElementById('signupPassword').value
                    })
                });
                const data = await res.json();
                if (res.ok) {
                    setAuth(data.token, data.user);
                    authModal.style.display = 'none';
                    signupForm.reset();
                    renderHistory();
                } else {
                    signupError.textContent = data.error || 'Registration failed';
                    signupError.style.display = 'block';
                }
            } catch {
                signupError.textContent = 'Connection error. Please try again.';
                signupError.style.display = 'block';
            }
        });
    }

    // Verify stored token on load
    async function verifyAuth() {
        const token = getToken();
        if (!token) { updateAuthUI(); return; }
        try {
            const res = await fetch('/api/auth/me', { headers: authHeaders() });
            if (res.ok) {
                const data = await res.json();
                localStorage.setItem(AUTH_USER_KEY, JSON.stringify(data.user));
                updateAuthUI();
            } else {
                clearAuth();
            }
        } catch {
            // Keep cached auth if server unavailable
            updateAuthUI();
        }
    }
    verifyAuth();

    // Theme toggle
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = document.getElementById('themeIcon');
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('medscan-theme', theme);
        themeIcon.className = theme === 'dark' ? 'bi bi-sun-fill' : 'bi bi-moon-fill';
    }
    if (localStorage.getItem('medscan-theme') === 'dark') {
        themeIcon.className = 'bi bi-sun-fill';
    }
    themeToggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        applyTheme(current === 'dark' ? 'light' : 'dark');
    });

    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadContent = document.getElementById('uploadContent');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const fileName = document.getElementById('fileName');
    const removeBtn = document.getElementById('removeBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingContainer = document.getElementById('loadingContainer');
    const resultsSection = document.getElementById('results-section');
    const uploadSection = document.getElementById('upload-section');
    const scanAgainBtn = document.getElementById('scanAgainBtn');

    let selectedFile = null;
    let lastResult = null;  // Store last analysis result for PDF/history

    // ============================================================
    // PATIENT INFO FORM
    // ============================================================
    const togglePatientBtn = document.getElementById('togglePatientForm');
    const patientForm = document.getElementById('patientForm');
    const patientChevron = document.getElementById('patientChevron');

    togglePatientBtn.addEventListener('click', () => {
        const visible = patientForm.style.display !== 'none';
        patientForm.style.display = visible ? 'none' : 'block';
        patientChevron.className = visible ? 'bi bi-chevron-down' : 'bi bi-chevron-up';
    });

    function getPatientInfo() {
        return {
            name: document.getElementById('patientName').value.trim(),
            age: document.getElementById('patientAge').value.trim(),
            gender: document.getElementById('patientGender').value,
            notes: document.getElementById('patientNotes').value.trim()
        };
    }

    // ============================================================
    // FILE HANDLING
    // ============================================================

    // Click to upload
    dropZone.addEventListener('click', (e) => {
        if (e.target === removeBtn || e.target.closest('.btn-remove')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Handle file selection
    function handleFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|bmp|tiff|webp|dcm)$/i)) {
            showError('Invalid file type. Please upload a chest X-ray image (JPG, PNG, BMP, TIFF, or WebP).');
            return;
        }

        // Validate file size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            showError('File too large. Maximum size is 16MB.');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            previewContainer.style.display = 'block';
            fileName.textContent = file.name;
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Remove file
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        uploadContent.style.display = 'block';
        analyzeBtn.disabled = true;
    }

    // ============================================================
    // ANALYSIS
    // ============================================================

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading
        analyzeBtn.style.display = 'none';
        loadingContainer.style.display = 'block';
        resultsSection.style.display = 'none';

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                displayResults(result);
            } else {
                showError(result.error || 'Analysis failed. Please try again.');
                loadingContainer.style.display = 'none';
                analyzeBtn.style.display = 'flex';
            }
        } catch (err) {
            showError('Connection error. Please make sure the server is running.');
            loadingContainer.style.display = 'none';
            analyzeBtn.style.display = 'flex';
        }
    });

    // ============================================================
    // DISPLAY RESULTS
    // ============================================================

    function displayResults(result) {
        loadingContainer.style.display = 'none';
        lastResult = result;  // Store for PDF download

        const info = result.disease_info;
        const prediction = result.prediction;

        // --- Result Header ---
        const resultIcon = document.getElementById('resultIcon');
        const iconMap = {
            'COVID-19': 'bi-virus',
            'Healthy': 'bi-heart-pulse',
            'Pneumonia': 'bi-lungs',
            'Tuberculosis': 'bi-bullseye'
        };
        const colorMap = {
            'COVID-19': '#ef4444',
            'Healthy': '#10b981',
            'Pneumonia': '#f59e0b',
            'Tuberculosis': '#8b5cf6'
        };
        const bgColorMap = {
            'COVID-19': 'rgba(239,68,68,0.2)',
            'Healthy': 'rgba(16,185,129,0.2)',
            'Pneumonia': 'rgba(245,158,11,0.2)',
            'Tuberculosis': 'rgba(139,92,246,0.2)'
        };

        resultIcon.innerHTML = `<i class="bi ${iconMap[prediction] || 'bi-lungs'}"></i>`;
        resultIcon.style.background = bgColorMap[prediction] || 'rgba(37,99,235,0.2)';
        resultIcon.style.color = colorMap[prediction] || 'white';

        document.getElementById('resultName').textContent = info.name || prediction;

        // Confidence badge
        const confBadge = document.getElementById('confidenceBadge');
        confBadge.textContent = `${result.confidence_pct} Confidence`;
        const confValue = result.confidence;
        if (confValue >= 0.9) {
            confBadge.style.background = 'rgba(16,185,129,0.2)';
            confBadge.style.color = '#10b981';
        } else if (confValue >= 0.7) {
            confBadge.style.background = 'rgba(245,158,11,0.2)';
            confBadge.style.color = '#f59e0b';
        } else {
            confBadge.style.background = 'rgba(239,68,68,0.2)';
            confBadge.style.color = '#ef4444';
        }

        document.getElementById('inferenceTime').textContent = `Inference: ${result.inference_time}`;
        document.getElementById('analysisDate').textContent = result.timestamp;

        // Ensemble badge
        let ensembleBadge = document.getElementById('ensembleBadge');
        if (!ensembleBadge) {
            ensembleBadge = document.createElement('span');
            ensembleBadge.id = 'ensembleBadge';
            ensembleBadge.className = 'ensemble-badge';
            document.querySelector('.result-meta').appendChild(ensembleBadge);
        }
        if (result.inference_mode === 'ensemble') {
            ensembleBadge.innerHTML = '<i class="bi bi-diagram-3"></i> Ensemble';
            ensembleBadge.title = 'ResNet18 + DenseNet121 ensemble prediction';
            ensembleBadge.style.display = 'inline-flex';
        } else {
            ensembleBadge.style.display = 'none';
        }

        // Severity
        const severityBar = document.getElementById('severityBar');
        const severityValue = document.getElementById('severityValue');
        severityValue.textContent = info.severity || 'N/A';
        severityValue.style.background = info.severity_color ? info.severity_color + '20' : '#eee';
        severityValue.style.color = info.severity_color || '#666';

        // Description
        document.getElementById('resultDescription').innerHTML = `<p>${info.description || ''}</p>`;

        // --- Probability Bars ---
        const probBars = document.getElementById('probBars');
        probBars.innerHTML = '';
        const barClassMap = {
            'COVID-19': 'bar-covid',
            'Healthy': 'bar-healthy',
            'Pneumonia': 'bar-pneumonia',
            'Tuberculosis': 'bar-tb'
        };

        result.all_predictions.forEach(pred => {
            const pct = (pred.probability * 100).toFixed(1);
            const barClass = barClassMap[pred.class] || 'bar-covid';
            const item = document.createElement('div');
            item.className = 'prob-item';
            item.innerHTML = `
                <div class="prob-label">
                    <span>${pred.class}</span>
                    <span>${pct}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill ${barClass}" style="width: 0%"></div>
                </div>
            `;
            probBars.appendChild(item);

            // Animate bar
            requestAnimationFrame(() => {
                setTimeout(() => {
                    item.querySelector('.prob-bar-fill').style.width = `${pct}%`;
                }, 100);
            });
        });

        // --- Symptoms ---
        const symptomsCard = document.getElementById('symptomsCard');
        const symptomsList = document.getElementById('symptomsList');
        if (info.symptoms && info.symptoms.length > 0) {
            symptomsCard.style.display = 'block';
            symptomsList.innerHTML = info.symptoms.map(s => `<li>${s}</li>`).join('');
        } else {
            symptomsCard.style.display = 'none';
        }

        // --- Recommendations ---
        const recoList = document.getElementById('recoList');
        recoList.innerHTML = (info.recommendations || []).map(r => `<li>${r}</li>`).join('');

        // --- Emergency ---
        const emergencyCard = document.getElementById('emergencyCard');
        const emergencyList = document.getElementById('emergencyList');
        if (info.when_to_seek_emergency && info.when_to_seek_emergency.length > 0) {
            emergencyCard.style.display = 'block';
            emergencyList.innerHTML = info.when_to_seek_emergency.map(e => `<li>${e}</li>`).join('');
        } else {
            emergencyCard.style.display = 'none';
        }

        // --- Treatment ---
        const treatmentCard = document.getElementById('treatmentCard');
        if (info.treatment_overview) {
            treatmentCard.style.display = 'block';
            document.getElementById('treatmentText').textContent = info.treatment_overview;
        } else {
            treatmentCard.style.display = 'none';
        }

        // --- Grad-CAM Heatmap ---
        const gradcamCard = document.getElementById('gradcamCard');
        if (result.gradcam_image) {
            gradcamCard.style.display = 'block';
            document.getElementById('gradcamOverlay').src = result.gradcam_image;
            // Set original image from the preview
            document.getElementById('gradcamOriginal').src = previewImage.src;
        } else {
            gradcamCard.style.display = 'none';
        }

        // Show results
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Show OOD alert if detected
        let oodContainer = document.getElementById('oodAlert');
        if (!oodContainer) {
            oodContainer = document.createElement('div');
            oodContainer.id = 'oodAlert';
            oodContainer.className = 'ood-alert-container';
            const resultCard = document.getElementById('resultCard');
            resultCard.parentNode.insertBefore(oodContainer, resultCard);
        }
        if (result.ood_detection && result.ood_detection.is_ood) {
            const ood = result.ood_detection;
            oodContainer.innerHTML = `
                <div class="ood-alert">
                    <div class="ood-alert-header">
                        <i class="bi bi-shield-exclamation"></i>
                        <h4>Out-of-Distribution Warning</h4>
                    </div>
                    <p>This image may <strong>not be a chest X-ray</strong>. The AI model was trained exclusively on chest X-ray images and results for other image types are unreliable.</p>
                    <div class="ood-metrics">
                        <span class="ood-metric"><strong>Energy:</strong> ${ood.energy}</span>
                        <span class="ood-metric"><strong>Max Prob:</strong> ${(ood.max_probability * 100).toFixed(1)}%</span>
                        <span class="ood-metric"><strong>Entropy:</strong> ${ood.entropy.toFixed(3)}</span>
                        <span class="ood-metric"><strong>Feature Norm:</strong> ${ood.feature_norm.toFixed(1)}</span>
                    </div>
                    <ul class="ood-reasons">
                        ${ood.reasons.map(r => `<li><i class="bi bi-exclamation-circle"></i> ${r}</li>`).join('')}
                    </ul>
                </div>
            `;
            oodContainer.style.display = 'block';
        } else {
            oodContainer.style.display = 'none';
        }

        // Show quality warnings if any
        if (result.quality_warnings && result.quality_warnings.length > 0) {
            const warningHtml = result.quality_warnings.map(w =>
                `<div class="quality-warning"><i class="bi bi-exclamation-triangle-fill"></i> ${w}</div>`
            ).join('');
            let warningContainer = document.getElementById('qualityWarnings');
            if (!warningContainer) {
                warningContainer = document.createElement('div');
                warningContainer.id = 'qualityWarnings';
                warningContainer.className = 'quality-warnings-container';
                const resultCard = document.getElementById('resultCard');
                resultCard.parentNode.insertBefore(warningContainer, resultCard);
            }
            warningContainer.innerHTML = warningHtml;
            warningContainer.style.display = 'block';
        } else {
            const wc = document.getElementById('qualityWarnings');
            if (wc) wc.style.display = 'none';
        }

        // Save to history (server-side)
        saveToHistory(result);
    }

    // ============================================================
    // ANALYSIS HISTORY (SQLite via API)
    // ============================================================

    async function fetchHistory() {
        try {
            const res = await fetch('/api/history?limit=50', { headers: authHeaders() });
            const data = await res.json();
            return data.history || [];
        } catch {
            return [];
        }
    }

    async function saveToHistory(result) {
        const patient = getPatientInfo();
        const entry = {
            prediction: result.prediction,
            confidence: result.confidence,
            confidence_pct: result.confidence_pct,
            all_predictions: result.all_predictions,
            timestamp: result.timestamp,
            inference_time: result.inference_time,
            inference_mode: result.inference_mode || 'single',
            device_used: result.device_used || '',
            is_ood: result.ood_detection ? result.ood_detection.is_ood : false,
            ood_score: result.ood_detection ? result.ood_detection.ood_score : 0,
            patient_name: patient.name || '',
            patient_age: patient.age || '',
            patient_gender: patient.gender || '',
            patient_notes: patient.notes || '',
            quality_warnings: result.quality_warnings || []
        };
        try {
            await fetch('/api/history', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...authHeaders() },
                body: JSON.stringify(entry)
            });
        } catch { /* silent fail */ }
        renderHistory();
    }

    async function renderHistory() {
        const history = await fetchHistory();
        const grid = document.getElementById('historyGrid');
        const empty = document.getElementById('historyEmpty');
        const count = document.getElementById('historyCount');

        count.textContent = `${history.length} scan${history.length !== 1 ? 's' : ''}`;

        if (history.length === 0) {
            empty.style.display = 'block';
            grid.querySelectorAll('.history-card').forEach(c => c.remove());
            return;
        }

        empty.style.display = 'none';
        grid.querySelectorAll('.history-card').forEach(c => c.remove());

        const iconMap = {
            'COVID-19': 'bi-virus', 'Healthy': 'bi-heart-pulse',
            'Pneumonia': 'bi-lungs', 'Tuberculosis': 'bi-bullseye'
        };
        const colorMap = {
            'COVID-19': '#ef4444', 'Healthy': '#10b981',
            'Pneumonia': '#f59e0b', 'Tuberculosis': '#8b5cf6'
        };

        history.forEach(entry => {
            const card = document.createElement('div');
            card.className = 'history-card';
            const icon = iconMap[entry.prediction] || 'bi-lungs';
            const color = colorMap[entry.prediction] || '#2563eb';
            const oodBadge = entry.is_ood ? '<span class="ood-badge" style="margin-left:6px;"><i class="bi bi-shield-exclamation"></i> OOD</span>' : '';
            const modeBadge = entry.inference_mode === 'ensemble'
                ? '<span class="ensemble-badge" style="margin-left:4px;font-size:9px;padding:2px 6px;"><i class="bi bi-diagram-3"></i></span>'
                : '';
            const patientLine = entry.patient_name
                ? `<span class="history-patient"><i class="bi bi-person"></i> ${entry.patient_name}</span>`
                : '';
            card.innerHTML = `
                <div class="history-card-icon" style="color:${color};background:${color}20;">
                    <i class="bi ${icon}"></i>
                </div>
                <div class="history-card-body">
                    <h4>${entry.prediction}${oodBadge}${modeBadge}</h4>
                    <span class="history-conf">${entry.confidence_pct} confidence</span>
                    ${patientLine}
                    <span class="history-time">${entry.timestamp}</span>
                </div>
                <button class="history-delete" data-id="${entry.id}" title="Delete">
                    <i class="bi bi-x"></i>
                </button>
            `;
            grid.appendChild(card);
        });
    }

    // Clear history
    document.getElementById('clearHistoryBtn').addEventListener('click', async () => {
        if (confirm('Clear all scan history?')) {
            try {
                await fetch('/api/history', { method: 'DELETE', headers: authHeaders() });
            } catch { /* silent */ }
            renderHistory();
        }
    });

    // Delete individual history entry
    document.getElementById('historyGrid').addEventListener('click', async (e) => {
        const btn = e.target.closest('.history-delete');
        if (!btn) return;
        const id = parseInt(btn.dataset.id);
        try {
            await fetch(`/api/history/${id}`, { method: 'DELETE', headers: authHeaders() });
        } catch { /* silent */ }
        renderHistory();
    });

    // Render history on load
    renderHistory();

    // Export history with auth headers
    async function exportHistory(format) {
        try {
            const res = await fetch(`/api/history/export/${format}`, { headers: authHeaders() });
            if (res.ok) {
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `medscan_history_${new Date().toISOString().slice(0,10)}.${format}`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);
            } else {
                showError('Failed to export history.');
            }
        } catch {
            showError('Failed to export history. Check server connection.');
        }
    }
    document.getElementById('exportJsonBtn').addEventListener('click', () => exportHistory('json'));
    document.getElementById('exportCsvBtn').addEventListener('click', () => exportHistory('csv'));

    // ============================================================
    // MODE TOGGLE (Single / Batch)
    // ============================================================
    const singleModeBtn = document.getElementById('singleModeBtn');
    const batchModeBtn = document.getElementById('batchModeBtn');
    const singleUploadArea = document.getElementById('singleUploadArea');
    const batchUploadArea = document.getElementById('batchUploadArea');

    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (btn.dataset.mode === 'single') {
                singleUploadArea.style.display = 'block';
                batchUploadArea.style.display = 'none';
                document.getElementById('batchResults').style.display = 'none';
            } else {
                singleUploadArea.style.display = 'none';
                batchUploadArea.style.display = 'block';
            }
        });
    });

    // ============================================================
    // BATCH UPLOAD
    // ============================================================
    const batchDropZone = document.getElementById('batchDropZone');
    const batchFileInput = document.getElementById('batchFileInput');
    const batchFileList = document.getElementById('batchFileList');
    const batchAnalyzeBtn = document.getElementById('batchAnalyzeBtn');
    const batchLoadingContainer = document.getElementById('batchLoadingContainer');
    let batchFiles = [];

    batchDropZone.addEventListener('click', () => batchFileInput.click());
    batchDropZone.addEventListener('dragover', (e) => { e.preventDefault(); batchDropZone.classList.add('drag-over'); });
    batchDropZone.addEventListener('dragleave', () => batchDropZone.classList.remove('drag-over'));
    batchDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        batchDropZone.classList.remove('drag-over');
        addBatchFiles(Array.from(e.dataTransfer.files));
    });
    batchFileInput.addEventListener('change', (e) => addBatchFiles(Array.from(e.target.files)));

    function addBatchFiles(files) {
        const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
        files.forEach(f => {
            if (batchFiles.length >= 20) return;
            if (validTypes.includes(f.type) || f.name.match(/\.(jpg|jpeg|png|bmp|tiff|webp)$/i)) {
                batchFiles.push(f);
            }
        });
        renderBatchFileList();
    }

    function renderBatchFileList() {
        batchFileList.innerHTML = batchFiles.map((f, i) => `
            <div class="batch-file-item">
                <i class="bi bi-file-earmark-image"></i>
                <span>${f.name}</span>
                <button class="batch-remove-btn" data-idx="${i}"><i class="bi bi-x"></i></button>
            </div>
        `).join('');
        batchAnalyzeBtn.disabled = batchFiles.length === 0;

        // Remove handlers
        batchFileList.querySelectorAll('.batch-remove-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                batchFiles.splice(parseInt(btn.dataset.idx), 1);
                renderBatchFileList();
            });
        });
    }

    batchAnalyzeBtn.addEventListener('click', async () => {
        if (batchFiles.length === 0) return;

        batchAnalyzeBtn.style.display = 'none';
        batchLoadingContainer.style.display = 'block';
        document.getElementById('batchProgress').textContent = `0/${batchFiles.length}`;

        const formData = new FormData();
        batchFiles.forEach(f => formData.append('files', f));

        try {
            const response = await fetch('/predict/batch', { method: 'POST', body: formData });
            const data = await response.json();

            if (response.ok && data.results) {
                displayBatchResults(data.results);
            } else {
                showError(data.error || 'Batch analysis failed.');
            }
        } catch (err) {
            showError('Connection error during batch analysis.');
        }

        batchLoadingContainer.style.display = 'none';
        batchAnalyzeBtn.style.display = 'flex';
    });

    function displayBatchResults(results) {
        const tbody = document.getElementById('batchTableBody');
        tbody.innerHTML = '';
        const colorMap = {
            'COVID-19': '#ef4444', 'Healthy': '#10b981',
            'Pneumonia': '#f59e0b', 'Tuberculosis': '#8b5cf6'
        };

        results.forEach((r, i) => {
            const tr = document.createElement('tr');
            if (r.error) {
                tr.innerHTML = `<td>${i + 1}</td><td>${r.filename}</td><td colspan="2">—</td><td><span class="batch-status error">Error</span></td>`;
            } else {
                const color = colorMap[r.prediction] || '#2563eb';
                const oodBadge = r.is_ood ? ' <span class="ood-badge" title="Out-of-distribution: may not be a chest X-ray"><i class="bi bi-shield-exclamation"></i> OOD</span>' : '';
                tr.innerHTML = `
                    <td>${i + 1}</td>
                    <td>${r.filename || 'Unknown'}</td>
                    <td><span style="color:${color};font-weight:600;">${r.prediction}</span>${oodBadge}</td>
                    <td>${r.confidence_pct}</td>
                    <td><span class="batch-status success">Done</span></td>
                `;
            }
            tbody.appendChild(tr);
        });

        document.getElementById('batchResults').style.display = 'block';
        document.getElementById('batchResults').scrollIntoView({ behavior: 'smooth' });
    }

    document.getElementById('batchResetBtn').addEventListener('click', () => {
        batchFiles = [];
        renderBatchFileList();
        document.getElementById('batchResults').style.display = 'none';
        batchFileInput.value = '';
    });

    // ============================================================
    // SCAN AGAIN
    // ============================================================

    scanAgainBtn.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        resetUpload();
        analyzeBtn.style.display = 'flex';
        uploadSection.scrollIntoView({ behavior: 'smooth' });
    });

    // ============================================================
    // PDF DOWNLOAD
    // ============================================================
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    downloadPdfBtn.addEventListener('click', async () => {
        if (!lastResult) return;
        downloadPdfBtn.disabled = true;
        downloadPdfBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Generating...';

        try {
            const pdfData = { ...lastResult, patient_info: getPatientInfo() };
            const response = await fetch('/api/generate-report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(pdfData)
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `MedScan_Report_${new Date().toISOString().slice(0,10)}.pdf`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);
            } else {
                showError('Failed to generate PDF report.');
            }
        } catch (err) {
            showError('Failed to generate PDF. Check server connection.');
        }

        downloadPdfBtn.disabled = false;
        downloadPdfBtn.innerHTML = '<i class="bi bi-file-earmark-pdf"></i> Download PDF Report';
    });

    // ============================================================
    // ERROR HANDLING
    // ============================================================

    function showError(message) {
        // Create a temporary error toast
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed;
            top: 80px;
            left: 50%;
            transform: translateX(-50%);
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            padding: 14px 24px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 500;
            z-index: 1000;
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            display: flex;
            align-items: center;
            gap: 10px;
            max-width: 500px;
            animation: fadeInUp 0.3s ease-out;
        `;
        toast.innerHTML = `<i class="bi bi-exclamation-circle" style="font-size:18px;color:#ef4444;"></i> ${message}`;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s';
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    // ============================================================
    // SCROLL REVEAL ANIMATIONS
    // ============================================================

    function initScrollReveal() {
        // Add reveal classes to elements
        const revealMappings = [
            { selector: '.section-header', cls: 'reveal' },
            { selector: '.upload-area', cls: 'reveal reveal-scale' },
            { selector: '.step-card', cls: 'reveal' },
            { selector: '.about-card', cls: 'reveal' },
            { selector: '.footer-content', cls: 'reveal' },
        ];

        revealMappings.forEach(({ selector, cls }) => {
            document.querySelectorAll(selector).forEach((el, i) => {
                cls.split(' ').forEach(c => el.classList.add(c));
                // Add stagger delay for sibling elements
                if (i > 0 && i <= 5) {
                    el.classList.add(`reveal-delay-${i}`);
                }
            });
        });

        // Alternate left/right for about cards
        document.querySelectorAll('.about-card').forEach((el, i) => {
            el.classList.remove('reveal-left', 'reveal-right');
            el.classList.add(i % 2 === 0 ? 'reveal-left' : 'reveal-right');
            if (i > 0 && i <= 5) el.classList.add(`reveal-delay-${i}`);
        });

        // Intersection Observer
        const observerOptions = {
            root: null,
            rootMargin: '0px 0px -60px 0px',
            threshold: 0.15
        };

        const revealObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                    revealObserver.unobserve(entry.target);
                }
            });
        }, observerOptions);

        document.querySelectorAll('.reveal').forEach(el => {
            revealObserver.observe(el);
        });
    }

    initScrollReveal();

    // ============================================================
    // CHATBOT
    // ============================================================

    const chatFab = document.getElementById('chatbotFab');
    const chatWindow = document.getElementById('chatbotWindow');
    const chatClose = document.getElementById('chatClose');
    const chatInput = document.getElementById('chatInput');
    const chatSend = document.getElementById('chatSend');
    const chatMessages = document.getElementById('chatMessages');
    const chatSuggestions = document.getElementById('chatSuggestions');

    let chatOpen = false;

    function toggleChat() {
        chatOpen = !chatOpen;
        if (chatOpen) {
            chatWindow.classList.add('open');
            chatFab.classList.add('active');
            chatInput.focus();
        } else {
            chatWindow.classList.remove('open');
            chatFab.classList.remove('active');
        }
    }

    chatFab.addEventListener('click', toggleChat);
    chatClose.addEventListener('click', toggleChat);

    // Send message
    function sendMessage() {
        const msg = chatInput.value.trim();
        if (!msg) return;

        // Add user message
        appendMessage('user', msg);
        chatInput.value = '';

        // Show typing indicator
        const typingEl = showTyping();

        // Call API
        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        })
        .then(res => res.json())
        .then(data => {
            removeTyping(typingEl);
            if (data.error) {
                appendMessage('bot', 'Sorry, something went wrong. Please try again.');
            } else {
                appendMessage('bot', data.response, data.topic, data.follow_up);
            }
        })
        .catch(() => {
            removeTyping(typingEl);
            appendMessage('bot', 'Connection error. Please check if the server is running.');
        });
    }

    chatSend.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // Suggestion chips
    chatSuggestions.addEventListener('click', (e) => {
        const chip = e.target.closest('.suggestion-chip');
        if (!chip) return;
        chatInput.value = chip.dataset.msg;
        sendMessage();
    });

    function appendMessage(role, text, topic = '', followUp = '') {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-msg ${role}`;

        const iconClass = role === 'bot' ? 'bi-robot' : 'bi-person-fill';

        // Format text: convert \n to <br>, **bold** to <strong>
        let formattedText = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');

        let topicHTML = topic ? `<div class="msg-topic">${topic}</div>` : '';
        let followUpHTML = followUp ? `<br><em style="font-size:12px;opacity:0.7;">${followUp}</em>` : '';

        msgDiv.innerHTML = `
            <div class="msg-avatar"><i class="bi ${iconClass}"></i></div>
            <div class="msg-bubble">
                ${topicHTML}
                <p>${formattedText}${followUpHTML}</p>
            </div>
        `;

        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTyping() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-msg bot';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="msg-avatar"><i class="bi bi-robot"></i></div>
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return typingDiv;
    }

    function removeTyping(el) {
        if (el && el.parentNode) el.remove();
    }
});
