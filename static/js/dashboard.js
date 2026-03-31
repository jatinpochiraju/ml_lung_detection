/**
 * MedScan AI - Analytics Dashboard
 * Chart.js powered analytics for admin users
 */

// Dark mode (prevent flash)
(function() {
    const saved = localStorage.getItem('medscan-theme');
    if (saved === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
})();

document.addEventListener('DOMContentLoaded', () => {
    const AUTH_TOKEN_KEY = 'medscan-auth-token';
    const AUTH_USER_KEY = 'medscan-auth-user';

    function getToken() { return localStorage.getItem(AUTH_TOKEN_KEY); }
    function getUser() {
        try { return JSON.parse(localStorage.getItem(AUTH_USER_KEY)); }
        catch { return null; }
    }
    function authHeaders() {
        const token = getToken();
        if (token) return { 'Authorization': `Bearer ${token}` };
        return {};
    }

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
        // Update charts on theme change
        setTimeout(() => { if (chartsLoaded) loadDashboardData(); }, 100);
    });

    const authGate = document.getElementById('dashboardAuthGate');
    const dashContent = document.getElementById('dashboardContent');
    const dashLoginModal = document.getElementById('dashLoginModal');
    const dashModalClose = document.getElementById('dashModalClose');
    const dashLoginBtn = document.getElementById('dashboardLoginBtn');
    const timeRangeSelect = document.getElementById('timeRangeSelect');
    const refreshBtn = document.getElementById('refreshBtn');

    let chartsLoaded = false;
    let scansChart = null, diseaseChart = null, confidenceChart = null, confByDiseaseChart = null;

    // Chart.js color scheme
    const DISEASE_COLORS = {
        'COVID-19': '#ef4444',
        'Healthy': '#10b981',
        'Pneumonia': '#f59e0b',
        'Tuberculosis': '#8b5cf6'
    };

    function isDark() {
        return document.documentElement.getAttribute('data-theme') === 'dark';
    }

    function chartTextColor() { return isDark() ? '#94a3b8' : '#64748b'; }
    function chartGridColor() { return isDark() ? '#334155' : '#e2e8f0'; }

    // Check auth
    async function checkAuth() {
        const token = getToken();
        if (!token) {
            showAuthGate();
            return;
        }
        try {
            const res = await fetch('/api/auth/me', { headers: authHeaders() });
            if (res.ok) {
                const data = await res.json();
                if (data.user && data.user.is_admin) {
                    showDashboard();
                    loadDashboardData();
                    return;
                }
            }
        } catch { /* fall through */ }
        showAuthGate();
    }

    function showAuthGate() {
        authGate.style.display = 'flex';
        dashContent.style.display = 'none';
    }

    function showDashboard() {
        authGate.style.display = 'none';
        dashContent.style.display = 'block';
    }

    // Login modal
    dashLoginBtn.addEventListener('click', () => { dashLoginModal.style.display = 'flex'; });
    dashModalClose.addEventListener('click', () => { dashLoginModal.style.display = 'none'; });
    dashLoginModal.addEventListener('click', (e) => {
        if (e.target === dashLoginModal) dashLoginModal.style.display = 'none';
    });

    document.getElementById('dashLoginForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const errorEl = document.getElementById('dashLoginError');
        errorEl.style.display = 'none';
        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    login: document.getElementById('dashLoginId').value.trim(),
                    password: document.getElementById('dashLoginPassword').value
                })
            });
            const data = await res.json();
            if (res.ok) {
                localStorage.setItem(AUTH_TOKEN_KEY, data.token);
                localStorage.setItem(AUTH_USER_KEY, JSON.stringify(data.user));
                dashLoginModal.style.display = 'none';
                if (data.user.is_admin) {
                    showDashboard();
                    loadDashboardData();
                } else {
                    errorEl.textContent = 'Admin access required. Your account is not an admin.';
                    errorEl.style.display = 'block';
                }
            } else {
                errorEl.textContent = data.error || 'Login failed';
                errorEl.style.display = 'block';
            }
        } catch {
            errorEl.textContent = 'Connection error';
            errorEl.style.display = 'block';
        }
    });

    // Time range & refresh
    timeRangeSelect.addEventListener('change', () => loadDashboardData());
    refreshBtn.addEventListener('click', () => {
        refreshBtn.classList.add('spinning');
        loadDashboardData().then(() => {
            setTimeout(() => refreshBtn.classList.remove('spinning'), 500);
        });
    });

    // Load all dashboard data
    async function loadDashboardData() {
        const days = timeRangeSelect.value;
        try {
            const [summaryRes, timelineRes, usersRes] = await Promise.all([
                fetch('/api/analytics/summary', { headers: authHeaders() }),
                fetch(`/api/analytics/timeline?days=${days}`, { headers: authHeaders() }),
                fetch('/api/analytics/users', { headers: authHeaders() })
            ]);

            if (!summaryRes.ok || !timelineRes.ok || !usersRes.ok) {
                if (summaryRes.status === 401 || summaryRes.status === 403) {
                    showAuthGate();
                    return;
                }
                return;
            }

            const summary = await summaryRes.json();
            const timeline = await timelineRes.json();
            const users = await usersRes.json();

            updateKPIs(summary);
            updateScansTimeline(timeline);
            updateDiseaseDistribution(summary);
            updateConfidenceTimeline(timeline);
            updateConfidenceByDisease(summary);
            updateRecentScans(timeline.recent_scans || []);
            updateUsersTable(users.users || []);
            chartsLoaded = true;
        } catch (err) {
            console.error('Dashboard load error:', err);
        }
    }

    // Update KPI cards
    function updateKPIs(data) {
        document.getElementById('kpiTotalScans').textContent = data.total_scans.toLocaleString();
        document.getElementById('kpiScansToday').textContent = data.scans_today.toLocaleString();
        document.getElementById('kpiAvgConfidence').textContent = (data.avg_confidence * 100).toFixed(1) + '%';
        document.getElementById('kpiTotalUsers').textContent = data.total_users.toLocaleString();
        document.getElementById('kpiOodRate').textContent = (data.ood_rate * 100).toFixed(1) + '%';
    }

    // Scans over time chart
    function updateScansTimeline(data) {
        const ctx = document.getElementById('scansTimelineChart').getContext('2d');
        const labels = data.daily_scans.map(d => d.date);
        const counts = data.daily_scans.map(d => d.count);

        if (scansChart) scansChart.destroy();
        scansChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Scans',
                    data: counts,
                    backgroundColor: 'rgba(37, 99, 235, 0.6)',
                    borderColor: '#2563eb',
                    borderWidth: 1,
                    borderRadius: 6,
                    barPercentage: 0.7,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                },
                scales: {
                    x: {
                        ticks: { color: chartTextColor(), maxTicksLimit: 10 },
                        grid: { display: false },
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { color: chartTextColor(), stepSize: 1 },
                        grid: { color: chartGridColor() },
                    }
                }
            }
        });
    }

    // Disease distribution pie chart
    function updateDiseaseDistribution(data) {
        const ctx = document.getElementById('diseaseDistChart').getContext('2d');
        const dist = data.disease_distribution;
        const labels = Object.keys(dist);
        const values = Object.values(dist);
        const colors = labels.map(l => DISEASE_COLORS[l] || '#64748b');

        if (diseaseChart) diseaseChart.destroy();
        diseaseChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels,
                datasets: [{
                    data: values,
                    backgroundColor: colors,
                    borderWidth: 2,
                    borderColor: isDark() ? '#1e293b' : '#ffffff',
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '55%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: chartTextColor(), padding: 16, usePointStyle: true, pointStyleWidth: 12 },
                    },
                }
            }
        });
    }

    // Confidence over time chart
    function updateConfidenceTimeline(data) {
        const ctx = document.getElementById('confidenceTimelineChart').getContext('2d');
        const labels = data.daily_confidence.map(d => d.date);
        const values = data.daily_confidence.map(d => (d.avg_confidence * 100).toFixed(1));

        if (confidenceChart) confidenceChart.destroy();
        confidenceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Avg Confidence (%)',
                    data: values,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4,
                    pointBackgroundColor: '#10b981',
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                },
                scales: {
                    x: {
                        ticks: { color: chartTextColor(), maxTicksLimit: 10 },
                        grid: { display: false },
                    },
                    y: {
                        min: 0,
                        max: 100,
                        ticks: { color: chartTextColor(), callback: v => v + '%' },
                        grid: { color: chartGridColor() },
                    }
                }
            }
        });
    }

    // Confidence by disease bar chart
    function updateConfidenceByDisease(data) {
        const ctx = document.getElementById('confidenceByDiseaseChart').getContext('2d');
        const avgPerDisease = data.avg_confidence_per_disease || {};
        const labels = Object.keys(avgPerDisease);
        const values = Object.values(avgPerDisease).map(v => (v * 100).toFixed(1));
        const colors = labels.map(l => DISEASE_COLORS[l] || '#64748b');

        if (confByDiseaseChart) confByDiseaseChart.destroy();
        confByDiseaseChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Avg Confidence (%)',
                    data: values,
                    backgroundColor: colors.map(c => c + '99'),
                    borderColor: colors,
                    borderWidth: 2,
                    borderRadius: 8,
                    barPercentage: 0.6,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { display: false },
                },
                scales: {
                    x: {
                        min: 0,
                        max: 100,
                        ticks: { color: chartTextColor(), callback: v => v + '%' },
                        grid: { color: chartGridColor() },
                    },
                    y: {
                        ticks: { color: chartTextColor() },
                        grid: { display: false },
                    }
                }
            }
        });
    }

    // Recent scans table
    function updateRecentScans(scans) {
        const tbody = document.getElementById('recentScansBody');
        if (scans.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-light);">No scans yet</td></tr>';
            return;
        }
        const colorMap = { 'COVID-19': '#ef4444', 'Healthy': '#10b981', 'Pneumonia': '#f59e0b', 'Tuberculosis': '#8b5cf6' };
        tbody.innerHTML = scans.map(s => `
            <tr>
                <td>${s.id}</td>
                <td><span style="color:${colorMap[s.prediction] || '#2563eb'};font-weight:600;">${s.prediction}</span></td>
                <td>${s.confidence_pct}</td>
                <td>${s.patient_name || '—'}</td>
                <td>${s.is_ood ? '<span class="ood-badge-sm"><i class="bi bi-shield-exclamation"></i></span>' : '—'}</td>
                <td>${s.timestamp || s.created_at || '—'}</td>
            </tr>
        `).join('');
    }

    // Users table
    function updateUsersTable(users) {
        const tbody = document.getElementById('usersTableBody');
        if (users.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--text-light);">No users</td></tr>';
            return;
        }
        tbody.innerHTML = users.map(u => `
            <tr>
                <td><strong>${u.username}</strong></td>
                <td>${u.is_admin ? '<span class="admin-badge-sm"><i class="bi bi-shield-check"></i> Admin</span>' : 'User'}</td>
                <td>${u.scan_count}</td>
            </tr>
        `).join('');
    }

    // Init
    checkAuth();
});
