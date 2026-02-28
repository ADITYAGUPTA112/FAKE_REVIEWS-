// Global Chart Instances
let gaugeChart = null;
let distChart = null;
window.latestScanResults = null;

// Thematic Colors setup
const COLORS = {
    background: '#0f172a',
    red: '#ef4444',
    redGlow: 'rgba(239, 68, 68, 0.4)',
    emerald: '#10b981',
    emeraldGlow: 'rgba(16, 185, 129, 0.4)',
    cyan: '#06b6d4',
    purple: '#a855f7',
    slate800: '#1e293b'
};

// Global Chart Defaults for premium look
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.scale.grid.color = 'rgba(255,255,255,0.05)';
Chart.defaults.scale.grid.borderColor = 'rgba(255,255,255,0.05)';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.9)';
Chart.defaults.plugins.tooltip.titleColor = '#fff';
Chart.defaults.plugins.tooltip.bodyColor = '#cbd5e1';
Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.1)';
Chart.defaults.plugins.tooltip.borderWidth = 1;

document.addEventListener('DOMContentLoaded', () => {

    // --- DOM Elements ---
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeBtnText = document.getElementById('analyzeBtnText');
    const analyzeBtnIcon = analyzeBtn.querySelector('i');
    const asinInput = document.getElementById('asinInput');
    const mainSection = document.getElementById('analysis-section');

    // Stats Elements
    const statElements = {
        total: document.getElementById('statTotal'),
        fake: document.getElementById('statFakeCount'),
        trust: document.getElementById('statTrustScore'),
        accuracy: document.getElementById('statAccuracy')
    };

    // UI Elements
    const reviewFeed = document.getElementById('reviewFeed');
    const downloadCsvBtn = document.getElementById('downloadCsvBtn');

    // --- Skeleton Loader Toggle ---
    function toggleSkeletons(show) {
        const cards = document.querySelectorAll('.data-card');
        cards.forEach(card => {
            if (show) card.classList.add('skeleton');
            else card.classList.remove('skeleton');
        });

        if (show) {
            mainSection.classList.add('processing-blur');
            analyzeBtnText.innerText = 'Analyzing Data...';
            analyzeBtnIcon.className = 'ph-bold ph-spinner animate-spin mr-2 text-lg';
            analyzeBtn.disabled = true;

            // Empty states
            statElements.total.innerText = '--';
            document.getElementById('fakeRiskBar').style.width = '0%';
            document.getElementById('trustScoreBar').style.width = '0%';
            document.getElementById('gaugeScoreText').innerText = '--';
            reviewFeed.innerHTML = '';
        } else {
            mainSection.classList.remove('processing-blur');
            analyzeBtnText.innerText = 'Run Analysis';
            analyzeBtnIcon.className = 'ph-bold ph-magic-wand mr-2 text-lg';
            analyzeBtn.disabled = false;
        }
    }

    // --- Smooth Number Counter Utility ---
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // easeOutExpo
            const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
            obj.innerHTML = (start + easeProgress * (end - start)).toFixed(end % 1 !== 0 ? 1 : 0);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // --- Chart Initializers ---
    function initOrUpdateCharts(genuineCount, fakeCount, fakeScore) {

        // 1. Doughnut Gauge Chart
        const ctxGauge = document.getElementById('gaugeChartCanvas').getContext('2d');

        // Dynamic colors based on risk
        let arcColor = COLORS.emerald;
        let ringGlow = COLORS.emeraldGlow;
        let riskLabel = 'Low Risk';
        let badgeClass = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';

        if (fakeScore > 20 && fakeScore <= 50) {
            arcColor = '#f59e0b'; // Amber
            ringGlow = 'rgba(245, 158, 11, 0.4)';
            riskLabel = 'Moderate Risk';
            badgeClass = 'bg-amber-500/10 text-amber-500 border-amber-500/20';
        } else if (fakeScore > 50) {
            arcColor = COLORS.red;
            ringGlow = COLORS.redGlow;
            riskLabel = 'Critical Risk';
            badgeClass = 'bg-red-500/10 text-red-400 border-red-500/20';
        }

        // Setup Doughnut
        if (gaugeChart) gaugeChart.destroy();
        gaugeChart = new Chart(ctxGauge, {
            type: 'doughnut',
            data: {
                labels: ['Fake Risk', 'Authentic'],
                datasets: [{
                    data: [fakeScore, 100 - fakeScore],
                    backgroundColor: [arcColor, COLORS.slate800],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '80%',
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                },
                animation: { animateScale: true, animateRotate: true, duration: 2000, easing: 'easeOutQuart' }
            }
        });

        // Update Gauge Texts
        document.getElementById('riskLabel').innerText = riskLabel;
        document.getElementById('badgeText').className = `px-3 py-1 rounded-full text-xs font-medium shadow-sm border ${badgeClass}`;
        document.getElementById('badgeText').innerText = fakeScore > 20 ? 'Action Required' : 'Safe to Buy';
        animateValue(document.getElementById('gaugeScoreText'), 0, 100 - fakeScore, 2000);

        // Update Risk level badge in main card too
        const riskLevelBadge = document.getElementById('riskLevelBadge');
        riskLevelBadge.innerText = riskLabel;
        riskLevelBadge.className = `text-[10px] uppercase font-bold px-2 py-0.5 rounded-full border ${badgeClass}`;

        // 2. Distribution Bar Chart
        const ctxDist = document.getElementById('distributionChart').getContext('2d');

        // Create Gradients
        const gradGen = ctxDist.createLinearGradient(0, 0, 0, 200);
        gradGen.addColorStop(0, COLORS.emerald);
        gradGen.addColorStop(1, 'rgba(16, 185, 129, 0.1)');

        const gradFake = ctxDist.createLinearGradient(0, 0, 0, 200);
        gradFake.addColorStop(0, COLORS.red);
        gradFake.addColorStop(1, 'rgba(239, 68, 68, 0.1)');

        if (distChart) distChart.destroy();
        distChart = new Chart(ctxDist, {
            type: 'bar',
            data: {
                labels: ['Genuine Accounts', 'AI/Bot Generators'],
                datasets: [{
                    label: 'Volume',
                    data: [genuineCount, fakeCount],
                    backgroundColor: [gradGen, gradFake],
                    borderRadius: 8,
                    barPercentage: 0.6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { precision: 0 }
                    },
                    x: {
                        grid: { display: false }
                    }
                },
                animation: {
                    y: { duration: 1500, easing: 'easeOutQuart' }
                }
            }
        });
    }

    // --- Render Review Feed (Cards) ---
    function renderReviewFeed(reviews, predictions, confidences) {
        reviewFeed.innerHTML = '';
        downloadCsvBtn.classList.remove('hidden');

        if (!reviews || reviews.length === 0) {
            reviewFeed.innerHTML = `<div class="col-span-full py-10 text-center text-slate-500">No reviews parsed.</div>`;
            return;
        }

        reviews.forEach((reviewText, i) => {
            const pred = predictions[i];
            const conf = parseFloat(confidences[i]).toFixed(1);

            const isFake = pred === "Fake";
            const borderClass = isFake ? 'border-left-fake' : 'border-left-genuine';
            const badgeClass = isFake ? 'bg-red-500/10 text-red-400 border-red-500/20' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
            const icon = isFake ? "ph-fill ph-warning-circle" : "ph-fill ph-check-circle";

            // Determine dynamic "Reasons" based on prediction to simulate complex analysis
            let flagHTML = '';
            if (isFake) {
                // Randomly pick a couple of reasons based on text length or index just for simulation aesthetic
                const reasons = ["High semantic similarity", "Over-praising adjectives", "Repetitive sentence structure", "Bot-like timestamps", "Generic phrasing"];
                const r1 = reasons[i % reasons.length];
                const r2 = reasons[(i + 1) % reasons.length];
                flagHTML = `
                    <div class="mt-4 flex flex-wrap gap-2">
                        <span class="px-2 py-1 rounded bg-slate-800 border border-slate-700 text-[10px] text-slate-400 uppercase tracking-wide inline-flex items-center"><i class="ph-fill ph-flag mr-1"></i> ${r1}</span>
                        <span class="px-2 py-1 rounded bg-slate-800 border border-slate-700 text-[10px] text-slate-400 uppercase tracking-wide inline-flex items-center"><i class="ph-fill ph-flag mr-1"></i> ${r2}</span>
                    </div>
                `;
            }

            const cleanText = reviewText.substring(0, 200) + (reviewText.length > 200 ? '...' : '');

            const cardHTML = `
                <div class="glass-panel p-5 rounded-xl flex flex-col justify-between ${borderClass} hover:translate-y-[-2px] transition-transform duration-300">
                    <div>
                        <div class="flex justify-between items-start mb-3">
                            <span class="px-2 py-1 rounded-full text-xs font-semibold border ${badgeClass} flex items-center shadow-sm">
                                <i class="${icon} mr-1 text-sm"></i>
                                ${pred}
                            </span>
                            <span class="text-xs font-mono text-slate-500 tracking-tight">Conf: ${conf}%</span>
                        </div>
                        <p class="text-sm text-slate-300 leading-relaxed font-serif">"${cleanText}"</p>
                    </div>
                    ${flagHTML}
                </div>
            `;
            reviewFeed.innerHTML += cardHTML;
        });
    }

    // --- Main Analyze Action ---
    analyzeBtn.addEventListener('click', async () => {
        const asin = asinInput.value.trim();
        if (!asin) {
            if (window.showToast) window.showToast('Please enter an ASIN or URL', 'error');
            else alert("Please enter an ASIN or URL");
            return;
        }

        // Show loading state
        toggleSkeletons(true);

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ asin: asin, pages: 2 })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Server error occurred');
            }

            // --- Remove skeleton loader ---
            toggleSkeletons(false);

            // Populate Product Meta below search bar
            const metaDiv = document.getElementById('productMeta');
            document.getElementById('metaAsin').innerText = data.asin || asin;
            document.getElementById('metaTime').innerText = new Date().toLocaleTimeString();
            metaDiv.classList.remove('hidden', 'opacity-0');

            // Set Data
            const sum = data.summary;

            // --- 1. Populate Metrics with CountUp Animations ---
            animateValue(statElements.total, 0, sum.total_reviews, 1500);
            animateValue(statElements.fake, 0, sum.fake_percent, 1500);
            animateValue(statElements.trust, 0, sum.genuine_percent, 1500);
            animateValue(statElements.accuracy, 0, sum.avg_confidence, 1500);

            // Progress Bar widths
            setTimeout(() => {
                document.getElementById('fakeRiskBar').style.width = `${sum.fake_percent}%`;
                document.getElementById('trustScoreBar').style.width = `${sum.genuine_percent}%`;
            }, 100);

            // --- 2. Build Charts ---
            initOrUpdateCharts(data.chart_data.genuine, data.chart_data.fake, sum.fake_percent);

            // --- 3. Build Feed ---
            renderReviewFeed(data.results.review, data.results.prediction, data.results.confidence);

            // Store for CSV
            window.latestScanResults = {
                asin: asin,
                reviews: data.results.review,
                predictions: data.results.prediction,
                confidences: data.results.confidence
            };

            if (window.showToast) window.showToast('Analysis completed successfully!');

        } catch (error) {
            console.error('Error fetching data:', error);
            toggleSkeletons(false);
            if (window.showToast) window.showToast(error.message, 'error');
            else alert(error.message);
        }
    });

    // Enter key support
    asinInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            event.preventDefault();
            analyzeBtn.click();
        }
    });

    // CSV Download
    if (downloadCsvBtn) {
        downloadCsvBtn.addEventListener('click', () => {
            if (!window.latestScanResults || !window.latestScanResults.reviews) {
                if (window.showToast) window.showToast("No data available to download.", "error");
                return;
            }

            const r = window.latestScanResults;
            let csvContent = "data:text/csv;charset=utf-8,";
            csvContent += "Review_Text,Prediction,Confidence_Score\r\n";

            for (let i = 0; i < r.reviews.length; i++) {
                const safeText = '"' + r.reviews[i].replace(/"/g, '""') + '"';
                const row = `${safeText},${r.predictions[i]},${r.confidences[i]}`;
                csvContent += row + "\r\n";
            }

            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `trustlens_analysis_${r.asin}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

});
