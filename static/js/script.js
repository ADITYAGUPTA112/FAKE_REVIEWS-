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

    // --- Variables ---
    let gaugeChart, distChart, temporalChart;

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
    function renderReviewFeed(reviews, predictions, confidences, dates, explanations) {
        reviewFeed.innerHTML = '';
        downloadCsvBtn.classList.remove('hidden');

        if (!reviews || reviews.length === 0) {
            reviewFeed.innerHTML = `<div class="col-span-full py-10 text-center text-slate-500">No reviews parsed.</div>`;
            return;
        }

        reviews.forEach((reviewText, i) => {
            const pred = predictions[i];
            const conf = parseFloat(confidences[i]).toFixed(1);
            const dateStr = dates[i] ? new Date(dates[i]).toLocaleDateString() : 'Unknown Date';
            const expList = explanations ? explanations[i] : [];

            const isFake = pred === "Fake";
            const borderClass = isFake ? 'border-left-fake' : 'border-left-genuine';
            const badgeClass = isFake ? 'bg-red-500/10 text-red-400 border-red-500/20' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
            const icon = isFake ? "ph-fill ph-warning-circle" : "ph-fill ph-check-circle";

            // Render LIME Explanations
            let flagHTML = '';
            if (expList && expList.length > 0) {
                let badges = expList.map(item => {
                    // Usually LIME gives negative weight to one class, positive to another.
                    // We just care about magnitude for importance
                    const imp = Math.abs(item.weight).toFixed(2);
                    return `<span class="px-2 py-1 rounded bg-slate-800 border border-slate-700 text-[10px] text-slate-400 uppercase tracking-wide inline-flex items-center" title="LIME Weight: ${imp}"><i class="ph-fill ph-push-pin mr-1"></i> ${item.word}</span>`;
                }).join('');

                flagHTML = `
                    <div class="mt-4 border-t border-white/5 pt-3">
                        <p class="text-[10px] text-slate-500 mb-2 uppercase tracking-wider font-semibold">AI Key Factors (Explainable AI)</p>
                        <div class="flex flex-wrap gap-2">
                            ${badges}
                        </div>
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
                            <div class="text-right">
                                <div class="text-[10px] text-slate-500 mb-0.5"><i class="ph ph-calendar mr-1"></i>${dateStr}</div>
                                <span class="text-xs font-mono text-slate-500 tracking-tight">Conf: ${conf}%</span>
                            </div>
                        </div>
                        <p class="text-sm text-slate-300 leading-relaxed font-serif">"${cleanText}"</p>
                    </div>
                    ${flagHTML}
                </div>
            `;
            reviewFeed.innerHTML += cardHTML;
        });
    }

    // --- Build Temporal Chart ---
    function buildTemporalChart(dates, predictions) {
        const ctxTemp = document.getElementById('temporalChart').getContext('2d');
        if (temporalChart) temporalChart.destroy();

        // 1. Group by Month-Year
        const grouped = {};
        for (let i = 0; i < dates.length; i++) {
            if (!dates[i]) continue;
            const d = new Date(dates[i]);
            if (isNaN(d)) continue;

            const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`; // YYYY-MM
            if (!grouped[key]) grouped[key] = { fake: 0, genuine: 0 };

            if (predictions[i] === 'Fake') grouped[key].fake++;
            else grouped[key].genuine++;
        }

        // Sort keys chronologically
        const sortedKeys = Object.keys(grouped).sort();
        if (sortedKeys.length === 0) return; // No valid dates

        const labels = sortedKeys.map(k => {
            const parts = k.split('-');
            const d = new Date(parts[0], parseInt(parts[1]) - 1);
            return d.toLocaleDateString(undefined, { month: 'short', year: 'numeric' });
        });

        const fakeData = sortedKeys.map(k => grouped[k].fake);
        const genData = sortedKeys.map(k => grouped[k].genuine);

        temporalChart = new Chart(ctxTemp, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Genuine Reviews',
                        data: genData,
                        borderColor: COLORS.emerald,
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Fake Reviews',
                        data: fakeData,
                        borderColor: COLORS.red,
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: COLORS.slate400, usePointStyle: true, boxWidth: 6 }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: COLORS.slate400, precision: 0 }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: COLORS.slate400 }
                    }
                }
            }
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
            renderReviewFeed(data.results.review, data.results.prediction, data.results.confidence, data.results.date, data.results.explanation);

            // --- 4. Build Temporal Diagram ---
            buildTemporalChart(data.results.date, data.results.prediction);

            // Store for CSV
            window.latestScanResults = {
                asin: asin,
                reviews: data.results.review,
                predictions: data.results.prediction,
                confidences: data.results.confidence,
                dates: data.results.date
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
