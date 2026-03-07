// Global Chart Instances
let gaugeChart = null;
let distChart = null;
window.latestScanResults = null;

// Thematic Colors setup
const COLORS = {
    background: '#050505',
    red: '#ff003c',
    redGlow: 'rgba(255, 0, 60, 0.4)',
    emerald: '#10b981',
    emeraldGlow: 'rgba(16, 185, 129, 0.4)',
    lime: '#39ff14',
    limeGlow: 'rgba(57, 255, 20, 0.4)',
    cyan: '#00f0ff',
    cyanGlow: 'rgba(0, 240, 255, 0.4)',
    slate800: '#1e293b',
    slate400: '#94a3b8'
};

// Global Chart Defaults for premium look
Chart.defaults.color = '#cbd5e1';
Chart.defaults.font.family = "'Space Mono', monospace";
Chart.defaults.scale.grid.color = 'rgba(255,255,255,0.05)';
Chart.defaults.scale.grid.borderColor = 'rgba(255,255,255,0.05)';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(10, 10, 10, 0.9)';
Chart.defaults.plugins.tooltip.titleColor = '#39ff14';
Chart.defaults.plugins.tooltip.titleFont = { family: "'Outfit', sans-serif", weight: 'bold' };
Chart.defaults.plugins.tooltip.bodyColor = '#f8fafc';
Chart.defaults.plugins.tooltip.borderColor = 'rgba(57, 255, 20, 0.3)';
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
            const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
            obj.innerHTML = (start + easeProgress * (end - start)).toFixed(end % 1 !== 0 ? 1 : 0);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    // --- Decorative Sparklines ---
    function drawDecorativeSparkline(canvasId, colorStr) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        canvas.width = w;
        canvas.height = h;

        ctx.clearRect(0, 0, w, h);
        ctx.beginPath();
        let pts = [];
        for (let i = 0; i <= w; i += 5) {
            pts.push({ x: i, y: h / 2 + (Math.random() * h / 2.5 * (Math.random() > 0.5 ? 1 : -1)) });
        }
        ctx.moveTo(0, h / 2);
        for (let i = 0; i < pts.length; i++) {
            ctx.lineTo(pts[i].x, pts[i].y);
        }
        ctx.strokeStyle = colorStr;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.fillStyle = colorStr.replace('rgb', 'rgba').replace(')', ', 0.1)');
        if (colorStr.startsWith('#')) ctx.fillStyle = colorStr + '20';
        ctx.fill();
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
            arcColor = COLORS.cyan;
            ringGlow = COLORS.cyanGlow;
            riskLabel = 'Moderate Risk';
            badgeClass = 'bg-cyan-500/10 text-[#00f0ff] border-[#00f0ff]/20';
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
            const borderClass = isFake ? 'card-fake' : 'card-genuine';
            const badgeClass = isFake ? 'bg-red-500/10 text-red-500 border-red-500/20' : 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20';
            const icon = isFake ? "ph-fill ph-warning-circle" : "ph-fill ph-check-circle";
            const stampHTML = isFake ? '<div class="stamp-fake">⚠ FLAGGED</div>' : '<div class="stamp-genuine">✓ VERIFIED</div>';

            // Render LIME Explanations
            let flagHTML = '';
            if (expList && expList.length > 0) {
                let badges = expList.map(item => {
                    const imp = Math.abs(item.weight).toFixed(2);
                    return `<span class="px-2 py-1 rounded bg-black border border-[#39ff14]/20 text-[10px] text-[#39ff14] font-mono tracking-widest inline-flex items-center" title="LIME Weight: ${imp}"><i class="ph-fill ph-sparkle mr-1 text-[#39ff14]"></i> ${item.word}</span>`;
                }).join('');

                flagHTML = `
                    <div class="mt-4 border-t border-[#39ff14]/20 pt-3 relative z-10">
                        <p class="text-[10px] text-[#39ff14]/70 mb-2 uppercase tracking-widest font-bold font-mono">AI Keyword Analysis</p>
                        <div class="flex flex-wrap gap-2">
                            ${badges}
                        </div>
                    </div>
                `;
            }

            const cleanText = reviewText.substring(0, 200) + (reviewText.length > 200 ? '...' : '');

            const cardHTML = `
                <div class="glass-panel p-5 rounded-xl flex flex-col justify-between ${borderClass} hover:translate-y-[-2px] transition-transform duration-300 relative overflow-hidden">
                    ${stampHTML}
                    <div class="relative z-10">
                        <div class="flex justify-between items-start mb-3">
                            <span class="px-2 py-1 rounded-full text-[10px] tracking-widest uppercase font-bold border ${badgeClass} flex items-center shadow-sm font-mono">
                                <i class="${icon} mr-1 text-sm"></i>
                                ${pred}
                            </span>
                            <div class="text-right">
                                <div class="text-[10px] text-slate-400 mb-0.5 font-mono"><i class="ph ph-calendar mr-1"></i>${dateStr}</div>
                                <span class="text-[10px] font-mono text-[#00f0ff] tracking-widest uppercase font-bold">Conf: ${conf}%</span>
                            </div>
                        </div>
                        <p class="text-sm text-slate-300 leading-relaxed font-mono">"${cleanText}"</p>
                    </div>
                    ${flagHTML}
                </div>
            `;
            reviewFeed.innerHTML += cardHTML;
        });
    }

    // --- 4. TEMPORAL CHART ---
    let temporalChartInstance = null;
    function buildTemporalChart(results) {
        if (temporalChartInstance) temporalChartInstance.destroy();
        const ctx = document.getElementById('temporalChart').getContext('2d');

        // Check if we have date fields and sort
        const timelineData = {};
        for (let i = 0; i < results.review.length; i++) {
            const dateStr = results.date[i];
            const pred = results.prediction[i];
            if (!dateStr) continue;

            const d = new Date(dateStr);
            if (isNaN(d)) continue;

            const yyyy_mm = d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, '0');

            if (!timelineData[yyyy_mm]) timelineData[yyyy_mm] = { fake: 0, genuine: 0 };

            if (pred === 'Fake') timelineData[yyyy_mm].fake++;
            else timelineData[yyyy_mm].genuine++;
        }

        const labels = Object.keys(timelineData).sort();
        const fakeData = labels.map(l => timelineData[l].fake);
        const genuineData = labels.map(l => timelineData[l].genuine);

        temporalChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Genuine Subs',
                        data: genuineData,
                        borderColor: COLORS.emerald,
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Fake Subs',
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

    // --- 4b. HEATMAP CHART ---
    let heatmapChartInstance = null;
    function buildHeatmapChart(results) {
        if (heatmapChartInstance) heatmapChartInstance.destroy();
        const ctx = document.getElementById('heatmapChart');
        if (!ctx) return;

        const timelineData = {};
        for (let i = 0; i < results.review.length; i++) {
            const dateStr = results.date[i];
            const pred = results.prediction[i];
            if (!dateStr) continue;

            const d = new Date(dateStr);
            if (isNaN(d)) continue;

            const yyyy_mm_dd = d.toISOString().split('T')[0];
            if (!timelineData[yyyy_mm_dd]) timelineData[yyyy_mm_dd] = { fake: 0, genuine: 0, total: 0 };

            if (pred === 'Fake') timelineData[yyyy_mm_dd].fake++;
            else timelineData[yyyy_mm_dd].genuine++;
            timelineData[yyyy_mm_dd].total++;
        }

        const labels = Object.keys(timelineData).sort();
        const freqData = labels.map(l => timelineData[l].total);
        const bgColors = labels.map(l => {
            const ratio = timelineData[l].fake / timelineData[l].total;
            if (ratio > 0.6) return 'rgba(255, 0, 60, 0.8)'; // Red
            if (ratio > 0.3) return 'rgba(245, 158, 11, 0.8)'; // Amber
            return 'rgba(57, 255, 20, 0.8)'; // Lime
        });

        heatmapChartInstance = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Review Density',
                    data: freqData,
                    backgroundColor: bgColors,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: COLORS.slate400 } },
                    y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: COLORS.slate400, precision: 0 } }
                }
            }
        });
    }

    // --- 5. WORD CLOUD ---
    function buildWordCloud(results) {
        if (!window.WordCloud) return; // Library not loaded

        const stopwords = new Set(["the", "and", "a", "to", "of", "in", "i", "is", "that", "it", "on", "you", "this", "for", "but", "with", "are", "have", "be", "at", "or", "as", "was", "so", "if", "out", "not", "my", "they", "we", "all", "just", "like", "very", "can", "will", "no", "there", "what", "when", "has", "do", "more", "me", "up", "an", "about", "which", "one", "from", "some", "would", "would", "get", "their"]);

        const counts = { fake: {}, genuine: {} };
        const combined = {};

        for (let i = 0; i < results.review.length; i++) {
            const text = results.review[i].toLowerCase();
            const pred = results.prediction[i]; // 'Fake' or 'Genuine'

            // Extract words (3 chars or more)
            const words = text.match(/\b[a-z]{3,}\b/g) || [];

            words.forEach(w => {
                if (stopwords.has(w)) return;

                if (pred === 'Fake') {
                    counts.fake[w] = (counts.fake[w] || 0) + 1;
                } else {
                    counts.genuine[w] = (counts.genuine[w] || 0) + 1;
                }
                combined[w] = (combined[w] || 0) + 1;
            });
        }

        // Convert to array format expected by wordcloud2.js: [word, size]
        const list = [];
        const maxWords = 60;

        // Sort combined to get top words
        const sortedWords = Object.keys(combined).sort((a, b) => combined[b] - combined[a]).slice(0, maxWords);

        sortedWords.forEach(w => {
            // we scale frequency to make it decent
            list.push([w, combined[w] * 3 + 12]);
        });

        // Clear canvas first
        const canvas = document.getElementById('wordCloudCanvas');
        if (!canvas) return;

        const wrapper = document.getElementById('wordCloudWrapper');
        canvas.width = wrapper.clientWidth;
        canvas.height = wrapper.clientHeight;

        WordCloud(canvas, {
            list: list,
            fontFamily: 'Space Mono, monospace',
            weightFactor: function (size) {
                return size; // scale factor
            },
            color: function (word, weight) {
                // Determine if this word is predominantly used in fake or genuine reviews
                const f = counts.fake[word] || 0;
                const g = counts.genuine[word] || 0;

                // Color by density
                const total = f + g;
                if (total === 0) return '#00f0ff';

                if (f / total >= 0.6) return '#ff003c'; // Red
                if (g / total >= 0.6) return '#10b981'; // Green
                return '#94a3b8'; // Neutral Slate
            },
            rotateRatio: 0.1,
            rotationSteps: 2,
            backgroundColor: 'transparent',
            shape: 'circle',
            drawOutOfBound: false,
            shrinkToFit: true
        });
    }

    // --- Threat Scanner & AI Log Simulation ---
    let aiLogInterval;
    const aiLogMessages = [
        "Establishing secure connection to Amazon data cluster...",
        "Bypassing standard captchas...",
        "Scraping review payloads... [OK]",
        "Initializing DistilBERT sequence classification...",
        "Running NLP model over extracted entities...",
        "Cross-referencing temporal activity spikes...",
        "Detecting bot patterns and repetitive syntax...",
        "Calculating authenticity confidence scores...",
        "Finalizing heuristic protocols...",
        "Data compiled. Preparing visual synthesis..."
    ];

    function showThreatScanner() {
        const overlay = document.getElementById('threatScannerOverlay');
        const logStream = document.getElementById('aiLogStream');
        if (!overlay || !logStream) return;

        logStream.innerHTML = '';
        overlay.classList.remove('hidden', 'opacity-0');

        let msgIndex = 0;
        aiLogInterval = setInterval(() => {
            if (msgIndex >= aiLogMessages.length) {
                // Keep repeating the last few or random to show activity if it takes long
                const randomMsg = aiLogMessages[Math.floor(Math.random() * (aiLogMessages.length - 2)) + 2];
                addLogLine(logStream, `[RETRY] ${randomMsg}`);
            } else {
                addLogLine(logStream, `> ${aiLogMessages[msgIndex]}`);
                msgIndex++;
            }
        }, 800);
    }

    function addLogLine(container, text) {
        const line = document.createElement('div');
        line.className = 'log-line';
        line.innerHTML = `<span class="text-[#39ff14]/70 mr-2">[${new Date().toISOString().split('T')[1].slice(0, -1)}]</span> ${text}`;
        container.appendChild(line);
        container.scrollTop = container.scrollHeight;
    }

    function hideThreatScanner() {
        const overlay = document.getElementById('threatScannerOverlay');
        if (!overlay) return;
        overlay.classList.add('opacity-0');
        clearInterval(aiLogInterval);
        setTimeout(() => {
            overlay.classList.add('hidden');
        }, 500);
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
        showThreatScanner();

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

            // --- Remove skeleton loader & overlay ---
            toggleSkeletons(false);
            hideThreatScanner();

            // Populate Product Meta below search bar
            const metaDiv = document.getElementById('productMeta');
            document.getElementById('metaAsin').innerText = data.asin || asin;
            document.getElementById('metaTime').innerText = new Date().toLocaleTimeString();

            // Set active case number
            const caseNum = "CASE-" + new Date().getTime().toString().slice(-6);
            document.getElementById('activeCaseNumber').innerText = caseNum;

            // Set Data
            const sum = data.summary;
            document.getElementById('metaTitle').innerText = sum.product_title || 'Unknown Product';
            const ratingEl = document.getElementById('metaRating');
            if (ratingEl) ratingEl.innerText = sum.product_rating || '0.0';
            const revTotEl = document.getElementById('metaReviewsTotal');
            if (revTotEl) revTotEl.innerText = sum.product_reviews_total || '0';

            const metaImg = document.getElementById('metaImage');
            const placeholder = document.getElementById('metaImagePlaceholder');
            if (metaImg && placeholder && sum.product_image) {
                metaImg.src = sum.product_image;
                metaImg.classList.remove('hidden');
                placeholder.classList.add('hidden');
            } else if (metaImg && placeholder) {
                metaImg.classList.add('hidden');
                placeholder.classList.remove('hidden');
            }

            metaDiv.classList.remove('hidden', 'opacity-0');

            // --- 1. Populate Metrics with CountUp Animations ---
            // progress bars cleanup
            document.querySelectorAll('.skeleton').forEach(el => el.classList.remove('skeleton'));

            animateValue(statElements.total, 0, sum.total_reviews, 1500);
            animateValue(statElements.fake, 0, sum.fake_percent, 1500);
            animateValue(statElements.trust, 0, sum.genuine_percent, 1500);
            animateValue(statElements.accuracy, 0, sum.avg_confidence, 1500);

            // Draw Decorative sparklines
            drawDecorativeSparkline('sparklineTotal', '#f8fafc');
            drawDecorativeSparkline('sparklineAccuracy', '#f59e0b');

            // Progress Bar widths
            setTimeout(() => {
                document.getElementById('fakeRiskBar').style.width = `${sum.fake_percent}%`;
                document.getElementById('trustScoreBar').style.width = `${sum.genuine_percent}%`;
            }, 100);

            // --- 2. Build Charts ---
            initOrUpdateCharts(data.chart_data.fake, data.chart_data.genuine, sum.fake_percent);

            // --- 3. Build Feed ---
            renderReviewFeed(data.results.review, data.results.prediction, data.results.confidence, data.results.date, data.results.explanation);

            // --- 4. Build Temporal Diagrams ---
            buildTemporalChart(data.results);
            buildHeatmapChart(data.results);

            // --- 5. Build Word Cloud ---
            buildWordCloud(data.results);

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
            console.error("Analysis failed:", error);
            if (window.showToast) window.showToast(error.message || "Failed to analyze product. Please try again.", 'error');
            else alert("Failed to analyze product. Please try again.");

            toggleSkeletons(false);
            hideThreatScanner();
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

