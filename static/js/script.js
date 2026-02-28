Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';
Chart.defaults.scale.grid.borderColor = 'rgba(255, 255, 255, 0.05)';

document.addEventListener('DOMContentLoaded', () => {
    // Check elements exist
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (!analyzeBtn) return;

    analyzeBtn.addEventListener('click', async () => {
        const asinInput = document.getElementById('asinInput');
        const analyzeBtnText = document.getElementById('analyzeBtnText');
        const icon = analyzeBtn.querySelector('i');
        const asin = asinInput.value.trim();

        if (!asin) {
            alert('Please enter an ASIN or Amazon URL');
            return;
        }

        // Loading State UI
        analyzeBtn.disabled = true;
        analyzeBtnText.innerText = 'Analyzing Data...';
        icon.className = 'ph ph-spinner animate-spin mr-2 text-lg';

        // Setup empty feed
        document.getElementById('reviewFeed').innerHTML = `
            <div class="space-y-4">
                <div class="glass-panel p-5 rounded-xl skeleton h-24 border-left-genuine"></div>
                <div class="glass-panel p-5 rounded-xl skeleton h-24 border-left-fake"></div>
                <div class="glass-panel p-5 rounded-xl skeleton h-24 border-left-genuine"></div>
            </div>
        `;

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ asin: asin })
            });

            const data = await response.json();

            if (data.error) {
                alert('Error: ' + data.error);
                resetBtn();
                return;
            }

            // --- 1. UPDATE 4 GRID METRICS ---
            animateValue('statTotal', 0, data.summary.total_reviews, 1000);
            animateValue('statFakeCount', 0, data.chart_data.fake, 1000);

            document.getElementById('statTrustScore').innerText = data.summary.genuine_percent + "%";
            const trustBar = document.getElementById('trustScoreBar');
            if (trustBar) {
                trustBar.style.width = data.summary.genuine_percent + "%";
                if (data.summary.genuine_percent < 50) trustBar.className = "bg-red-500 h-1.5 rounded-full transition-all duration-1000";
                else if (data.summary.genuine_percent < 80) trustBar.className = "bg-yellow-400 h-1.5 rounded-full transition-all duration-1000";
                else trustBar.className = "bg-emerald-400 h-1.5 rounded-full transition-all duration-1000";
            }

            document.getElementById('statAccuracy').innerText = data.summary.avg_confidence + "%";

            // --- 2. UPDATE CIRCULAR GAUGE (SVG) ---
            updateGauge(data.summary.genuine_percent);

            // --- 3. RENDERING CHARTS ---
            renderDistributionChart(data.chart_data);

            // --- 4. REVIEW INTELLIGENCE FEED (Cards) ---
            renderReviewFeed(data.results.review, data.results.prediction, data.results.confidence);

        } catch (error) {
            console.error('Error fetching data:', error);
            alert('Failed to analyze product. Check console.');
        } finally {
            resetBtn();
        }

        function resetBtn() {
            analyzeBtn.disabled = false;
            analyzeBtnText.innerText = 'Analyze Now';
            icon.className = 'ph ph-magic-wand mr-2 text-lg';
        }
    });
});

// Animation helper for numbers
function animateValue(id, start, end, duration) {
    if (start === end) return;
    const obj = document.getElementById(id);
    if (!obj) return;

    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            obj.innerHTML = end; // ensure exact final value
        }
    };
    window.requestAnimationFrame(step);
}

// Update SVG Gauge
function updateGauge(score) {
    const gaugeScore = document.getElementById('gaugeScore');
    const gaugePath = document.getElementById('gaugePath');
    const riskLabel = document.getElementById('riskLabel');
    if (!gaugeScore || !gaugePath) return;

    animateValue('gaugeScore', 0, score, 1500);

    // Calculate stroke dasharray (circumference of r=15.9155 is 100)
    setTimeout(() => {
        gaugePath.style.strokeDasharray = `${score}, 100`;

        let colorClass = "text-emerald-500";
        let label = "Safe / Authentic";
        let labelColor = "text-emerald-400";

        if (score < 50) {
            colorClass = "text-red-500";
            label = "High Risk Level";
            labelColor = "text-red-400";
        } else if (score < 80) {
            colorClass = "text-yellow-400";
            label = "Moderate Suspicion";
            labelColor = "text-yellow-400";
        }

        gaugePath.setAttribute('class', `transition-all duration-1000 ease-out ${colorClass}`);
        if (riskLabel) {
            riskLabel.innerText = label;
            riskLabel.className = `ml-2 font-semibold ${labelColor}`;
        }
    }, 100);
}

// Render Review Feed
function renderReviewFeed(reviews, predictions, confidences) {
    const feed = document.getElementById('reviewFeed');
    if (!feed) return;

    feed.innerHTML = '';

    // Sort logic or just show top 10 for performance
    const limit = Math.min(reviews.length, 50);

    // Create elements
    for (let i = 0; i < limit; i++) {
        const text = reviews[i];
        const isFake = predictions[i] === 'Fake';
        const conf = confidences[i];

        const cardClass = isFake ? 'border-left-fake' : 'border-left-genuine';
        const badgeColor = isFake ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
        const icon = isFake ? 'ph-shield-warning' : 'ph-check-circle';

        // Truncate long reviews
        const limitStr = 150;
        const truncated = text.length > limitStr ? text.substring(0, limitStr) + '...' : text;
        const hasMore = text.length > limitStr;

        const html = `
            <div class="glass-panel p-5 rounded-xl ${cardClass} hover:bg-slate-800/80 transition-colors">
                <div class="flex justify-between items-start mb-3">
                    <div class="flex items-center space-x-3">
                        <div class="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center border border-slate-700">
                            <i class="ph ph-user text-slate-400"></i>
                        </div>
                        <div>
                            <span class="text-sm font-medium text-slate-200">Amazon Customer</span>
                            <div class="flex items-center mt-0.5 space-x-2">
                                <span class="text-xs px-2 py-0.5 rounded-full ${badgeColor} flex items-center">
                                    <i class="ph ${icon} mr-1"></i> ${isFake ? 'Fake' : 'Genuine'}
                                </span>
                                <span class="text-xs text-slate-500">Conf: ${conf}%</span>
                            </div>
                        </div>
                    </div>
                    <button class="text-slate-500 hover:text-cyan-400 transition" title="View Details">
                        <i class="ph ph-dots-three-circle text-xl"></i>
                    </button>
                </div>
                
                <p class="text-slate-300 text-sm leading-relaxed whitespace-pre-line review-text" data-full="${escapeHTML(text)}">
                    ${escapeHTML(truncated)}
                </p>
                
                ${hasMore ? `<button class="text-cyan-400 hover:text-cyan-300 text-xs mt-2 font-medium" onclick="expandReview(this)">Read full review</button>` : ''}
                
                ${isFake ? `
                <div class="mt-4 pt-3 border-t border-slate-700/50 flex flex-wrap gap-2">
                    <span class="text-xs text-slate-500">Flags:</span>
                    <span class="text-xs px-2 py-0.5 rounded-md bg-slate-800 text-slate-400 border border-slate-700">Generic Language</span>
                    <span class="text-xs px-2 py-0.5 rounded-md bg-slate-800 text-slate-400 border border-slate-700">Missing detail</span>
                </div>
                ` : ''}
            </div>
        `;
        feed.insertAdjacentHTML('beforeend', html);
    }
}

// Helper to prevent XSS
function escapeHTML(str) {
    return str.replace(/[&<>'"]/g,
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

// Handle expanding text inline
window.expandReview = function (btn) {
    const p = btn.previousElementSibling;
    p.innerHTML = p.getAttribute('data-full');
    btn.remove();
}

// Global Dist Chart instance definition
let distChartInstance = null;

function renderDistributionChart(chartData) {
    const ctx = document.getElementById('distributionChart');
    if (!ctx) return;

    if (distChartInstance) {
        distChartInstance.destroy();
    }

    // A bar chart to show distribution of Fake vs Genuine predictions
    distChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Authentic', 'Fake / AI Gen'],
            datasets: [{
                label: 'Number of Reviews',
                data: [chartData.genuine, chartData.fake],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)', // Emerald
                    'rgba(239, 68, 68, 0.8)'   // Red
                ],
                borderColor: [
                    '#10b981',
                    '#ef4444'
                ],
                borderWidth: 1,
                borderRadius: 4,
                barPercentage: 0.6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 10
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutQuart'
            }
        }
    });
}
