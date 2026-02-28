// Set defaults for Chart.js
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';
Chart.defaults.scale.grid.borderColor = 'rgba(255, 255, 255, 0.05)';

// Main Trend Chart
const trendCtx = document.getElementById('trendChart').getContext('2d');

// Create gradient for the line
const gradientGenuine = trendCtx.createLinearGradient(0, 0, 0, 300);
gradientGenuine.addColorStop(0, 'rgba(16, 185, 129, 0.4)');
gradientGenuine.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

const gradientFake = trendCtx.createLinearGradient(0, 0, 0, 300);
gradientFake.addColorStop(0, 'rgba(239, 68, 68, 0.4)');
gradientFake.addColorStop(1, 'rgba(239, 68, 68, 0.0)');

new Chart(trendCtx, {
    type: 'line',
    data: {
        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        datasets: [
            {
                label: 'Genuine Reviews',
                data: [650, 720, 680, 850, 920, 880, 950],
                borderColor: '#10b981',
                backgroundColor: gradientGenuine,
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointBackgroundColor: '#0a0f1c',
                pointBorderColor: '#10b981',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6
            },
            {
                label: 'Fake Detected',
                data: [120, 150, 110, 190, 240, 180, 210],
                borderColor: '#ef4444',
                backgroundColor: gradientFake,
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointBackgroundColor: '#0a0f1c',
                pointBorderColor: '#ef4444',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(17, 24, 39, 0.9)',
                titleColor: '#f8fafc',
                bodyColor: '#e2e8f0',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                padding: 12,
                boxPadding: 6,
                usePointStyle: true,
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    drawBorder: false,
                },
                ticks: {
                    padding: 10
                }
            },
            x: {
                grid: {
                    display: false,
                    drawBorder: false
                },
                ticks: {
                    padding: 10
                }
            }
        },
        interaction: {
            intersect: false,
            mode: 'index',
        },
    }
});

// Distribution Doughnut Chart
const distCtx = document.getElementById('distributionChart').getContext('2d');

new Chart(distCtx, {
    type: 'doughnut',
    data: {
        labels: ['Genuine', 'Suspicious', 'Fake'],
        datasets: [{
            data: [65, 22, 13],
            backgroundColor: [
                '#10b981',
                '#f59e0b',
                '#ef4444'
            ],
            borderWidth: 0,
            hoverOffset: 4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '75%',
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                backgroundColor: 'rgba(17, 24, 39, 0.9)',
                padding: 12,
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                callbacks: {
                    label: function (context) {
                        return ' ' + context.label + ': ' + context.raw + '%';
                    }
                }
            }
        }
    },
    plugins: [{
        id: 'textCenter',
        beforeDraw: function (chart) {
            var width = chart.width,
                height = chart.height,
                ctx = chart.ctx;

            ctx.restore();
            var fontSize = (height / 114).toFixed(2);
            ctx.font = "600 " + fontSize + "em Inter";
            ctx.textBaseline = "middle";
            ctx.fillStyle = "#f8fafc";

            var text = "18.3K",
                textX = Math.round((width - ctx.measureText(text).width) / 2),
                textY = height / 2 - 10;

            ctx.fillText(text, textX, textY);

            ctx.font = "400 " + (fontSize * 0.4).toFixed(2) + "em Inter";
            ctx.fillStyle = "#94a3b8";
            var text2 = "Total Flags",
                textX2 = Math.round((width - ctx.measureText(text2).width) / 2),
                textY2 = height / 2 + 15;

            ctx.fillText(text2, textX2, textY2);
            ctx.save();
        }
    }]
});
