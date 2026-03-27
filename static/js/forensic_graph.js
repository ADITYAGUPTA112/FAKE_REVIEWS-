/**
 * Obsidian Lens – Forensic Network Graph Renderer
 * Uses vis.js Network to visualize reviewer coordination clusters.
 */

class ForensicGraph {
    constructor(containerId) {
        this.containerId = containerId;
        this.network = null;
        this.nodes = null;
        this.edges = null;
    }

    render(networkData) {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        const clusters = networkData.suspicious_clusters || [];
        const graphSummary = networkData.graph_summary || {};

        if (clusters.length === 0) {
            container.innerHTML = `
                <div class="flex flex-col items-center justify-center h-full opacity-60">
                    <span class="material-symbols-outlined text-5xl text-primary/40 mb-3">hub</span>
                    <p class="text-sm text-slate-500">No suspicious reviewer clusters detected</p>
                    <p class="text-xs text-slate-600 mt-1">Network appears clean for this product</p>
                </div>`;
            return;
        }

        // Build vis.js dataset
        const nodeSet = new Map();
        const edgeList = [];
        let edgeId = 0;

        clusters.forEach((cluster, ci) => {
            const hue = ci * 60; // Rotate colors per cluster
            const color = `hsl(${180 + hue}, 90%, 60%)`;
            const bgColor = `hsla(${180 + hue}, 90%, 60%, 0.15)`;

            cluster.users.forEach(user => {
                if (!nodeSet.has(user)) {
                    nodeSet.set(user, {
                        id: user,
                        label: user.length > 12 ? user.substring(0, 12) + '…' : user,
                        color: {
                            background: bgColor,
                            border: color,
                            highlight: { background: color, border: '#fff' },
                            hover: { background: color, border: '#fff' }
                        },
                        borderWidth: 2,
                        font: { color: '#e2e8f0', size: 10, face: 'Inter' },
                        shape: 'dot',
                        size: 14 + (cluster.risk_score / 10),
                        shadow: { enabled: true, color: color, size: 10, x: 0, y: 0 },
                        title: `Reviewer: ${user}\nCluster Risk: ${cluster.risk_score}%\nDensity: ${cluster.density}`
                    });
                }
            });

            // Create edges between all users in this cluster
            for (let i = 0; i < cluster.users.length; i++) {
                for (let j = i + 1; j < cluster.users.length; j++) {
                    edgeList.push({
                        id: edgeId++,
                        from: cluster.users[i],
                        to: cluster.users[j],
                        color: { color: `hsla(${180 + hue}, 80%, 50%, 0.3)`, highlight: color, hover: color },
                        width: 1 + (cluster.avg_edge_weight / 3),
                        smooth: { type: 'continuous' },
                        title: `Shared products: ${cluster.products.length}\nCo-review events: ${cluster.total_co_review_events}`
                    });
                }
            }
        });

        this.nodes = new vis.DataSet(Array.from(nodeSet.values()));
        this.edges = new vis.DataSet(edgeList);

        const options = {
            physics: {
                enabled: true,
                forceAtlas2Based: {
                    gravitationalConstant: -30,
                    centralGravity: 0.005,
                    springLength: 120,
                    springConstant: 0.06,
                    damping: 0.4
                },
                solver: 'forceAtlas2Based',
                stabilization: { iterations: 80, fit: true }
            },
            interaction: {
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true,
                navigationButtons: false
            },
            nodes: {
                shape: 'dot',
                font: { color: '#94a3b8', size: 10, face: 'Inter, sans-serif' }
            },
            edges: {
                smooth: { type: 'continuous' }
            },
            layout: { improvedLayout: true }
        };

        container.innerHTML = '';
        this.network = new vis.Network(container, { nodes: this.nodes, edges: this.edges }, options);

        // Add cluster legend
        const legend = document.createElement('div');
        legend.className = 'absolute top-3 left-3 z-10 space-y-1';
        legend.innerHTML = clusters.map((c, i) => {
            const hue = i * 60;
            return `<div class="flex items-center gap-2 text-xs">
                <span class="w-3 h-3 rounded-full" style="background:hsl(${180 + hue},90%,60%)"></span>
                <span class="text-slate-400">Cluster ${c.cluster_id} · Risk: <span class="font-bold text-red-400">${c.risk_score}%</span> · ${c.cluster_size} users</span>
            </div>`;
        }).join('');
        container.style.position = 'relative';
        container.appendChild(legend);
    }

    destroy() {
        if (this.network) {
            this.network.destroy();
            this.network = null;
        }
    }
}

// Export for global use
window.ForensicGraph = ForensicGraph;
