GRAPH_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Engram - Knowledge Graph</title>
    <link rel="icon" href="/favicon.ico">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #05060d; font-family: -apple-system, sans-serif; overflow: hidden; }
        #canvas { width: 100vw; height: 100vh; display: block; }
        #label-canvas { position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 5; }

        #info { position: absolute; top: 16px; left: 16px; color: #8b949e; z-index: 10; pointer-events: none; }
        #info h1 { font-size: 11px; color: #5eead480; font-weight: 400; text-transform: uppercase; letter-spacing: 3px; margin-top: 6px; }
        #info .brand { font-family: 'Orbitron', sans-serif; font-size: 28px; font-weight: 700; color: #5eead4; text-shadow: 0 0 20px #5eead440; letter-spacing: 3px; }

        #controls {
            position: absolute; bottom: 16px; left: 16px;
            background: rgba(10,10,18,0.95); padding: 12px 16px;
            border-radius: 8px; border: 1px solid #1a1a2e;
            font-size: 11px; color: #8b949e; z-index: 100;
        }
        .legend-item { display: flex; align-items: center; padding: 5px 0; cursor: pointer; border-radius: 4px; padding: 6px 8px; margin: 2px -8px; }
        .legend-item:hover { background: rgba(255,255,255,0.05); }
        .legend-item.active { background: rgba(94,234,212,0.15); }
        .legend-item.dimmed { opacity: 0.4; }
        .legend-dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; }
        .legend-count { margin-left: auto; padding-left: 16px; color: #5eead4; font-weight: 500; }
        .control-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; padding-top: 10px; border-top: 1px solid #1a1a2e; }
        .toggle-btn {
            padding: 6px 12px; background: #0f0f1a; border: 1px solid #1a1a2e;
            border-radius: 4px; color: #8b949e; cursor: pointer; font-size: 11px;
        }
        .toggle-btn:hover { border-color: #5eead4; color: #5eead4; }
        .toggle-btn.active { background: #5eead420; border-color: #5eead4; color: #5eead4; }

        #search-box { position: absolute; top: 16px; left: 50%; transform: translateX(-50%); z-index: 100; }
        #search-input {
            width: 350px; padding: 10px 16px;
            background: rgba(10,10,18,0.95); border: 1px solid #1a1a2e;
            border-radius: 6px; color: #e0e0ff; font-size: 14px;
        }
        #search-input:focus { outline: none; border-color: #5eead4; }
        #search-input::placeholder { color: #4a4a6a; }
        #search-results {
            position: absolute; top: 100%; left: 0; right: 0;
            max-height: 300px; overflow-y: auto;
            background: rgba(10,10,18,0.98); border: 1px solid #1a1a2e;
            border-top: none; border-radius: 0 0 6px 6px; display: none;
        }
        #search-results.visible { display: block; }
        .search-result { padding: 10px 16px; cursor: pointer; border-bottom: 1px solid #1a1a2e; display: flex; align-items: center; }
        .search-result:hover { background: #1a1a2e; }
        .search-result-name { flex: 1; color: #e0e0ff; font-size: 12px; }
        .search-result-type { font-size: 10px; padding: 2px 6px; border-radius: 4px; text-transform: uppercase; }

        #stats { position: absolute; top: 16px; right: 16px; background: rgba(10,10,18,0.95); padding: 12px 16px; border-radius: 6px; border: 1px solid #1a1a2e; font-size: 11px; color: #8b8ba0; z-index: 10; }
        #loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #5eead4; z-index: 10; }

        #tooltip {
            position: absolute; pointer-events: none; z-index: 1000;
            background: rgba(10,10,18,0.95); padding: 8px 12px;
            border-radius: 6px; border: 1px solid #5eead4;
            font-size: 12px; color: #e0e0ff; max-width: 300px;
            display: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        #tooltip.visible { display: block; }

        #node-info {
            position: absolute; top: 100px; right: 16px; width: 300px; max-height: 500px; overflow-y: auto;
            background: rgba(10,10,18,0.95); padding: 16px; border-radius: 8px; border: 1px solid #1a1a2e;
            font-size: 12px; color: #e0e0ff; display: none; z-index: 10;
        }
        #node-info.visible { display: block; }
        #node-info h3 { margin: 0 0 12px 0; font-size: 14px; color: #fff; word-break: break-word; }
        #node-info .type-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 500; text-transform: uppercase; margin-bottom: 12px; }
        #node-info .info-row { margin: 8px 0; padding: 8px 0; border-top: 1px solid #1a1a2e; }
        #node-info .info-label { color: #6b6b8a; font-size: 10px; text-transform: uppercase; margin-bottom: 4px; }
        #node-info .info-value { color: #e0e0ff; word-break: break-word; }
        #node-info .close-btn { position: absolute; top: 8px; right: 8px; background: none; border: none; color: #6b6b8a; cursor: pointer; font-size: 16px; }
        #node-info .close-btn:hover { color: #5eead4; }
        .neighbor-item { padding: 6px 8px; margin: 4px 0; background: rgba(255,255,255,0.03); border-radius: 4px; cursor: pointer; display: flex; align-items: center; }
        .neighbor-item:hover { background: rgba(94,234,212,0.1); }
        .neighbor-dot { width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }
        .neighbor-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

        #mode-indicator { position: absolute; bottom: 16px; right: 16px; background: rgba(94,234,212,0.1); border: 1px solid #5eead4; padding: 8px 12px; border-radius: 6px; font-size: 10px; color: #5eead4; z-index: 10; min-width: 140px; text-align: center; }

        /* Chat Panel */
        #chat-toggle {
            position: absolute; top: 50%; right: 0; transform: translateY(-50%);
            background: rgba(10,10,18,0.95); border: 1px solid #5eead4; border-right: none;
            padding: 12px 8px; border-radius: 8px 0 0 8px; cursor: pointer;
            color: #5eead4; font-size: 18px; z-index: 200;
            transition: right 0.3s ease;
        }
        #chat-toggle:hover { background: rgba(94,234,212,0.2); }
        #chat-toggle.open { right: 380px; }

        #chat-panel {
            position: absolute; top: 0; right: -400px; width: 380px; height: 100vh;
            background: rgba(10,10,18,0.98); border-left: 1px solid #1a1a2e;
            display: flex; flex-direction: column; z-index: 150;
            transition: right 0.3s ease;
        }
        #chat-panel.open { right: 0; }

        #chat-header {
            padding: 16px; border-bottom: 1px solid #1a1a2e;
            display: flex; align-items: center; justify-content: space-between;
        }
        #chat-header h2 { font-size: 14px; color: #5eead4; font-weight: 500; margin: 0; }
        #chat-header .chat-actions { display: flex; gap: 8px; }
        .chat-action-btn {
            padding: 4px 10px; background: #0f0f1a; border: 1px solid #1a1a2e;
            border-radius: 4px; color: #8b949e; cursor: pointer; font-size: 10px;
        }
        .chat-action-btn:hover { border-color: #5eead4; color: #5eead4; }
        .chat-action-btn.active { background: #5eead420; border-color: #5eead4; color: #5eead4; }

        /* Debug Panel Styles */
        .debug-section {
            margin-top: 10px; padding: 10px; background: rgba(15,15,26,0.8);
            border: 1px solid #2a2a4e; border-radius: 8px; font-size: 11px;
        }
        .debug-section.collapsed .debug-content { display: none; }
        .debug-header {
            display: flex; align-items: center; justify-content: space-between;
            cursor: pointer; padding: 4px; color: #8b949e;
        }
        .debug-header:hover { color: #5eead4; }
        .debug-header .toggle-icon { transition: transform 0.2s; }
        .debug-section:not(.collapsed) .debug-header .toggle-icon { transform: rotate(90deg); }
        .debug-content { margin-top: 10px; }
        .debug-tab-bar { display: flex; gap: 4px; margin-bottom: 10px; }
        .debug-tab-bar button {
            padding: 4px 12px; background: #0f0f1a; border: 1px solid #2a2a4e;
            border-radius: 4px; color: #8b949e; cursor: pointer; font-size: 10px;
        }
        .debug-tab-bar button:hover { border-color: #5eead4; }
        .debug-tab-bar button.active { background: #5eead420; border-color: #5eead4; color: #5eead4; }
        .debug-list { max-height: 250px; overflow-y: auto; }
        .debug-item {
            display: flex; align-items: center; gap: 8px; padding: 6px 8px;
            margin: 4px 0; background: rgba(255,255,255,0.02); border-radius: 4px;
            position: relative; cursor: pointer;
        }
        .debug-item:hover { background: rgba(94,234,212,0.1); }
        .debug-item.filtered { opacity: 0.5; }
        .debug-item.filtered:hover { opacity: 0.8; }
        .debug-item .score-bar {
            position: absolute; left: 0; top: 0; height: 100%;
            background: rgba(94,234,212,0.15); border-radius: 4px; z-index: 0;
        }
        .debug-item .name {
            flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            position: relative; z-index: 1; color: #e0e0ff;
        }
        .debug-item .score { color: #5eead4; font-weight: 500; position: relative; z-index: 1; min-width: 40px; text-align: right; }
        .debug-item .sources { display: flex; gap: 3px; position: relative; z-index: 1; }
        .debug-item .source-badge {
            padding: 1px 4px; background: #2a2a4e; border-radius: 3px;
            font-size: 9px; color: #8b949e; text-transform: uppercase;
        }
        .debug-item .source-badge.vector { background: #5eead420; color: #5eead4; }
        .debug-item .source-badge.bm25 { background: #a78bfa20; color: #a78bfa; }
        .debug-item .source-badge.graph { background: #f472b620; color: #f472b6; }
        .debug-item .source-badge.forced { background: #fbbf2420; color: #fbbf24; }
        .debug-item .test-btn {
            padding: 2px 6px; background: #0f0f1a; border: 1px solid #2a2a4e;
            border-radius: 3px; color: #8b949e; cursor: pointer; font-size: 12px;
            position: relative; z-index: 1;
        }
        .debug-item .test-btn:hover { border-color: #5eead4; color: #5eead4; }
        .debug-item .hop-badge {
            padding: 1px 4px; background: #2a2a4e; border-radius: 3px;
            font-size: 9px; color: #8b949e;
        }
        .sources-legend {
            margin-top: 8px; padding-top: 8px; border-top: 1px solid #2a2a4e;
            font-size: 9px; color: #6b6b8a;
        }
        .retest-indicator {
            padding: 8px; margin: 8px 0; background: #fbbf2410; border: 1px solid #fbbf2440;
            border-radius: 6px; font-size: 11px; color: #fbbf24; text-align: center;
        }

        #chat-messages {
            flex: 1; overflow-y: auto; padding: 16px;
            display: flex; flex-direction: column; gap: 12px;
        }
        .chat-message {
            max-width: 90%; padding: 10px 14px; border-radius: 12px;
            font-size: 13px; line-height: 1.5; word-wrap: break-word;
        }
        .chat-message.user {
            align-self: flex-end; background: #5eead420; color: #e0e0ff;
            border-bottom-right-radius: 4px;
        }
        .chat-message.assistant {
            align-self: flex-start; background: #1a1a2e; color: #e0e0ff;
            border-bottom-left-radius: 4px;
        }
        .chat-message.system {
            align-self: center; background: transparent; color: #6b6b8a;
            font-size: 11px; text-align: center;
        }
        .chat-message .activated-info {
            margin-top: 8px; padding-top: 8px; border-top: 1px solid #2a2a4e;
            font-size: 11px; color: #5eead4; cursor: pointer;
        }
        .chat-message .activated-info:hover { text-decoration: underline; }

        #chat-input-area {
            padding: 12px 16px; border-top: 1px solid #1a1a2e;
            display: flex; gap: 8px;
        }
        #chat-input {
            flex: 1; padding: 10px 14px; background: #0f0f1a;
            border: 1px solid #1a1a2e; border-radius: 8px;
            color: #e0e0ff; font-size: 13px; resize: none;
        }
        #chat-input:focus { outline: none; border-color: #5eead4; }
        #chat-input::placeholder { color: #4a4a6a; }
        #chat-send {
            padding: 10px 16px; background: #5eead4; border: none;
            border-radius: 8px; color: #0a0a12; font-weight: 600;
            cursor: pointer; font-size: 13px;
        }
        #chat-send:hover { background: #4dd4c0; }
        #chat-send:disabled { background: #2a2a4e; color: #6b6b8a; cursor: not-allowed; }

        /* Activation glow animation */
        @keyframes activationPulse {
            0% { opacity: 0.3; }
            50% { opacity: 1; }
            100% { opacity: 0.3; }
        }
        .activation-glow { animation: activationPulse 1.5s ease-in-out infinite; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <canvas id="label-canvas"></canvas>
    <div id="info"><div class="brand">Engram</div><h1>Memory Graph</h1></div>

    <div id="search-box">
        <input type="text" id="search-input" placeholder="Search all nodes... (Enter to search)">
        <div id="search-results"></div>
    </div>

    <div id="tooltip"></div>

    <div id="node-info">
        <button class="close-btn" onclick="closeNodeInfo()">&times;</button>
        <div id="node-info-content"></div>
    </div>

    <div id="controls">
        <div class="legend-item" data-type="concept" onclick="toggleTypeFilter('concept')">
            <span class="legend-dot" style="background:#5eead4"></span>Concept<span class="legend-count" id="c-count">0</span>
        </div>
        <div class="legend-item" data-type="semantic" onclick="toggleTypeFilter('semantic')">
            <span class="legend-dot" style="background:#a78bfa"></span>Semantic<span class="legend-count" id="s-count">0</span>
        </div>
        <div class="legend-item" data-type="episodic" onclick="toggleTypeFilter('episodic')">
            <span class="legend-dot" style="background:#f472b6"></span>Episodic<span class="legend-count" id="e-count">0</span>
        </div>
        <div class="control-row">
        </div>
    </div>

    <div id="stats">Loading...</div>
    <div id="loading">Loading graph...</div>
    <div id="mode-indicator">WebGL</div>

    <button id="chat-toggle" onclick="toggleChat()">💬</button>
    <div id="chat-panel">
        <div id="chat-header">
            <h2>Chat with Memory</h2>
            <div class="chat-actions">
                <button class="chat-action-btn" id="debug-toggle" onclick="toggleDebugMode()" title="Toggle Debug Mode">🔍 Debug</button>
                <button class="chat-action-btn" id="show-activation-btn" onclick="showLastActivation()">Show Activation</button>
                <button class="chat-action-btn" onclick="clearChat()">Clear</button>
            </div>
        </div>
        <div id="chat-messages">
            <div class="chat-message system">Ask questions about your knowledge graph. Activated memories will be highlighted.</div>
        </div>
        <div id="chat-input-area">
            <textarea id="chat-input" placeholder="Ask a question..." rows="1"></textarea>
            <button id="chat-send" onclick="sendChat()">Send</button>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const labelCanvas = document.getElementById('label-canvas');
        const gl = canvas.getContext('webgl', { antialias: true, alpha: false });
        const labelCtx = labelCanvas.getContext('2d');

        if (!gl) {
            document.getElementById('loading').innerHTML = 'WebGL not supported';
            throw new Error('WebGL not supported');
        }

        const typeColors = { concept: [0.37, 0.92, 0.83], semantic: [0.65, 0.55, 0.98], episodic: [0.96, 0.45, 0.71] };
        const typeColorsHex = { concept: '#5eead4', semantic: '#a78bfa', episodic: '#f472b6' };

        const clusterPalette = [
            [0.37, 0.92, 0.83], [0.65, 0.55, 0.98], [0.96, 0.45, 0.71], [0.98, 0.75, 0.14],
            [0.20, 0.83, 0.60], [0.38, 0.65, 0.98], [0.98, 0.44, 0.52], [0.64, 0.90, 0.21],
            [0.13, 0.83, 0.93], [0.75, 0.52, 0.99], [0.98, 0.45, 0.09], [0.18, 0.83, 0.75]
        ];

        let nodes = [], links = [], nodeMap = {};
        let bounds = { min_x: -1000, max_x: 1000, min_y: -1000, max_y: 1000 };
        let viewX = 0, viewY = 0, scale = 1;
        let selectedNode = null, hoveredNode = null;
        let highlightedNodes = new Set();
        let neighborNodes = new Set();
        let activatedNodes = new Set(); // Nodes activated by chat
        let typeFilters = new Set(); // empty = show all
        let clusterCenters = {}; // { clusterId: { x, y, name, nodeCount, topNodes } }
        let interClusterEdges = []; // [{ from: clusterId, to: clusterId, count: n }]
        let connStats = { max: 1, p999: 1, p99: 1, p90: 1, p50: 1 }; // Connection percentiles for LOD

        // Hierarchical cluster data for semantic zoom (5 levels)
        let level0Centers = {}; // { level0Id: { x, y, radius, name, nodeCount } }
        let level1Centers = {}; // { "level0-level1": { x, y, radius, name, nodeCount, parent } }
        let level2Centers = {}; // { "l0-l1-l2": { x, y, radius, name, nodeCount, parent } }
        let level3Centers = {}; // { "l0-l1-l2-l3": { x, y, radius, name, nodeCount, parent } }
        let level0Edges = []; // Inter-level0 cluster edges
        let level1Edges = []; // Inter-level1 cluster edges
        let level2Edges = []; // Inter-level2 cluster edges
        let level3Edges = []; // Inter-level3 cluster edges

        let isDragging = false, lastMouseX = 0, lastMouseY = 0, dragStartX = 0, dragStartY = 0;
        let animationFrameId = null;
        let loadingViewport = false, lastViewport = null, viewportDebounceTimer = null;

        // Animation state for smooth zoom
        let zoomAnimation = null;
        let initialView = { x: 0, y: 0, scale: 1 }; // Stored on init


        function animateToOverview() {
            // Zoom out to initial view (as it was when page loaded)
            animateTo(initialView.x, initialView.y, initialView.scale);
        }

        function animateTo(targetX, targetY, targetScale, duration = 800) {
            // Cancel any existing animation
            if (zoomAnimation) {
                cancelAnimationFrame(zoomAnimation.frameId);
            }

            const startX = viewX;
            const startY = viewY;
            const startScale = scale;
            const startTime = performance.now();

            function easeInOutCubic(t) {
                return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
            }

            function step(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = easeInOutCubic(progress);

                viewX = startX + (targetX - startX) * eased;
                viewY = startY + (targetY - startY) * eased;
                scale = startScale + (targetScale - startScale) * eased;

                render();

                if (progress < 1) {
                    zoomAnimation = { frameId: requestAnimationFrame(step) };
                } else {
                    zoomAnimation = null;
                    scheduleViewportLoad();
                }
            }

            zoomAnimation = { frameId: requestAnimationFrame(step) };
        }

        function findClusterAtPosition(screenX, screenY) {
            // Check if click is near a hierarchical cluster (when zoomed out)
            const semanticZoomLevel = scale < 0.008 ? 0 : (scale < 0.02 ? 1 : (scale < 0.045 ? 2 : (scale < 0.09 ? 3 : 4)));
            const world = screenToWorld(screenX, screenY);

            // Helper to find cluster hit
            function checkCenters(centers, minCount) {
                for (const [key, center] of Object.entries(centers)) {
                    if (center.nodeCount < minCount) continue;
                    const dx = world.x - center.x;
                    const dy = world.y - center.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    const clickRadius = Math.max(center.radius * 1.5, 50 / scale);
                    if (dist < clickRadius) {
                        return { id: key, center: center };
                    }
                }
                return null;
            }

            // Check appropriate level
            if (semanticZoomLevel === 0) {
                const hit = checkCenters(level0Centers, 3);
                if (hit) return { type: 'level0', ...hit };
            } else if (semanticZoomLevel === 1) {
                const hit = checkCenters(level1Centers, 2);
                if (hit) return { type: 'level1', ...hit };
            } else if (semanticZoomLevel === 2) {
                const hit = checkCenters(level2Centers, 2);
                if (hit) return { type: 'level2', ...hit };
            } else if (semanticZoomLevel === 3) {
                const hit = checkCenters(level3Centers, 2);
                if (hit) return { type: 'level3', ...hit };
            }

            return null;
        }

        // WebGL shaders
        const vertexShaderSrc = `
            attribute vec2 a_position;
            attribute vec4 a_color;
            attribute float a_size;
            uniform vec2 u_resolution;
            uniform vec2 u_view;
            uniform float u_scale;
            varying vec4 v_color;
            void main() {
                vec2 pos = (a_position - u_view) * u_scale;
                vec2 clipSpace = (pos / u_resolution) * 2.0;
                gl_Position = vec4(clipSpace, 0, 1);
                gl_PointSize = a_size * u_scale;
                v_color = a_color;
            }
        `;

        const fragmentShaderSrc = `
            precision mediump float;
            varying vec4 v_color;
            void main() {
                vec2 coord = gl_PointCoord - vec2(0.5);
                float dist = length(coord);
                if (dist > 0.5) discard;

                // Outer glow (soft halo around node)
                float outerGlow = smoothstep(0.5, 0.3, dist) * 0.4;

                // Border/outline effect
                float borderOuter = 0.42;
                float borderInner = 0.35;
                float border = smoothstep(borderInner, borderOuter, dist) * smoothstep(0.5, borderOuter, dist);
                vec3 borderColor = v_color.rgb * 1.5; // Brighter border

                // Core fill with slight gradient
                float coreFill = 1.0 - smoothstep(0.0, borderInner, dist) * 0.2;

                // Combine: core + border + glow
                vec3 finalColor = mix(v_color.rgb * coreFill, borderColor, border * 0.6);
                finalColor += outerGlow * v_color.rgb;

                // Extra glow for important nodes (alpha > 0.95)
                float importantGlow = v_color.a > 0.95 ? (1.0 - dist * 1.5) * 0.25 : 0.0;
                finalColor += importantGlow;

                float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
                gl_FragColor = vec4(finalColor, alpha * v_color.a);
            }
        `;

        const lineVertexShaderSrc = `
            attribute vec2 a_position;
            attribute vec4 a_color;
            uniform vec2 u_resolution;
            uniform vec2 u_view;
            uniform float u_scale;
            varying vec4 v_color;
            void main() {
                vec2 pos = (a_position - u_view) * u_scale;
                vec2 clipSpace = (pos / u_resolution) * 2.0;
                gl_Position = vec4(clipSpace, 0, 1);
                v_color = a_color;
            }
        `;

        const lineFragmentShaderSrc = `
            precision mediump float;
            varying vec4 v_color;
            void main() {
                gl_FragColor = v_color;
            }
        `;

        function createShader(type, source) {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                console.error(gl.getShaderInfoLog(shader));
                return null;
            }
            return shader;
        }

        function createProgram(vertexSrc, fragmentSrc) {
            const program = gl.createProgram();
            gl.attachShader(program, createShader(gl.VERTEX_SHADER, vertexSrc));
            gl.attachShader(program, createShader(gl.FRAGMENT_SHADER, fragmentSrc));
            gl.linkProgram(program);
            return program;
        }

        const nodeProgram = createProgram(vertexShaderSrc, fragmentShaderSrc);
        const lineProgram = createProgram(lineVertexShaderSrc, lineFragmentShaderSrc);
        const nodeBuffer = gl.createBuffer();
        const lineBuffer = gl.createBuffer();

        function resize() {
            const dpr = window.devicePixelRatio;
            canvas.width = window.innerWidth * dpr;
            canvas.height = window.innerHeight * dpr;
            canvas.style.width = window.innerWidth + 'px';
            canvas.style.height = window.innerHeight + 'px';
            labelCanvas.width = window.innerWidth * dpr;
            labelCanvas.height = window.innerHeight * dpr;
            labelCanvas.style.width = window.innerWidth + 'px';
            labelCanvas.style.height = window.innerHeight + 'px';
            gl.viewport(0, 0, canvas.width, canvas.height);
            render();
        }

        function screenToWorld(sx, sy) {
            // Match shader: clipSpace = (pos / u_resolution) * 2.0, with u_resolution = size/2
            // This means effective multiplier is 2 in the transform
            return {
                x: (sx - window.innerWidth / 2) / (scale * 2) + viewX,
                y: -(sy - window.innerHeight / 2) / (scale * 2) + viewY
            };
        }

        function worldToScreen(wx, wy) {
            // Match shader transform: pos * 2 / resolution * 2 = pos * 4 / size
            // Screen = (clip + 1) * size/2, so final = pos * 2 + size/2
            return {
                x: (wx - viewX) * scale * 2 + window.innerWidth / 2,
                y: -(wy - viewY) * scale * 2 + window.innerHeight / 2
            };
        }

        function getViewportBounds() {
            const margin = 200 / scale;
            const tl = screenToWorld(0, 0);
            const br = screenToWorld(window.innerWidth, window.innerHeight);
            return {
                min_x: Math.min(tl.x, br.x) - margin,
                max_x: Math.max(tl.x, br.x) + margin,
                min_y: Math.min(tl.y, br.y) - margin,
                max_y: Math.max(tl.y, br.y) + margin
            };
        }

        function viewportChanged(v1, v2) {
            if (!v1 || !v2) return true;
            const threshold = 100 / scale;
            return Math.abs(v1.min_x - v2.min_x) > threshold ||
                   Math.abs(v1.max_x - v2.max_x) > threshold;
        }

        async function loadViewportData() {
            // With lazy loading, we don't fetch from server on viewport change
            // Nodes are loaded on-demand when user clicks into clusters
            // Just re-render with currently loaded data
            render();
        }

        function scheduleViewportLoad() {
            if (viewportDebounceTimer) clearTimeout(viewportDebounceTimer);
            viewportDebounceTimer = setTimeout(() => {
                // Auto-load clusters that are in view (optional enhancement)
                // For now, just re-render
                render();
            }, 100);
        }

        function isNodeVisible(node) {
            if (typeFilters.size > 0 && !typeFilters.has(node.type)) return false;
            return true;
        }

        function getNodeColor(node) {
            if (!isNodeVisible(node)) return [0.1, 0.1, 0.12, 0.2];

            const isHighlighted = highlightedNodes.size === 0 || highlightedNodes.has(node.id);
            const isNeighbor = neighborNodes.has(node.id);
            const isSelected = node === selectedNode;
            const isActivated = activatedNodes.has(node.id);

            // Activated nodes from chat get bright glow
            if (isActivated) {
                // Pulsing glow effect using time
                const pulse = 0.7 + 0.3 * Math.sin(Date.now() / 300);
                return [1.0, 0.95, 0.3, pulse]; // Golden glow for activated
            }

            if (!isHighlighted && !isNeighbor && !isSelected && highlightedNodes.size > 0) {
                return [0.15, 0.15, 0.2, 0.3];
            }
            if (!isHighlighted && !isNeighbor && !isSelected && neighborNodes.size > 0) {
                return [0.15, 0.15, 0.2, 0.3];
            }

            const color = typeColors[node.type] || typeColors.concept;
            // Add glow (alpha > 0.95) for important nodes
            const importance = node.conn > 10 ? 1.0 : 0.9;
            return [color[0], color[1], color[2], importance];
        }

        function render() {
            gl.clearColor(0.02, 0.024, 0.05, 1); // Darker background for more contrast
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

            const w = canvas.width / window.devicePixelRatio;
            const h = canvas.height / window.devicePixelRatio;
            const dpr = window.devicePixelRatio;

            // Clear label canvas
            labelCtx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);

            // Draw cluster edges with thickness and glow based on connection count
            // Use the same 5-level thresholds as semantic zoom
            const edgeSemanticLevel = scale < 0.008 ? 0 : (scale < 0.02 ? 1 : (scale < 0.045 ? 2 : (scale < 0.09 ? 3 : 4)));
            if (edgeSemanticLevel <= 3 && (level0Edges.length > 0 || level1Edges.length > 0 || level2Edges.length > 0 || level3Edges.length > 0)) {
                labelCtx.save();
                labelCtx.scale(dpr, dpr);

                // Calculate max edge count for scaling line width
                const maxL0Count = Math.max(1, ...level0Edges.map(e => e.count));
                const maxL1Count = Math.max(1, ...level1Edges.map(e => e.count));
                const maxL2Count = level2Edges.length > 0 ? Math.max(1, ...level2Edges.map(e => e.count)) : 1;
                const maxL3Count = level3Edges.length > 0 ? Math.max(1, ...level3Edges.map(e => e.count)) : 1;

                // Draw Level 0 edges (between super-clusters)
                if (edgeSemanticLevel === 0) {
                    for (const edge of level0Edges) {
                        const c1 = level0Centers[edge.from];
                        const c2 = level0Centers[edge.to];
                        if (!c1 || !c2) continue;

                        const pos1 = worldToScreen(c1.x, c1.y);
                        const pos2 = worldToScreen(c2.x, c2.y);

                        // Skip if off-screen
                        if (pos1.x < -100 && pos2.x < -100) continue;
                        if (pos1.x > w + 100 && pos2.x > w + 100) continue;
                        if (pos1.y < -100 && pos2.y < -100) continue;
                        if (pos1.y > h + 100 && pos2.y > h + 100) continue;

                        const color1 = clusterPalette[edge.from % clusterPalette.length];
                        const color2 = clusterPalette[edge.to % clusterPalette.length];

                        // Line width based on connection count (2px to 10px)
                        const relativeCount = edge.count / maxL0Count;
                        const lineWidth = 2 + relativeCount * 8;

                        // Curved line
                        const mx = (pos1.x + pos2.x) / 2;
                        const my = (pos1.y + pos2.y) / 2;
                        const dx = pos2.x - pos1.x, dy = pos2.y - pos1.y;
                        const dist = Math.sqrt(dx*dx + dy*dy);
                        const nx = dist > 0 ? -dy / dist : 0;
                        const ny = dist > 0 ? dx / dist : 0;
                        const curveAmount = Math.min(dist * 0.15, 80);
                        const cx = mx + nx * curveAmount;
                        const cy = my + ny * curveAmount;

                        // Create gradient
                        const gradient = labelCtx.createLinearGradient(pos1.x, pos1.y, pos2.x, pos2.y);
                        gradient.addColorStop(0, `rgba(${color1[0]*255}, ${color1[1]*255}, ${color1[2]*255}, 0.7)`);
                        gradient.addColorStop(1, `rgba(${color2[0]*255}, ${color2[1]*255}, ${color2[2]*255}, 0.7)`);

                        // Subtle edge glow - stronger for shorter edges
                        const glowIntensity = Math.max(0, 1 - dist / 500) * relativeCount;
                        const mixedColor = [
                            (color1[0] + color2[0]) / 2,
                            (color1[1] + color2[1]) / 2,
                            (color1[2] + color2[2]) / 2
                        ];
                        labelCtx.shadowColor = `rgba(${mixedColor[0]*255}, ${mixedColor[1]*255}, ${mixedColor[2]*255}, ${0.3 * glowIntensity})`;
                        labelCtx.shadowBlur = 8 + glowIntensity * 10;

                        labelCtx.strokeStyle = gradient;
                        labelCtx.lineWidth = lineWidth;
                        labelCtx.lineCap = 'round';
                        labelCtx.beginPath();
                        labelCtx.moveTo(pos1.x, pos1.y);
                        labelCtx.quadraticCurveTo(cx, cy, pos2.x, pos2.y);
                        labelCtx.stroke();
                    }
                    labelCtx.shadowBlur = 0;
                }

                // Draw Level 1 edges (between sub-clusters)
                if (edgeSemanticLevel === 1) {
                    const maxL1Edges = Math.min(level1Edges.length, 150);
                    for (let i = 0; i < maxL1Edges; i++) {
                        const edge = level1Edges[i];
                        const c1 = level1Centers[edge.from];
                        const c2 = level1Centers[edge.to];
                        if (!c1 || !c2) continue;

                        const pos1 = worldToScreen(c1.x, c1.y);
                        const pos2 = worldToScreen(c2.x, c2.y);

                        // Skip if off-screen
                        if (pos1.x < -50 && pos2.x < -50) continue;
                        if (pos1.x > w + 50 && pos2.x > w + 50) continue;
                        if (pos1.y < -50 && pos2.y < -50) continue;
                        if (pos1.y > h + 50 && pos2.y > h + 50) continue;

                        const color1 = clusterPalette[(c1.parent || 0) % clusterPalette.length];
                        const color2 = clusterPalette[(c2.parent || 0) % clusterPalette.length];

                        // Line width based on connection count (1px to 5px)
                        const relativeCount = edge.count / maxL1Count;
                        const lineWidth = 1 + relativeCount * 4;

                        // Curved line
                        const mx = (pos1.x + pos2.x) / 2;
                        const my = (pos1.y + pos2.y) / 2;
                        const dx = pos2.x - pos1.x, dy = pos2.y - pos1.y;
                        const dist = Math.sqrt(dx*dx + dy*dy);
                        const nx = dist > 0 ? -dy / dist : 0;
                        const ny = dist > 0 ? dx / dist : 0;
                        const curveAmount = Math.min(dist * 0.12, 40);
                        const cx = mx + nx * curveAmount;
                        const cy = my + ny * curveAmount;

                        // Create gradient
                        const gradient = labelCtx.createLinearGradient(pos1.x, pos1.y, pos2.x, pos2.y);
                        gradient.addColorStop(0, `rgba(${color1[0]*255}, ${color1[1]*255}, ${color1[2]*255}, 0.5)`);
                        gradient.addColorStop(1, `rgba(${color2[0]*255}, ${color2[1]*255}, ${color2[2]*255}, 0.5)`);

                        // Very subtle glow for L1 edges
                        const glowIntensity = Math.max(0, 1 - dist / 300) * relativeCount;
                        const mixedColor = [
                            (color1[0] + color2[0]) / 2,
                            (color1[1] + color2[1]) / 2,
                            (color1[2] + color2[2]) / 2
                        ];
                        labelCtx.shadowColor = `rgba(${mixedColor[0]*255}, ${mixedColor[1]*255}, ${mixedColor[2]*255}, ${0.2 * glowIntensity})`;
                        labelCtx.shadowBlur = 4 + glowIntensity * 6;

                        labelCtx.strokeStyle = gradient;
                        labelCtx.lineWidth = lineWidth;
                        labelCtx.lineCap = 'round';
                        labelCtx.beginPath();
                        labelCtx.moveTo(pos1.x, pos1.y);
                        labelCtx.quadraticCurveTo(cx, cy, pos2.x, pos2.y);
                        labelCtx.stroke();
                    }
                    labelCtx.shadowBlur = 0;
                }

                // Draw Level 2 edges (between L2 sub-clusters)
                if (edgeSemanticLevel === 2 && level2Edges.length > 0) {
                    const maxL2Edges = Math.min(level2Edges.length, 200);
                    for (let i = 0; i < maxL2Edges; i++) {
                        const edge = level2Edges[i];
                        const c1 = level2Centers[edge.from];
                        const c2 = level2Centers[edge.to];
                        if (!c1 || !c2) continue;

                        const pos1 = worldToScreen(c1.x, c1.y);
                        const pos2 = worldToScreen(c2.x, c2.y);

                        // Skip if off-screen
                        if (pos1.x < -50 && pos2.x < -50) continue;
                        if (pos1.x > w + 50 && pos2.x > w + 50) continue;
                        if (pos1.y < -50 && pos2.y < -50) continue;
                        if (pos1.y > h + 50 && pos2.y > h + 50) continue;

                        const parentL0 = c1.parent || 0;
                        const color = clusterPalette[parentL0 % clusterPalette.length];

                        // Line width based on connection count (1px to 3px)
                        const relativeCount = edge.count / maxL2Count;
                        const lineWidth = 0.5 + relativeCount * 2.5;

                        // Straight lines for L2 (simpler, faster)
                        labelCtx.strokeStyle = `rgba(${color[0]*255}, ${color[1]*255}, ${color[2]*255}, 0.4)`;
                        labelCtx.lineWidth = lineWidth;
                        labelCtx.beginPath();
                        labelCtx.moveTo(pos1.x, pos1.y);
                        labelCtx.lineTo(pos2.x, pos2.y);
                        labelCtx.stroke();
                    }
                }

                // Draw Level 3 edges (between L3 groups)
                if (edgeSemanticLevel === 3 && level3Edges.length > 0) {
                    const maxL3Edges = Math.min(level3Edges.length, 300);
                    for (let i = 0; i < maxL3Edges; i++) {
                        const edge = level3Edges[i];
                        const c1 = level3Centers[edge.from];
                        const c2 = level3Centers[edge.to];
                        if (!c1 || !c2) continue;

                        const pos1 = worldToScreen(c1.x, c1.y);
                        const pos2 = worldToScreen(c2.x, c2.y);

                        // Skip if off-screen
                        if (pos1.x < -50 && pos2.x < -50) continue;
                        if (pos1.x > w + 50 && pos2.x > w + 50) continue;
                        if (pos1.y < -50 && pos2.y < -50) continue;
                        if (pos1.y > h + 50 && pos2.y > h + 50) continue;

                        const parentL0 = c1.parent || 0;
                        const color = clusterPalette[parentL0 % clusterPalette.length];

                        // Line width based on connection count (0.5px to 2px)
                        const relativeCount = edge.count / maxL3Count;
                        const lineWidth = 0.5 + relativeCount * 1.5;

                        // Straight lines for L3 (simpler, faster)
                        labelCtx.strokeStyle = `rgba(${color[0]*255}, ${color[1]*255}, ${color[2]*255}, 0.3)`;
                        labelCtx.lineWidth = lineWidth;
                        labelCtx.beginPath();
                        labelCtx.moveTo(pos1.x, pos1.y);
                        labelCtx.lineTo(pos2.x, pos2.y);
                        labelCtx.stroke();
                    }
                }

                labelCtx.restore();
            }

            // Draw lines with gradient colors
            if (links.length > 0) {
                gl.useProgram(lineProgram);
                gl.uniform2f(gl.getUniformLocation(lineProgram, 'u_resolution'), w / 2, h / 2);
                gl.uniform2f(gl.getUniformLocation(lineProgram, 'u_view'), viewX, viewY);
                gl.uniform1f(gl.getUniformLocation(lineProgram, 'u_scale'), scale);

                const hasSelection = selectedNode || neighborNodes.size > 0;
                // Line data format: x, y, r, g, b, a (6 floats per vertex)
                const normalLineData = [];
                const glowLineData = [];

                // Helper to get node color for edges
                function getEdgeNodeColor(node) {
                    return typeColors[node.type] || typeColors.concept;
                }

                // Helper to draw a bezier curve segment (quadratic)
                function drawBezierSegment(targetArray, s, t, cx, cy, sColor, tColor, opacity) {
                    const segments = 6;
                    for (let i = 0; i < segments; i++) {
                        const t1 = i / segments, t2 = (i + 1) / segments;
                        const x1 = (1-t1)*(1-t1)*s.x + 2*(1-t1)*t1*cx + t1*t1*t.x;
                        const y1 = (1-t1)*(1-t1)*s.y + 2*(1-t1)*t1*cy + t1*t1*t.y;
                        const x2 = (1-t2)*(1-t2)*s.x + 2*(1-t2)*t2*cx + t2*t2*t.x;
                        const y2 = (1-t2)*(1-t2)*s.y + 2*(1-t2)*t2*cy + t2*t2*t.y;
                        const r1 = sColor[0] + (tColor[0] - sColor[0]) * t1;
                        const g1 = sColor[1] + (tColor[1] - sColor[1]) * t1;
                        const b1 = sColor[2] + (tColor[2] - sColor[2]) * t1;
                        const r2 = sColor[0] + (tColor[0] - sColor[0]) * t2;
                        const g2 = sColor[1] + (tColor[1] - sColor[1]) * t2;
                        const b2 = sColor[2] + (tColor[2] - sColor[2]) * t2;
                        targetArray.push(x1, y1, r1, g1, b1, opacity);
                        targetArray.push(x2, y2, r2, g2, b2, opacity);
                    }
                }

                // Helper for hierarchical edge bundling with multiple control points
                function drawHierarchicalBundledEdge(targetArray, s, t, sColor, tColor, opacity, bundleStrength) {
                    // Find lowest common ancestor (LCA) level
                    const sL0 = s.level0 || 0, sL1 = s.level1 || 0;
                    const tL0 = t.level0 || 0, tL1 = t.level1 || 0;

                    // Determine LCA level: 0=same L0, 1=same L1, 2=same node
                    let lcaLevel = 0;
                    if (sL0 === tL0) {
                        lcaLevel = 1;
                        if (sL1 === tL1) {
                            lcaLevel = 2;
                        }
                    }

                    // If same L2 cluster or bundling disabled, use simple curve
                    if (lcaLevel === 2 || bundleStrength < 0.1) {
                        const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2;
                        const dx = t.x - s.x, dy = t.y - s.y;
                        const dist = Math.sqrt(dx*dx + dy*dy);
                        const nx = dist > 0 ? -dy / dist : 0;
                        const ny = dist > 0 ? dx / dist : 0;
                        const curveAmount = Math.min(dist * 0.15, 80);
                        drawBezierSegment(targetArray, s, t, mx + nx * curveAmount, my + ny * curveAmount, sColor, tColor, opacity);
                        return;
                    }

                    // Get cluster centers for routing
                    const sL1Key = `${sL0}-${sL1}`;
                    const tL1Key = `${tL0}-${tL1}`;
                    const sL1Center = level1Centers[sL1Key];
                    const tL1Center = level1Centers[tL1Key];
                    const sL0Center = level0Centers[sL0];
                    const tL0Center = level0Centers[tL0];

                    // Build control points path based on LCA level
                    let controlPoints = [];

                    if (lcaLevel === 0) {
                        // Different L0: route through both L0 centers
                        if (sL1Center) controlPoints.push({ x: sL1Center.x, y: sL1Center.y });
                        if (sL0Center) controlPoints.push({ x: sL0Center.x, y: sL0Center.y });
                        if (tL0Center) controlPoints.push({ x: tL0Center.x, y: tL0Center.y });
                        if (tL1Center) controlPoints.push({ x: tL1Center.x, y: tL1Center.y });
                    } else if (lcaLevel === 1) {
                        // Same L0, different L1: route through L1 centers
                        if (sL1Center) controlPoints.push({ x: sL1Center.x, y: sL1Center.y });
                        if (tL1Center) controlPoints.push({ x: tL1Center.x, y: tL1Center.y });
                    }

                    // If no valid control points, fall back to simple curve
                    if (controlPoints.length === 0) {
                        const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2;
                        const dx = t.x - s.x, dy = t.y - s.y;
                        const dist = Math.sqrt(dx*dx + dy*dy);
                        const nx = dist > 0 ? -dy / dist : 0;
                        const ny = dist > 0 ? dx / dist : 0;
                        drawBezierSegment(targetArray, s, t, mx + nx * dist * 0.15, my + ny * dist * 0.15, sColor, tColor, opacity);
                        return;
                    }

                    // Blend control points toward direct path based on bundleStrength
                    const directMx = (s.x + t.x) / 2, directMy = (s.y + t.y) / 2;
                    controlPoints = controlPoints.map(cp => ({
                        x: cp.x * bundleStrength + directMx * (1 - bundleStrength),
                        y: cp.y * bundleStrength + directMy * (1 - bundleStrength)
                    }));

                    // Draw through all control points using Catmull-Rom spline
                    const allPoints = [s, ...controlPoints, t];
                    const segments = 4 * (allPoints.length - 1);

                    for (let i = 0; i < segments; i++) {
                        const tParam1 = i / segments;
                        const tParam2 = (i + 1) / segments;

                        const pos1 = catmullRomPoint(allPoints, tParam1);
                        const pos2 = catmullRomPoint(allPoints, tParam2);

                        const r1 = sColor[0] + (tColor[0] - sColor[0]) * tParam1;
                        const g1 = sColor[1] + (tColor[1] - sColor[1]) * tParam1;
                        const b1 = sColor[2] + (tColor[2] - sColor[2]) * tParam1;
                        const r2 = sColor[0] + (tColor[0] - sColor[0]) * tParam2;
                        const g2 = sColor[1] + (tColor[1] - sColor[1]) * tParam2;
                        const b2 = sColor[2] + (tColor[2] - sColor[2]) * tParam2;

                        targetArray.push(pos1.x, pos1.y, r1, g1, b1, opacity);
                        targetArray.push(pos2.x, pos2.y, r2, g2, b2, opacity);
                    }
                }

                // Catmull-Rom spline interpolation
                function catmullRomPoint(points, t) {
                    const n = points.length - 1;
                    const segment = Math.min(Math.floor(t * n), n - 1);
                    const localT = (t * n) - segment;

                    // Get 4 control points (with clamping at boundaries)
                    const p0 = points[Math.max(0, segment - 1)];
                    const p1 = points[segment];
                    const p2 = points[Math.min(n, segment + 1)];
                    const p3 = points[Math.min(n, segment + 2)];

                    // Catmull-Rom basis functions
                    const t2 = localT * localT;
                    const t3 = t2 * localT;

                    const x = 0.5 * (
                        (2 * p1.x) +
                        (-p0.x + p2.x) * localT +
                        (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
                        (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3
                    );

                    const y = 0.5 * (
                        (2 * p1.y) +
                        (-p0.y + p2.y) * localT +
                        (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
                        (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3
                    );

                    return { x, y };
                }

                // Semantic zoom for edges - aligned with node rendering
                // scale < 0.025: Level 0 inter-super-cluster edges
                // scale 0.025-0.06: Level 1 inter-sub-cluster edges
                // scale > 0.06: Individual edges with LOD
                const edgeSemanticLevel = scale < 0.025 ? 0 : (scale < 0.06 ? 1 : 2);

                // Transition factors for smooth edge blending
                const l0ToL1EdgeTransition = scale < 0.02 ? 0 : (scale < 0.03 ? (scale - 0.02) / 0.01 : 1);
                const l1ToFullEdgeTransition = scale < 0.05 ? 0 : (scale < 0.07 ? (scale - 0.05) / 0.02 : 1);

                // Draw Level 0 and Level 1 edges on 2D canvas (for variable line width)
                // Skip WebGL edge drawing for cluster edges - handled below in canvas section

                // Draw individual edges when zoomed in enough
                const showIndividualEdges = edgeSemanticLevel === 2 || l1ToFullEdgeTransition > 0;
                if (showIndividualEdges) {
                    const edgeOpacity = edgeSemanticLevel === 2 ? 1.0 : l1ToFullEdgeTransition;

                    for (const link of links) {
                        const s = nodeMap[link.source];
                        const t = nodeMap[link.target];
                        if (!s || !t || !isNodeVisible(s) || !isNodeVisible(t)) continue;

                        // During transition, dim intra-cluster edges
                        const sameCluster = (s.cluster || 0) === (t.cluster || 0);
                        const isTransitioning = l1ToFullEdgeTransition < 1;
                        const thisEdgeOpacity = isTransitioning && sameCluster ? edgeOpacity * 0.5 : edgeOpacity;
                        if (thisEdgeOpacity < 0.05) continue;

                        const isGlowLink = selectedNode && (
                            (s === selectedNode && neighborNodes.has(t.id)) ||
                            (t === selectedNode && neighborNodes.has(s.id))
                        );

                        const targetArray = isGlowLink ? glowLineData : normalLineData;
                        const baseOpacity = isGlowLink ? 0.9 : (hasSelection ? 0.35 : 0.7);
                        const opacity = baseOpacity * thisEdgeOpacity;
                        const sColor = getEdgeNodeColor(s);
                        const tColor = getEdgeNodeColor(t);

                        const dx = t.x - s.x, dy = t.y - s.y;
                        const dist = Math.sqrt(dx*dx + dy*dy);
                        if (dist < 0.001) continue;

                        // Use hierarchical bundling when zoomed out
                        if (scale < 0.15 && Object.keys(level1Centers).length > 0) {
                            // Bundle strength based on zoom level (stronger when zoomed out)
                            const bundleStrength = Math.min(1, Math.max(0.3, 1.5 - scale * 5));
                            drawHierarchicalBundledEdge(targetArray, s, t, sColor, tColor, opacity, bundleStrength);
                        } else {
                            // Simple curved edge
                            const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2;
                            const curveAmount = Math.min(dist * 0.2, 150);
                            const nx = -dy / dist, ny = dx / dist;
                            const cx = mx + nx * curveAmount, cy = my + ny * curveAmount;
                            drawBezierSegment(targetArray, s, t, cx, cy, sColor, tColor, opacity);
                        }
                    }
                }

                const posLoc = gl.getAttribLocation(lineProgram, 'a_position');
                const colorLoc = gl.getAttribLocation(lineProgram, 'a_color');
                const stride = 6 * 4; // 6 floats * 4 bytes

                // Draw normal lines with gradients
                if (normalLineData.length > 0) {
                    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
                    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normalLineData), gl.DYNAMIC_DRAW);
                    gl.enableVertexAttribArray(posLoc);
                    gl.enableVertexAttribArray(colorLoc);
                    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
                    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 2 * 4);
                    gl.drawArrays(gl.LINES, 0, normalLineData.length / 6);
                }

                // Draw glowing lines for selected node connections
                if (glowLineData.length > 0) {
                    gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
                    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(glowLineData), gl.DYNAMIC_DRAW);
                    gl.enableVertexAttribArray(posLoc);
                    gl.enableVertexAttribArray(colorLoc);
                    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
                    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 2 * 4);
                    gl.drawArrays(gl.LINES, 0, glowLineData.length / 6);
                }
            }

            // ============ SEMANTIC ZOOM: 5 levels of hierarchy ============
            // Level 0: scale < 0.008 - super-clusters
            // Level 1: scale 0.008-0.02 - large clusters
            // Level 2: scale 0.02-0.045 - medium clusters
            // Level 3: scale 0.045-0.09 - small clusters
            // Level 4: scale > 0.09 - individual nodes

            const semanticZoomLevel = scale < 0.008 ? 0 : (scale < 0.02 ? 1 : (scale < 0.045 ? 2 : (scale < 0.09 ? 3 : 4)));
            const showLevel0 = semanticZoomLevel === 0;
            const showLevel1 = semanticZoomLevel === 1;
            const showLevel2 = semanticZoomLevel === 2;
            const showLevel3 = semanticZoomLevel === 3;
            const showNodes = semanticZoomLevel === 4;

            // Transition factors for smooth blending between levels
            const l0ToL1Transition = scale < 0.006 ? 0 : (scale < 0.01 ? (scale - 0.006) / 0.004 : 1);
            const l1ToL2Transition = scale < 0.016 ? 0 : (scale < 0.024 ? (scale - 0.016) / 0.008 : 1);
            const l2ToL3Transition = scale < 0.038 ? 0 : (scale < 0.052 ? (scale - 0.038) / 0.014 : 1);
            const l3ToNodesTransition = scale < 0.08 ? 0 : (scale < 0.1 ? (scale - 0.08) / 0.02 : 1);

            // Pulsing animation timing (very subtle)
            const pulseTime = Date.now() / 1000;

            gl.useProgram(nodeProgram);
            gl.uniform2f(gl.getUniformLocation(nodeProgram, 'u_resolution'), w / 2, h / 2);
            gl.uniform2f(gl.getUniformLocation(nodeProgram, 'u_view'), viewX, viewY);
            gl.uniform1f(gl.getUniformLocation(nodeProgram, 'u_scale'), scale);

            const nodeData = [];
            const labelsToRender = [];

            // Draw Level 0 super-clusters when very zoomed out
            if (showLevel0 || (showLevel1 && l0ToL1Transition < 1)) {
                const opacity = showLevel0 ? 1.0 : (1 - l0ToL1Transition);

                // Calculate max node count for relative sizing
                const l0Entries = Object.entries(level0Centers);
                const maxNodeCount = Math.max(1, ...l0Entries.map(([_, c]) => c.nodeCount));

                for (const [l0Id, center] of l0Entries) {
                    if (center.nodeCount < 5) continue;

                    const pos = worldToScreen(center.x, center.y);
                    if (pos.x < -300 || pos.x > w + 300 || pos.y < -300 || pos.y > h + 300) continue;

                    const isLoaded = loadedClusters.has(parseInt(l0Id));
                    const color = clusterPalette[l0Id % clusterPalette.length];
                    // Size proportional to node count - use sqrt for visual area scaling
                    const relativeSize = Math.sqrt(center.nodeCount / maxNodeCount);

                    // Very subtle pulsing
                    const pulseOffset = parseInt(l0Id) * 0.7;
                    const pulse = 0.97 + 0.03 * Math.sin(pulseTime * 1.2 + pulseOffset);

                    // Minimum screen pixels scales with relative size (20px min, 120px max)
                    const minScreenPx = (20 + relativeSize * 100) * pulse;
                    // Convert to world units for current scale
                    const size = minScreenPx / scale;

                    // Unloaded clusters are dimmer/more transparent
                    const clusterOpacity = isLoaded ? opacity : opacity * 0.6;
                    nodeData.push(center.x, center.y, color[0], color[1], color[2], clusterOpacity, size);

                    // Labels for level 0
                    if (showLevel0 && center.nodeCount >= 5) {
                        labelsToRender.push({
                            type: 'level0',
                            x: center.x, y: center.y,
                            name: center.name,
                            count: center.nodeCount,
                            color: color,
                            size: size,
                            loaded: isLoaded
                        });
                    }
                }
            }

            // Draw Level 1 clusters
            if (showLevel1 || (showLevel0 && l0ToL1Transition > 0) || (showLevel2 && l1ToL2Transition < 1)) {
                const opacity = showLevel1 ? 1.0 : (showLevel0 ? l0ToL1Transition : (1 - l1ToL2Transition));
                const l1Entries = Object.entries(level1Centers);
                const maxNodeCount = Math.max(1, ...l1Entries.map(([_, c]) => c.nodeCount));

                for (const [key, center] of l1Entries) {
                    if (center.nodeCount < 3) continue;
                    const pos = worldToScreen(center.x, center.y);
                    if (pos.x < -200 || pos.x > w + 200 || pos.y < -200 || pos.y > h + 200) continue;

                    const color = clusterPalette[(center.parent || 0) % clusterPalette.length];
                    const relativeSize = Math.sqrt(center.nodeCount / maxNodeCount);
                    const minScreenPx = 15 + relativeSize * 50;
                    const size = minScreenPx / scale;

                    nodeData.push(center.x, center.y, color[0], color[1], color[2], opacity * 0.95, size);

                    if (showLevel1 && center.nodeCount >= 3) {
                        labelsToRender.push({ type: 'level1', x: center.x, y: center.y, name: center.name, count: center.nodeCount, color: color, size: size });
                    }
                }
            }

            // Draw Level 2 clusters
            if (showLevel2 || (showLevel1 && l1ToL2Transition > 0) || (showLevel3 && l2ToL3Transition < 1)) {
                const opacity = showLevel2 ? 1.0 : (showLevel1 ? l1ToL2Transition : (1 - l2ToL3Transition));
                const l2Entries = Object.entries(level2Centers);
                const maxNodeCount = Math.max(1, ...l2Entries.map(([_, c]) => c.nodeCount));

                for (const [key, center] of l2Entries) {
                    if (center.nodeCount < 2) continue;
                    const pos = worldToScreen(center.x, center.y);
                    if (pos.x < -150 || pos.x > w + 150 || pos.y < -150 || pos.y > h + 150) continue;

                    const color = clusterPalette[(center.parent || 0) % clusterPalette.length];
                    const relativeSize = Math.sqrt(center.nodeCount / maxNodeCount);
                    const minScreenPx = 12 + relativeSize * 40;
                    const size = minScreenPx / scale;

                    nodeData.push(center.x, center.y, color[0], color[1], color[2], opacity * 0.9, size);

                    if (showLevel2 && center.nodeCount >= 2) {
                        labelsToRender.push({ type: 'level2', x: center.x, y: center.y, name: center.name, count: center.nodeCount, color: color, size: size });
                    }
                }
            }

            // Draw Level 3 clusters
            if (showLevel3 || (showLevel2 && l2ToL3Transition > 0) || (showNodes && l3ToNodesTransition < 1)) {
                const opacity = showLevel3 ? 1.0 : (showLevel2 ? l2ToL3Transition : (1 - l3ToNodesTransition));
                const l3Entries = Object.entries(level3Centers);
                const maxNodeCount = Math.max(1, ...l3Entries.map(([_, c]) => c.nodeCount));

                for (const [key, center] of l3Entries) {
                    if (center.nodeCount < 2) continue;
                    const pos = worldToScreen(center.x, center.y);
                    if (pos.x < -100 || pos.x > w + 100 || pos.y < -100 || pos.y > h + 100) continue;

                    const color = clusterPalette[(center.parent || 0) % clusterPalette.length];
                    const relativeSize = Math.sqrt(center.nodeCount / maxNodeCount);
                    const minScreenPx = 10 + relativeSize * 30;
                    const size = minScreenPx / scale;

                    nodeData.push(center.x, center.y, color[0], color[1], color[2], opacity * 0.85, size);

                    if (showLevel3 && center.nodeCount >= 1) {
                        labelsToRender.push({ type: 'level3', x: center.x, y: center.y, name: center.name, count: center.nodeCount, color: color, size: size });
                    }
                }
            }

            // Draw individual nodes at ALL zoom levels with LOD filtering
            // L0: top 1%, L1: top 5%, L2: top 25%, L3: top 50%, L4: 100%
            {
                // LOD: Minimum connections required to be visible at each zoom level
                const minConnByLevel = {
                    0: connStats.p999, // L0: top 0.1%
                    1: connStats.p99,  // L1: top 1%
                    2: connStats.p90,  // L2: top 10%
                    3: connStats.p50,  // L3: top 50%
                    4: 0               // L4: all nodes
                };
                const minConnForVisibility = minConnByLevel[semanticZoomLevel] || 0;

                for (const node of nodes) {
                    // LOD: Skip nodes below connection threshold (unless activated/selected)
                    const nodeConn = node.conn || 0;
                    if (nodeConn < minConnForVisibility && !activatedNodes.has(node.id) && node !== selectedNode) {
                        continue;
                    }

                    const color = getNodeColor(node);
                    // Node size: scale with zoom level
                    const baseSize = Math.max(48, Math.min(192, Math.sqrt(nodeConn || 1) * 32));
                    const minScreenPx = semanticZoomLevel < 2 ? 8 : 6;
                    const size = Math.max(minScreenPx / scale, baseSize * Math.max(1, Math.min(4, scale)));

                    let finalColor = color;
                    if (node === selectedNode) {
                        finalColor = [1, 1, 1, 1];
                    } else if (node === hoveredNode) {
                        finalColor = [1, 1, 1, 0.9];
                    } else if (neighborNodes.has(node.id)) {
                        finalColor = [color[0] * 1.2, color[1] * 1.2, color[2] * 1.2, 1];
                    }

                    nodeData.push(node.x, node.y, finalColor[0], finalColor[1], finalColor[2], finalColor[3], size);

                    // Collect labels for important nodes at all zoom levels
                    if (isNodeVisible(node) && (nodeConn >= minConnForVisibility || scale > 0.1)) {
                        const pos = worldToScreen(node.x, node.y);
                        labelsToRender.push({
                            type: 'node',
                            node: node,
                            pos: pos,
                            size: size
                        });
                    }
                }
            }

            // Render all collected nodes
            if (nodeData.length > 0) {
                gl.bindBuffer(gl.ARRAY_BUFFER, nodeBuffer);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(nodeData), gl.DYNAMIC_DRAW);

                const stride = 7 * 4;
                const posLoc = gl.getAttribLocation(nodeProgram, 'a_position');
                const colorLoc = gl.getAttribLocation(nodeProgram, 'a_color');
                const sizeLoc = gl.getAttribLocation(nodeProgram, 'a_size');

                gl.enableVertexAttribArray(posLoc);
                gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
                gl.enableVertexAttribArray(colorLoc);
                gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 8);
                gl.enableVertexAttribArray(sizeLoc);
                gl.vertexAttribPointer(sizeLoc, 1, gl.FLOAT, false, stride, 24);

                gl.drawArrays(gl.POINTS, 0, nodeData.length / 7);
            }

            // Render labels on 2D canvas
            labelCtx.save();
            labelCtx.scale(dpr, dpr);
            labelCtx.textAlign = 'center';

            // Helper to draw text with background
            function drawTextWithBg(text, x, y, font, textColor, bgColor = 'rgba(10, 10, 18, 0.8)', padding = 3) {
                labelCtx.font = font;
                const metrics = labelCtx.measureText(text);
                const textWidth = metrics.width;
                // Extract font size from font string (e.g., "bold 14px ..." -> 14)
                const fontSizeMatch = font.match(/(\d+)px/);
                const fontSize = fontSizeMatch ? parseInt(fontSizeMatch[1]) : 12;

                // Background positioned around text (text baseline is at y)
                const bgX = x - textWidth / 2 - padding;
                const bgY = y - fontSize + 1;
                const bgW = textWidth + padding * 2;
                const bgH = fontSize + padding + 2;

                // Draw background
                labelCtx.fillStyle = bgColor;
                labelCtx.beginPath();
                labelCtx.roundRect(bgX, bgY, bgW, bgH, 2);
                labelCtx.fill();

                // Draw text
                labelCtx.fillStyle = textColor;
                labelCtx.fillText(text, x, y);
            }

            for (const label of labelsToRender) {
                if (label.type === 'level0') {
                    const pos = worldToScreen(label.x, label.y);
                    if (pos.x < -200 || pos.x > w + 200 || pos.y < -200 || pos.y > h + 200) continue;

                    const color = label.color;
                    const fontSize = Math.min(32, Math.max(14, 12 + Math.log(label.count) * 3));
                    const isLoaded = label.loaded;

                    // Loaded clusters are bright, unloaded are dimmer with indicator
                    const textColor = isLoaded
                        ? `rgb(${color[0]*255}, ${color[1]*255}, ${color[2]*255})`
                        : `rgba(${color[0]*255}, ${color[1]*255}, ${color[2]*255}, 0.6)`;

                    labelCtx.shadowBlur = 0;
                    drawTextWithBg(label.name, pos.x, pos.y - 10, `bold ${fontSize}px -apple-system, sans-serif`, textColor);

                    // Show different indicator for loaded vs unloaded
                    if (isLoaded) {
                        drawTextWithBg(`${label.count} nodes`, pos.x, pos.y + fontSize - 5, `${Math.max(10, fontSize * 0.6)}px -apple-system, sans-serif`, 'rgba(255,255,255,0.7)');
                    } else {
                        drawTextWithBg(`${label.count} nodes · click to load`, pos.x, pos.y + fontSize - 5, `${Math.max(10, fontSize * 0.6)}px -apple-system, sans-serif`, 'rgba(200,200,255,0.5)');
                    }

                } else if (label.type === 'level1') {
                    const pos = worldToScreen(label.x, label.y);
                    if (pos.x < -150 || pos.x > w + 150 || pos.y < -150 || pos.y > h + 150) continue;

                    const color = label.color;
                    const fontSize = Math.min(18, Math.max(10, 8 + Math.log(label.count) * 2));
                    const textColor = `rgb(${color[0]*255}, ${color[1]*255}, ${color[2]*255})`;

                    labelCtx.shadowBlur = 0;
                    drawTextWithBg(label.name, pos.x, pos.y - 5, `600 ${fontSize}px -apple-system, sans-serif`, textColor);
                    drawTextWithBg(`${label.count}`, pos.x, pos.y + fontSize, `${Math.max(8, fontSize * 0.6)}px -apple-system, sans-serif`, 'rgba(255,255,255,0.6)');

                } else if (label.type === 'level2') {
                    const pos = worldToScreen(label.x, label.y);
                    if (pos.x < -100 || pos.x > w + 100 || pos.y < -100 || pos.y > h + 100) continue;

                    const color = label.color;
                    const fontSize = Math.min(14, Math.max(9, 7 + Math.log(label.count) * 1.5));
                    const textColor = `rgb(${color[0]*255}, ${color[1]*255}, ${color[2]*255})`;

                    labelCtx.shadowBlur = 0;
                    drawTextWithBg(label.name, pos.x, pos.y - 3, `500 ${fontSize}px -apple-system, sans-serif`, textColor);

                } else if (label.type === 'level3') {
                    const pos = worldToScreen(label.x, label.y);
                    if (pos.x < -80 || pos.x > w + 80 || pos.y < -80 || pos.y > h + 80) continue;

                    const color = label.color;
                    const fontSize = Math.min(12, Math.max(8, 6 + Math.log(label.count) * 1.2));
                    const textColor = `rgb(${color[0]*255}, ${color[1]*255}, ${color[2]*255})`;

                    labelCtx.shadowBlur = 0;
                    drawTextWithBg(label.name, pos.x, pos.y, `${fontSize}px -apple-system, sans-serif`, textColor);

                } else if (label.type === 'node') {
                    const { node, pos, size } = label;
                    if (pos.x < -100 || pos.x > w + 100 || pos.y < -100 || pos.y > h + 100) continue;

                    const nodeName = node.name.length > 20 ? node.name.slice(0, 20) + '...' : node.name;
                    const screenSize = size * scale;

                    labelCtx.shadowBlur = 0;
                    drawTextWithBg(nodeName, pos.x, pos.y + screenSize / 2 + 14, '11px -apple-system, sans-serif', 'rgba(255,255,255,0.9)');
                }
            }
            labelCtx.restore();

            // Show zoom level indicator
            const zoomIndicators = ['L0: Super-clusters', 'L1: Clusters', 'L2: Sub-clusters', 'L3: Groups', 'L4: Nodes'];
            const zoomIndicator = zoomIndicators[semanticZoomLevel] || 'L4: Nodes';
            document.getElementById('mode-indicator').textContent = `WebGL | ${zoomIndicator} | ${scale.toFixed(3)}`;

            // Debug: log what's being rendered
            if (nodeData.length > 0 && Math.random() < 0.01) { // 1% chance to log
                console.log(`Rendering: scale=${scale.toFixed(4)}, level=${semanticZoomLevel}, points=${nodeData.length/7}, nodes=${nodes.length}`);
            }

        }

        function findNodeAt(sx, sy) {
            const world = screenToWorld(sx, sy);
            let closest = null, closestDist = Infinity;

            for (const node of nodes) {
                if (!isNodeVisible(node)) continue;
                const dx = node.x - world.x;
                const dy = node.y - world.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                // Match bigger node sizes
                const baseSize = Math.max(48, Math.min(192, Math.sqrt(node.conn || 1) * 32));
                const threshold = (baseSize * 2) / scale + 30;

                if (dist < threshold && dist < closestDist) {
                    closest = node;
                    closestDist = dist;
                }
            }
            return closest;
        }

        // Tooltip
        const tooltip = document.getElementById('tooltip');
        function showTooltip(node, x, y) {
            tooltip.textContent = node.name;
            tooltip.style.left = (x + 15) + 'px';
            tooltip.style.top = (y + 15) + 'px';
            tooltip.classList.add('visible');
        }
        function hideTooltip() { tooltip.classList.remove('visible'); }

        // Node info panel with neighbors
        async function showNodeInfo(node) {
            const panel = document.getElementById('node-info');
            const content = document.getElementById('node-info-content');
            const color = typeColorsHex[node.type];

            // Fetch neighbors
            let neighborsHtml = '<div style="color:#6b6b8a;padding:8px 0;">Loading...</div>';
            try {
                const res = await fetch(`/admin/graph/neighbors?node_id=${encodeURIComponent(node.id)}`);
                const data = await res.json();

                neighborNodes.clear();
                data.neighbors.forEach(n => neighborNodes.add(n.id));
                render();

                if (data.neighbors.length > 0) {
                    neighborsHtml = data.neighbors.map(n => `
                        <div class="neighbor-item" data-x="${n.x}" data-y="${n.y}" data-id="${n.id}">
                            <span class="neighbor-dot" style="background:${typeColorsHex[n.type]}"></span>
                            <span class="neighbor-name">${n.name}</span>
                        </div>
                    `).join('');
                } else {
                    neighborsHtml = '<div style="color:#6b6b8a;padding:8px 0;">No connections</div>';
                }
            } catch (e) {
                neighborsHtml = '<div style="color:#6b6b8a;padding:8px 0;">Failed to load</div>';
            }

            // Determine what content to show
            const displayContent = node.fullContent || node.name || '';
            const hasContent = displayContent.length > 0;

            content.innerHTML = `
                <span class="type-badge" style="background:${color}20;color:${color}">${node.type}${node.subtype ? ' · ' + node.subtype : ''}</span>
                <h3>${node.name}</h3>
                ${hasContent ? `
                <div class="info-row">
                    <div class="info-label">Content</div>
                    <div class="info-value" style="max-height:150px;overflow-y:auto;white-space:pre-wrap;font-size:12px;line-height:1.5;background:rgba(0,0,0,0.2);padding:8px;border-radius:4px;margin-top:4px;">${displayContent}</div>
                </div>` : ''}
                <div class="info-row" style="display:flex;gap:16px;">
                    <div>
                        <div class="info-label">Connections</div>
                        <div class="info-value">${node.conn || 0}</div>
                    </div>
                    <div>
                        <div class="info-label">Cluster</div>
                        <div class="info-value">${node.cluster}</div>
                    </div>
                </div>
                <div class="info-row">
                    <div class="info-label">Connected Nodes</div>
                    <div style="max-height:200px;overflow-y:auto;margin-top:8px;">${neighborsHtml}</div>
                </div>
            `;
            panel.classList.add('visible');

            // Add click handlers for neighbors
            content.querySelectorAll('.neighbor-item').forEach(item => {
                item.addEventListener('click', () => {
                    const x = parseFloat(item.dataset.x);
                    const y = parseFloat(item.dataset.y);
                    const id = item.dataset.id;
                    viewX = x;
                    viewY = y;
                    if (nodeMap[id]) {
                        selectedNode = nodeMap[id];
                        showNodeInfo(selectedNode);
                    }
                    render();
                    scheduleViewportLoad();
                });
            });
        }

        function closeNodeInfo() {
            document.getElementById('node-info').classList.remove('visible');
            selectedNode = null;
            neighborNodes.clear();
            render();
        }

        // Type filter
        function toggleTypeFilter(type) {
            const item = document.querySelector(`.legend-item[data-type="${type}"]`);

            if (typeFilters.has(type)) {
                typeFilters.delete(type);
                item.classList.remove('active');
            } else {
                // If clicking same type that's the only active one, clear all
                if (typeFilters.size === 1 && typeFilters.has(type)) {
                    typeFilters.clear();
                    document.querySelectorAll('.legend-item').forEach(el => el.classList.remove('active', 'dimmed'));
                } else {
                    typeFilters.add(type);
                    item.classList.add('active');
                }
            }

            // Update dimmed state
            document.querySelectorAll('.legend-item').forEach(el => {
                const t = el.dataset.type;
                if (typeFilters.size > 0 && !typeFilters.has(t)) {
                    el.classList.add('dimmed');
                } else {
                    el.classList.remove('dimmed');
                }
            });

            render();
        }

        function computeClusterCenters() {
            clusterCenters = {};
            const clusterNodes = {};

            // Group nodes by cluster
            for (const node of nodes) {
                const c = node.cluster || 0;
                if (!clusterNodes[c]) clusterNodes[c] = [];
                clusterNodes[c].push(node);
            }

            // Compute centers, radius, and gather top nodes
            for (const [clusterId, clusterNodeList] of Object.entries(clusterNodes)) {
                const sumX = clusterNodeList.reduce((s, n) => s + n.x, 0);
                const sumY = clusterNodeList.reduce((s, n) => s + n.y, 0);
                const count = clusterNodeList.length;
                const cx = sumX / count;
                const cy = sumY / count;

                // Compute radius as max distance from center to any node + padding
                let maxDist = 0;
                for (const node of clusterNodeList) {
                    const dx = node.x - cx;
                    const dy = node.y - cy;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist > maxDist) maxDist = dist;
                }
                const radius = maxDist + 100;

                // Sort by connections to find top nodes
                const sorted = [...clusterNodeList].sort((a, b) => (b.conn || 0) - (a.conn || 0));
                const topNodes = sorted.slice(0, 5).map(n => n.name || n.id);

                clusterCenters[clusterId] = {
                    x: cx,
                    y: cy,
                    radius: radius,
                    nodeCount: count,
                    topNodes: topNodes,
                    name: topNodes[0]?.length > 20 ? topNodes[0].substring(0, 20) + '...' : (topNodes[0] || 'Cluster')
                };
            }
        }

        function computeInterClusterEdges() {
            // Count edges between different clusters
            const edgeCounts = {};

            for (const link of links) {
                const s = nodeMap[link.source];
                const t = nodeMap[link.target];
                if (!s || !t) continue;

                const c1 = s.cluster || 0;
                const c2 = t.cluster || 0;

                if (c1 === c2) continue;

                const key = c1 < c2 ? `${c1}-${c2}` : `${c2}-${c1}`;
                edgeCounts[key] = (edgeCounts[key] || 0) + 1;
            }

            interClusterEdges = [];
            for (const [key, count] of Object.entries(edgeCounts)) {
                const [from, to] = key.split('-').map(Number);
                interClusterEdges.push({ from, to, count });
            }

            interClusterEdges.sort((a, b) => b.count - a.count);
        }

        function computeConnStats() {
            // Compute connection percentiles for LOD filtering
            // L0: top 0.1%, L1: top 1%, L2: top 10%, L3: top 50%, L4: all
            if (nodes.length === 0) {
                connStats = { max: 1, p999: 1, p99: 1, p90: 1, p50: 1 };
                return;
            }

            const conns = nodes.map(n => n.conn || 0).sort((a, b) => b - a);
            const n = conns.length;

            connStats = {
                max: conns[0] || 1,
                p999: conns[Math.floor(n * 0.001)] || 1,  // top 0.1%
                p99: conns[Math.floor(n * 0.01)] || 1,    // top 1%
                p90: conns[Math.floor(n * 0.1)] || 1,     // top 10%
                p50: conns[Math.floor(n * 0.5)] || 1      // top 50%
            };

            console.log('Connection stats for LOD:', connStats);
        }

        function computeHierarchicalCenters() {
            // Compute centers for all 5 hierarchy levels
            level0Centers = {};
            level1Centers = {};
            level2Centers = {};
            level3Centers = {};
            level0Edges = [];
            level1Edges = [];
            level2Edges = [];
            level3Edges = [];

            // Helper to compute centers for a level
            function computeLevelCenters(levelKeys, radiusPadding) {
                const grouped = {};
                for (const node of nodes) {
                    const key = levelKeys.map(k => node[k] || 0).join('-');
                    if (!grouped[key]) grouped[key] = [];
                    grouped[key].push(node);
                }

                const centers = {};
                for (const [key, nodeList] of Object.entries(grouped)) {
                    if (nodeList.length === 0) continue;

                    const sumX = nodeList.reduce((s, n) => s + n.x, 0);
                    const sumY = nodeList.reduce((s, n) => s + n.y, 0);
                    const count = nodeList.length;
                    const cx = sumX / count;
                    const cy = sumY / count;

                    let maxDist = 0;
                    for (const node of nodeList) {
                        const dx = node.x - cx;
                        const dy = node.y - cy;
                        const dist = Math.sqrt(dx*dx + dy*dy);
                        if (dist > maxDist) maxDist = dist;
                    }

                    const sorted = [...nodeList].sort((a, b) => (b.conn || 0) - (a.conn || 0));
                    const topName = sorted[0]?.name || 'Cluster';
                    const parent = parseInt(key.split('-')[0]) || 0;

                    centers[key] = {
                        x: cx, y: cy,
                        radius: maxDist + radiusPadding,
                        nodeCount: count,
                        parent: parent,
                        name: topName.length > 25 ? topName.substring(0, 25) + '...' : topName
                    };
                }
                return centers;
            }

            // Compute centers for each level
            level0Centers = computeLevelCenters(['level0'], 200);
            level1Centers = computeLevelCenters(['level0', 'level1'], 150);
            level2Centers = computeLevelCenters(['level0', 'level1', 'level2'], 100);
            level3Centers = computeLevelCenters(['level0', 'level1', 'level2', 'level3'], 50);

            console.log(`Hierarchical centers: ${Object.keys(level0Centers).length} L0, ${Object.keys(level1Centers).length} L1, ${Object.keys(level2Centers).length} L2, ${Object.keys(level3Centers).length} L3`);

            // Helper to compute inter-cluster edges
            function computeLevelEdges(levelKeys) {
                const edgeCounts = {};
                for (const link of links) {
                    const s = nodeMap[link.source];
                    const t = nodeMap[link.target];
                    if (!s || !t) continue;

                    const keyS = levelKeys.map(k => s[k] || 0).join('-');
                    const keyT = levelKeys.map(k => t[k] || 0).join('-');
                    if (keyS === keyT) continue;

                    const edgeKey = keyS < keyT ? `${keyS}|${keyT}` : `${keyT}|${keyS}`;
                    edgeCounts[edgeKey] = (edgeCounts[edgeKey] || 0) + 1;
                }

                const edges = [];
                for (const [key, count] of Object.entries(edgeCounts)) {
                    const [from, to] = key.split('|');
                    edges.push({ from, to, count });
                }
                edges.sort((a, b) => b.count - a.count);
                return edges;
            }

            level0Edges = computeLevelEdges(['level0']);
            level1Edges = computeLevelEdges(['level0', 'level1']);
            level2Edges = computeLevelEdges(['level0', 'level1', 'level2']);
            level3Edges = computeLevelEdges(['level0', 'level1', 'level2', 'level3']);

            // Compute inter-level0 edges
            const l0EdgeCounts = {};
            for (const link of links) {
                const s = nodeMap[link.source];
                const t = nodeMap[link.target];
                if (!s || !t) continue;

                const l0s = s.level0 || 0;
                const l0t = t.level0 || 0;
                if (l0s === l0t) continue;

                const key = l0s < l0t ? `${l0s}-${l0t}` : `${l0t}-${l0s}`;
                l0EdgeCounts[key] = (l0EdgeCounts[key] || 0) + 1;
            }

            for (const [key, count] of Object.entries(l0EdgeCounts)) {
                const [from, to] = key.split('-').map(Number);
                level0Edges.push({ from, to, count });
            }
            level0Edges.sort((a, b) => b.count - a.count);

            // Compute inter-level1 edges
            const l1EdgeCounts = {};
            for (const link of links) {
                const s = nodeMap[link.source];
                const t = nodeMap[link.target];
                if (!s || !t) continue;

                const keyS = `${s.level0 || 0}-${s.level1 || 0}`;
                const keyT = `${t.level0 || 0}-${t.level1 || 0}`;
                if (keyS === keyT) continue;

                const edgeKey = keyS < keyT ? `${keyS}:${keyT}` : `${keyT}:${keyS}`;
                l1EdgeCounts[edgeKey] = (l1EdgeCounts[edgeKey] || 0) + 1;
            }

            for (const [key, count] of Object.entries(l1EdgeCounts)) {
                const [from, to] = key.split(':');
                level1Edges.push({ from, to, count });
            }
            level1Edges.sort((a, b) => b.count - a.count);

            console.log(`Hierarchical: ${Object.keys(level0Centers).length} L0, ${Object.keys(level1Centers).length} L1 clusters`);
        }

        // Search
        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');
        let searchTimeout = null;

        searchInput.addEventListener('input', () => {
            if (searchTimeout) clearTimeout(searchTimeout);
            const q = searchInput.value.trim();
            if (q.length < 2) {
                searchResults.classList.remove('visible');
                highlightedNodes.clear();
                render();
                return;
            }
            searchTimeout = setTimeout(() => doSearch(q), 300);
        });

        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const q = searchInput.value.trim();
                if (q.length >= 2) doSearch(q);
            }
            if (e.key === 'Escape') {
                searchInput.value = '';
                searchResults.classList.remove('visible');
                highlightedNodes.clear();
                render();
            }
        });

        async function doSearch(q) {
            try {
                const data = await (await fetch(`/admin/graph/search?q=${encodeURIComponent(q)}`)).json();

                highlightedNodes.clear();
                data.results.forEach(r => highlightedNodes.add(r.id));

                searchResults.innerHTML = data.results.length === 0
                    ? '<div style="padding:12px;color:#6b6b8a;">No results</div>'
                    : data.results.slice(0, 20).map(r => `
                        <div class="search-result" data-x="${r.x}" data-y="${r.y}" data-id="${r.id}">
                            <span class="search-result-name">${r.name}</span>
                            <span class="search-result-type" style="background:${typeColorsHex[r.type]}20;color:${typeColorsHex[r.type]}">${r.type}</span>
                        </div>
                    `).join('');

                searchResults.classList.add('visible');
                render();
            } catch (e) {
                console.error('Search failed:', e);
            }
        }

        searchResults.addEventListener('click', (e) => {
            const result = e.target.closest('.search-result');
            if (result) {
                viewX = parseFloat(result.dataset.x);
                viewY = parseFloat(result.dataset.y);
                scale = 2;
                searchResults.classList.remove('visible');
                render();
                scheduleViewportLoad();
            }
        });

        // Mouse events
        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            dragStartX = lastMouseX = e.clientX;
            dragStartY = lastMouseY = e.clientY;
        });

        canvas.addEventListener('mousemove', (e) => {
            if (isDragging) {
                viewX -= (e.clientX - lastMouseX) / scale;
                viewY += (e.clientY - lastMouseY) / scale;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                render();
                scheduleViewportLoad();
            } else {
                // Check for hierarchical cluster hover
                const clusterHit = findClusterAtPosition(e.clientX, e.clientY);
                if (clusterHit !== null) {
                    canvas.style.cursor = 'pointer';
                    const center = clusterHit.center;
                    const levelNames = { level0: 'L0 super-cluster', level1: 'L1 cluster', level2: 'L2 cluster', level3: 'L3 cluster' };
                    const levelName = levelNames[clusterHit.type] || 'cluster';
                    showTooltip({ name: `Click to zoom into "${center.name}" (${center.nodeCount} nodes, ${levelName})` }, e.clientX, e.clientY);
                    hoveredNode = null;
                    return;
                }

                canvas.style.cursor = 'default';
                const node = findNodeAt(e.clientX, e.clientY);
                if (node !== hoveredNode) {
                    hoveredNode = node;
                    node ? showTooltip(node, e.clientX, e.clientY) : hideTooltip();
                    render();
                }
            }
        });

        canvas.addEventListener('mouseup', async (e) => {
            if (Math.abs(e.clientX - dragStartX) < 5 && Math.abs(e.clientY - dragStartY) < 5) {
                // Check for hierarchical cluster click first (when zoomed out)
                const clusterHit = findClusterAtPosition(e.clientX, e.clientY);
                if (clusterHit !== null) {
                    // Target scales to reach the next zoom level (zoom in extra for better visibility):
                    // L0 → L1: need scale >= 0.008, target 0.025
                    // L1 → L2: need scale >= 0.02, target 0.055
                    // L2 → L3: need scale >= 0.045, target 0.1
                    // L3 → L4 (nodes): need scale >= 0.09, target 0.2
                    const targetScales = {
                        level0: 0.006,
                        level1: 0.015,
                        level2: 0.035,
                        level3: 0.065
                    };
                    const targetScale = targetScales[clusterHit.type] || 0.2;

                    // Extract L0 cluster ID for lazy loading
                    const l0Id = clusterHit.type === 'level0' ? parseInt(clusterHit.id) :
                                 parseInt(clusterHit.id.split('-')[0]);

                    await animateToHierarchicalCluster(clusterHit.center, targetScale, l0Id);
                    isDragging = false;
                    return;
                }

                // Then check for node click
                const node = findNodeAt(e.clientX, e.clientY);
                if (node) {
                    selectedNode = node;
                    showNodeInfo(node);
                } else {
                    closeNodeInfo();
                    highlightedNodes.clear();
                }
                render();
            }
            isDragging = false;
        });

        async function animateToHierarchicalCluster(center, targetScale, clusterId = null) {
            // Animate to center of hierarchical cluster with appropriate zoom
            const padding = 1.1;
            const fitScale = Math.min(
                window.innerWidth / (center.radius * 4 * padding),
                window.innerHeight / (center.radius * 4 * padding)
            );
            const finalScale = Math.max(targetScale, fitScale);
            console.log(`Zooming to cluster: ${center.name}, x=${center.x.toFixed(0)}, y=${center.y.toFixed(0)}, radius=${center.radius.toFixed(0)}, nodes=${center.nodeCount}, finalScale=${finalScale.toFixed(4)}`);

            // Lazy load: If this is an L0 cluster and not yet loaded, load it
            if (clusterId !== null && !loadedClusters.has(clusterId)) {
                console.log(`Lazy loading cluster ${clusterId}...`);
                await loadClusterData(clusterId);

                // Also load cross-cluster links if this is the first cluster
                if (loadedClusters.size === 1) {
                    loadCrossClusterLinks();  // Fire and forget
                }
            }

            animateTo(center.x, center.y, finalScale);
        }


        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const mouseWorld = screenToWorld(e.clientX, e.clientY);
            scale = Math.max(0.001, Math.min(10, scale * zoomFactor));
            const newMouseWorld = screenToWorld(e.clientX, e.clientY);
            viewX += mouseWorld.x - newMouseWorld.x;
            viewY += mouseWorld.y - newMouseWorld.y;
            render();
            scheduleViewportLoad();
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                // Always zoom out to initial overview
                animateToOverview();
                closeNodeInfo();
                highlightedNodes.clear();
                stopActivationAnimation();
                typeFilters.clear();
                document.querySelectorAll('.legend-item').forEach(el => el.classList.remove('active', 'dimmed'));
                searchInput.value = '';
                searchResults.classList.remove('visible');
                render();
            }
        });

        // ============ CHAT FUNCTIONALITY ============
        let lastActivation = { concepts: [], memories: [] };
        let chatOpen = false;
        let debugMode = false;
        let lastDebugInfo = null;
        let lastQuery = '';
        let forceIncludeNodes = [];
        let forceExcludeNodes = [];

        function toggleDebugMode() {
            debugMode = !debugMode;
            document.getElementById('debug-toggle').classList.toggle('active', debugMode);
        }

        function toggleChat() {
            chatOpen = !chatOpen;
            document.getElementById('chat-panel').classList.toggle('open', chatOpen);
            document.getElementById('chat-toggle').classList.toggle('open', chatOpen);
        }

        function clearChat() {
            document.getElementById('chat-messages').innerHTML =
                '<div class="chat-message system">Ask questions about your knowledge graph. Activated memories will be highlighted.</div>';
            lastActivation = { concepts: [], memories: [] };
            stopActivationAnimation();
            highlightedNodes.clear();
            render();
        }

        function showLastActivation() {
            if (lastActivation.concepts.length === 0 && lastActivation.memories.length === 0) {
                return;
            }
            activatedNodes.clear();
            lastActivation.concepts.forEach(id => activatedNodes.add(id));
            lastActivation.memories.forEach(id => activatedNodes.add(id));
            highlightedNodes.clear();
            activatedNodes.forEach(id => highlightedNodes.add(id));

            // Find center of activated nodes and pan to them
            let sumX = 0, sumY = 0, count = 0;
            for (const node of nodes) {
                if (activatedNodes.has(node.id)) {
                    sumX += node.x;
                    sumY += node.y;
                    count++;
                }
            }
            if (count > 0) {
                viewX = sumX / count;
                viewY = sumY / count;
                scale = Math.min(2, scale * 1.5);
            }
            render();
            scheduleViewportLoad();
        }

        function highlightActivation(concepts, memories) {
            lastActivation = { concepts, memories };
            activatedNodes.clear();
            concepts.forEach(id => activatedNodes.add(id));
            memories.forEach(id => activatedNodes.add(id));
            highlightedNodes.clear();
            activatedNodes.forEach(id => highlightedNodes.add(id));
            startActivationAnimation();
        }

        function startActivationAnimation() {
            if (animationFrameId) cancelAnimationFrame(animationFrameId);

            function animate() {
                if (activatedNodes.size === 0) {
                    animationFrameId = null;
                    return;
                }
                render();
                animationFrameId = requestAnimationFrame(animate);
            }
            animate();
        }

        function stopActivationAnimation() {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
            activatedNodes.clear();
            render();
        }

        // Continuous animation for cluster glow pulsing
        let continuousAnimationId = null;
        function startContinuousAnimation() {
            if (continuousAnimationId) return;

            function animate() {
                const semanticZoomLevel = scale < 0.008 ? 0 : (scale < 0.02 ? 1 : (scale < 0.045 ? 2 : (scale < 0.09 ? 3 : 4)));
                // Animate when showing clusters (for pulsing) - levels 0-3
                if (semanticZoomLevel <= 3) {
                    render();
                }
                continuousAnimationId = requestAnimationFrame(animate);
            }
            animate();
        }

        function stopContinuousAnimation() {
            if (continuousAnimationId) {
                cancelAnimationFrame(continuousAnimationId);
                continuousAnimationId = null;
            }
        }

        async function sendChat(isRetest = false) {
            const input = document.getElementById('chat-input');
            const sendBtn = document.getElementById('chat-send');
            const messages = document.getElementById('chat-messages');
            const query = isRetest ? lastQuery : input.value.trim();

            if (!query) return;

            // Store for retest
            if (!isRetest) {
                lastQuery = query;
                forceIncludeNodes = [];
                forceExcludeNodes = [];
            }

            // Add user message (only if not a retest)
            if (!isRetest) {
                const userMsg = document.createElement('div');
                userMsg.className = 'chat-message user';
                userMsg.textContent = query;
                messages.appendChild(userMsg);
            } else {
                // Add retest indicator
                const retestIndicator = document.createElement('div');
                retestIndicator.className = 'retest-indicator';
                const forceInfo = [];
                if (forceIncludeNodes.length) forceInfo.push(`+${forceIncludeNodes.length} forced`);
                if (forceExcludeNodes.length) forceInfo.push(`-${forceExcludeNodes.length} excluded`);
                retestIndicator.textContent = `Re-testing with: ${forceInfo.join(', ')}`;
                messages.appendChild(retestIndicator);
            }

            // Clear input and disable
            if (!isRetest) input.value = '';
            sendBtn.disabled = true;
            input.disabled = true;

            // Add loading message
            const loadingMsg = document.createElement('div');
            loadingMsg.className = 'chat-message assistant';
            loadingMsg.innerHTML = '<span class="activation-glow">Thinking...</span>';
            messages.appendChild(loadingMsg);
            messages.scrollTop = messages.scrollHeight;

            try {
                const requestBody = {
                    model: 'engram',
                    messages: [{ role: 'user', content: query }],
                    top_k_memories: 20,
                    top_k_episodes: 5,
                    debug: debugMode
                };

                if (forceIncludeNodes.length > 0) {
                    requestBody.force_include_nodes = forceIncludeNodes;
                }
                if (forceExcludeNodes.length > 0) {
                    requestBody.force_exclude_nodes = forceExcludeNodes;
                }

                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });

                const data = await response.json();

                // Remove loading message
                messages.removeChild(loadingMsg);

                // Extract response
                const answer = data.choices[0].message.content;
                const concepts = data.concepts_activated || [];
                const memories = data.memories_used || [];
                const memoriesCount = data.memories_count || 0;

                // Store debug info
                if (data.debug_info) {
                    lastDebugInfo = data.debug_info;
                }

                // Add assistant message
                const assistantMsg = document.createElement('div');
                assistantMsg.className = 'chat-message assistant';

                // Format answer (basic markdown-like formatting)
                let formattedAnswer = answer
                    .replace(/\\n/g, '<br>')
                    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.+?)\*/g, '<em>$1</em>');

                assistantMsg.innerHTML = formattedAnswer;

                // Add activation info if there are activated nodes
                if (concepts.length > 0 || memories.length > 0) {
                    const activationInfo = document.createElement('div');
                    activationInfo.className = 'activated-info';
                    activationInfo.textContent = `Activated: ${concepts.length} concepts, ${memoriesCount} memories`;
                    activationInfo.onclick = () => highlightActivation(concepts, memories);
                    assistantMsg.appendChild(activationInfo);

                    // Auto-highlight
                    highlightActivation(concepts, memories);
                }

                // Add debug section if debug mode is on
                if (debugMode && data.debug_info) {
                    const debugSection = createDebugSection(data.debug_info);
                    assistantMsg.appendChild(debugSection);
                }

                messages.appendChild(assistantMsg);
                messages.scrollTop = messages.scrollHeight;

            } catch (error) {
                messages.removeChild(loadingMsg);
                const errorMsg = document.createElement('div');
                errorMsg.className = 'chat-message system';
                errorMsg.textContent = 'Error: ' + error.message;
                messages.appendChild(errorMsg);
            } finally {
                sendBtn.disabled = false;
                input.disabled = false;
                input.focus();
            }
        }

        function createDebugSection(debugInfo) {
            const section = document.createElement('div');
            section.className = 'debug-section collapsed';

            const memCount = debugInfo.retrieved_memories?.length || 0;
            const conceptCount = debugInfo.activated_concepts?.length || 0;

            section.innerHTML = `
                <div class="debug-header" onclick="toggleDebugSection(this.parentElement)">
                    <span><span class="toggle-icon">▶</span> Debug: ${conceptCount} concepts, ${memCount} memories</span>
                </div>
                <div class="debug-content">
                    <div class="debug-tab-bar">
                        <button class="active" onclick="switchDebugTab(this, 'memories')">Memories</button>
                        <button onclick="switchDebugTab(this, 'concepts')">Concepts</button>
                    </div>
                    <div class="debug-list" data-tab="memories">
                        ${renderMemoryList(debugInfo.retrieved_memories || [])}
                    </div>
                    <div class="debug-list" data-tab="concepts" style="display:none">
                        ${renderConceptList(debugInfo.activated_concepts || [])}
                    </div>
                    <div class="sources-legend">
                        Sources: <span style="color:#5eead4">V</span>=Vector <span style="color:#a78bfa">B</span>=BM25 <span style="color:#f472b6">G</span>=Graph <span style="color:#fbbf24">F</span>=Forced
                    </div>
                </div>
            `;

            return section;
        }

        function renderMemoryList(memories) {
            if (!memories.length) return '<div style="color:#6b6b8a;padding:8px;">No memories retrieved</div>';

            return memories.map(m => {
                const scorePercent = Math.min(100, m.score * 100);
                const sources = (m.sources || []).map(s => {
                    const letter = s === 'vector' ? 'V' : s === 'bm25' ? 'B' : s === 'graph' ? 'G' : 'F';
                    return `<span class="source-badge ${s}">${letter}</span>`;
                }).join('');
                const filteredClass = m.included ? '' : 'filtered';
                const testBtn = m.included
                    ? `<button class="test-btn" onclick="testExcludeNode('${m.id}')" title="Test without this">−</button>`
                    : `<button class="test-btn" onclick="testIncludeNode('${m.id}')" title="Test with this">+</button>`;

                return `
                    <div class="debug-item ${filteredClass}" data-id="${m.id}" onclick="highlightNodeInGraph('${m.id}', event)">
                        <span class="score-bar" style="width:${scorePercent}%"></span>
                        <span class="name" title="${escapeHtml(m.name)}">${escapeHtml(m.name)}</span>
                        <span class="score">${m.score.toFixed(2)}</span>
                        <span class="sources">${sources}</span>
                        ${testBtn}
                    </div>
                `;
            }).join('');
        }

        function renderConceptList(concepts) {
            if (!concepts.length) return '<div style="color:#6b6b8a;padding:8px;">No concepts activated</div>';

            return concepts.map(c => {
                const scorePercent = Math.min(100, c.activation * 100);
                const filteredClass = c.included ? '' : 'filtered';
                const testBtn = c.included
                    ? `<button class="test-btn" onclick="testExcludeNode('${c.id}')" title="Test without this">−</button>`
                    : `<button class="test-btn" onclick="testIncludeNode('${c.id}')" title="Test with this">+</button>`;

                return `
                    <div class="debug-item ${filteredClass}" data-id="${c.id}" onclick="highlightNodeInGraph('${c.id}', event)">
                        <span class="score-bar" style="width:${scorePercent}%"></span>
                        <span class="name" title="${escapeHtml(c.name)}">${escapeHtml(c.name)}</span>
                        <span class="score">${c.activation.toFixed(2)}</span>
                        <span class="hop-badge">hop ${c.hop}</span>
                        ${testBtn}
                    </div>
                `;
            }).join('');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function toggleDebugSection(section) {
            section.classList.toggle('collapsed');
        }

        function switchDebugTab(btn, tab) {
            const tabBar = btn.parentElement;
            const content = tabBar.parentElement;

            tabBar.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            content.querySelectorAll('.debug-list').forEach(list => {
                list.style.display = list.dataset.tab === tab ? 'block' : 'none';
            });
        }

        function highlightNodeInGraph(nodeId, event) {
            if (event && event.target.classList.contains('test-btn')) return;

            const node = nodeMap[nodeId];
            if (node) {
                viewX = node.x;
                viewY = node.y;
                scale = Math.max(scale, 1.5);
                selectedNode = node;
                highlightedNodes.clear();
                highlightedNodes.add(nodeId);
                render();
                scheduleViewportLoad();
                showNodeInfo(node);
            }
        }

        function testIncludeNode(nodeId) {
            event.stopPropagation();
            if (!forceIncludeNodes.includes(nodeId)) {
                forceIncludeNodes.push(nodeId);
            }
            forceExcludeNodes = forceExcludeNodes.filter(id => id !== nodeId);
            sendChat(true);
        }

        function testExcludeNode(nodeId) {
            event.stopPropagation();
            if (!forceExcludeNodes.includes(nodeId)) {
                forceExcludeNodes.push(nodeId);
            }
            forceIncludeNodes = forceIncludeNodes.filter(id => id !== nodeId);
            sendChat(true);
        }

        // Chat input handlers
        document.getElementById('chat-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChat();
            }
        });

        // Auto-resize chat input
        document.getElementById('chat-input').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
        });

        // Lazy loading state
        let loadedClusters = new Set();  // L0 cluster IDs that have been loaded
        let clusterNodes = {};  // L0 ID -> array of nodes
        let clusterLinks = {};  // L0 ID -> array of internal links
        let crossClusterLinks = [];  // Links between clusters
        let clusterCentersData = null;  // Cached cluster centers from API
        let isLoadingCluster = false;

        async function init() {
            resize();
            window.addEventListener('resize', resize);

            const boundsData = await (await fetch('/admin/graph/bounds')).json();
            if (!boundsData.has_layout) {
                document.getElementById('loading').innerHTML = 'No layout. Run: uv run python scripts/compute_layout.py';
                return;
            }

            bounds = boundsData;
            viewX = (bounds.min_x + bounds.max_x) / 2;
            viewY = (bounds.min_y + bounds.max_y) / 2;
            const graphWidth = bounds.max_x - bounds.min_x;
            const graphHeight = bounds.max_y - bounds.min_y;
            scale = Math.max(0.001, Math.min(1, Math.min(
                window.innerWidth * 0.7 / (graphWidth * 2),
                window.innerHeight * 0.7 / (graphHeight * 2)
            )));

            initialView = { x: viewX, y: viewY, scale: scale };

            const stats = await (await fetch('/admin/graph/stats')).json();
            document.getElementById('c-count').textContent = stats.concepts;
            document.getElementById('s-count').textContent = stats.semantic;
            document.getElementById('e-count').textContent = stats.episodic;
            document.getElementById('stats').innerHTML = `<div>Total: <strong style="color:#5eead4">${stats.total}</strong></div><div>Clusters: <strong style="color:#a78bfa">${stats.clusters}</strong></div>`;

            // Lazy load: Only load cluster centers initially (very fast)
            await loadClusterCentersOnly();

            document.getElementById('loading').style.display = 'none';

            // Start with overview (no nodes loaded yet, just cluster bubbles)
            render();

            startContinuousAnimation();
        }

        async function loadClusterCentersOnly() {
            const loadingEl = document.getElementById('loading');
            loadingEl.innerHTML = 'Loading cluster map...';

            try {
                clusterCentersData = await (await fetch('/admin/graph/clusters')).json();

                // Build hierarchical centers from API data (no nodes loaded yet)
                buildCentersFromClusterData(clusterCentersData);

                console.log(`Loaded ${clusterCentersData.l0.length} L0 clusters, ${clusterCentersData.l1.length} L1, ${clusterCentersData.l2.length} L2 (${clusterCentersData.total_nodes} total nodes available)`);
            } catch (e) {
                console.error('Failed to load cluster centers:', e);
            }
        }

        function buildCentersFromClusterData(data) {
            // Build level0Centers from API data
            level0Centers = {};
            for (const c of data.l0) {
                level0Centers[c.level0] = {
                    x: c.center_x,
                    y: c.center_y,
                    radius: c.radius || 200,
                    nodeCount: c.node_count,
                    name: c.name || 'Cluster',
                    loaded: loadedClusters.has(c.level0)
                };
            }

            // Build level1Centers
            level1Centers = {};
            for (const c of data.l1) {
                const key = `${c.level0}-${c.level1}`;
                level1Centers[key] = {
                    x: c.center_x,
                    y: c.center_y,
                    radius: 150,
                    nodeCount: c.node_count,
                    parent: c.level0,
                    name: c.name || `Group ${c.level1}`
                };
            }

            // Build level2Centers
            level2Centers = {};
            for (const c of data.l2) {
                const key = `${c.level0}-${c.level1}-${c.level2}`;
                level2Centers[key] = {
                    x: c.center_x,
                    y: c.center_y,
                    radius: 100,
                    nodeCount: c.node_count,
                    parent: c.level0,
                    name: c.name || `Set ${c.level2}`
                };
            }

            // Build L0 edges
            level0Edges = data.edges.map(e => ({
                from: String(e.from),
                to: String(e.to),
                count: e.weight
            }));

            level1Edges = [];
            level2Edges = [];
            level3Edges = [];
            level3Centers = {};
        }

        async function loadClusterData(level0Id) {
            if (loadedClusters.has(level0Id) || isLoadingCluster) return;

            isLoadingCluster = true;
            console.log(`Loading cluster ${level0Id}...`);

            try {
                const data = await (await fetch(`/admin/graph/data/cluster/${level0Id}`)).json();

                clusterNodes[level0Id] = data.nodes;
                clusterLinks[level0Id] = data.links;
                loadedClusters.add(level0Id);

                // Mark cluster as loaded in centers
                if (level0Centers[level0Id]) {
                    level0Centers[level0Id].loaded = true;
                }

                // Rebuild nodes/links arrays from all loaded clusters
                rebuildNodesFromLoadedClusters();

                console.log(`Loaded cluster ${level0Id}: ${data.nodes.length} nodes, ${data.links.length} links`);
                render();
            } catch (e) {
                console.error(`Failed to load cluster ${level0Id}:`, e);
            } finally {
                isLoadingCluster = false;
            }
        }

        function rebuildNodesFromLoadedClusters() {
            // Combine all loaded cluster data into nodes/links arrays
            nodes = [];
            nodeMap = {};

            for (const l0Id of loadedClusters) {
                const clusterNodeList = clusterNodes[l0Id] || [];
                for (const n of clusterNodeList) {
                    if (!nodeMap[n.id]) {
                        nodes.push(n);
                        nodeMap[n.id] = n;
                    }
                }
            }

            // Combine internal links from loaded clusters
            links = [];
            for (const l0Id of loadedClusters) {
                const clusterLinkList = clusterLinks[l0Id] || [];
                for (const link of clusterLinkList) {
                    if (nodeMap[link.source] && nodeMap[link.target]) {
                        links.push(link);
                    }
                }
            }

            // Add cross-cluster links between loaded clusters
            for (const link of crossClusterLinks) {
                if (nodeMap[link.source] && nodeMap[link.target]) {
                    links.push(link);
                }
            }

            // Recompute centers for loaded data
            if (nodes.length > 0) {
                computeDetailedCentersForLoadedNodes();
            }

            computeConnStats();
            computeClusterCenters();
            computeInterClusterEdges();
        }

        function computeDetailedCentersForLoadedNodes() {
            // Update level1-3 centers with actual loaded node data
            const tempLevel1 = {};
            const tempLevel2 = {};
            const tempLevel3 = {};

            for (const node of nodes) {
                // L1
                const l1Key = `${node.level0}-${node.level1}`;
                if (!tempLevel1[l1Key]) tempLevel1[l1Key] = [];
                tempLevel1[l1Key].push(node);

                // L2
                const l2Key = `${node.level0}-${node.level1}-${node.level2}`;
                if (!tempLevel2[l2Key]) tempLevel2[l2Key] = [];
                tempLevel2[l2Key].push(node);

                // L3
                const l3Key = `${node.level0}-${node.level1}-${node.level2}-${node.level3}`;
                if (!tempLevel3[l3Key]) tempLevel3[l3Key] = [];
                tempLevel3[l3Key].push(node);
            }

            // Compute L1 centers from actual nodes
            for (const [key, nodeList] of Object.entries(tempLevel1)) {
                const cx = nodeList.reduce((s, n) => s + n.x, 0) / nodeList.length;
                const cy = nodeList.reduce((s, n) => s + n.y, 0) / nodeList.length;
                let maxDist = 0;
                for (const n of nodeList) {
                    const d = Math.sqrt((n.x - cx)**2 + (n.y - cy)**2);
                    if (d > maxDist) maxDist = d;
                }
                const sorted = [...nodeList].sort((a, b) => (b.conn || 0) - (a.conn || 0));
                level1Centers[key] = {
                    x: cx, y: cy,
                    radius: maxDist + 100,
                    nodeCount: nodeList.length,
                    parent: parseInt(key.split('-')[0]) || 0,
                    name: (sorted[0]?.name || 'Cluster').substring(0, 25)
                };
            }

            // Compute L2 centers
            for (const [key, nodeList] of Object.entries(tempLevel2)) {
                const cx = nodeList.reduce((s, n) => s + n.x, 0) / nodeList.length;
                const cy = nodeList.reduce((s, n) => s + n.y, 0) / nodeList.length;
                let maxDist = 0;
                for (const n of nodeList) {
                    const d = Math.sqrt((n.x - cx)**2 + (n.y - cy)**2);
                    if (d > maxDist) maxDist = d;
                }
                const sorted = [...nodeList].sort((a, b) => (b.conn || 0) - (a.conn || 0));
                level2Centers[key] = {
                    x: cx, y: cy,
                    radius: maxDist + 50,
                    nodeCount: nodeList.length,
                    parent: parseInt(key.split('-')[0]) || 0,
                    name: (sorted[0]?.name || 'Group').substring(0, 25)
                };
            }

            // Compute L3 centers
            for (const [key, nodeList] of Object.entries(tempLevel3)) {
                const cx = nodeList.reduce((s, n) => s + n.x, 0) / nodeList.length;
                const cy = nodeList.reduce((s, n) => s + n.y, 0) / nodeList.length;
                let maxDist = 0;
                for (const n of nodeList) {
                    const d = Math.sqrt((n.x - cx)**2 + (n.y - cy)**2);
                    if (d > maxDist) maxDist = d;
                }
                const sorted = [...nodeList].sort((a, b) => (b.conn || 0) - (a.conn || 0));
                level3Centers[key] = {
                    x: cx, y: cy,
                    radius: maxDist + 30,
                    nodeCount: nodeList.length,
                    parent: parseInt(key.split('-')[0]) || 0,
                    name: (sorted[0]?.name || 'Set').substring(0, 25)
                };
            }
        }

        async function loadCrossClusterLinks() {
            if (crossClusterLinks.length > 0) return;

            try {
                const data = await (await fetch('/admin/graph/links/cross')).json();
                crossClusterLinks = data.links;
                console.log(`Loaded ${crossClusterLinks.length} cross-cluster links`);
                rebuildNodesFromLoadedClusters();
            } catch (e) {
                console.error('Failed to load cross-cluster links:', e);
            }
        }

        init();
    </script>
</body>
</html>
"""
