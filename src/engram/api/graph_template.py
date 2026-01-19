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

        // LEVEL-BASED NAVIGATION STATE
        // Level 0: L0 clusters, Level 1: L1 clusters, Level 2: L2 clusters, Level 3: actual nodes
        let currentLevel = 0;
        let parentPath = { l0: null, l1: null, l2: null };
        let currentItems = [];  // clusters or nodes at current level
        let currentEdges = [];  // edges between items at current level
        let bounds = { min_x: -1000, max_x: 1000, min_y: -1000, max_y: 1000 };
        let viewX = 0, viewY = 0, scale = 1;

        // Node-level state (only used at level 3)
        let nodes = [], nodeMap = {};
        let selectedNode = null, hoveredNode = null;
        let highlightedNodes = new Set();
        let neighborNodes = new Set();
        let activatedNodes = new Set();
        let typeFilters = new Set();

        let isDragging = false, lastMouseX = 0, lastMouseY = 0, dragStartX = 0, dragStartY = 0;
        let animationFrameId = null;
        let zoomAnimation = null;
        let initialView = { x: 0, y: 0, scale: 1 };
        let viewportDebounceTimer = null;

        function animateTo(targetX, targetY, targetScale, duration = 600) {
            if (zoomAnimation) cancelAnimationFrame(zoomAnimation.frameId);

            const startX = viewX, startY = viewY, startScale = scale;
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
                }
            }

            zoomAnimation = { frameId: requestAnimationFrame(step) };
        }

        function findItemAtPosition(screenX, screenY) {
            const world = screenToWorld(screenX, screenY);

            // At node level, find nodes
            if (currentLevel === 3) {
                let closest = null, closestDist = Infinity;
                for (const node of nodes) {
                    const dx = node.x - world.x, dy = node.y - world.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    const threshold = 50 / scale + 30;
                    if (dist < threshold && dist < closestDist) {
                        closest = node;
                        closestDist = dist;
                    }
                }
                return closest ? { type: 'node', item: closest } : null;
            }

            // At cluster levels, find clusters
            for (const item of currentItems) {
                const dx = world.x - item.x, dy = world.y - item.y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                const clickRadius = Math.max(50, 60 / scale);
                if (dist < clickRadius) {
                    return { type: 'cluster', item: item };
                }
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
                float outerGlow = smoothstep(0.5, 0.3, dist) * 0.4;
                float borderOuter = 0.42, borderInner = 0.35;
                float border = smoothstep(borderInner, borderOuter, dist) * smoothstep(0.5, borderOuter, dist);
                vec3 borderColor = v_color.rgb * 1.5;
                float coreFill = 1.0 - smoothstep(0.0, borderInner, dist) * 0.2;
                vec3 finalColor = mix(v_color.rgb * coreFill, borderColor, border * 0.6);
                finalColor += outerGlow * v_color.rgb;
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
            gl.clearColor(0.02, 0.024, 0.05, 1);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

            const w = canvas.width / window.devicePixelRatio;
            const h = canvas.height / window.devicePixelRatio;
            const dpr = window.devicePixelRatio;
            labelCtx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);

            // Build cluster lookup for edges (at cluster levels 0-2)
            const itemMap = {};
            for (const item of currentItems) {
                itemMap[item.id] = item;
            }

            // Draw edges on 2D canvas
            labelCtx.save();
            labelCtx.scale(dpr, dpr);

            if (currentLevel < 3 && currentEdges.length > 0) {
                const maxWeight = Math.max(1, ...currentEdges.map(e => e.weight || 1));
                for (const edge of currentEdges) {
                    const c1 = itemMap[edge.from];
                    const c2 = itemMap[edge.to];
                    if (!c1 || !c2) continue;

                    const pos1 = worldToScreen(c1.x, c1.y);
                    const pos2 = worldToScreen(c2.x, c2.y);
                    if (pos1.x < -100 && pos2.x < -100) continue;
                    if (pos1.x > w + 100 && pos2.x > w + 100) continue;

                    const color1 = clusterPalette[edge.from % clusterPalette.length];
                    const color2 = clusterPalette[edge.to % clusterPalette.length];
                    const relWeight = (edge.weight || 1) / maxWeight;
                    const lineWidth = 2 + relWeight * 6;

                    // Curved line
                    const mx = (pos1.x + pos2.x) / 2, my = (pos1.y + pos2.y) / 2;
                    const dx = pos2.x - pos1.x, dy = pos2.y - pos1.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    const nx = dist > 0 ? -dy / dist : 0, ny = dist > 0 ? dx / dist : 0;
                    const cx = mx + nx * Math.min(dist * 0.15, 60);
                    const cy = my + ny * Math.min(dist * 0.15, 60);

                    const gradient = labelCtx.createLinearGradient(pos1.x, pos1.y, pos2.x, pos2.y);
                    gradient.addColorStop(0, `rgba(${color1[0]*255}, ${color1[1]*255}, ${color1[2]*255}, 0.6)`);
                    gradient.addColorStop(1, `rgba(${color2[0]*255}, ${color2[1]*255}, ${color2[2]*255}, 0.6)`);

                    labelCtx.strokeStyle = gradient;
                    labelCtx.lineWidth = lineWidth;
                    labelCtx.lineCap = 'round';
                    labelCtx.beginPath();
                    labelCtx.moveTo(pos1.x, pos1.y);
                    labelCtx.quadraticCurveTo(cx, cy, pos2.x, pos2.y);
                    labelCtx.stroke();
                }
            }

            // Draw node-level edges at level 3
            if (currentLevel === 3 && currentEdges.length > 0) {
                for (const edge of currentEdges) {
                    const s = nodeMap[edge.from];
                    const t = nodeMap[edge.to];
                    if (!s || !t) continue;

                    const pos1 = worldToScreen(s.x, s.y);
                    const pos2 = worldToScreen(t.x, t.y);
                    const sColor = typeColors[s.type] || typeColors.concept;
                    const tColor = typeColors[t.type] || typeColors.concept;

                    const gradient = labelCtx.createLinearGradient(pos1.x, pos1.y, pos2.x, pos2.y);
                    gradient.addColorStop(0, `rgba(${sColor[0]*255}, ${sColor[1]*255}, ${sColor[2]*255}, 0.5)`);
                    gradient.addColorStop(1, `rgba(${tColor[0]*255}, ${tColor[1]*255}, ${tColor[2]*255}, 0.5)`);

                    labelCtx.strokeStyle = gradient;
                    labelCtx.lineWidth = 1.5;
                    labelCtx.beginPath();
                    labelCtx.moveTo(pos1.x, pos1.y);
                    labelCtx.lineTo(pos2.x, pos2.y);
                    labelCtx.stroke();
                }
            }
            labelCtx.restore();

            // Render points (clusters or nodes)
            gl.useProgram(nodeProgram);
            gl.uniform2f(gl.getUniformLocation(nodeProgram, 'u_resolution'), w / 2, h / 2);
            gl.uniform2f(gl.getUniformLocation(nodeProgram, 'u_view'), viewX, viewY);
            gl.uniform1f(gl.getUniformLocation(nodeProgram, 'u_scale'), scale);

            const nodeData = [];
            const labelsToRender = [];

            if (currentLevel < 3) {
                // Draw clusters
                const maxCount = Math.max(1, ...currentItems.map(c => c.count || 1));
                for (const item of currentItems) {
                    const pos = worldToScreen(item.x, item.y);
                    if (pos.x < -200 || pos.x > w + 200 || pos.y < -200 || pos.y > h + 200) continue;

                    const color = clusterPalette[item.id % clusterPalette.length];
                    const relSize = Math.sqrt((item.count || 1) / maxCount);
                    const size = (30 + relSize * 80) / scale;

                    nodeData.push(item.x, item.y, color[0], color[1], color[2], 1.0, size);
                    labelsToRender.push({
                        type: 'cluster', x: item.x, y: item.y,
                        name: item.name, count: item.count, color: color
                    });
                }
            } else {
                // Draw actual nodes
                for (const node of nodes) {
                    const color = getNodeColor(node);
                    const size = Math.max(30, Math.sqrt(node.conn || 1) * 15) / scale;
                    let finalColor = color;
                    if (node === selectedNode) finalColor = [1, 1, 1, 1];
                    else if (node === hoveredNode) finalColor = [1, 1, 1, 0.9];
                    else if (neighborNodes.has(node.id)) finalColor = [color[0]*1.2, color[1]*1.2, color[2]*1.2, 1];

                    nodeData.push(node.x, node.y, finalColor[0], finalColor[1], finalColor[2], finalColor[3], size);
                    labelsToRender.push({
                        type: 'node', x: node.x, y: node.y,
                        name: node.name, color: typeColors[node.type] || typeColors.concept
                    });
                }
            }

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

            // Render labels
            labelCtx.save();
            labelCtx.scale(dpr, dpr);
            labelCtx.textAlign = 'center';

            for (const label of labelsToRender) {
                const pos = worldToScreen(label.x, label.y);
                if (pos.x < -100 || pos.x > w + 100 || pos.y < -100 || pos.y > h + 100) continue;

                const color = label.color;
                const textColor = `rgb(${color[0]*255}, ${color[1]*255}, ${color[2]*255})`;

                if (label.type === 'cluster') {
                    const fontSize = Math.min(20, Math.max(12, 10 + Math.log(label.count || 1) * 2));
                    const countFontSize = Math.max(12, fontSize * 0.75);

                    // Measure both texts for background
                    labelCtx.font = `bold ${fontSize}px -apple-system, sans-serif`;
                    const nameMetrics = labelCtx.measureText(label.name);
                    labelCtx.font = `bold ${countFontSize}px -apple-system, sans-serif`;
                    const countMetrics = labelCtx.measureText(`${label.count} nodes`);
                    const maxWidth = Math.max(nameMetrics.width, countMetrics.width);

                    // Background - sized for both lines
                    labelCtx.fillStyle = 'rgba(10, 10, 18, 0.9)';
                    labelCtx.fillRect(pos.x - maxWidth/2 - 8, pos.y - fontSize - 2, maxWidth + 16, fontSize + countFontSize + 12);

                    // Cluster name
                    labelCtx.font = `bold ${fontSize}px -apple-system, sans-serif`;
                    labelCtx.fillStyle = textColor;
                    labelCtx.fillText(label.name, pos.x, pos.y);

                    // Node count - prominent
                    labelCtx.font = `bold ${countFontSize}px -apple-system, sans-serif`;
                    labelCtx.fillStyle = '#5eead4';
                    labelCtx.fillText(`${label.count} nodes`, pos.x, pos.y + countFontSize + 4);
                } else if (label.type === 'node') {
                    const fontSize = 11;
                    labelCtx.font = `${fontSize}px -apple-system, sans-serif`;
                    const metrics = labelCtx.measureText(label.name);
                    labelCtx.fillStyle = 'rgba(10, 10, 18, 0.8)';
                    labelCtx.fillRect(pos.x - metrics.width/2 - 3, pos.y + 8, metrics.width + 6, fontSize + 4);
                    labelCtx.fillStyle = textColor;
                    labelCtx.fillText(label.name, pos.x, pos.y + 18);
                }
            }
            labelCtx.restore();

            // Show level indicator
            const levelNames = ['L0 Clusters', 'L1 Clusters', 'L2 Clusters', 'Nodes'];
            const backPath = currentLevel > 0 ? ` (ESC to go back)` : '';
            const itemCount = currentLevel < 3 ? currentItems.length : nodes.length;
            document.getElementById('mode-indicator').textContent = `${levelNames[currentLevel]}${backPath} | ${itemCount} items`;
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
                            <span class="neighbor-name">${escapeHtml(n.name)}</span>
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
                <span class="type-badge" style="background:${color}20;color:${color}">${escapeHtml(node.type)}${node.subtype ? ' · ' + escapeHtml(node.subtype) : ''}</span>
                <h3>${escapeHtml(node.name)}</h3>
                ${hasContent ? `
                <div class="info-row">
                    <div class="info-label">Content</div>
                    <div class="info-value" style="max-height:150px;overflow-y:auto;white-space:pre-wrap;font-size:12px;line-height:1.5;background:rgba(0,0,0,0.2);padding:8px;border-radius:4px;margin-top:4px;">${escapeHtml(displayContent)}</div>
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
                const results = data.results || [];

                highlightedNodes.clear();
                results.forEach(r => highlightedNodes.add(r.id));

                searchResults.innerHTML = results.length === 0
                    ? '<div style="padding:12px;color:#6b6b8a;">No results</div>'
                    : results.slice(0, 20).map(r => `
                        <div class="search-result" data-x="${r.x}" data-y="${r.y}" data-id="${r.id}">
                            <span class="search-result-name">${escapeHtml(r.name)}</span>
                            <span class="search-result-type" style="background:${typeColorsHex[r.type]}20;color:${typeColorsHex[r.type]}">${escapeHtml(r.type)}</span>
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
            } else {
                const hit = findItemAtPosition(e.clientX, e.clientY);
                if (hit) {
                    canvas.style.cursor = 'pointer';
                    if (hit.type === 'cluster') {
                        showTooltip({ name: `Click to expand: ${escapeHtml(hit.item.name)} (${hit.item.count} nodes)` }, e.clientX, e.clientY);
                    } else {
                        hoveredNode = hit.item;
                        showTooltip(hit.item, e.clientX, e.clientY);
                    }
                } else {
                    canvas.style.cursor = 'default';
                    hoveredNode = null;
                    hideTooltip();
                }
                render();
            }
        });

        canvas.addEventListener('mouseup', async (e) => {
            if (Math.abs(e.clientX - dragStartX) < 5 && Math.abs(e.clientY - dragStartY) < 5) {
                const hit = findItemAtPosition(e.clientX, e.clientY);

                if (hit && hit.type === 'cluster') {
                    // Drill into cluster
                    await drillIntoCluster(hit.item.id);
                } else if (hit && hit.type === 'node') {
                    selectedNode = hit.item;
                    showNodeInfo(hit.item);
                } else {
                    closeNodeInfo();
                    highlightedNodes.clear();
                }
                render();
            }
            isDragging = false;
        });

        // Level navigation functions
        async function drillIntoCluster(itemId) {
            console.log(`Drilling into cluster ${itemId} at level ${currentLevel}`);

            try {
                if (currentLevel === 0) {
                    // L0 → L1: Load L1 clusters
                    parentPath.l0 = itemId;
                    const data = await (await fetch(`/admin/graph/clusters/l1/${itemId}`)).json();
                    currentItems = data.clusters || [];
                    currentEdges = data.edges || [];
                    currentLevel = 1;
                } else if (currentLevel === 1) {
                    // L1 → L2: Load L2 clusters
                    parentPath.l1 = itemId;
                    const data = await (await fetch(`/admin/graph/clusters/l2/${parentPath.l0}/${itemId}`)).json();
                    currentItems = data.clusters || [];
                    currentEdges = data.edges || [];
                    currentLevel = 2;
                } else if (currentLevel === 2) {
                    // L2 → Nodes: Load actual nodes
                    parentPath.l2 = itemId;
                    const data = await (await fetch(`/admin/graph/clusters/nodes/${parentPath.l0}/${parentPath.l1}/${itemId}`)).json();
                    nodes = data.nodes || [];
                    nodeMap = {};
                    for (const n of nodes) { nodeMap[n.id] = n; }
                    currentEdges = data.edges || [];
                    currentItems = [];  // Clear cluster items
                    currentLevel = 3;
                }

                // Fit view to new items
                fitViewToCurrentLevel();
                render();
            } catch (e) {
                console.error('Failed to load cluster data:', e);
            }
        }

        async function goBack() {
            if (currentLevel === 0) return;  // Already at top

            console.log(`Going back from level ${currentLevel}`);

            try {
                if (currentLevel === 1) {
                    // Back to L0
                    const data = await (await fetch('/admin/graph/clusters')).json();
                    currentItems = data.clusters || [];
                    currentEdges = data.edges || [];
                    parentPath = { l0: null, l1: null, l2: null };
                    currentLevel = 0;
                } else if (currentLevel === 2) {
                    // Back to L1
                    const data = await (await fetch(`/admin/graph/clusters/l1/${parentPath.l0}`)).json();
                    currentItems = data.clusters || [];
                    currentEdges = data.edges || [];
                    parentPath.l1 = null;
                    parentPath.l2 = null;
                    currentLevel = 1;
                } else if (currentLevel === 3) {
                    // Back to L2
                    const data = await (await fetch(`/admin/graph/clusters/l2/${parentPath.l0}/${parentPath.l1}`)).json();
                    currentItems = data.clusters || [];
                    currentEdges = data.edges || [];
                    nodes = [];
                    nodeMap = {};
                    parentPath.l2 = null;
                    currentLevel = 2;
                }

                fitViewToCurrentLevel();
                render();
            } catch (e) {
                console.error('Failed to go back:', e);
            }
        }

        function fitViewToCurrentLevel() {
            const items = currentLevel < 3 ? currentItems : nodes;
            if (items.length === 0) return;

            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (const item of items) {
                minX = Math.min(minX, item.x);
                maxX = Math.max(maxX, item.x);
                minY = Math.min(minY, item.y);
                maxY = Math.max(maxY, item.y);
            }

            const padding = 0.15;
            // Prevent divide by zero when all items at same position
            const gw = Math.max(100, (maxX - minX) * (1 + padding));
            const gh = Math.max(100, (maxY - minY) * (1 + padding));
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;

            const newScale = Math.min(
                window.innerWidth / (gw * 2) * 0.8,
                window.innerHeight / (gh * 2) * 0.8
            );

            animateTo(centerX, centerY, Math.max(0.001, Math.min(2, newScale)));
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
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (currentLevel > 0) {
                    goBack();
                } else {
                    closeNodeInfo();
                    highlightedNodes.clear();
                    searchInput.value = '';
                    searchResults.classList.remove('visible');
                }
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

                // Format answer (escape HTML first, then apply markdown-like formatting)
                let formattedAnswer = escapeHtml(answer)
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
                    ? `<button class="test-btn" onclick="testExcludeNode('${m.id}', event)" title="Test without this">−</button>`
                    : `<button class="test-btn" onclick="testIncludeNode('${m.id}', event)" title="Test with this">+</button>`;

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
                    ? `<button class="test-btn" onclick="testExcludeNode('${c.id}', event)" title="Test without this">−</button>`
                    : `<button class="test-btn" onclick="testIncludeNode('${c.id}', event)" title="Test with this">+</button>`;

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

        function testIncludeNode(nodeId, evt) {
            if (evt) evt.stopPropagation();
            if (!forceIncludeNodes.includes(nodeId)) {
                forceIncludeNodes.push(nodeId);
            }
            forceExcludeNodes = forceExcludeNodes.filter(id => id !== nodeId);
            sendChat(true);
        }

        function testExcludeNode(nodeId, evt) {
            if (evt) evt.stopPropagation();
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

        async function init() {
            resize();
            window.addEventListener('resize', resize);

            try {
                const boundsData = await (await fetch('/admin/graph/bounds')).json();
                if (!boundsData.has_layout) {
                    document.getElementById('loading').innerHTML = 'No layout. Run: uv run python scripts/compute_layout.py';
                    return;
                }

                bounds = boundsData;

                const stats = await (await fetch('/admin/graph/stats')).json();
                document.getElementById('c-count').textContent = stats.concepts || 0;
                document.getElementById('s-count').textContent = stats.semantic || 0;
                document.getElementById('e-count').textContent = stats.episodic || 0;
                document.getElementById('stats').innerHTML = `<div>Total: <strong style="color:#5eead4">${stats.total || 0}</strong></div><div>Clusters: <strong style="color:#a78bfa">${stats.clusters || 0}</strong></div>`;

                // Load L0 clusters (initial view)
                document.getElementById('loading').innerHTML = 'Loading clusters...';
                const data = await (await fetch('/admin/graph/clusters')).json();
                currentItems = data.clusters || [];
                currentEdges = data.edges || [];
                currentLevel = 0;
                parentPath = { l0: null, l1: null, l2: null };

                console.log(`Loaded ${currentItems.length} L0 clusters, ${data.total_nodes || 0} total nodes`);

                document.getElementById('loading').style.display = 'none';

                // Fit view to initial clusters
                fitViewToCurrentLevel();
                initialView = { x: viewX, y: viewY, scale: scale };

                render();
            } catch (e) {
                console.error('Failed to initialize graph:', e);
                document.getElementById('loading').textContent = 'Error loading graph: ' + e.message;
            }
        }

        init();
    </script>
</body>
</html>
"""
