"""
Void 3D Map Generator
====================

Generate interactive HTML/Three.js visualization of cosmic void networks.

Creates an interactive 3D map showing:
- Voids as ellipsoids with proper orientation and size
- Colors mapped from local clustering coefficients
- Network edges (when available)
- Camera controls for exploration
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def generate_void_3d_map_html(
    data_path: str = "results/figures/void/void_map_data.json",
    output_path: str = "results/figures/void/void_3d_map.html",
    template_path: Optional[str] = None
) -> str:
    """
    Generate interactive 3D void map HTML file.

    Parameters:
        data_path: Path to JSON data file
        output_path: Path to output HTML file
        template_path: Optional custom HTML template

    Returns:
        str: Path to generated HTML file
    """
    logger.info("Generating 3D void map HTML...")

    # Load visualization data
    try:
        with open(data_path, 'r') as f:
            viz_data = json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Could not load visualization data: {e}")

    # Generate HTML content
    html_content = _generate_html_content(viz_data)

    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

    logger.info(f"Generated 3D void map at {output_path}")
    return str(output_path)


def _generate_html_content(viz_data: Dict[str, Any]) -> str:
    """
    Generate complete HTML content with embedded Three.js visualization.

    Parameters:
        viz_data: Visualization data dictionary

    Returns:
        Complete HTML string
    """
    # Extract data
    voids = viz_data.get("voids", [])
    edges = viz_data.get("edges", [])
    metadata = viz_data.get("metadata", {})

    # Generate JavaScript data
    voids_js = json.dumps(voids, separators=(',', ':'))
    edges_js = json.dumps(edges, separators=(',', ':'))
    metadata_js = json.dumps(metadata, separators=(',', ':'))

    # HTML template with Three.js
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cosmic Void Network - 3D Interactive Map</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #000;
            color: #fff;
            overflow: hidden;
        }}

        #container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}

        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            max-width: 300px;
            z-index: 100;
        }}

        #legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
            font-size: 11px;
            z-index: 100;
        }}

        #controls {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            z-index: 100;
        }}

        .color-bar {{
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, #2E86AB, #F18F01, #F24236);
            border-radius: 3px;
            margin: 5px 0;
        }}

        .stats {{
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
            font-size: 11px;
            position: absolute;
            bottom: 10px;
            right: 10px;
            max-width: 300px;
            z-index: 100;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="container"></div>

    <div id="info">
        <h3>Cosmic Void Network</h3>
        <p><strong>Voids:</strong> {metadata.get('n_voids', 0):,}</p>
        <p><strong>Global Clustering:</strong> {metadata.get('global_clustering_coefficient', 0):.3f}</p>
        <p><strong>Linking Length:</strong> {metadata.get('linking_length', 0):.1f} Mpc</p>
    </div>

    <div id="legend">
        <h4>Clustering Coefficient</h4>
        <div class="color-bar"></div>
        <div style="display: flex; justify-content: space-between; font-size: 10px;">
            <span>Low (0.0)</span>
            <span>η_natural (0.443)</span>
            <span>High (1.0)</span>
        </div>
    </div>

    <div id="controls">
        <h4>Controls</h4>
        <p><strong>Mouse:</strong> Orbit camera</p>
        <p><strong>Scroll:</strong> Zoom in/out</p>
        <p><strong>Arrow Keys:</strong> Move through space</p>
        <p><strong>WASD:</strong> Alternative movement</p>
    </div>

    <div class="stats" id="stats">
        <div id="stats-content">Initializing 3D visualization...</div>
    </div>

    <script>
        // Visualization data
        const voidsData = {voids_js};
        const edgesData = {edges_js};
        const metadata = {metadata_js};

        // Scene setup
        let scene, camera, renderer, controls;
        let voidsGroup, edgesGroup;
        let raycaster, mouse, tooltip;
        let statsElement;

        // Movement controls
        let keys = {{}};
        const moveSpeed = 5;

        init();
        animate();

        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000011);

            // Camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
            camera.position.set(200, 200, 200);

            // Renderer
            renderer = new THREE.WebGLRenderer({{antialias: true}});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.getElementById('container').appendChild(renderer.domElement);

            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = false;
            controls.minDistance = 10;
            controls.maxDistance = 1000;

            // Lighting
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);

            // Groups for organization
            voidsGroup = new THREE.Group();
            scene.add(voidsGroup);

            edgesGroup = new THREE.Group();
            scene.add(edgesGroup);

            // Raycaster for mouse interaction
            raycaster = new THREE.Raycaster();
            mouse = new THREE.Vector2();

            // Tooltip
            tooltip = document.createElement('div');
            tooltip.style.position = 'absolute';
            tooltip.style.background = 'rgba(0, 0, 0, 0.8)';
            tooltip.style.color = 'white';
            tooltip.style.padding = '8px';
            tooltip.style.borderRadius = '4px';
            tooltip.style.fontSize = '12px';
            tooltip.style.pointerEvents = 'none';
            tooltip.style.zIndex = '1000';
            tooltip.style.display = 'none';
            document.body.appendChild(tooltip);

            // Stats element
            statsElement = document.getElementById('stats-content');

            // Create voids
            createVoids();

            // Create edges (if any)
            if (edgesData.length > 0) {{
                createEdges();
            }}

            // Event listeners
            window.addEventListener('resize', onWindowResize, false);
            window.addEventListener('mousemove', onMouseMove, false);
            window.addEventListener('keydown', onKeyDown, false);
            window.addEventListener('keyup', onKeyUp, false);

            updateStats();
        }}

        function createVoids() {{
            const geometry = new THREE.SphereGeometry(1, 16, 12);

            // Color mapping function
            function getColor(clustering) {{
                if (clustering < 0.443) {{
                    // Blue to yellow (low to η_natural)
                    const t = clustering / 0.443;
                    return new THREE.Color().setHSL(0.6 - t * 0.6, 1, 0.5);
                }} else {{
                    // Yellow to red (η_natural to high)
                    const t = (clustering - 0.443) / (1.0 - 0.443);
                    return new THREE.Color().setHSL(0.1 - t * 0.1, 1, 0.5);
                }}
            }}

            voidsData.forEach((voidData, index) => {{
                const material = new THREE.MeshLambertMaterial({{
                    color: getColor(voidData.clustering),
                    transparent: true,
                    opacity: 0.7
                }});

                const mesh = new THREE.Mesh(geometry, material);

                // Position
                mesh.position.set(voidData.x, voidData.y, voidData.z);

                // Scale by radius
                const scale = voidData.radius / 10; // Normalize for visibility
                mesh.scale.setScalar(Math.max(scale, 0.1));

                // Rotate by orientation (around local z-axis)
                mesh.rotation.z = (voidData.orientation * Math.PI) / 180;

                // Store void data for tooltip
                mesh.userData = voidData;
                mesh.userData.index = index;

                voidsGroup.add(mesh);
            }});

            console.log(`Created ${{voidsData.length}} void ellipsoids`);
        }}

        function createEdges() {{
            const material = new THREE.LineBasicMaterial({{
                color: 0x444444,
                transparent: true,
                opacity: 0.3
            }});

            edgesData.forEach(edge => {{
                const [i, j] = edge;
                if (i < voidsData.length && j < voidsData.length) {{
                    const void1 = voidsData[i];
                    const void2 = voidsData[j];

                    const geometry = new THREE.BufferGeometry().setFromPoints([
                        new THREE.Vector3(void1.x, void1.y, void1.z),
                        new THREE.Vector3(void2.x, void2.y, void2.z)
                    ]);

                    const line = new THREE.Line(geometry, material);
                    edgesGroup.add(line);
                }}
            }});

            console.log(`Created ${{edgesData.length}} network edges`);
        }}

        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);

            const intersects = raycaster.intersectObjects(voidsGroup.children);

            if (intersects.length > 0) {{
                const intersected = intersects[0].object;
                const voidData = intersected.userData;

                tooltip.style.left = event.clientX + 10 + 'px';
                tooltip.style.top = event.clientY - 10 + 'px';
                tooltip.innerHTML = `
                    <strong>Void ${{voidData.index}}</strong><br>
                    Survey: ${{voidData.survey}}<br>
                    Position: (${{voidData.x.toFixed(1)}}, ${{voidData.y.toFixed(1)}}, ${{voidData.z.toFixed(1)}}) Mpc<br>
                    Radius: ${{voidData.radius.toFixed(1)}} Mpc<br>
                    Redshift: ${{voidData.redshift.toFixed(3)}}<br>
                    Clustering: ${{voidData.clustering.toFixed(3)}}
                `;
                tooltip.style.display = 'block';
            }} else {{
                tooltip.style.display = 'none';
            }}
        }}

        function onKeyDown(event) {{
            keys[event.code] = true;
        }}

        function onKeyUp(event) {{
            keys[event.code] = false;
        }}

        function updateMovement() {{
            const direction = new THREE.Vector3();
            camera.getWorldDirection(direction);
            direction.normalize();

            const right = new THREE.Vector3();
            right.crossVectors(camera.up, direction);
            right.normalize();

            if (keys['ArrowUp'] || keys['KeyW']) {{
                camera.position.add(direction.clone().multiplyScalar(moveSpeed));
            }}
            if (keys['ArrowDown'] || keys['KeyS']) {{
                camera.position.add(direction.clone().multiplyScalar(-moveSpeed));
            }}
            if (keys['ArrowLeft'] || keys['KeyA']) {{
                camera.position.add(right.clone().multiplyScalar(moveSpeed));
            }}
            if (keys['ArrowRight'] || keys['KeyD']) {{
                camera.position.add(right.clone().multiplyScalar(-moveSpeed));
            }}
            if (keys['Space']) {{
                camera.position.y += moveSpeed;
            }}
            if (keys['ShiftLeft'] || keys['ShiftRight']) {{
                camera.position.y -= moveSpeed;
            }}
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function updateStats() {{
            if (statsElement) {{
                const pos = camera.position;
                statsElement.innerHTML = `
                    Camera: (${{pos.x.toFixed(0)}}, ${{pos.y.toFixed(0)}}, ${{pos.z.toFixed(0))}}<br>
                    Voids: ${{voidsData.length.toLocaleString()}}<br>
                    Edges: ${{edgesData.length.toLocaleString()}}<br>
                    FPS: TBD
                `;
            }}
        }}

        function animate() {{
            requestAnimationFrame(animate);

            controls.update();
            updateMovement();
            updateStats();

            renderer.render(scene, camera);
        }}
    </script>
</body>
</html>"""

    return html_template
