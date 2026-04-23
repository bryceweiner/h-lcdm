"""
Void 3D Map Generator
====================

Generate interactive HTML/Three.js visualization of cosmic void networks.

Creates an interactive 3D map showing:
- Voids as ellipsoids with proper orientation and size
- Colors mapped from redshift (blue = nearby, red = distant)
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
    Generate complete HTML content with embedded visualization data.

    Parameters:
        viz_data: Visualization data dictionary

    Returns:
        Complete HTML string
    """
    # Extract and embed all data
    voids = viz_data.get("voids", [])
    edges = viz_data.get("edges", [])
    metadata = viz_data.get("metadata", {})
    
    # Sample 1 in 10 edges to avoid stack overflow
    sampled_edges = edges[::10]  # Every 10th edge
    
    # Generate JavaScript data (embedded)
    voids_js = json.dumps(voids, separators=(',', ':'))
    edges_js = json.dumps(sampled_edges, separators=(',', ':'))
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
            background: linear-gradient(to right, #2E86AB, #2AA876, #F18F01, #F24236);
            border-radius: 3px;
            margin: 5px 0;
        }}
        
        .boundary-legend {{
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .boundary-item {{
            display: flex;
            align-items: center;
            margin: 3px 0;
            font-size: 10px;
        }}
        
        .boundary-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 5px;
            border: 1px solid rgba(255, 255, 255, 0.3);
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
        <h4>Redshift (z)</h4>
        <div class="color-bar"></div>
        <div style="display: flex; justify-content: space-between; font-size: 10px;">
            <span>Nearby ({metadata.get('redshift_range', {}).get('min', 0):.3f})</span>
            <span>Mid ({(metadata.get('redshift_range', {}).get('min', 0) + metadata.get('redshift_range', {}).get('max', 0.4)) / 2:.3f})</span>
            <span>Distant ({metadata.get('redshift_range', {}).get('max', 0.4):.3f})</span>
        </div>
        
        <div class="boundary-legend">
            <h4 style="margin: 5px 0;">Survey Boundaries</h4>
            <div class="boundary-item">
                <div class="boundary-color" style="background: #00AAFF;"></div>
                <span>DESI (44-676 Mpc)</span>
            </div>
            <div class="boundary-item">
                <div class="boundary-color" style="background: #FF6600;"></div>
                <span>SDSS Clampitt (712-1477 Mpc)</span>
            </div>
            <div class="boundary-item">
                <div class="boundary-color" style="background: #88FF00;"></div>
                <span>SDSS Other (31-327 Mpc)</span>
            </div>
        </div>
    </div>

    <div id="controls">
        <h4>Controls</h4>
        <p><strong>Mouse:</strong> Orbit camera</p>
        <p><strong>Scroll:</strong> Zoom in/out</p>
        <p><strong>Arrow Keys:</strong> Move through space</p>
        <p><strong>WASD:</strong> Alternative movement</p>
        <p style="margin-top: 10px;">
            <label>
                <input type="checkbox" id="toggleVoids" checked> 
                Show All Voids
            </label>
        </p>
        <div id="surveyToggles" style="margin-left: 20px; font-size: 12px;">
            <!-- Survey-specific toggles will be added dynamically -->
        </div>
        <p>
            <label>
                <input type="checkbox" id="toggleBoundaries" checked> 
                Show Survey Boundaries
            </label>
        </p>
        <p>
            <label>
                <input type="checkbox" id="toggleEdges" checked> 
                Show Network Edges
            </label>
        </p>
    </div>

    <div class="stats" id="stats">
        <div id="stats-content">Initializing 3D visualization...</div>
    </div>

    <script>
        // Visualization data (embedded)
        const voidsData = {voids_js};
        const edgesData = {edges_js};
        const metadata = {metadata_js};

        // Scene setup
        let scene, camera, renderer, controls;
        let voidsGroup, edgesGroup, boundariesGroup;
        let voidsBySurvey = {{}}; // Store void meshes grouped by survey
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
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 3000);
            camera.position.set(-1800, 0, 1000);
            camera.lookAt(0, 0, 0);
            console.log('Initial camera position:', camera.position);

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
            controls.maxDistance = 2500;  // Increased to allow viewing all voids
            controls.target.set(0, 0, 0);  // Ensure looking at origin
            controls.update();  // Update controls after setting camera position

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

            boundariesGroup = new THREE.Group();
            scene.add(boundariesGroup);

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

            // Create survey boundaries
            createSurveyBoundaries();

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
            
            // Toggle all voids checkbox
            const toggleVoids = document.getElementById('toggleVoids');
            if (toggleVoids) {{
                toggleVoids.addEventListener('change', function() {{
                    // Toggle all survey groups
                    Object.keys(voidsBySurvey).forEach(survey => {{
                        voidsBySurvey[survey].visible = this.checked;
                        const checkbox = document.getElementById(`toggle-${{survey}}`);
                        if (checkbox) checkbox.checked = this.checked;
                    }});
                }});
            }}
            
            // Toggle boundaries checkbox
            const toggleBoundaries = document.getElementById('toggleBoundaries');
            if (toggleBoundaries) {{
                toggleBoundaries.addEventListener('change', function() {{
                    boundariesGroup.visible = this.checked;
                }});
            }}
            
            // Toggle edges checkbox
            const toggleEdges = document.getElementById('toggleEdges');
            if (toggleEdges) {{
                toggleEdges.addEventListener('change', function() {{
                    edgesGroup.visible = this.checked;
                }});
            }}

            updateStats();
        }}

        function createSurveyToggles(surveys) {{
            const container = document.getElementById('surveyToggles');
            if (!container) return;

            // Sort surveys alphabetically
            const sortedSurveys = surveys.sort();

            sortedSurveys.forEach(survey => {{
                const label = document.createElement('label');
                label.style.display = 'block';
                label.style.marginBottom = '5px';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `toggle-${{survey}}`;
                checkbox.checked = true;
                checkbox.addEventListener('change', function() {{
                    if (voidsBySurvey[survey]) {{
                        voidsBySurvey[survey].visible = this.checked;
                    }}
                }});

                // Clean up survey name for display
                const displayName = survey
                    .replace(/_/g, ' ')
                    .replace(/DR(\\d+)/g, 'DR$1')
                    .replace(/NGC/g, '(NGC)')
                    .replace(/SGC/g, '(SGC)');

                label.appendChild(checkbox);
                label.appendChild(document.createTextNode(` ${{displayName}}`));
                container.appendChild(label);
            }});
        }}

        function createSurveyBoundaries() {{
            // Calculate survey boundaries from actual void data
            const surveyBoundaries = {{}};
            
            voidsData.forEach(voidData => {{
                const r = Math.sqrt(voidData.x**2 + voidData.y**2 + voidData.z**2);
                
                if (!surveyBoundaries[voidData.survey]) {{
                    surveyBoundaries[voidData.survey] = {{
                        min: r,
                        max: r,
                        count: 0
                    }};
                }}
                
                surveyBoundaries[voidData.survey].min = Math.min(surveyBoundaries[voidData.survey].min, r);
                surveyBoundaries[voidData.survey].max = Math.max(surveyBoundaries[voidData.survey].max, r);
                surveyBoundaries[voidData.survey].count++;
            }});
            
            // Define colors for major survey groups
            const surveyColors = {{
                'DESI': 0x00AAFF,     // Cyan for DESI
                'SDSS_DR7_CLAMPITT': 0xFF6600,  // Orange for CLAMPITT
                'SDSS': 0x88FF00      // Green for other SDSS
            }};
            
            // Group surveys by type
            const majorSurveys = [];
            Object.keys(surveyBoundaries).forEach(survey => {{
                let color, label;
                
                if (survey.includes('DESI')) {{
                    // Aggregate all DESI surveys
                    let existing = majorSurveys.find(s => s.label === 'DESI');
                    if (existing) {{
                        existing.min = Math.min(existing.min, surveyBoundaries[survey].min);
                        existing.max = Math.max(existing.max, surveyBoundaries[survey].max);
                        existing.count += surveyBoundaries[survey].count;
                    }} else {{
                        majorSurveys.push({{
                            label: 'DESI',
                            min: surveyBoundaries[survey].min,
                            max: surveyBoundaries[survey].max,
                            color: surveyColors['DESI'],
                            count: surveyBoundaries[survey].count
                        }});
                    }}
                }} else if (survey === 'SDSS_DR7_CLAMPITT') {{
                    majorSurveys.push({{
                        label: 'SDSS_DR7_CLAMPITT',
                        min: surveyBoundaries[survey].min,
                        max: surveyBoundaries[survey].max,
                        color: surveyColors['SDSS_DR7_CLAMPITT'],
                        count: surveyBoundaries[survey].count
                    }});
                }} else if (survey.includes('SDSS')) {{
                    // Aggregate other SDSS surveys
                    let existing = majorSurveys.find(s => s.label === 'SDSS (Other)');
                    if (existing) {{
                        existing.min = Math.min(existing.min, surveyBoundaries[survey].min);
                        existing.max = Math.max(existing.max, surveyBoundaries[survey].max);
                        existing.count += surveyBoundaries[survey].count;
                    }} else {{
                        majorSurveys.push({{
                            label: 'SDSS (Other)',
                            min: surveyBoundaries[survey].min,
                            max: surveyBoundaries[survey].max,
                            color: surveyColors['SDSS'],
                            count: surveyBoundaries[survey].count
                        }});
                    }}
                }}
            }});
            
            // Create wireframe spheres for survey boundaries
            majorSurveys.forEach(survey => {{
                // Inner boundary (min distance)
                const innerGeometry = new THREE.SphereGeometry(survey.min, 32, 32);
                const innerMaterial = new THREE.MeshBasicMaterial({{
                    color: survey.color,
                    wireframe: true,
                    transparent: true,
                    opacity: 0.08
                }});
                const innerSphere = new THREE.Mesh(innerGeometry, innerMaterial);
                boundariesGroup.add(innerSphere);
                
                // Outer boundary (max distance)
                const outerGeometry = new THREE.SphereGeometry(survey.max, 32, 32);
                const outerMaterial = new THREE.MeshBasicMaterial({{
                    color: survey.color,
                    wireframe: true,
                    transparent: true,
                    opacity: 0.12
                }});
                const outerSphere = new THREE.Mesh(outerGeometry, outerMaterial);
                boundariesGroup.add(outerSphere);
            }});
            
            console.log('Survey boundaries:');
            majorSurveys.forEach(s => {{
                console.log(`  ${{s.label}}: ${{s.min.toFixed(1)}}-${{s.max.toFixed(1)}} Mpc (${{s.count}} voids)`);
            }});
        }}

        function createVoids() {{
            const geometry = new THREE.SphereGeometry(1, 16, 12);

            // Get redshift range from metadata
            const minZ = {metadata.get('redshift_range', {}).get('min', 0)};
            const maxZ = {metadata.get('redshift_range', {}).get('max', 0.4)};

            // Color mapping function: redshift to color
            function getColor(redshift) {{
                // Normalize redshift to 0-1 range
                const t = (redshift - minZ) / (maxZ - minZ);
                // Blue (nearby, low z) to red (distant, high z)
                return new THREE.Color().setHSL(0.6 - t * 0.6, 1, 0.5);
            }}

            // Group voids by survey
            const voidsBySurveyTemp = {{}};
            voidsData.forEach((voidData, index) => {{
                const survey = voidData.survey || 'UNKNOWN';
                if (!voidsBySurveyTemp[survey]) {{
                    voidsBySurveyTemp[survey] = [];
                }}
                voidsBySurveyTemp[survey].push({{ voidData, index }});
            }});

            // Create groups for each survey
            Object.keys(voidsBySurveyTemp).forEach(survey => {{
                const surveyGroup = new THREE.Group();
                surveyGroup.name = survey;
                
                voidsBySurveyTemp[survey].forEach(item => {{
                    const voidData = item.voidData;
                    const index = item.index;
                    
                    const material = new THREE.MeshLambertMaterial({{
                        color: getColor(voidData.redshift),
                        transparent: true,
                        opacity: 0.9,
                        emissive: getColor(voidData.redshift),
                        emissiveIntensity: 0.3
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

                    surveyGroup.add(mesh);
                }});

                voidsGroup.add(surveyGroup);
                voidsBySurvey[survey] = surveyGroup;
            }});

            console.log(`Created ${{voidsData.length}} void ellipsoids from ${{Object.keys(voidsBySurveyTemp).length}} surveys`);
            
            // Create survey toggles
            createSurveyToggles(Object.keys(voidsBySurveyTemp));
        }}

        function createEdges() {{
            if (edgesData.length === 0) {{
                console.log('No network edges to display');
                return;
            }}
            
            // Use a single geometry with all edge lines for better performance
            const positions = [];
            
            edgesData.forEach(edge => {{
                const [i, j] = edge;
                if (i < voidsData.length && j < voidsData.length) {{
                    const void1 = voidsData[i];
                    const void2 = voidsData[j];
                    
                    positions.push(void1.x, void1.y, void1.z);
                    positions.push(void2.x, void2.y, void2.z);
                }}
            }});
            
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            
            const material = new THREE.LineBasicMaterial({{
                color: 0xFFFFFF,  // White edges
                transparent: true,
                opacity: 0.15,
                linewidth: 1
            }});
            
            const lineSegments = new THREE.LineSegments(geometry, material);
            edgesGroup.add(lineSegments);

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
                    Camera: (${{pos.x.toFixed(0)}}, ${{pos.y.toFixed(0)}}, ${{pos.z.toFixed(0)}})<br>
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

