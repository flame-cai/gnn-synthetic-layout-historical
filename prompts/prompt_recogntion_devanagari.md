As an expert software development, specializing front-end sofware development using .vue, and backend developend using python. Please assist me in making precise improvements to data annotation tool below, which helps historians digitize historical manuscripts.

This code is part of a larger system which performs layout analysis on manuscript images using a Graph Neural Network (GNN). The layout analysis problem is formulated in a graph based manner, where characters are treated as nodes and characters of the same text lines are connected with edges. Thus nodes containing the same textline have the same text line label. The user can also label nodes with textbox labels, marking nodes of each text box with the same integer label. Once labelled (using gnn layout inference + manual corrections), the system generates PAGE XML files containing textbox and text line bounding polygons, along with visualizations. The system also saves textline images, for each textbox.
After performing the layout analysis semi-automatically using the graph neural network, we use Gemini API to recognize text context from each textline in the "Recognition Mode".

I want your help in updating how keystrokes are handled in the text input fields in the recognition mode, where the user will make manual corrections to Gemini output. Right now the key board only records latin characters, but we want to support devanagari script now. Note that this feature is optional, and user should be able to toggle default behaviour, with the new "devanagari input keyboard"

Please make precise changes, without any unnecessary changes. Write robust code, with good debugging. Reuse the files devanagariInputUtils.js, inputClusterCode.js and CharacterPalette.vue shown below, and only make changes precise changes to the ManuscriptViewer.vue file by importing the handleInput(event, devanagariRef).

Please study the below code, and integrate the devanagari keyboard function export function handleInput(event, devanagariRef) in the Recognition Mode front end. Please find the relevant files below:



<template>
  <div class="manuscript-viewer">
    
    <!-- TOP RAIL: Navigation & Global Actions -->
    <div class="top-bar">
      <div class="top-bar-left">
        <button class="nav-btn secondary" @click="$emit('back')">Back</button>
        <span class="page-title">{{ manuscriptNameForDisplay }} <span class="divider">/</span> Page {{ currentPageForDisplay }}</span>
      </div>

      <!-- NEW: Auto-Recognition Controls in Center/Right -->
      <div class="top-bar-center" style="display:flex; align-items:center; gap: 10px; margin-left: 20px;">
          <input 
            v-model="geminiKey" 
            type="password" 
            placeholder="Gemini API Key" 
            class="api-input-small"
            @change="saveKeyToStorage"
          />
          <label class="toggle-switch">
             <input type="checkbox" v-model="autoRecogEnabled">
             <span class="slider"></span>
          </label>
          <span style="font-size: 0.8rem; color: #ccc;">Auto-Recognize on Save</span>
      </div>

      <div class="top-bar-right">
        <div class="action-group">
           <button class="nav-btn" @click="previousPage" :disabled="loading || isProcessingSave || isFirstPage">
            Previous
          </button>
          <button class="nav-btn" @click="nextPage" :disabled="loading || isProcessingSave || isLastPage">
            Next
          </button>
        </div>

        <div class="separator"></div>

        <div class="action-group">
           <button class="action-btn primary" @click="saveAndGoNext" :disabled="loading || isProcessingSave">
            {{ autoRecogEnabled ? 'Save, Recog & Next (S)' : 'Save & Next (S)' }}
          </button>
          <button class="action-btn" @click="downloadResults" :disabled="loading || isProcessingSave">
            Download PAGE-XMLs
          </button>
          <button class="action-btn" @click="runHeuristic" :disabled="loading || recognitionModeActive">
            Auto-Link
          </button>
        </div>
      </div>
    </div>

    <!-- MAIN CONTENT: Visualization Area -->
    <div class="visualization-container" ref="container">
      
      <!-- 1. Unified Overlay for Saving OR Mode Switching -->
      <div v-if="isProcessingSave" class="processing-save-notice">
        Processing... Please wait.
      </div>

      <div v-if="error" class="error-message">
        {{ error }}
      </div>

      <!-- 2. Loading Indicator (Only for initial page load) -->
      <div v-if="loading" class="loading">Loading Page Data...</div>

      <!-- 3. Image Container: Changed v-else to v-show to prevent DOM destruction -->
      <div
        v-show="!loading && imageData" 
        class="image-container"
        :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }"
      >
        <img
          v-if="imageData"
          :src="`data:image/jpeg;base64,${imageData}`"
          :width="scaledWidth"
          :height="scaledHeight"
          class="manuscript-image"
          @load="imageLoaded = true"
        />
        <div
          v-else
          class="placeholder-image"
          :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }"
        >
          No image available
        </div>

        <!-- SVG Graph Layer (Visible in Graph Modes) -->
        <svg
          v-if="graphIsLoaded && !recognitionModeActive"
          class="graph-overlay"
          :class="{ 'is-visible': textlineModeActive || textboxModeActive || nodeModeActive }"
          :width="scaledWidth"
          :height="scaledHeight"
          :viewBox="`0 0 ${scaledWidth} ${scaledHeight}`"
          :style="{ cursor: svgCursor }"
          @click="onBackgroundClick($event)"
          @contextmenu.prevent 
          @mousemove="handleSvgMouseMove"
          @mouseleave="handleSvgMouseLeave"
          ref="svgOverlayRef"
        >
          <line
            v-for="(edge, index) in workingGraph.edges"
            :key="`edge-${index}`"
            :x1="scaleX(workingGraph.nodes[edge.source].x)"
            :y1="scaleY(workingGraph.nodes[edge.source].y)"
            :x2="scaleX(workingGraph.nodes[edge.target].x)"
            :y2="scaleY(workingGraph.nodes[edge.target].y)"
            :stroke="getEdgeColor(edge)"
            :stroke-width="isEdgeSelected(edge) ? 3 : 2.5"
            @click.stop="textlineModeActive && onEdgeClick(edge, $event)"
          />

          <circle
            v-for="(node, nodeIndex) in workingGraph.nodes"
            :key="`node-${nodeIndex}`"
            :cx="scaleX(node.x)"
            :cy="scaleY(node.y)"
            :r="getNodeRadius(nodeIndex)"
            :fill="getNodeColor(nodeIndex)"
            @click.stop="onNodeClick(nodeIndex, $event)"
            @contextmenu.stop.prevent="onNodeRightClick(nodeIndex, $event)"
          />

          <line
            v-if="
              textlineModeActive &&
              selectedNodes.length === 1 &&
              tempEndPoint &&
              !isAKeyPressed &&
              !isDKeyPressed
            "
            :x1="scaleX(workingGraph.nodes[selectedNodes[0]].x)"
            :y1="scaleY(workingGraph.nodes[selectedNodes[0]].y)"
            :x2="tempEndPoint.x"
            :y2="tempEndPoint.y"
            stroke="#ff9500"
            stroke-width="2.5"
            stroke-dasharray="5,5"
          />
        </svg>

        <!-- SVG Polygon Layer (Visible in Recognition Mode) -->
        <svg
          v-if="recognitionModeActive"
          class="graph-overlay is-visible"
          :width="scaledWidth"
          :height="scaledHeight"
          :viewBox="`0 0 ${scaledWidth} ${scaledHeight}`"
          @click.stop
        >
          <!-- Draw inactive polygons faintly so user knows where lines are -->
          <polygon
            v-for="(points, lineId) in pagePolygons"
            :key="`poly-bg-${lineId}`"
            :points="pointsToSvgString(points)"
            fill="transparent"
            stroke="rgba(255, 255, 255, 0.2)"
            stroke-width="1"
            class="polygon-inactive"
            @click="activateInput(lineId)"
          />

          <!-- Draw Active Polygon Highlighted -->
          <polygon
            v-if="focusedLineId && pagePolygons[focusedLineId]"
            :points="pointsToSvgString(pagePolygons[focusedLineId])"
            fill="rgba(0, 255, 255, 0.1)"
            stroke="#00e5ff"
            stroke-width="2"
            class="polygon-active"
          />
        </svg>

        <!-- Recognition Input Overlay Layer (Single Floating Input) -->
        <div
            v-if="recognitionModeActive && focusedLineId && pagePolygons[focusedLineId]"
            class="input-floater"
            :style="getActiveInputStyle()"
        >
            <input 
                ref="activeInput"
                v-model="localTextContent[focusedLineId]" 
                class="line-input active"
                @blur="handleInputBlur"
                @keydown.tab.prevent="focusNextLine(false)"
                @keydown.shift.tab.prevent="focusNextLine(true)"
                placeholder="Type text here..."
                :style="{ fontSize: getDynamicFontSize() }"
            />
        </div>

      </div>
    </div>

    <!-- BOTTOM RAIL: Controls & Help Center -->
    <div class="bottom-panel" :class="{ 'is-collapsed': isPanelCollapsed }">
      
      <!-- Mode Tabs (Always Visible) -->
      <div class="mode-tabs">
         <button 
           class="mode-tab" 
           :class="{ active: !textlineModeActive && !textboxModeActive && !nodeModeActive && !recognitionModeActive }"
           @click="setMode('view')"
           :disabled="isProcessingSave">
           View Mode
         </button>
          <button 
           class="mode-tab" 
           :class="{ active: nodeModeActive }"
           @click="setMode('node')"
           :disabled="isProcessingSave || !graphIsLoaded">
           Node Mode (N)
         </button>
         <button 
           class="mode-tab" 
           :class="{ active: textlineModeActive }"
           @click="setMode('edge')"
           :disabled="isProcessingSave">
           Edge Edit (W)
         </button>
         <button 
           class="mode-tab" 
           :class="{ active: textboxModeActive }"
           @click="setMode('region')"
           :disabled="isProcessingSave || !graphIsLoaded">
           Region Labeling (R)
         </button>
         <!-- NEW RECOGNITION TAB -->

         <button 
           class="mode-tab" 
           :class="{ active: recognitionModeActive }"
           @click="requestSwitchToRecognition" 
           :disabled="isProcessingSave">
           Recognize (T)
         </button>

         <div class="tab-spacer"></div>

         <button class="panel-toggle-btn" @click="isPanelCollapsed = !isPanelCollapsed" title="Toggle Help Panel">
            <span v-if="isPanelCollapsed">▲ Show Help</span>
            <span v-else>▼ Hide</span>
         </button>
      </div>

      <!-- Help & Actions Content Area -->
      <div class="help-content-area" v-show="!isPanelCollapsed">
        
        <div v-if="!textlineModeActive && !textboxModeActive && !nodeModeActive && !recognitionModeActive" class="help-section">
          <div class="instructions-container">
            <h3>View Mode</h3>
            <p>Pan and zoom to inspect the manuscript. Select a mode above or use hotkeys to start annotating.</p>
          </div>
        </div>

        <div v-if="textlineModeActive" class="help-section">
          <div class="media-container">
            <video :src="edgeWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
          </div>
          <div class="instructions-container">
            <h3>Text-Line Mode</h3>
            <ul>
              <li><strong>Connect:</strong> Hold <code>'a'</code> and hover over nodes to connect.</li>
              <li><strong>Delete:</strong> Hold <code>'d'</code> and hover over edges to delete.</li>
            </ul>
          </div>
        </div>

        <div v-if="textboxModeActive" class="help-section">
           <div class="media-container">
            <video :src="regionWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
          </div>
          <div class="instructions-container">
            <h3>Text-Box Mode</h3>
            <p>
              Hold <code>'e'</code> and hover over lines to label them. Release and press again for new box.
            </p>
          </div>
        </div>

        <div v-if="nodeModeActive" class="help-section">
           <div class="media-container">
            <video :src="nodeWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
          </div>
          <div class="instructions-container">
            <h3>Node Mode</h3>
            <ul>
              <li><strong>Add/Delete:</strong> Left-click to add, Right-click to remove nodes.</li>
            </ul>
          </div>
        </div>

        <!-- RECOGNITION MODE HELP -->
        <div v-if="recognitionModeActive" class="help-section">
           <div class="media-container">
             <div class="webm-placeholder">
              <span>Recognition Mode</span>
            </div>
           </div>
           <div class="instructions-container">
             <h3>Recognition Mode</h3>
             <p>Transcribe line-by-line using Gemini AI.</p>
             <div class="form-group-inline">
                <input v-model="geminiKey" type="password" placeholder="Enter Gemini API Key" class="api-input" />
                <button class="action-btn primary" @click="triggerRecognition" :disabled="isRecognizing || !geminiKey">
                    {{ isRecognizing ? 'Processing...' : 'Auto-Recognize' }}
                </button>
             </div>
             <ul>
               <li><strong>Navigate:</strong> Press <code>Tab</code> to move to the next line automatically.</li>
               <li><strong>Edit:</strong> Type in the box below the highlighted line.</li>
               <li><strong>Focus:</strong> Click any faint box to jump to that line.</li>
             </ul>
           </div>
        </div>
        
        <div v-if="modifications.length > 0" class="log-sidebar">
            <div class="log-header">
              <span>Changes: {{ modifications.length }}</span>
              <button class="text-btn" @click="resetModifications" :disabled="loading">Reset All</button>
            </div>
            <ul class="log-list">
              <li v-for="(mod, index) in modifications.slice().reverse()" :key="index">
                <small>{{ mod.type }}</small>
                <button @click="undoModification(modifications.length - 1 - index)" class="undo-icon">↺</button>
              </li>
            </ul>
        </div>

      </div>
    </div>

  </div>
</template>

<script setup>
  
import { ref, onMounted, onBeforeUnmount, computed, watch, reactive, nextTick } from 'vue'
import { generateLayoutGraph } from '../layout-analysis-utils/LayoutGraphGenerator.js'
import { useRouter } from 'vue-router'
import edgeWebm from '../tutorial/_edge.webm'
import regionWebm from '../tutorial/_textbox.webm'
import nodeWebm from '../tutorial/_node.webm'



const props = defineProps({
  manuscriptName: { type: String, default: null },
  pageName: { type: String, default: null },
})

const emit = defineEmits(['page-changed', 'back'])
const router = useRouter()

// UI State
const isPanelCollapsed = ref(false)
const activeInput = ref(null) // DOM Ref for input

const setMode = (mode) => {
  // 1. Reset ALL modes to false immediately
  textlineModeActive.value = false
  textboxModeActive.value = false
  nodeModeActive.value = false
  recognitionModeActive.value = false
  
  // 2. Activate the specific mode
  if (mode === 'edge') {
    textlineModeActive.value = true
  } else if (mode === 'region') {
    textboxModeActive.value = true
  } else if (mode === 'node') {
    nodeModeActive.value = true
  } else if (mode === 'recognition') {
    recognitionModeActive.value = true
    
    // Recognition specific init
    sortLinesTopToBottom()
    if(sortedLineIds.value.length > 0 && !focusedLineId.value) {
        activateInput(sortedLineIds.value[0])
    }
  }
  // If mode === 'view', everything stays false, which is correct.
  
  isPanelCollapsed.value = false
}


const isEditModeFlow = computed(() => !!props.manuscriptName && !!props.pageName)

// --- DATA ---
const nodeModeActive = ref(false)
const localManuscriptName = ref('')
const localCurrentPage = ref('')
const localPageList = ref([])
const loading = ref(true)
const isProcessingSave = ref(false)
const error = ref(null)
const imageData = ref('')
const imageLoaded = ref(false)
const textlineModeActive = ref(false)
const textboxModeActive = ref(false)
const recognitionModeActive = ref(false)

// Graph Data
const dimensions = ref([0, 0])
const points = ref([])
const graph = ref({ nodes: [], edges: [] })
const workingGraph = reactive({ nodes: [], edges: [] })
const modifications = ref([])
const nodeEdgeCounts = ref({})
const selectedNodes = ref([])
const tempEndPoint = ref(null)
const isDKeyPressed = ref(false)
const isAKeyPressed = ref(false)
const isEKeyPressed = ref(false) 
const hoveredNodesForMST = reactive(new Set())
const container = ref(null)
const svgOverlayRef = ref(null)

// Labeling Data
const textlineLabels = reactive({}) 
const textlines = ref({}) 
const nodeToTextlineMap = ref({}) 
const hoveredTextlineId = ref(null)
const textboxLabels = ref(0) 
const labelColors = ['#448aff', '#ffeb3b', '#4CAF50', '#f44336', '#9c27b0', '#ff9800'] 

// Recognition Data
const geminiKey = ref(localStorage.getItem('gemini_key') || '')
const isRecognizing = ref(false)
const localTextContent = reactive({}) // Map: lineId -> string
const pagePolygons = ref({}) // Map: lineId -> [[x,y],...]
const focusedLineId = ref(null)
const sortedLineIds = ref([])
const autoRecogEnabled = ref(false)

const saveKeyToStorage = () => {
    localStorage.setItem('gemini_key', geminiKey.value)
}

const scaleFactor = 0.7
const NODE_HOVER_RADIUS = 7
const EDGE_HOVER_THRESHOLD = 5

const manuscriptNameForDisplay = computed(() => localManuscriptName.value)
const currentPageForDisplay = computed(() => localCurrentPage.value)
const isFirstPage = computed(() => localPageList.value.indexOf(localCurrentPage.value) === 0)
const isLastPage = computed(() => localPageList.value.indexOf(localCurrentPage.value) === localPageList.value.length - 1)

const scaledWidth = computed(() => Math.floor(dimensions.value[0] * scaleFactor))
const scaledHeight = computed(() => Math.floor(dimensions.value[1] * scaleFactor))
const scaleX = (x) => x * scaleFactor
const scaleY = (y) => y * scaleFactor
const graphIsLoaded = computed(() => workingGraph.nodes && workingGraph.nodes.length > 0)


// --- RECOGNITION MODE LOGIC ---

// Helper: Convert polygon point list to SVG string
const pointsToSvgString = (pts) => {
    if(!pts) return "";
    return pts.map(p => `${scaleX(p[0])},${scaleY(p[1])}`).join(" ");
}

// Helper: Sort lines Top -> Bottom for navigation
const sortLinesTopToBottom = () => {
    const ids = Object.keys(pagePolygons.value);
    if(ids.length === 0) {
        sortedLineIds.value = [];
        return;
    }
    
    // Compute simple centroid Y for sorting
    const stats = ids.map(id => {
        const pts = pagePolygons.value[id];
        const ys = pts.map(p => p[1]);
        const xs = pts.map(p => p[0]);
        return {
            id,
            minY: Math.min(...ys),
            minX: Math.min(...xs)
        }
    });
    
    // Sort primarily by Y, secondarily by X
    stats.sort((a,b) => {
        const diffY = a.minY - b.minY;
        if(Math.abs(diffY) > 20) return diffY; // Threshold for same-line detection
        return a.minX - b.minX;
    });
    
    sortedLineIds.value = stats.map(s => s.id);
}

// Calculate style for the floating input
const getActiveInputStyle = () => {
    if(!focusedLineId.value || !pagePolygons.value[focusedLineId.value]) return { display: 'none' };
    
    const pts = pagePolygons.value[focusedLineId.value];
    const xs = pts.map(p => p[0]);
    const ys = pts.map(p => p[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const maxY = Math.max(...ys); // Position below bottom edge
    
    const width = (maxX - minX);
    
    return {
        position: 'absolute',
        top: `${scaleY(maxY) + 5}px`, // 5px buffer
        left: `${scaleX(minX)}px`,
        width: `${scaleX(width)}px`,
        height: 'auto',
        zIndex: 100
    }
}

// Dynamic Font Size Calculation
const getDynamicFontSize = () => {
    if(!focusedLineId.value) return '16px';
    
    const text = localTextContent[focusedLineId.value] || "";
    const charCount = Math.max(text.length, 10); // avoid div by zero
    
    const pts = pagePolygons.value[focusedLineId.value];
    if(!pts) return '16px';
    
    const xs = pts.map(p => p[0]);
    const width = (Math.max(...xs) - Math.min(...xs)) * scaleFactor;
    
    // Heuristic: Width / Chars gives px per char. Font size is roughly 1.6x char width for monospace, 
    // but for variable width font, let's try a factor.
    let calcSize = (width / charCount) * 1.8;
    
    // Clamp values
    calcSize = Math.max(14, Math.min(calcSize, 40));
    
    return `${calcSize}px`;
}

const activateInput = (lineId) => {
    focusedLineId.value = lineId;
    nextTick(() => {
        if(activeInput.value) {
            activeInput.value.focus();
        }
    });
}

const handleInputBlur = () => {
    // Optional: Only clear focus if user clicks completely outside?
    // For now, we keep the variable set so the polygon stays highlighted, 
    // unless the user clicks another polygon.
    // Setting focusedLineId = null here makes the box disappear on click-away, which is cleaner.
    setTimeout(() => {
       // Check if focus moved to another input or button, if not, clear.
       if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
       focusedLineId.value = null; 
    }, 200);
}

const focusNextLine = (reverse = false) => {
    if(sortedLineIds.value.length === 0) return;
    
    let currentIdx = sortedLineIds.value.indexOf(focusedLineId.value);
    
    let nextIdx;
    if (currentIdx === -1) {
        nextIdx = 0;
    } else {
        if(reverse) {
             nextIdx = currentIdx - 1;
             if(nextIdx < 0) nextIdx = sortedLineIds.value.length - 1;
        } else {
             nextIdx = currentIdx + 1;
             if(nextIdx >= sortedLineIds.value.length) nextIdx = 0; // Loop or stop? Loop is nice.
        }
    }
    
    activateInput(sortedLineIds.value[nextIdx]);
}

const triggerRecognition = async () => {
    if(!geminiKey.value) return alert("Please enter an API Key");
    localStorage.setItem('gemini_key', geminiKey.value);
    
    isRecognizing.value = true;
    try {
        const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/recognize-text`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                manuscript: localManuscriptName.value,
                page: localCurrentPage.value,
                apiKey: geminiKey.value
            })
        });
        
        const data = await res.json();
        
        if (!res.ok) throw new Error(data.error || "Unknown server error");
        
        if (data.transcriptions) {
            let count = 0;
            Object.entries(data.transcriptions).forEach(([id, text]) => {
                if (text) {
                    localTextContent[id] = text;
                    count++;
                }
            });
            if (count > 0 && sortedLineIds.value.length > 0) {
                // Start review process at top
                activateInput(sortedLineIds.value[0]);
            } else if (count === 0) {
                alert("Gemini finished but returned no text.");
            }
        }
    } catch(e) {
        console.error(e);
        alert("Recognition failed: " + e.message);
    } finally {
        isRecognizing.value = false;
    }
}


// --- EXISTING GRAPH LOGIC ---

const getAverageNodeSize = () => {
    if (!workingGraph.nodes || workingGraph.nodes.length === 0) return 10;
    const sum = workingGraph.nodes.reduce((acc, n) => acc + (n.s || 10), 0);
    return sum / workingGraph.nodes.length;
}

const addNode = (clientX, clientY) => {
    if (!svgOverlayRef.value) return;
    const rect = svgOverlayRef.value.getBoundingClientRect();
    const x = (clientX - rect.left) / scaleFactor;
    const y = (clientY - rect.top) / scaleFactor;
    workingGraph.nodes.push({ x: x, y: y, s: getAverageNodeSize() });
    modifications.value.push({ type: 'node_add' });
}

const deleteNode = (nodeIndex) => {
    if (nodeIndex < 0 || nodeIndex >= workingGraph.nodes.length) return;
    workingGraph.nodes.splice(nodeIndex, 1);
    workingGraph.edges = workingGraph.edges.filter(e => e.source !== nodeIndex && e.target !== nodeIndex);
    workingGraph.edges.forEach(e => {
        if (e.source > nodeIndex) e.source--;
        if (e.target > nodeIndex) e.target--;
    });
    const newLabels = {};
    Object.keys(textlineLabels).forEach(key => {
        const idx = parseInt(key);
        if (idx < nodeIndex) {
            newLabels[idx] = textlineLabels[idx];
        } else if (idx > nodeIndex) {
            newLabels[idx - 1] = textlineLabels[idx];
        }
    });
    for (const key in textlineLabels) delete textlineLabels[key];
    Object.assign(textlineLabels, newLabels);
    resetSelection();
    modifications.value.push({ type: 'node_delete' });
}

const svgCursor = computed(() => {
  if (textboxModeActive.value) return isEKeyPressed.value ? 'crosshair' : 'pointer'
  if (!textlineModeActive.value && !nodeModeActive.value) return 'default'
  if (nodeModeActive.value) return 'cell'; 
  if (isAKeyPressed.value) return 'crosshair'
  if (isDKeyPressed.value) return 'not-allowed'
  return 'default'
})

const downloadResults = async () => {
    try {
        const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/download-results/${localManuscriptName.value}`);
        if (!response.ok) throw new Error('Download failed');
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${localManuscriptName.value}_results.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    } catch (e) {
        alert("Error downloading results: " + e.message);
    }
}

const computeTextlines = () => {
  if (!graphIsLoaded.value) {
    textlines.value = {}
    nodeToTextlineMap.value = {}
    return
  }
  const numNodes = workingGraph.nodes.length
  const adj = Array(numNodes).fill(0).map(() => [])
  for (const edge of workingGraph.edges) {
    if (adj[edge.source] && adj[edge.target]) {
      adj[edge.source].push(edge.target)
      adj[edge.target].push(edge.source)
    }
  }
  const visited = new Array(numNodes).fill(false)
  const newTextlines = {}
  const newNodeToTextlineMap = {}
  let currentTextlineId = 0
  for (let i = 0; i < numNodes; i++) {
    if (!visited[i]) {
      const component = []
      const stack = [i]
      visited[i] = true
      while (stack.length > 0) {
        const u = stack.pop()
        component.push(u)
        newNodeToTextlineMap[u] = currentTextlineId
        for (const v of adj[u]) {
          if (!visited[v]) {
            visited[v] = true
            stack.push(v)
          }
        }
      }
      newTextlines[currentTextlineId] = component
      currentTextlineId++
    }
  }
  textlines.value = newTextlines
  nodeToTextlineMap.value = newNodeToTextlineMap
}

const fetchPageData = async (manuscript, page, isRefresh = false) => {
  if (!manuscript || !page) return;
  
  // Only trigger full loading state if this is a NEW page load
  if (!isRefresh) {
      loading.value = true;
      imageData.value = ''; // Only clear image if changing pages
  }

  error.value = null
  modifications.value = []
  
  // Clear Data States (Keep these to ensure fresh data)
  Object.keys(textlineLabels).forEach(k => delete textlineLabels[k])
  Object.keys(localTextContent).forEach(k => delete localTextContent[k])
  pagePolygons.value = {}
  sortedLineIds.value = []

  try {
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscript}/${page}`
    )
    if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch page data')
    const data = await response.json()

    dimensions.value = data.dimensions
    
    // Update image only if we need to (or if it wasn't there)
    if (data.image) imageData.value = data.image;
    points.value = data.points.map((p) => ({ coordinates: [p[0], p[1]], segment: null }))

    if (data.graph) {
      graph.value = data.graph
    } else if (data.points?.length > 0) {
      graph.value = generateLayoutGraph(data.points)
      if (!isEditModeFlow.value) await saveGeneratedGraph(manuscript, page, graph.value)
    }
    
    if (data.textline_labels) {
      data.textline_labels.forEach((label, index) => { if (label !== -1) textlineLabels[index] = label })
    }
    if (data.textbox_labels?.length > 0) {
       data.textbox_labels.forEach((label, index) => { textlineLabels[index] = label })
       textboxLabels.value = Math.max(...data.textbox_labels) + 1; 
    }
    
    // Process new Polygon/Text data from backend
    if (data.polygons) pagePolygons.value = data.polygons;
    if (data.textContent) {
        Object.assign(localTextContent, data.textContent);
    }

    resetWorkingGraph()
    sortLinesTopToBottom()
  } catch (err) {
    console.error(err)
    error.value = err.message
  } finally {
    loading.value = false
  }
}

const fetchPageList = async (manuscript) => {
  if (!manuscript) return
  try {
    const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/manuscript/${manuscript}/pages`)
    if (!response.ok) throw new Error('Failed to fetch page list')
    localPageList.value = await response.json()
  } catch (err) {
    localPageList.value = []
  }
}

const updateUniqueNodeEdgeCounts = () => {
  const counts = {}
  if (!workingGraph.nodes) return
  workingGraph.nodes.forEach((_, index) => { counts[index] = 0 })
  if (!workingGraph.edges) {
    nodeEdgeCounts.value = counts
    return
  }
  const uniqueEdges = new Set()
  for (const edge of workingGraph.edges) {
    const key = `${Math.min(edge.source, edge.target)}-${Math.max(edge.source, edge.target)}`
    uniqueEdges.add(key)
  }
  for (const key of uniqueEdges) {
    const [source, target] = key.split('-').map(Number)
    if (counts[source] !== undefined) counts[source]++
    if (counts[target] !== undefined) counts[target]++
  }
  nodeEdgeCounts.value = counts
}

watch([() => workingGraph.edges, () => workingGraph.nodes], () => {
    updateUniqueNodeEdgeCounts()
    computeTextlines()
  },{ deep: true, immediate: true }
)

const resetWorkingGraph = () => {
  workingGraph.nodes = JSON.parse(JSON.stringify(graph.value.nodes || []))
  workingGraph.edges = JSON.parse(JSON.stringify(graph.value.edges || []))
  resetSelection()
  computeTextlines()
}

// Colors & Styling
const getNodeColor = (nodeIndex) => {
  if (textboxModeActive.value) {
    const textlineId = nodeToTextlineMap.value[nodeIndex]
    if (hoveredTextlineId.value === textlineId) return '#ff4081' 
    const label = textlineLabels[nodeIndex]
    return (label !== undefined && label > -1) ? labelColors[label % labelColors.length] : '#9e9e9e' 
  }
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return '#00bcd4'
  if (isNodeSelected(nodeIndex)) return '#ff9500'
  const edgeCount = nodeEdgeCounts.value[nodeIndex]
  if (edgeCount < 2) return '#f44336'
  if (edgeCount === 2) return '#4CAF50'
  return '#2196F3'
}

const getNodeRadius = (nodeIndex) => {
  if (textboxModeActive.value) {
    return (hoveredTextlineId.value === nodeToTextlineMap.value[nodeIndex]) ? 7 : 5
  }
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return 7
  if (isNodeSelected(nodeIndex)) return 6
  return nodeEdgeCounts.value[nodeIndex] < 2 ? 5 : 3
}
const getEdgeColor = (edge) => (edge.modified ? '#f44336' : '#ffffff')
const isNodeSelected = (nodeIndex) => selectedNodes.value.includes(nodeIndex)
const isEdgeSelected = (edge) => {
  return selectedNodes.value.length === 2 &&
    ((selectedNodes.value[0] === edge.source && selectedNodes.value[1] === edge.target) ||
      (selectedNodes.value[0] === edge.target && selectedNodes.value[1] === edge.source))
}

const resetSelection = () => {
  selectedNodes.value = []
  tempEndPoint.value = null
}

const onEdgeClick = (edge, event) => {
  if (isAKeyPressed.value || isDKeyPressed.value || textboxModeActive.value || recognitionModeActive.value) return
  event.stopPropagation()
  selectedNodes.value = [edge.source, edge.target]
}

const onBackgroundClick = (event) => {
    if (recognitionModeActive.value) return; // Handled by polygons
    if (nodeModeActive.value) {
        addNode(event.clientX, event.clientY);
        return;
    }
    if (!isAKeyPressed.value && !isDKeyPressed.value) resetSelection();
}

const onNodeClick = (nodeIndex, event) => {
    event.stopPropagation(); 
    if (nodeModeActive.value || recognitionModeActive.value) return;
    if (isAKeyPressed.value || isDKeyPressed.value || textboxModeActive.value) return;
    const existingIndex = selectedNodes.value.indexOf(nodeIndex);
    if (existingIndex !== -1) selectedNodes.value.splice(existingIndex, 1);
    else selectedNodes.value.length < 2 ? selectedNodes.value.push(nodeIndex) : (selectedNodes.value = [nodeIndex]);
}

const onNodeRightClick = (nodeIndex, event) => {
    if (nodeModeActive.value) {
        event.preventDefault(); 
        deleteNode(nodeIndex);
    }
}

const handleSvgMouseMove = (event) => {
  if (!svgOverlayRef.value) return
  const { left, top } = svgOverlayRef.value.getBoundingClientRect()
  const mouseX = event.clientX - left
  const mouseY = event.clientY - top

  if (textboxModeActive.value) {
    // Hover logic for regions
    let newHoveredTextlineId = null
    for (let i = 0; i < workingGraph.nodes.length; i++) {
      const node = workingGraph.nodes[i]
      if (Math.hypot(mouseX - scaleX(node.x), mouseY - scaleY(node.y)) < NODE_HOVER_RADIUS) {
        newHoveredTextlineId = nodeToTextlineMap.value[i]
        break 
      }
    }
    // Check Edges if node not found
    if (newHoveredTextlineId === null) {
        for(const edge of workingGraph.edges) {
             const n1 = workingGraph.nodes[edge.source], n2 = workingGraph.nodes[edge.target];
             if(n1 && n2 && distanceToLineSegment(mouseX, mouseY, scaleX(n1.x), scaleY(n1.y), scaleX(n2.x), scaleY(n2.y)) < EDGE_HOVER_THRESHOLD) {
                 newHoveredTextlineId = nodeToTextlineMap.value[edge.source];
                 break;
             }
        }
    }
    hoveredTextlineId.value = newHoveredTextlineId
    if (hoveredTextlineId.value !== null && isEKeyPressed.value) labelTextline()
    return
  }

  if (!textlineModeActive.value) return
  if (isDKeyPressed.value) handleEdgeHoverDelete(mouseX, mouseY)
  else if (isAKeyPressed.value) handleNodeHoverCollect(mouseX, mouseY)
  else if (selectedNodes.value.length === 1) tempEndPoint.value = { x: mouseX, y: mouseY }
  else tempEndPoint.value = null
}

const handleSvgMouseLeave = () => {
  if (selectedNodes.value.length === 1) tempEndPoint.value = null
  hoveredTextlineId.value = null
}

const labelTextline = () => {
  if (hoveredTextlineId.value === null) return
  const nodesToLabel = textlines.value[hoveredTextlineId.value]
  if (nodesToLabel) {
    nodesToLabel.forEach((nodeIndex) => { textlineLabels[nodeIndex] = textboxLabels.value })
  }
}

const handleGlobalKeyDown = (e) => {
  const tagName = e.target.tagName.toLowerCase();
  if (tagName === 'input' || tagName === 'textarea') return; 

  const key = e.key.toLowerCase()
  if (key === 's' && !e.repeat) {
    e.preventDefault()
    saveAndGoNext()
    return
  }
  if (key === 'w' && !e.repeat) { e.preventDefault(); setMode('edge'); return }
  if (key === 'r' && !e.repeat) { e.preventDefault(); setMode('region'); return }
  if (key === 'n' && !e.repeat) { e.preventDefault(); setMode('node'); return }
  if (key === 't' && !e.repeat) { e.preventDefault(); requestSwitchToRecognition(); return }
  if (textboxModeActive.value && key === 'e' && !e.repeat) { e.preventDefault(); isEKeyPressed.value = true; return }
  
  if (textlineModeActive.value && !e.repeat) {
    if (key === 'd') { e.preventDefault(); isDKeyPressed.value = true; resetSelection(); }
    if (key === 'a') { e.preventDefault(); isAKeyPressed.value = true; hoveredNodesForMST.clear(); resetSelection(); }
  }
}

const handleGlobalKeyUp = (e) => {
  const key = e.key.toLowerCase()
  if (textboxModeActive.value && key === 'e') {
    isEKeyPressed.value = false
    textboxLabels.value++ 
  }
  if (textlineModeActive.value) {
    if (key === 'd') isDKeyPressed.value = false
    if (key === 'a') {
      isAKeyPressed.value = false
      if (hoveredNodesForMST.size >= 2) addMSTEdges()
      hoveredNodesForMST.clear()
    }
  }
}

const edgeExists = (nodeA, nodeB) =>
  workingGraph.edges.some(
    (e) => (e.source === nodeA && e.target === nodeB) || (e.source === nodeB && e.target === nodeA)
  )
const addEdge = () => {
  if (selectedNodes.value.length !== 2 || edgeExists(...selectedNodes.value)) return
  const [source, target] = selectedNodes.value
  const newEdge = { source, target, label: 0, modified: true }
  workingGraph.edges.push(newEdge)
  modifications.value.push({ type: 'add', source, target, label: 0 })
  resetSelection()
}
const deleteEdge = () => {
  if (selectedNodes.value.length !== 2) return
  const [source, target] = selectedNodes.value
  const edgeIndex = workingGraph.edges.findIndex(
    (e) => (e.source === source && e.target === target) || (e.source === target && e.target === source)
  )
  if (edgeIndex === -1) return
  const removedEdge = workingGraph.edges.splice(edgeIndex, 1)[0]
  modifications.value.push({
    type: 'delete',
    source: removedEdge.source,
    target: removedEdge.target,
    label: removedEdge.label,
  })
  resetSelection()
}


const undoModification = (index) => {
  const mod = modifications.value.splice(index, 1)[0]
  if (mod.type === 'add') {
    const edgeIndex = workingGraph.edges.findIndex(
      (e) => e.source === mod.source && e.target === mod.target
    )
    if (edgeIndex !== -1) workingGraph.edges.splice(edgeIndex, 1)
  } else if (mod.type === 'delete') {
    workingGraph.edges.push({
      source: mod.source,
      target: mod.target,
      label: mod.label,
      modified: true,
    })
  } else if (mod.type === 'node_add') {
      workingGraph.nodes.pop();
  } else if (mod.type === 'node_delete') {
      // Complex undo for node delete omitted for brevity
      alert("Undo node delete not fully implemented, reload page.")
  }
}


const resetModifications = () => {
  resetWorkingGraph()
  modifications.value = []
}

const distanceToLineSegment = (px, py, x1, y1, x2, y2) =>
  Math.hypot(
    px - (x1 + Math.max(0, Math.min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1))) * (x2 - x1)),
    py - (y1 + Math.max(0, Math.min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1))) * (y2 - y1))
  )
const handleEdgeHoverDelete = (mouseX, mouseY) => {
  for (let i = workingGraph.edges.length - 1; i >= 0; i--) {
    const edge = workingGraph.edges[i]
    const n1 = workingGraph.nodes[edge.source], n2 = workingGraph.nodes[edge.target]
    if (n1 && n2 && distanceToLineSegment(mouseX, mouseY, scaleX(n1.x), scaleY(n1.y), scaleX(n2.x), scaleY(n2.y)) < EDGE_HOVER_THRESHOLD) {
      const removed = workingGraph.edges.splice(i, 1)[0]
      modifications.value.push({
        type: 'delete',
        source: removed.source,
        target: removed.target,
        label: removed.label,
      })
    }
  }
}
const handleNodeHoverCollect = (mouseX, mouseY) => {
  workingGraph.nodes.forEach((node, index) => {
    if (Math.hypot(mouseX - scaleX(node.x), mouseY - scaleY(node.y)) < NODE_HOVER_RADIUS)
      hoveredNodesForMST.add(index)
  })
}


const calculateMST = (indices, nodes) => {
  const points = indices.map((i) => ({ ...nodes[i], originalIndex: i }))
  const edges = []
  // Create complete graph between selected points
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++) {
      edges.push({
        source: points[i].originalIndex,
        target: points[j].originalIndex,
        weight: Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y),
      })
    }
  // Sort by weight
  edges.sort((a, b) => a.weight - b.weight)
  
  // Kruskal's Algorithm
  const parent = {}
  indices.forEach((i) => (parent[i] = i))
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])))
  const union = (i, j) => {
    const rootI = find(i), rootJ = find(j)
    if (rootI !== rootJ) {
      parent[rootJ] = rootI
      return true
    }
    return false
  }
  return edges.filter((e) => union(e.source, e.target))
}

const addMSTEdges = () => {
  // Calculate MST based on hovered nodes
  const newEdges = calculateMST(Array.from(hoveredNodesForMST), workingGraph.nodes)
  
  newEdges.forEach((edge) => {
    if (!edgeExists(edge.source, edge.target)) {
      const newEdge = { source: edge.source, target: edge.target, label: 0, modified: true }
      workingGraph.edges.push(newEdge)
      modifications.value.push({ type: 'add', ...newEdge })
    }
  })
}

const saveGeneratedGraph = async (name, page, g) => {
  try {
    await fetch(`${import.meta.env.VITE_BACKEND_URL}/save-graph/${name}/${page}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: g }),
    })
  } catch (e) { console.error(e) }
}

const saveModifications = async () => {
  const numNodes = workingGraph.nodes.length
  const labelsToSend = new Array(numNodes).fill(0) 
  for (const nodeIndex in textlineLabels) {
    if (nodeIndex < numNodes) labelsToSend[nodeIndex] = textlineLabels[nodeIndex]
  }
  const dummyTextlineLabels = new Array(numNodes).fill(-1);
  const requestBody = {
    graph: workingGraph, 
    modifications: modifications.value,
    textlineLabels: dummyTextlineLabels, 
    textboxLabels: labelsToSend,
    textContent: localTextContent,
    runRecognition: autoRecogEnabled.value,
    apiKey: geminiKey.value
  }
  try {
    const res = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${localManuscriptName.value}/${localCurrentPage.value}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      }
    )
    if (!res.ok) throw new Error((await res.json()).error || 'Save failed')

    // Check if auto-recognition returned new text
    const data = await res.json()
    if (data.recognizedText) {
        Object.assign(localTextContent, data.recognizedText)
    }

    if (data.recognizedText) { 
    // This will simply be undefined in the async version, which is correct.
    // The UI won't update text immediately, but the user is navigating away anyway.
    Object.assign(localTextContent, data.recognizedText) 
    }

    modifications.value = []
    error.value = null
  } catch (err) {
    error.value = err.message
    throw err
  }
}


// ManuscriptViewer.vue

const requestSwitchToRecognition = async () => {
    if (recognitionModeActive.value) return;

    // 1. Show the unified "Processing..." overlay
    isProcessingSave.value = true;

    try {
        // Scenario A: Unsaved Changes - Save first
        if (modifications.value.length > 0) {
            // Optional: You can keep the confirm here if you want, 
            // or perform auto-save since the user explicitly clicked the tool.
            // For smoothness, we'll assume auto-save or prompt. 
            // If you want the prompt back:
            if (!confirm("Save changes and switch to Recognition Mode?")) {
                isProcessingSave.value = false;
                return;
            }

            await saveModifications(); 
        }

        // Scenario B: Load Data (Silent Refresh)
        // Pass 'true' for isRefresh to avoid the "Loading..." spinner 
        // and avoid clearing the image
        await fetchPageData(localManuscriptName.value, localCurrentPage.value, true);
        
        // Switch Mode
        setMode('recognition');

    } catch (e) {
        alert("Error switching mode: " + e.message);
    } finally {
        // 2. Hide the overlay only when EVERYTHING is done
        isProcessingSave.value = false;
    }
}


const confirmAndNavigate = async (navAction) => {
  if (isProcessingSave.value) return
  if (modifications.value.length > 0 || (recognitionModeActive.value && Object.keys(localTextContent).length > 0)) {
    if (confirm('Do you want to save changes before navigating?')) {
      isProcessingSave.value = true
      try {
        await saveModifications()
        navAction()
      } catch (err) {
        alert('Save failed, navigation cancelled.')
      } finally {
        isProcessingSave.value = false
      }
    } else {
      modifications.value = []
      navAction()
    }
  } else {
    navAction()
  }
}

const navigateToPage = (page) => emit('page-changed', page)
const previousPage = () => confirmAndNavigate(() => {
    const idx = localPageList.value.indexOf(localCurrentPage.value)
    if (idx > 0) navigateToPage(localPageList.value[idx - 1])
})
const nextPage = () => confirmAndNavigate(() => {
    const idx = localPageList.value.indexOf(localCurrentPage.value)
    if (idx < localPageList.value.length - 1) navigateToPage(localPageList.value[idx + 1])
})

const saveAndGoNext = async () => {
  if (loading.value || isProcessingSave.value) return
  isProcessingSave.value = true
  try {
    await saveModifications()
    const idx = localPageList.value.indexOf(localCurrentPage.value)
    if (idx < localPageList.value.length - 1) navigateToPage(localPageList.value[idx + 1])
    else alert('Last page saved!')
  } catch (err) { alert(`Save failed: ${err.message}`) } 
  finally { isProcessingSave.value = false }
}

const runHeuristic = () => {
  if(!points.value.length) return;
  const rawPoints = points.value.map(p => [p.coordinates[0], p.coordinates[1], 10]); 
  const heuristicGraph = generateLayoutGraph(rawPoints);
  workingGraph.edges = heuristicGraph.edges.map(e => ({ source: e.source, target: e.target, label: e.label, modified: true }));
  modifications.value.push({ type: 'reset_heuristic' }); 
  computeTextlines();
}

onMounted(async () => {
  if (props.manuscriptName && props.pageName) {
    localManuscriptName.value = props.manuscriptName
    localCurrentPage.value = props.pageName
    await fetchPageList(props.manuscriptName)
    await fetchPageData(props.manuscriptName, props.pageName)
  }
  window.addEventListener('keydown', handleGlobalKeyDown)
  window.addEventListener('keyup', handleGlobalKeyUp)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleGlobalKeyDown)
  window.removeEventListener('keyup', handleGlobalKeyUp)
})

watch(() => props.pageName, (newPageName) => {
    if (newPageName && newPageName !== localCurrentPage.value) {
      localCurrentPage.value = newPageName
      fetchPageData(localManuscriptName.value, newPageName)
    }
})

watch(recognitionModeActive, (val) => {
    if(val) {
        textlineModeActive.value = false; textboxModeActive.value = false; nodeModeActive.value = false;
        resetSelection();
    }
})
</script>

<style scoped>
/* Basic Layout */
.manuscript-viewer {
  display: flex; flex-direction: column; height: 100vh; width: 100%;
  background-color: #1e1e1e; color: #e0e0e0; font-family: 'Roboto', sans-serif; overflow: hidden;
}

/* Top Bar */
.top-bar {
  display: flex; justify-content: space-between; align-items: center; padding: 0 16px;
  height: 60px; background-color: #2c2c2c; border-bottom: 1px solid #3d3d3d; flex-shrink: 0; z-index: 10;
}
.top-bar-left, .top-bar-right, .action-group { display: flex; align-items: center; gap: 16px; }
.page-title { font-size: 1.1rem; color: #fff; white-space: nowrap; }
.separator { width: 1px; height: 24px; background-color: #555; margin: 0 4px; }
button { border: none; cursor: pointer; border-radius: 4px; font-size: 0.9rem; transition: all 0.2s; }
.nav-btn { background: transparent; color: #aaa; padding: 8px 12px; display: flex; align-items: center; }
.nav-btn:hover:not(:disabled) { background: rgba(255,255,255,0.1); color: #fff; }
.action-btn { background: #424242; color: #fff; padding: 8px 16px; border: 1px solid #555; }
.action-btn.primary { background-color: #4CAF50; border-color: #43a047; }
.action-btn:hover:not(:disabled) { background-color: #505050; }
.action-btn.primary:hover:not(:disabled) { background-color: #5cb860; }
button:disabled { opacity: 0.5; cursor: not-allowed; }

/* Main Visualization */
.visualization-container {
  position: relative; overflow: auto; flex-grow: 1; display: flex;
  justify-content: center; align-items: flex-start; padding: 2rem; background-color: #121212;
}
.image-container { position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.6); }
.manuscript-image { display: block; user-select: none; opacity: 0.7; }
.graph-overlay { position: absolute; top: 0; left: 0; opacity: 0; pointer-events: none; transition: opacity 0.2s; }
.graph-overlay.is-visible { opacity: 1; pointer-events: auto; }

/* Input Floater (NEW) */
.input-floater {
    z-index: 100;
}
.line-input {
    width: 100%;
    background: rgba(0, 0, 0, 0.85);
    color: #fff;
    border: 1px solid #00e5ff; /* Cyan focus color */
    padding: 8px 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    border-radius: 4px;
    font-family: monospace;
    outline: none;
    transition: font-size 0.2s;
}

/* Polygons */
.polygon-inactive {
    cursor: pointer;
    pointer-events: auto;
    transition: stroke 0.2s;
}
.polygon-inactive:hover {
    stroke: rgba(255,255,255,0.6);
    stroke-width: 2;
}
.polygon-active {
    pointer-events: none; /* Let clicks pass through to input if overlapping? or keep blocking? */
    animation: pulse-border 2s infinite;
}

@keyframes pulse-border {
    0% { stroke-opacity: 1; }
    50% { stroke-opacity: 0.6; }
    100% { stroke-opacity: 1; }
}

/* Loading/Error */
.processing-save-notice, .loading, .error-message {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  padding: 20px 30px; border-radius: 8px; z-index: 10000; text-align: center;
  box-shadow: 0 4px 15px rgba(0,0,0,0.5);
}
.processing-save-notice { background: rgba(33,33,33,0.95); border: 1px solid #444; color: #fff; }
.error-message { background: #c62828; color: white; }
.loading { font-size: 1.2rem; color: #aaa; background: rgba(0,0,0,0.5); }

/* Bottom Rail */
.bottom-panel {
  background-color: #2c2c2c; border-top: 1px solid #3d3d3d; flex-shrink: 0; display: flex; flex-direction: column;
  height: 280px; transition: height 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.bottom-panel.is-collapsed { height: 45px; }
.mode-tabs { display: flex; background: #212121; height: 45px; flex-shrink: 0; }
.mode-tab { flex: 1; border-bottom: 3px solid transparent; color: #888; text-transform: uppercase; display: flex; align-items: center; justify-content: center; background: transparent; }
.mode-tab:hover:not(:disabled) { background: #2a2a2a; color: #bbb; }
.mode-tab.active { background: #2c2c2c; color: #448aff; border-bottom-color: #448aff; font-weight: 500; }
.tab-spacer { flex-grow: 1; background: #212121; }
.panel-toggle-btn { background: #333; color: #aaa; border-left: 1px solid #444; padding: 0 16px; min-width: 100px; }

/* Help Area */
.help-content-area { padding: 16px 24px; display: flex; gap: 24px; height: 100%; overflow: hidden; }
.help-section { display: flex; gap: 24px; flex-grow: 1; height: 100%; }
.media-container { width: 200px; height: 200px; background: #000; border: 1px solid #444; flex-shrink: 0; }
.tutorial-video { width: 100%; height: 100%; object-fit: contain; }
.instructions-container { flex-grow: 1; max-width: 600px; overflow-y: auto; color: #ccc; }
.instructions-container h3 { color: #fff; margin-top: 0; }
code { background: #424242; color: #ffb74d; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
.form-group-inline { display: flex; gap: 10px; margin-bottom: 10px; }
.api-input { background: #444; border: 1px solid #555; color: #fff; padding: 5px 10px; flex-grow: 1; }
.webm-placeholder { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #777; background: #3a3a3a; }

/* Sidebar Log */
.log-sidebar { width: 200px; background: #222; border: 1px solid #444; display: flex; flex-direction: column; }
.log-header { padding: 8px 10px; background: #333; border-bottom: 1px solid #444; display: flex; justify-content: space-between; }
.log-list { list-style: none; padding: 0; margin: 0; overflow-y: auto; max-height: 120px; }
.log-list li { padding: 6px 10px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; color: #aaa; }
.undo-icon { background: none; color: #777; font-size: 1.1rem; }
.undo-icon:hover { color: #fff; }

.api-input-small {
    background: #444; border: 1px solid #555; color: #fff; padding: 4px 8px; 
    border-radius: 4px; font-size: 0.8rem; width: 120px;
}
.toggle-switch {
  position: relative; display: inline-block; width: 34px; height: 20px;
}
.toggle-switch input { opacity: 0; width: 0; height: 0; }
.slider {
  position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
  background-color: #ccc; transition: .4s; border-radius: 34px;
}
.slider:before {
  position: absolute; content: ""; height: 14px; width: 14px; left: 3px; bottom: 3px;
  background-color: white; transition: .4s; border-radius: 50%;
}
input:checked + .slider { background-color: #4CAF50; }
input:checked + .slider:before { transform: translateX(14px); }


</style>


import {
    singleConsonantMap, doubleCharMap, tripleCharMap,
    dependentVowelMap, independentVowelMap, combinedVowelMap,
    potentialVowelKeys, vowelReplacementMap, sequencePrefixes,
    miscMap, simpleInsertMap, // Import new maps
    handleSingleConsonant, insertCharacter, replacePreviousChars,
    applyDependentVowel, insertConsonantSequence, replaceConsonantSequence,
    applyNukta, // Import new helper
    logCharactersBeforeCursor,
    HALANT, ZWNJ, ZWJ, NUKTA, ANUSVARA, VISARGA, CANDRABINDU, DANDA, DOUBLE_DANDA, OM // Import constants
  } from './InputClusterCode'
  
  let lastEffectiveKey = null;
  
  export function handleInput(event, devanagariRef) {
    const key = event.key;
    const input = event.target;
    const cursorPosition = input.selectionStart;
    const currentValue = input.value;


    // Helper to check if a character is a "bare" Devanagari consonant
    // (not a matra, not halant, not modifier, etc., just the consonant character itself)
    const isBareDevanagariConsonant = (char) => {
        if (!char || char.length !== 1) return false; // Must be a single character
        const cp = char.charCodeAt(0);
        // Devanagari Unicode block range for consonants:
        // Main consonants: U+0915 (क) to U+0939 (ह)
        // Additional consonants (e.g., ळ, or nukta forms like क़ if they are single codepoints): U+0958 to U+095F
        if ((cp >= 0x0915 && cp <= 0x0939) || (cp >= 0x0958 && cp <= 0x095F)) {
            return true;
        }
        return false;
    };
  
    // --- Basic Filtering ---
    if (event.metaKey || event.ctrlKey || event.altKey) {
        console.log("Ignoring Ctrl/Meta/Alt key press");
        return;
    }
  
    let effectiveKey = key;
    if (event.shiftKey && key.length === 1 && !key.match(/[a-zA-Z]/)) {
          // Allow Shift + '.' for Nukta trigger? Or other specific combos?
          // For now, treat Shift + non-letter as potentially ignorable or map explicitly
          if (key === '.') { // Allow Shift + '.' if needed later for something else?
               // effectiveKey = '>'; // Example if Shift+. has meaning
               console.log("Shift + . detected, treating as '.' for now");
               effectiveKey = '.'; // Treat as period for now, can change if needed
          } else {
              console.log("Ignoring Shift + Symbol key press:", key);
              lastEffectiveKey = null;
              return;
          }
     } else if (key.length > 1 && key !== 'Backspace') {
         console.log(`Ignoring functional key: ${key}`);
         lastEffectiveKey = null;
         return;
     } else if (event.shiftKey && key.length === 1 && key.match(/[A-Z]/)) {
         effectiveKey = key; // Uppercase letter
     } else if (key.length === 1 && key.match(/[a-z]/)) {
         effectiveKey = key; // Lowercase letter
     } else if (key === 'Backspace') {
          effectiveKey = 'Backspace';
     } else if (simpleInsertMap[key] !== undefined) { // Check if it's a simple insert key (digit, space, ., etc.)
          effectiveKey = key;
     } else if (key === '`') { // Keep explicit halant trigger
         effectiveKey = '`';
     } else if (key === '.') { // Allow period
          effectiveKey = '.';
     } else {
          console.log(`Key "${key}" might pass to fallback or be ignored`);
          // Decide whether to ignore unmapped symbols or let them pass
          // Let's ignore unmapped symbols for now to avoid unexpected chars
          // You can remove this 'return' to allow them.
          // lastEffectiveKey = null; // Reset if ignoring
          // return;
          effectiveKey = key; // Allow pass-through for now
     }
  
  
    console.log("-------------------------");
    console.log(`Effective Key: ${effectiveKey} (at pos ${cursorPosition}) | Last Key: ${lastEffectiveKey}`);
    console.log("State BEFORE processing:");
    logCharactersBeforeCursor(input);
  
    const charM1 = currentValue[cursorPosition - 1];
    const charM2 = currentValue[cursorPosition - 2];
    const charM3 = currentValue[cursorPosition - 3];
    const charM4 = currentValue[cursorPosition - 4];
    const charM5 = currentValue[cursorPosition - 5];
  
    // --- Explicit Halant + ZWNJ ('`' key) ---
    // Insert HALANT + ZWNJ (useful for controlling conjuncts explicitly)
    if (effectiveKey === '`') {
        event.preventDefault();
        const sequence = HALANT + ZWNJ;
        insertCharacter(input, devanagariRef, sequence, cursorPosition);
        console.log('Inserted explicit halant + ZWNJ');
        lastEffectiveKey = effectiveKey;
        return;
    }
  
    // --- Backspace Handling (Keep existing logic) ---
    if (effectiveKey === 'Backspace') {
        lastEffectiveKey = null; // Reset sequence tracking
        if (charM1 === ZWNJ && charM2 === HALANT && cursorPosition >=3 ) {
            event.preventDefault();
            console.log('Backspace: removing Base/Modifier + Halant + ZWNJ'); // Nukta case C+Nukta+H+ZWNJ needs different handling? No, 3 chars works.
            const newValue = currentValue.slice(0, cursorPosition - 3) + currentValue.slice(cursorPosition);
            devanagariRef.value = newValue; input.value = newValue;
            input.setSelectionRange(cursorPosition - 3, cursorPosition - 3);
            logCharactersBeforeCursor(input); return;
        }
         else if (charM2 === HALANT && cursorPosition >= 2) {
             event.preventDefault();
             const newValue = currentValue.slice(0, cursorPosition - 1) + ZWNJ + currentValue.slice(cursorPosition);
             console.log('Backspace: Removed last char, Inserted ZWNJ after halant (original logic)');
             devanagariRef.value = newValue; input.value = newValue;
             input.setSelectionRange(cursorPosition, cursorPosition);
             logCharactersBeforeCursor(input); return;
         }
        else {
            console.log('Backspace: Default behavior');
            queueMicrotask(() => { devanagariRef.value = input.value; logCharactersBeforeCursor(input); });
            return;
        }
    }
  
    // --- Simple Insertions (Space, Digits, ZWJ, ZWNJ, Period, Avagraha etc.) ---
    if (simpleInsertMap[effectiveKey] !== undefined) {
        event.preventDefault();
        const charToInsert = simpleInsertMap[effectiveKey];
        insertCharacter(input, devanagariRef, charToInsert, cursorPosition);
        // Reset last key for space and punctuation, but maybe not for ZWJ/ZWNJ?
        if (charToInsert === ' ' || charToInsert === '.' || charToInsert === AVAGRAHA ) {
            lastEffectiveKey = null;
        } else {
            lastEffectiveKey = effectiveKey; // Keep sequence potential for digits? or ZWJ/ZWNJ? Let's update.
        }
        return;
    }
  
    // --- Miscellaneous Sequence Handling (MM, ff, .N, om) ---
    let potentialMiscSequence = '';
    let miscSequenceHandled = false;
    if (lastEffectiveKey && sequencePrefixes[lastEffectiveKey]?.includes(effectiveKey)) {
          potentialMiscSequence = lastEffectiveKey + effectiveKey;
          console.log("Potential Misc sequence:", potentialMiscSequence);
  
          // Check for MM (Chandrabindu)
          if (potentialMiscSequence === 'MM' && charM1 === ANUSVARA) {
               event.preventDefault();
               replacePreviousChars(input, devanagariRef, 1, CANDRABINDU, cursorPosition);
               miscSequenceHandled = true;
          }
          // Check for ff (Double Danda) - Note conflict with consonant 'f'
          // Prioritize 'ff' if previous was DANDA.
          else if (potentialMiscSequence === 'ff' && charM1 === DANDA) {
               event.preventDefault();
               replacePreviousChars(input, devanagariRef, 1, DOUBLE_DANDA, cursorPosition);
               miscSequenceHandled = true;
          }
           // Check for .N (Nukta)
           else if (potentialMiscSequence === '.N') {
              // Requires C+H+ZWNJ context
              if (cursorPosition >= 3 && charM1 === ZWNJ && charM2 === HALANT && !potentialVowelKeys.has(charM3) /* Ensure it's a consonant base */ ) {
                   event.preventDefault();
                   applyNukta(input, devanagariRef, cursorPosition); // Use helper
                   miscSequenceHandled = true;
              } else {
                  console.log("Nukta (.N) sequence detected but invalid context.");
                  // Prevent default insertion of 'N'? Or allow 'N'? Let's prevent.
                  event.preventDefault();
                  // Don't set miscSequenceHandled = true, let 'N' be potentially handled later if needed
                  lastEffectiveKey = effectiveKey; // Update last key to N
                  return; // Exit early, nukta cannot be applied here
              }
           }
           // Check for om
           else if (potentialMiscSequence === 'om' && miscMap[potentialMiscSequence]) {
               event.preventDefault();
               insertCharacter(input, devanagariRef, OM, cursorPosition);
               miscSequenceHandled = true;
               lastEffectiveKey = null; // Reset sequence after om
               return; // Handled 'om'
           }
    }
  
    if (miscSequenceHandled) {
          lastEffectiveKey = effectiveKey; // Update last key
          return; // Exit if a misc sequence was handled
    }
  
    // --- Explicit HALANT Insertion ('q' key) ---
    // Only inserts HALANT, potentially removing ZWNJ if present
    if (effectiveKey === 'q') {
        event.preventDefault();
        if (charM1 === ZWNJ && charM2 === HALANT) {
            // We are after C + H + ZWNJ. Replace ZWNJ with H. Net effect: remove ZWNJ.
             replacePreviousChars(input, devanagariRef, 1, '', cursorPosition); // Remove ZWNJ
             console.log("Applied explicit halant (q): Removed ZWNJ after existing Halant.");
        } else if (charM1 === ZWNJ) {
             // After explicit HALANT+ZWNJ (` key). Replace ZWNJ with just HALANT.
             replacePreviousChars(input, devanagariRef, 1, HALANT, cursorPosition);
             console.log("Applied explicit halant (q): Replaced ZWNJ with Halant.");
        }
        else {
            // Insert HALANT after a vowel or a consonant+matra
            insertCharacter(input, devanagariRef, HALANT, cursorPosition);
            console.log("Applied explicit halant (q): Inserted Halant.");
        }
        lastEffectiveKey = effectiveKey;
        return;
    }
  
    // --- Custom 'a' Vowel Handling (Schwa Deletion & Matra Application) ---
    // Define which keys trigger schwa deletion (C+H+ZWNJ -> C)
    const isSchwaDeletionKey = (effectiveKey === 'a' || effectiveKey === 'A');

    // Define which keys trigger 'aa' matra (C -> C+ ा) and what that matra is.
    let aaMatra = null;
    if (effectiveKey === 'a' || effectiveKey === 'A') {
        aaMatra = dependentVowelMap['a']; // Should be 'ा'
    } else if (effectiveKey === 'aa' || effectiveKey === 'AA') { // If you have 'aa'/'AA' mapping
        aaMatra = dependentVowelMap['aa']; // Should also be 'ा'
    }
    // Add more else if for other 'a'-like keys if necessary

    // 1. Handle Schwa Deletion: C + Halant + ZWNJ + 'a'/'A'  --->  C
    if (isSchwaDeletionKey && cursorPosition >= 2 && charM1 === ZWNJ && charM2 === HALANT) {
        // Context: charM3 (Base Consonant) + charM2 (Halant) + charM1 (ZWNJ)
        // Action: Pressing 'a' or 'A' removes Halant + ZWNJ, leaving just charM3.
        event.preventDefault();
        replacePreviousChars(input, devanagariRef, 2, '', cursorPosition); // Removes the last 2 chars (Halant + ZWNJ)
        console.log(`Schwa Deletion by '${effectiveKey}': Removed H+ZWNJ after '${charM3}' to form full consonant.`);
        lastEffectiveKey = effectiveKey; // Update last key
        return; // Crucial: exit after handling
    }

    // 2. Handle 'aa' Matra Application: C + 'a'/'A'/'aa'/'AA'  --->  C + ा
    // This executes if the schwa deletion didn't happen (e.g., cursor is after a full consonant).
    if (aaMatra === 'ा' && charM1 && isBareDevanagariConsonant(charM1)) {
        // Context: charM1 is a bare consonant (e.g., 'क', 'ख')
        // Action: Pressing an 'a'-like key appends the 'ा' matra.
        event.preventDefault();
        replacePreviousChars(input, devanagariRef, 1, charM1 + aaMatra, cursorPosition); // Replaces charM1 with charM1 + 'ा'
        console.log(`'${effectiveKey}' applied Matra '${aaMatra}' to bare consonant '${charM1}'.`);
        lastEffectiveKey = effectiveKey; // Update last key
        return; // Crucial: exit after handling
    }
    // --- END Custom 'a' Vowel Handling ---


    // --- Single Anusvara / Visarga ('M', 'H') Application ---
    if (effectiveKey === 'M' || effectiveKey === 'H') {
        event.preventDefault();
        const modifier = (effectiveKey === 'M') ? ANUSVARA : VISARGA;
  
        if (cursorPosition >= 3 && charM1 === ZWNJ && charM2 === HALANT) {
            // Preceded by C + H + ZWNJ. Replace H+ZWNJ with modifier.
            // Base char is charM3
            const baseChar = charM3;
            replacePreviousChars(input, devanagariRef, 2, modifier, cursorPosition); // Remove H+ZWNJ, add modifier
            console.log(`Applied modifier ${modifier} after ${baseChar} (replacing H+ZWNJ)`);
        } else if (cursorPosition > 0 && charM1 !== HALANT) {
             // Preceded by a full character (Vowel or C+Matra). Append modifier.
             insertCharacter(input, devanagariRef, modifier, cursorPosition);
             console.log(`Appended modifier ${modifier} after ${charM1}`);
        } else {
            // Context not suitable (e.g., start of input, after halant without ZWNJ)
            console.log(`Cannot apply modifier ${modifier} in current context.`);
            // Optionally insert with dotted circle: insertCharacter(input, devanagariRef, '\u25CC' + modifier, cursorPosition);
        }
        lastEffectiveKey = effectiveKey;
        return;
    }
  
      // --- Single Danda Insertion ('f' key) ---
      // Needs careful handling due to 'f' also mapping to consonant 'फ'
      // Rule: If 'f' is pressed AND it wasn't part of 'ff', treat as DANDA *unless*
      // the context implies the consonant 'फ'.
      // Let's prioritize DANDA if not after C+H+ZWNJ.
      if (effectiveKey === 'f') {
          const isConsonantContext = cursorPosition >= 3 && charM1 === ZWNJ && charM2 === HALANT;
          const isConsonantReplacementContext = doubleCharMap['h']?.[charM3] !== undefined && charM1 === ZWNJ && charM2 === HALANT; // e.g., p+h -> ph
  
          // If not likely forming 'फ' or 'ph', insert Danda.
          if (!isConsonantContext && !isConsonantReplacementContext) {
              event.preventDefault();
              insertCharacter(input, devanagariRef, DANDA, cursorPosition);
              console.log("Inserted Danda (|)");
              lastEffectiveKey = effectiveKey; // Treat as sequence starter for 'ff'
              return;
          }
          // Otherwise, let it fall through to consonant handling below.
          console.log("'f' key pressed in consonant context, will be handled as 'फ'");
      }
  
  
    // --- Consonant Sequence Completion (Triples, Doubles) ---
    // Keep this logic exactly as it was
    const tripleMappings = tripleCharMap[effectiveKey];
    if (tripleMappings && cursorPosition >= 5) { /* ... triple check logic ... */
        if (charM1 === ZWNJ && charM2 === HALANT && charM4 === HALANT) {
            const precedingSequence = charM5 + charM3;
            if (tripleMappings[precedingSequence]) {
                const mapping = tripleMappings[precedingSequence];
                event.preventDefault();
                replaceConsonantSequence(input, devanagariRef, mapping.resultChar, cursorPosition, mapping.remove);
                lastEffectiveKey = effectiveKey; return;
            }
        }
         if (effectiveKey === 'r' && charM1 === ZWNJ && charM2 === HALANT && charM3 === 'श' && tripleMappings['श']) {
              const mapping = tripleMappings['श'];
              event.preventDefault();
              replaceConsonantSequence(input, devanagariRef, mapping.resultChar, cursorPosition, mapping.remove);
              lastEffectiveKey = effectiveKey; return;
         }
    }
    const doubleMappings = doubleCharMap[effectiveKey];
    if (doubleMappings && cursorPosition >= 3) { /* ... double check logic ... */
         if (charM1 === ZWNJ && charM2 === HALANT) {
            const precedingBase = charM3;
            if (doubleMappings[precedingBase]) {
                const mapping = doubleMappings[precedingBase];
                event.preventDefault();
                replaceConsonantSequence(input, devanagariRef, mapping.resultChar, cursorPosition, mapping.remove);
                lastEffectiveKey = effectiveKey; return;
            }
        }
    }
  
    // --- Vowel Handling Logic (Keep existing logic) ---
    let potentialVowelSequence = '';
    if (lastEffectiveKey && sequencePrefixes[lastEffectiveKey]?.includes(effectiveKey) && potentialVowelKeys.has(effectiveKey[0])) {
         potentialVowelSequence = lastEffectiveKey + effectiveKey;
          console.log("Potential Vowel sequence:", potentialVowelSequence);
          if (combinedVowelMap[potentialVowelSequence]) {
              const isDependentContext = charM1 === ZWNJ && charM2 === HALANT && cursorPosition >= 3;
              const isVowelReplacementContext = vowelReplacementMap[charM1]?.[effectiveKey];
  
              if (isVowelReplacementContext) {
                  event.preventDefault();
                  const replacementChar = vowelReplacementMap[charM1][effectiveKey];
                  replacePreviousChars(input, devanagariRef, 1, replacementChar, cursorPosition);
                  console.log(`Vowel Replacement: ${charM1} + ${effectiveKey} -> ${replacementChar}`);
                  lastEffectiveKey = effectiveKey; return;
              } else if (isDependentContext && dependentVowelMap[potentialVowelSequence]) {
                  event.preventDefault();
                  applyDependentVowel(input, devanagariRef, dependentVowelMap[potentialVowelSequence], cursorPosition);
                  console.log(`Applied complex matra: ${dependentVowelMap[potentialVowelSequence]}`);
                  lastEffectiveKey = effectiveKey; return;
              } else if (!isDependentContext && independentVowelMap[potentialVowelSequence]) {
                  event.preventDefault();
                  insertCharacter(input, devanagariRef, independentVowelMap[potentialVowelSequence], cursorPosition);
                  console.log(`Inserted complex independent vowel: ${independentVowelMap[potentialVowelSequence]}`);
                  lastEffectiveKey = effectiveKey; return;
              } else {
                   console.log(`Sequence ${potentialVowelSequence} valid but context mismatch?`);
              }
        }
    }
     // Vowel Replacement Check (single key)
     if (potentialVowelKeys.has(effectiveKey) && charM1 && vowelReplacementMap[charM1]?.[effectiveKey]) {
          event.preventDefault();
          const replacementChar = vowelReplacementMap[charM1][effectiveKey];
          replacePreviousChars(input, devanagariRef, 1, replacementChar, cursorPosition);
          console.log(`Vowel Replacement (single key): ${charM1} + ${effectiveKey} -> ${replacementChar}`);
          lastEffectiveKey = effectiveKey; return;
     }
     // Single Vowel / Single Consonant Handling
    const isDepContext = charM1 === ZWNJ && charM2 === HALANT && cursorPosition >= 3;
    const devDep = dependentVowelMap[effectiveKey];
    const devIndep = independentVowelMap[effectiveKey];
    const devCons = singleConsonantMap[effectiveKey];
  
    if (isDepContext) {
        if (devDep) {
            event.preventDefault(); applyDependentVowel(input, devanagariRef, devDep, cursorPosition);
            lastEffectiveKey = effectiveKey; return;
        } else if (devCons) {
            event.preventDefault();
            replacePreviousChars(input, devanagariRef, 1, devCons + HALANT + ZWNJ, cursorPosition);
            console.log(`Forming conjunct: Removed ZWNJ, added ${devCons}+H+ZWNJ`);
            lastEffectiveKey = effectiveKey; return;
        } else if (devIndep) {
             event.preventDefault();
             replacePreviousChars(input, devanagariRef, 2, devIndep, cursorPosition);
             console.log(`WARN: Independent vowel after C+H+ZWNJ. Replaced H+ZWNJ with ${devIndep}`);
             lastEffectiveKey = effectiveKey; return;
        }
    } else {
        if (devIndep) {
            event.preventDefault(); insertCharacter(input, devanagariRef, devIndep, cursorPosition);
            lastEffectiveKey = effectiveKey; return;
        } else if (devCons) {
            // Check if it's 'f' which should have been handled as Danda already if appropriate
            if (effectiveKey === 'f') {
                 // If we reached here, 'f' should be treated as consonant 'फ'
                 event.preventDefault();
                 handleSingleConsonant(event, devanagariRef, devCons);
                 lastEffectiveKey = effectiveKey; return;
            } else {
               // Handle other consonants normally
               event.preventDefault(); handleSingleConsonant(event, devanagariRef, devCons);
               lastEffectiveKey = effectiveKey; return;
            }
        } else if (devDep) {
            event.preventDefault();
            const standaloneMatra = '\u25CC' + devDep;
            insertCharacter(input, devanagariRef, standaloneMatra, cursorPosition);
            console.log(`WARN: Dependent vowel in independent context. Inserted ${standaloneMatra}`);
            lastEffectiveKey = effectiveKey; return;
        }
    }
  
    // --- Handle 'h' as a single consonant if it didn't form a double/triple ---
    if (effectiveKey === 'h' && !doubleMappings?.[charM3] && !tripleMappings?.[charM5+charM3]) {
         event.preventDefault();
         handleSingleConsonant(event, devanagariRef, 'ह');
         lastEffectiveKey = effectiveKey;
         return;
    }
  
  
    // --- Fallback ---
    console.log(`Key "${effectiveKey}" not handled by custom logic. Default behavior might occur.`);
    lastEffectiveKey = effectiveKey; // Update last key even if default occurs
    queueMicrotask(() => {
        devanagariRef.value = input.value;
        logCharactersBeforeCursor(input);
    });
  }


  // InputClusterCode.js

export function logCharactersBeforeCursor(input) {
  const cursorPosition = input.selectionStart;
  const currentValue = input.value;
  // Log more characters for debugging multi-char sequences
  console.log({
    '-5': currentValue[cursorPosition - 5],
    '-4': currentValue[cursorPosition - 4],
    '-3': currentValue[cursorPosition - 3],
    '-2': currentValue[cursorPosition - 2],
    '-1': currentValue[cursorPosition - 1]
  });
  return;
}

// --- Character Constants ---
export const HALANT = '\u094D';
export const ZWNJ = '\u200C'; // Zero-Width Non-Joiner
export const ZWJ = '\u200D';  // Zero-Width Joiner
export const NUKTA = '\u093C'; // Combining Dot Below (Nukta)
export const ANUSVARA = '\u0902'; // ं
export const VISARGA = '\u0903'; // ः
export const CANDRABINDU = '\u0901'; // ँ
export const AVAGRAHA = '\u093D'; // ऽ
export const DANDA = '\u0964'; // ।
export const DOUBLE_DANDA = '\u0965'; // ॥
export const OM = '\u0950'; // ॐ
// Add constants for other special characters if keys are assigned
// export const DEVANAGARI_ABBREVIATION_SIGN = '\u0970';
// export const DEVANAGARI_SIGN_HIGH_SPACING_DOT = '\u0971';
// export const DEVANAGARI_SIGN_INVERTED_CANDRABINDU = '\u0900';
// export const DEVANAGARI_STRESS_SIGN_UDATTA = '\u0951';
// export const DEVANAGARI_STRESS_SIGN_ANUDATTA = '\u0952';

// --- Consonant Mappings ---
// Maps single Roman keys directly to Devanagari base consonants
export const singleConsonantMap = {
  'k': 'क', 'g': 'ग', 'c': 'च', 'j': 'ज', 'T': 'ट', 't': 'त', 'D': 'ड',
  'd': 'द', 'N': 'ण', 'n': 'न', 'p': 'प', 'b': 'ब', 'm': 'म', 'y': 'य',
  'r': 'र', 'l': 'ल', 'v': 'व', 'V': 'ङ', 'S': 'ष', 's': 'स', 'h': 'ह',
  'L': 'ळ', 'Y': 'ञ',
  'f': 'फ', // Note: 'f' also used for DANDA in miscMap
  'z': 'ज', // Note: 'z' also used for vowel prefixes
  'q': 'क', // Note: 'q' also used for HALANT in miscMap
};

// Structure: triggerKey: { precedingDevanagariBase: { resultChar: devanagariBase, remove: count } }
export const doubleCharMap = {
  'h': { // Aspirates + sh
    'क': { resultChar: 'ख', remove: 3 }, 'ग': { resultChar: 'घ', remove: 3 },
    'च': { resultChar: 'छ', remove: 3 }, 'ज': { resultChar: 'झ', remove: 3 },
    'ट': { resultChar: 'ठ', remove: 3 }, 'ड': { resultChar: 'ढ', remove: 3 },
    'त': { resultChar: 'थ', remove: 3 }, 'द': { resultChar: 'ध', remove: 3 },
    'प': { resultChar: 'फ', remove: 3 }, 'ब': { resultChar: 'भ', remove: 3 },
    'स': { resultChar: 'श', remove: 3 },
  },
  's': {
    'क': { resultChar: 'क्ष', remove: 3 }, // k + s -> ks (maps to kS = क्ष)
  },
  'S': {
    'क': { resultChar: 'क्ष', remove: 3 }  // k + S -> kS
  },
};

// Structure: triggerKey: { precedingDevSequence: { resultChar: devanagariBase, remove: count } }
export const tripleCharMap = {
  'y': {
    'दन': { resultChar: 'ज्ञ', remove: 5 }, // d + n + y -> dny (ज्ञ)
    'गञ': { resultChar: 'ज्ञ', remove: 5 }, // g + Y + y -> gny (ज्ञ)
    'गन': { resultChar: 'ज्ञ', remove: 5 }, // g + n + y -> gny (ज्ञ)
  },
  'r': {
    'श': { resultChar: 'श्र', remove: 3 }, // sh + r -> shr
  },
};


// --- Vowel Mappings ---
// Dependent Vowels (Matras)
export const dependentVowelMap = {
    'a':'ा', 'e':'े', 'i':'ि', 'o':'ो', 'u':'ु',
    'aa': 'ा', 'ee': 'ी', 'ii': 'ी', 'uu': 'ू', 'oo': 'ू',
    'ai':'ै', 'au':'ौ', 'ou':'ौ',
    'Rri':'ृ', 'RrI':'ॄ', 'Lli':'ॢ', 'LlI':'ॣ',
    'ze':'ॆ', 'zo':'ॊ', 'aE':'ॅ', 'aO':'ॉ',
    'zau':'\u094F', // Kashmiri/Bihari Au Matra
};

// Independent Vowels
export const independentVowelMap = {
    'a':'अ', 'A':'अ', 'i':'इ', 'I':'इ', 'u':'उ', 'U':'उ',
    'e':'ए', 'E':'ए', 'o':'ओ', 'O':'ओ',
    'aa':'आ', 'AA':'आ', 'ii':'ई', 'II':'ई', 'ee':'ई',
    'uu':'ऊ', 'UU':'ऊ', 'oo':'ऊ',
    'ai':'ऐ', 'AI':'ऐ', 'au':'औ', 'AU':'औ', 'ou':'औ',
    'RRi':'ऋ', 'RRI':'ॠ', 'LLi':'ऌ', 'LLI':'ॡ',
    'AE':'ॲ', // Marathi AE
    'AO':'ऑ', // Marathi/Borrowed AO
    // 'aE':'ऍ', // Alternate AE - choose one or handle contextually
    // 'aO':'ऑ', // Alternate AO - choose one or handle contextually
    'zEE':'ऎ', // South Indian Short E
    'zO':'ऒ',  // South Indian Short O
    'zA':'ऄ', // Historic/Regional A
    'zAU':'ॵ', // Historic/Regional Au
};

// Combined lookup for potential vowel starting keys/sequences
export const potentialVowelKeys = new Set([
    'a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U',
    'R', 'L', 'z' // Covers Rri, Lli, ze, zo, zau etc.
]);

// Combined map for resolving full vowel sequences
export const combinedVowelMap = { ...dependentVowelMap, ...independentVowelMap };

// --- Vowel Sequence Handling Logic ---
// Map for replacements like i+i -> ii, e+i -> ai, etc.
// Structure: { precedingDevChar: { currentKey: replacementDevChar } }
export const vowelReplacementMap = {
    // Dependent Matra Replacements
    'ि': { 'i': 'ी', 'e': 'ी' }, // short i + i/e -> long ii/ee
    'ु': { 'u': 'ू', 'o': 'ू' }, // short u + u/o -> long uu/oo
    'े': { 'e': 'ी', 'i': 'ै' }, // e + e -> ee, e + i -> ai
    'ो': { 'o': 'ू', 'u': 'ौ', 'i': 'ौ' }, // o + o -> oo, o + u/i -> au
    'ृ': { 'I': 'ॄ', 'i': 'ॄ' }, // Rri + I/i -> RrI
    'ॢ': { 'I': 'ॣ', 'i': 'ॣ' }, // Lli + I/i -> LlI
    'ा': { 'a': 'ा', 'E': 'ॅ', 'O': 'ॉ' }, // aa + a -> aa, aa + E -> aE Candra, aa + O -> aO Candra
    // Independent Vowel Replacements
    'इ': { 'i': 'ई', 'I': 'ई', 'e': 'ई', 'E': 'ई' }, // short I + i/I/e/E -> long II/EE
    'उ': { 'u': 'ऊ', 'U': 'ऊ', 'o': 'ऊ', 'O': 'ऊ' }, // short U + u/U/o/O -> long UU/OO
    'ए': { 'e': 'ई', 'E': 'ई', 'i': 'ऐ', 'I': 'ऐ' }, // E + e/E -> EE, E + i/I -> AI
    'ओ': { 'o': 'ऊ', 'O': 'ऊ', 'u': 'औ', 'U': 'औ' }, // O + o/O -> OO, O + u/U -> AU
    'अ': { 'a': 'आ', 'A': 'आ', 'E': 'ॲ', 'O': 'ऑ'}, // A + a/A -> AA, A + E -> AE(Marathi), A + O -> AO(Marathi)
    'ऋ': { 'I': 'ॠ' }, // RRi + I -> RRI
    'ऌ': { 'I': 'ॡ' }, // LLi + I -> LLI
};


// --- Miscellaneous Mappings ---
export const miscMap = { // VOWEL MODIFIERS(m), HALANT(H), NUKTA(N), NUMBERS, CURRENCY etc.
    // Single Key Modifiers / Symbols
    'M': ANUSVARA,      // 'ं'
    'H': VISARGA,       // 'ः'
    'F': AVAGRAHA,      // 'ऽ'
    'q': HALANT,        // '्' (Explicit Halant ONLY - applies differently than Halant+ZWNJ)
    ' ': ' ',
    '.': '.',           // Period
    'f': DANDA,         // '।', Note: 'f' is also consonant 'फ'
    '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
    '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
    'W': ZWJ,           // '\u200D' (Zero Width Joiner)
    'w': ZWNJ,          // '\u200C' (Zero Width Non-Joiner)

    // Sequences (Handled in handleInput based on last key)
    'MM': CANDRABINDU,  // 'ँ' (Replaces Anusvara)
    '.N': NUKTA,        // '◌़' (Applies to preceding consonant)
    'ff': DOUBLE_DANDA, // '॥' (Replaces Danda)
    'om': OM,           // 'ॐ'

    // --- Keys needing assignment for unmapped chars ---
    // Choose appropriate keys and uncomment/add here if needed
    // Example assignments:
    // '\'': '\u0970', // DEVANAGARI ABBREVIATION SIGN
    // '_': '\u0971',  // DEVANAGARI SIGN HIGH SPACING DOT
    // '^': '\u0900',  // DEVANAGARI SIGN INVERTED CANDRABINDU
    // '+': '\u0951',  // DEVANAGARI STRESS SIGN UDATTA
    // '=': '\u0952',  // DEVANAGARI STRESS SIGN ANUDATTA
};

// Helper Map for simple direct insertions (no context needed beyond the key itself)
// Includes digits, space, period, ZWJ, ZWNJ, Avagraha, and any assigned simple symbols
export const simpleInsertMap = {
    ' ': ' ', '.': '.',
    '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
    '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
    'W': ZWJ, 'w': ZWNJ,
    'F': AVAGRAHA, // Avagraha can usually be inserted directly
    // Add keys for other simple insertions if assigned in miscMap
    // '\'': '\u0970', '_': '\u0971', '^': '\u0900', '+': '\u0951', '=': '\u0952',
};

// --- Sequence Prefix Information ---
// Helps identify potential multi-character sequences
// Structure: { key: potentialNextKey[] }
// ** Define the base object first **
export const sequencePrefixes = {
    // Vowel prefixes
    'R': ['r', 'R', 'i', 'I'], // For Rr, RR, Rri, RRI
    'L': ['l', 'L', 'i', 'I'], // For Ll, LL, Lli, LLI
    'z': ['e', 'o', 'a', 'E', 'A', 'O', 'U'], // For ze, zo, za, zE, zA etc.
    'a': ['a', 'e', 'i', 'u', 'E', 'O'], // For aa, ae, ai, au, aE, aO
    'A': ['A', 'E', 'I', 'O', 'U'], // For AA, AE, AI, AO, AU
    'e': ['e', 'i'], // For ee, ei (ai)
    'E': ['E', 'I'], // For EE, EI (ai)
    'i': ['i', 'e'], // For ii, ie (ee)
    'I': ['I', 'E'], // For II, IE (ee)
    'o': ['o', 'u', 'i'], // For oo, ou (au), oi (au?) - ** Initial definition **
    'O': ['O', 'U', 'I'], // For OO, OU (au), OI (au?)
    'u': ['u', 'o'], // For uu, uo (oo)
    'U': ['U', 'O'], // For UU, UO (oo)

    // Misc prefixes
    '.': ['N'], // For Nukta sequence .N
    'M': ['M'], // For Chandrabindu sequence MM
    'f': ['f'], // For Double Danda sequence ff
    // Add 'A', 'U' prefixes if needed for 'AUM' later
};

// ** Modify the object after definition **
// Add 'm' to the potential keys following 'o' for the 'om' sequence
sequencePrefixes['o'] = [...(sequencePrefixes['o'] || []), 'm'];
// If you were implementing AUM:
// sequencePrefixes['A'] = [...(sequencePrefixes['A'] || []), 'U']; // If A can start AU and AUM
// sequencePrefixes['U'] = [...(sequencePrefixes['U'] || []), 'M']; // If U can start UU and follow A in AUM


// --- Helper Functions ---

// Insert Character Sequence (Generic)
export function insertCharacter(input, devanagariRef, charToInsert, cursorPosition) {
    const currentValue = input.value;
    const newValue =
      currentValue.slice(0, cursorPosition) +
      charToInsert +
      currentValue.slice(cursorPosition);
    const newCursorPosition = cursorPosition + charToInsert.length;

    devanagariRef.value = newValue;
    input.value = newValue;
    input.setSelectionRange(newCursorPosition, newCursorPosition);
    console.log(`Inserted: ${charToInsert}`);
    logCharactersBeforeCursor(input);
}

// Replace Previous Characters (Generic)
export function replacePreviousChars(input, devanagariRef, charsToRemove, charToInsert, cursorPosition) {
    const currentValue = input.value;
    const startReplacePos = cursorPosition - charsToRemove;

    // Ensure we don't go below index 0
    if (startReplacePos < 0) {
        console.error(`replacePreviousChars: Attempting to remove ${charsToRemove} chars from pos ${cursorPosition}.`);
        return; // Or handle differently
    }

    const newValue =
      currentValue.slice(0, startReplacePos) +
      charToInsert +
      currentValue.slice(cursorPosition); // Slice from original cursor pos

    // New cursor position: start of replacement + length of inserted char
    const newCursorPosition = startReplacePos + charToInsert.length;

    devanagariRef.value = newValue;
    input.value = newValue;
    input.setSelectionRange(newCursorPosition, newCursorPosition);
    console.log(`Replaced ${charsToRemove} chars with ${charToInsert}`);
    logCharactersBeforeCursor(input);
}

// Helper to apply Dependent Vowel (Matra)
export function applyDependentVowel(input, devanagariRef, matra, cursorPosition) {
    const currentValue = input.value;
    // Context assumes: Base (charM3) + Halant (charM2) + ZWNJ (charM1) before cursor
    const baseConsonant = currentValue[cursorPosition - 3];
    const charsToRemove = 3; // Base + Halant + ZWNJ
    const charToInsert = baseConsonant + matra;

    // Use the generic replace function
    replacePreviousChars(input, devanagariRef, charsToRemove, charToInsert, cursorPosition);
    console.log(`Applied Matra: ${matra} to ${baseConsonant}`);
}

// Insert Consonant Sequence (Base + Halant + ZWNJ)
export function insertConsonantSequence(input, devanagariRef, baseChar, cursorPosition) {
    const currentValue = input.value;
    const sequence = baseChar + HALANT + ZWNJ;
    const sequenceLength = sequence.length; // Should be 3

    const newValue =
      currentValue.slice(0, cursorPosition) +
      sequence +
      currentValue.slice(cursorPosition);

    const newCursorPosition = cursorPosition + sequenceLength;

    devanagariRef.value = newValue;
    input.value = newValue;
    input.setSelectionRange(newCursorPosition, newCursorPosition);
    console.log(`Inserted ${baseChar} + Halant + ZWNJ`);
    logCharactersBeforeCursor(input);
}

// Replace previous sequence with new Consonant Sequence (Base + Halant + ZWNJ)
export function replaceConsonantSequence(input, devanagariRef, baseChar, cursorPosition, charsToRemove) {
    const currentValue = input.value;
    const sequence = baseChar + HALANT + ZWNJ;
    const sequenceLength = sequence.length; // Should be 3

    const newValue =
      currentValue.slice(0, cursorPosition - charsToRemove) +
      sequence +
      currentValue.slice(cursorPosition);

    // New cursor position: original position - removed chars + inserted chars
    const newCursorPosition = cursorPosition - charsToRemove + sequenceLength;

    devanagariRef.value = newValue;
    input.value = newValue;
    input.setSelectionRange(newCursorPosition, newCursorPosition);
    console.log(`Replaced ${charsToRemove} chars with ${baseChar} + Halant + ZWNJ`);
    logCharactersBeforeCursor(input);
}

// Handle insertion of a single consonant character
export function handleSingleConsonant(event, devanagariRef, devanagariChar) {
  const input = event.target;
  const cursorPosition = input.selectionStart;
  const currentValue = input.value;
  const characterRelativeMinus1 = currentValue[cursorPosition - 1];

  // No preventDefault needed here, it's handled in handleInput

  if (characterRelativeMinus1 === ZWNJ) {
    // If ZWNJ is just before cursor (e.g., after explicit H+ZWNJ),
    // replace the ZWNJ with the new consonant sequence.
    replacePreviousChars(input, devanagariRef, 1, devanagariChar + HALANT + ZWNJ, cursorPosition);
    console.log(`Replaced ZWNJ with ${devanagariChar} + Halant + ZWNJ`);

  } else {
    // Standard insertion: Append Consonant + Halant + ZWNJ
    insertConsonantSequence(input, devanagariRef, devanagariChar, cursorPosition);
  }
}

// Apply Nukta to the preceding consonant (C+H+ZWNJ -> C+Nukta+H+ZWNJ)
export function applyNukta(input, devanagariRef, cursorPosition) {
    const currentValue = input.value;
    // Context: Base (charM3) + Halant (charM2) + ZWNJ (charM1)
     // Basic check to prevent errors if context is wrong, though handleInput should verify
    if (cursorPosition < 3 || currentValue[cursorPosition - 1] !== ZWNJ || currentValue[cursorPosition - 2] !== HALANT) {
        console.error("applyNukta called with invalid context.");
        return;
    }
    const baseConsonant = currentValue[cursorPosition - 3];
    const charsToRemove = 3; // Base + Halant + ZWNJ
    // Insert Base + Nukta + Halant + ZWNJ
    const charToInsert = baseConsonant + NUKTA + HALANT + ZWNJ;

    replacePreviousChars(input, devanagariRef, charsToRemove, charToInsert, cursorPosition);
    console.log(`Applied Nukta to ${baseConsonant}`);
}



<script setup>
import { ref } from 'vue'

const togglePalette = ref(false)
const copied = ref()

function copyToClipboard(char) {
  navigator.clipboard
    .writeText(char)
    .then(() => {
      copied.value = char; 
    })
    .catch((err) => {
      console.error('Failed to copy:', err)
    })
}

const characters = [
  'ऀ',
  'ऄ',
  'ऎ',
  'ऒ',
  'ऴ',
  'ऺ',
  'ऻ',
  'ऽ',
  'ॆ',
  'ॊ',
  'ॎ',
  'ॏ',
  '॑',
  '॒',
  'ॕ',
  'ॖ',
  'ॗ',
  'ऌ',
  'ॡ',
  'ॢ',
  'ॢ',
  '॥',
  '॰',
  'ॲ',
  'ॳ',
  'ॴ',
  'ॵ',
  'ॶ',
  'ॷ',
  'ॸ',
  'ॹ',
  'ॺ',
  'ॻ',
  'ॼ',
  'ॽ',
  'ॾ',
  'ॿ',
  '꣠',
  '꣡',
  '꣢',
  '꣣',
  '꣤',
  '꣥',
  '꣦',
  '꣧',
  '꣨',
  '꣩',
  '꣪',
  '꣫',
  '꣬',
  '꣭',
  '꣮',
  '꣯',
  '꣰',
  '꣱',
  'ꣲ',
  'ꣳ',
  'ꣴ',
  'ꣵ',
  'ꣶ',
  'ꣷ',
  '꣸',
  '꣹',
  '꣺',
  'ꣻ',
  '꣼',
  'ꣽ',
  'ꣾ',
  'ꣿ',
]
</script>

<template>
  <div class="characterPalette-container">
    <button @click="togglePalette = !togglePalette" class="btn btn-outline-warning me-2">
      ॳ Rare Characters
    </button>
    <span v-if="copied">Copied &zwnj;{{ copied }} !</span>
    <div v-if="togglePalette" class="characterPalette mt-2">
      <button
        class="btn btn-outline-secondary character-button"
        v-for="character in characters"
        :key="character"
        @click="copyToClipboard(character)"
      >
        {{ character }}
      </button>
    </div>
  </div>
</template>

<style>
.characterPalette-container {
  display: inline-block;
  /* max-width: 20rem; */
}

.characterPalette {
  /* display: flex; */
  position: absolute;
  max-width: 20rem;
  /* flex-shrink: 1; */
  padding: 0.5em;
  background-color: var(--color-background);
  border: var(--bs-border-width) solid var(--bs-border-color);
  border-radius: var(--bs-border-radius);
  justify-content: space-around;
}

.character-button {
  width: 2rem;
  margin-left: 0.5em;
  margin-bottom: 0.5em;
  padding: 6px 0px 6px 0px;
  text-align: center;
  background-color: var(--bs-body-bg);
}
</style>
