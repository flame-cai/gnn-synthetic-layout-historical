As an expert software development, specializing front-end sofware development using .vue, and backend developend using python. Please assist me in making precise improvements to data annotation tool below, which helps historians analyze layout and digitize text from historical manuscripts.

This code is part of a larger system which performs layout analysis on manuscript images using a Graph Neural Network (GNN). The layout analysis problem is formulated in a graph based manner, where characters are treated as nodes and characters of the same text lines are connected with edges. Thus nodes containing the same textline have the same text line label. The user can also add/delete nodes, label nodes with textbox labels by marking nodes of each text box with the same integer label. Once labelled (using gnn layout inference + manual corrections), the system generates PAGE XML files containing textbox and text line bounding polygons. The system also saves textline images, for each textbox.
After performing the layout analysis semi-automatically using the graph neural network, we use Gemini API to recognize text context from each textline in the "Recognition Mode".

I want you help in making the following upgrades to the code base:

- please remove the "view mode" as it is redundant. Instead we want to add a new hotkey "v". once v is pressed and help, all the nodes and edges in the layout mode, and text input fields in recognition mode should disappear. Please add this hot key tutorial text in the layout mode.

- once a manuscript is loaded, or new manuscript is uploaded, allow the user to jump to a particular page using a drop down (instead of just previous and next button). When user loads a manuscript, by default please load the page which has been the most recently edited. We can know this by looking at the PAGE-XML files of the manuscript.

- add a "save" button in addition to "save and next button" to the frontend. This will save the page data and stay on the same page.

- in the recognition mode, when the user is manually making corrections in the text input fields, we want to introduce an auto save functionality which will auto save (only the PAGE-XML data) every 20 seconds. This saving should happen in the background and should not interfere in the users typing experience using the recogition mode. Note, this feature is only in the recognition mode. Implement this carefully so as to not break any existing logic.

Please make precise changes to the code. DO NOT make unnecessary change. Rewrite the entire frontend file.


Please think very carefully to not break any existing functionality and to not introduce subtle bugs. Consider all changes to be made, their implications, by performing a tree of thought as a red team software testing specialist, revise the plan, rethink, and then implement the final robust code, with good logging for easy debugging.

Please first understand the code below:

# ManuscriptViewer.vue
<template>
  <div class="manuscript-viewer">
    
    <!-- TOP RAIL: Navigation & Global Actions -->
    <div class="top-bar">
      <div class="top-bar-left">
        <button class="nav-btn secondary" @click="$emit('back')">Back</button>
        <span class="page-title">{{ manuscriptNameForDisplay }} <span class="divider">/</span> Page {{ currentPageForDisplay }}</span>
      </div>

      <!-- Auto-Recognition Controls in Center -->
      <div class="top-bar-center" style="display:flex; align-items:center; gap: 10px; margin-left: 20px;">
          <!-- Auto-Recog Toggle -->
          <label class="toggle-switch">
             <input type="checkbox" v-model="autoRecogEnabled">
             <span class="slider"></span>
          </label>
          <span style="font-size: 0.8rem; color: #ccc;">Auto-Recognize on Save</span>

          <!-- Devanagari Keyboard Toggle -->
          <div class="divider-vertical" style="width:1px; height:20px; background:#444; margin:0 5px;"></div>
          <label class="toggle-switch">
             <input type="checkbox" v-model="devanagariModeEnabled">
             <span class="slider"></span>
          </label>
          <span style="font-size: 0.8rem; color: #ccc;">Devanagari Keyboard</span>
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

      <!-- 3. Image Container -->
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

        <!-- SVG Graph Layer (Visible in Layout Mode and View Mode) -->
        <!-- is-visible class allows pointer events only in Layout Mode -->
        <svg
          v-if="graphIsLoaded && !recognitionModeActive"
          class="graph-overlay"
          :class="{ 'is-visible': layoutModeActive }"
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
            @click.stop="layoutModeActive && onEdgeClick(edge, $event)"
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
              layoutModeActive &&
              selectedNodes.length === 1 &&
              tempEndPoint &&
              !isAKeyPressed &&
              !isDKeyPressed &&
              !isEKeyPressed
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
            stroke-width="0"
            class="polygon-active"
          />
        </svg>

        <!-- Recognition Input Overlay Layer -->
        <div
            v-if="recognitionModeActive && focusedLineId && pagePolygons[focusedLineId]"
            class="input-floater"
            :style="getActiveInputStyle()"
        >
            <input 
                ref="activeInput"
                v-model="localTextContent[focusedLineId]" 
                class="line-input active"
                @keydown="handleRecognitionInput"
                @blur="handleInputBlur"
                @keydown.tab.prevent="focusNextLine(false)"
                @keydown.shift.tab.prevent="focusNextLine(true)"
                placeholder="Type text here..."
                :style="{ 
                    fontSize: getDynamicFontSize(),
                    fontFamily: devanagariModeEnabled ? 'Arial, sans-serif' : 'monospace',
                    marginBottom: '4px' 
                }"
            />
            <div 
                v-if="localTextConfidence[focusedLineId]" 
                class="confidence-strip"
            >
                <span 
                    v-for="(char, idx) in localTextContent[focusedLineId]" 
                    :key="idx"
                    class="conf-char"
                    :style="{ 
                        color: getConfidenceColor(localTextConfidence[focusedLineId][idx]),
                        fontSize: getDynamicFontSize()
                    }"
                >{{ char }}</span>
            </div>
        </div>

      </div>
    </div>

    <!-- BOTTOM RAIL: Controls & Help Center -->
    <div class="bottom-panel" :class="{ 'is-collapsed': isPanelCollapsed }">
      
      <!-- Mode Tabs (Always Visible) -->
      <div class="mode-tabs">
         <button 
           class="mode-tab" 
           :class="{ active: !layoutModeActive && !recognitionModeActive }"
           @click="setMode('view')"
           :disabled="isProcessingSave">
           View Mode
         </button>
          <button 
           class="mode-tab" 
           :class="{ active: layoutModeActive }"
           @click="setMode('layout')"
           :disabled="isProcessingSave || !graphIsLoaded">
           Layout Mode (W)
         </button>
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
        
        <!-- View Mode Help -->
        <div v-if="!layoutModeActive && !recognitionModeActive" class="help-section">
          <div class="instructions-container">
            <h3>View Mode</h3>
            <p>Pan and zoom to inspect the manuscript. Press <code>W</code> for Layout Mode or <code>T</code> for Recognition.</p>
          </div>
        </div>

        <!-- Unified Layout Mode Help -->
        <div v-if="layoutModeActive" class="help-section full-width">
          <div class="help-grid">
            
            <!-- Nodes Card -->
            <div class="help-card">
              <div class="media-container-small">
                <video :src="nodeWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
              </div>
              <div class="card-text">
                <h4>Nodes</h4>
                <p><span class="key-badge">L-Click</span> Add Node</p>
                <p><span class="key-badge">R-Click</span> Delete Node</p>
              </div>
            </div>

            <!-- Edges Card -->
            <div class="help-card">
              <div class="media-container-small">
                <video :src="edgeWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
              </div>
              <div class="card-text">
                <h4>Edges</h4>
                <p>Hold <span class="key-badge">A</span> + Hover to Connect</p>
                <p>Hold <span class="key-badge">D</span> + Hover to Delete</p>
              </div>
            </div>

            <!-- Regions Card -->
            <div class="help-card">
              <div class="media-container-small">
                <video :src="regionWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
              </div>
              <div class="card-text">
                <h4>Regions</h4>
                <p>Hold <span class="key-badge">E</span> + Hover to Label</p>
                <p>Release & Press <span class="key-badge">E</span> for New Box</p>
              </div>
            </div>

          </div>
        </div>

        <!-- RECOGNITION MODE HELP -->
        <div v-if="recognitionModeActive" class="help-section">
           <div class="media-container">
             <div class="webm-placeholder" style="flex-direction:column; gap:10px;">
              <span>Recognition Mode</span>
              <span v-if="devanagariModeEnabled" style="color:#4CAF50; font-size:0.8rem;">Devanagari ON</span>
            </div>
           </div>
           <div class="instructions-container">
             <h3>Recognition Mode</h3>
             <p>Transcribe line-by-line using Gemini AI or Manual Input.</p>
             <ul>
               <li><strong>Navigate:</strong> Press <code>Tab</code> to move to the next line automatically.</li>
               <li><strong>Toggle Script:</strong> Use the switch in the top bar to enable Devanagari input.</li>
               <li v-if="devanagariModeEnabled"><strong>Keys:</strong> Type phonetically (e.g., 'k' -> 'क'). Use '`' for Halant+ZWNJ.</li>
             </ul>
             
             <div v-if="devanagariModeEnabled" style="margin-top: 15px; border-top: 1px solid #444; padding-top: 10px;">
                 <CharacterPalette />
             </div>
           </div>
        </div>
        
        <!-- Logs -->
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


// Import Devanagari logic and Palette
import { handleInput as handleDevanagariInput } from '../typing-utils/devanagariInputUtils.js'
import CharacterPalette from '../typing-utils/CharacterPalette.vue'

const props = defineProps({
  manuscriptName: { type: String, default: null },
  pageName: { type: String, default: null },
})

const emit = defineEmits(['page-changed', 'back'])
const router = useRouter()

// UI State
const isPanelCollapsed = ref(false)
const activeInput = ref(null) // DOM Ref for input

// Consolidated Mode Switching
const setMode = (mode) => {
  // Reset all modes first
  layoutModeActive.value = false
  recognitionModeActive.value = false
  
  // Clean up any lingering key states when switching modes
  isAKeyPressed.value = false
  isDKeyPressed.value = false
  isEKeyPressed.value = false
  resetSelection()

  if (mode === 'layout') {
    layoutModeActive.value = true
  } else if (mode === 'recognition') {
    recognitionModeActive.value = true
    // Recognition specific init
    sortLinesTopToBottom()
    if(sortedLineIds.value.length > 0 && !focusedLineId.value) {
        activateInput(sortedLineIds.value[0])
    }
  }
  // 'view' leaves both false, which is the correct state
  
  isPanelCollapsed.value = false
}


const isEditModeFlow = computed(() => !!props.manuscriptName && !!props.pageName)

// --- DATA ---
// Consolidated Layout Mode
const layoutModeActive = ref(false)
// Recognition Mode
const recognitionModeActive = ref(false)

const localManuscriptName = ref('')
const localCurrentPage = ref('')
const localPageList = ref([])
const loading = ref(true)
const isProcessingSave = ref(false)
const error = ref(null)
const imageData = ref('')
const imageLoaded = ref(false)

// Graph Data
const dimensions = ref([0, 0])
const points = ref([])
const graph = ref({ nodes: [], edges: [] })
const workingGraph = reactive({ nodes: [], edges: [] })
const modifications = ref([])
const nodeEdgeCounts = ref({})
const selectedNodes = ref([])
const tempEndPoint = ref(null)

// Key states for Pseudo-Modes inside Layout Mode
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
const devanagariModeEnabled = ref(true) 
const localTextConfidence = reactive({}) 
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

// Wrapper for Devanagari Input Handling
const handleRecognitionInput = (event) => {
    if (!devanagariModeEnabled.value) return; 
    if (event.ctrlKey || event.metaKey || event.altKey) return; 
    if (!focusedLineId.value) return;

    const textRef = {
        get value() {
            return localTextContent[focusedLineId.value] || '';
        },
        set value(val) {
            localTextContent[focusedLineId.value] = val;
        }
    };
    handleDevanagariInput(event, textRef);
}


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
        if(Math.abs(diffY) > 20) return diffY; 
        return a.minX - b.minX;
    });
    
    sortedLineIds.value = stats.map(s => s.id);
}

// Calculate style for the floating input (Adaptive for Vertical/Horizontal)
const getActiveInputStyle = () => {
    if(!focusedLineId.value || !pagePolygons.value[focusedLineId.value]) return { display: 'none' };
    
    const pts = pagePolygons.value[focusedLineId.value];
    const xs = pts.map(p => p[0]);
    const ys = pts.map(p => p[1]);
    
    // Raw coordinates
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    
    const rawWidth = maxX - minX;
    const rawHeight = maxY - minY;

    // Determine Orientation: Vertical if height is significantly larger than width
    const isVertical = rawHeight > (rawWidth * 1.2); 

    const style = {
        position: 'absolute',
        height: 'auto',
        zIndex: 100
    };

    if (isVertical) {
        // VERTICAL LOGIC
        // Decide Side: Check if polygon is on Left or Right half of the image
        const pageCenterX = dimensions.value[0] / 2;
        const polyCenterX = minX + (rawWidth / 2);
        
        // Fixed width for vertical inputs creates a better UX than trying to match the skinny line width
        const INPUT_WIDTH_PX = 250; 
        
        style.top = `${scaleY(minY)}px`; // Align with top of the line
        style.width = `${INPUT_WIDTH_PX}px`;

        if (polyCenterX > pageCenterX) {
            // Line is on Right side -> Place Input to the LEFT of the line
            style.left = `${scaleX(minX) - INPUT_WIDTH_PX - 10}px`;
        } else {
            // Line is on Left side -> Place Input to the RIGHT of the line
            style.left = `${scaleX(maxX) + 10}px`;
        }
    } else {
        // HORIZONTAL LOGIC (Standard)
        style.top = `${scaleY(maxY) + 5}px`;
        style.left = `${scaleX(minX)}px`;
        style.width = `${scaleX(rawWidth)}px`;
    }

    return style;
}

// Dynamic Font Size Calculation
const getDynamicFontSize = () => {
    if(!focusedLineId.value) return '16px';
    
    const text = localTextContent[focusedLineId.value] || "";
    const charCount = Math.max(text.length, 10); 
    
    const pts = pagePolygons.value[focusedLineId.value];
    if(!pts) return '16px';
    
    const xs = pts.map(p => p[0]);
    const width = (Math.max(...xs) - Math.min(...xs)) * scaleFactor;
    
    let calcSize = (width / charCount) * 1.8;
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
    setTimeout(() => {
       if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
       if (document.activeElement && document.activeElement.classList.contains('character-button')) return;
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
             if(nextIdx >= sortedLineIds.value.length) nextIdx = 0; 
        }
    }
    
    activateInput(sortedLineIds.value[nextIdx]);
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
  if (!layoutModeActive.value) return 'default'
  
  // Logic inside Layout Mode based on held keys
  if (isEKeyPressed.value) return 'crosshair' // Region Mode
  if (isAKeyPressed.value) return 'crosshair' // Edge Connect
  if (isDKeyPressed.value) return 'not-allowed' // Edge Delete
  
  // Default Layout Mode is Node editing
  return 'cell'; 
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
  
  if (!isRefresh) {
      loading.value = true;
      imageData.value = ''; 
  }

  error.value = null
  modifications.value = []
  
  Object.keys(textlineLabels).forEach(k => delete textlineLabels[k])
  Object.keys(localTextContent).forEach(k => delete localTextContent[k])
  Object.keys(localTextConfidence).forEach(k => delete localTextConfidence[k]) 
  pagePolygons.value = {}
  sortedLineIds.value = []

  try {
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscript}/${page}`
    )
    if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch page data')
    const data = await response.json()

    dimensions.value = data.dimensions
    
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
    
    if (data.polygons) pagePolygons.value = data.polygons;
    if (data.textContent) {
        Object.assign(localTextContent, data.textContent);
    }
    if (data.textConfidences) {
        Object.assign(localTextConfidence, data.textConfidences);
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

const getConfidenceColor = (score) => {
    if (score === undefined || score === null) return '#fff'; 
    if (score >= 0.8) return '#4CAF50'; 
    if (score >= 0.5) return '#FFC107'; 
    return '#FF5252';                   
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
  // If 'e' is held, show Region labeling colors
  if (layoutModeActive.value && isEKeyPressed.value) {
    const textlineId = nodeToTextlineMap.value[nodeIndex]
    if (hoveredTextlineId.value === textlineId) return '#ff4081' 
    const label = textlineLabels[nodeIndex]
    return (label !== undefined && label > -1) ? labelColors[label % labelColors.length] : '#9e9e9e' 
  }
  
  // Standard Graph Colors
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return '#00bcd4'
  if (isNodeSelected(nodeIndex)) return '#ff9500'
  const edgeCount = nodeEdgeCounts.value[nodeIndex]
  if (edgeCount < 2) return '#f44336'
  if (edgeCount === 2) return '#4CAF50'
  return '#2196F3'
}

const getNodeRadius = (nodeIndex) => {
  // Larger nodes for Region Labeling
  if (layoutModeActive.value && isEKeyPressed.value) {
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
  if (isAKeyPressed.value || isDKeyPressed.value || isEKeyPressed.value || recognitionModeActive.value) return
  event.stopPropagation()
  selectedNodes.value = [edge.source, edge.target]
}

const onBackgroundClick = (event) => {
    if (recognitionModeActive.value) return; 
    
    // In Layout Mode, clicking empty space adds a node unless keys are held
    if (layoutModeActive.value && !isAKeyPressed.value && !isDKeyPressed.value && !isEKeyPressed.value) {
        addNode(event.clientX, event.clientY);
        return;
    }
    
    resetSelection();
}

const onNodeClick = (nodeIndex, event) => {
    event.stopPropagation(); 
    if (!layoutModeActive.value || recognitionModeActive.value) return;
    
    // If modifiers are held, click behavior changes (or is disabled)
    if (isAKeyPressed.value || isDKeyPressed.value || isEKeyPressed.value) return;
    
    // Standard Selection for Edges logic (click 1, click 2)
    const existingIndex = selectedNodes.value.indexOf(nodeIndex);
    if (existingIndex !== -1) selectedNodes.value.splice(existingIndex, 1);
    else selectedNodes.value.length < 2 ? selectedNodes.value.push(nodeIndex) : (selectedNodes.value = [nodeIndex]);
}

const onNodeRightClick = (nodeIndex, event) => {
    // Only delete node if in Layout Mode and NO modifiers are held
    if (layoutModeActive.value && !isAKeyPressed.value && !isDKeyPressed.value && !isEKeyPressed.value) {
        event.preventDefault(); 
        deleteNode(nodeIndex);
    }
}

const handleSvgMouseMove = (event) => {
  if (!svgOverlayRef.value || !layoutModeActive.value) return
  const { left, top } = svgOverlayRef.value.getBoundingClientRect()
  const mouseX = event.clientX - left
  const mouseY = event.clientY - top

  // 1. Region Labeling Mode ('E' held)
  if (isEKeyPressed.value) {
    let newHoveredTextlineId = null
    for (let i = 0; i < workingGraph.nodes.length; i++) {
      const node = workingGraph.nodes[i]
      if (Math.hypot(mouseX - scaleX(node.x), mouseY - scaleY(node.y)) < NODE_HOVER_RADIUS) {
        newHoveredTextlineId = nodeToTextlineMap.value[i]
        break 
      }
    }
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
    if (hoveredTextlineId.value !== null) labelTextline()
    return
  }

  // 2. Edge Deletion ('D' held)
  if (isDKeyPressed.value) {
      handleEdgeHoverDelete(mouseX, mouseY)
      return
  }

  // 3. Edge Connection ('A' held)
  if (isAKeyPressed.value) {
      handleNodeHoverCollect(mouseX, mouseY)
      return
  }

  // 4. Standard Layout Mode (Dragging Edge Selection)
  if (selectedNodes.value.length === 1) tempEndPoint.value = { x: mouseX, y: mouseY }
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
  
  // Mode Switching Hotkeys
  if (key === 'w' && !e.repeat) { e.preventDefault(); setMode('layout'); return }
  if (key === 't' && !e.repeat) { e.preventDefault(); requestSwitchToRecognition(); return }
  
  // Layout Mode Modifiers
  if (layoutModeActive.value && !e.repeat) {
      if (key === 'e') { e.preventDefault(); isEKeyPressed.value = true; return }
      if (key === 'd') { e.preventDefault(); isDKeyPressed.value = true; resetSelection(); return }
      if (key === 'a') { e.preventDefault(); isAKeyPressed.value = true; hoveredNodesForMST.clear(); resetSelection(); return }
  }
}

const handleGlobalKeyUp = (e) => {
  const key = e.key.toLowerCase()
  if (layoutModeActive.value) {
      if (key === 'e') {
        isEKeyPressed.value = false
        textboxLabels.value++ 
      }
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

    modifications.value = []
    error.value = null
  } catch (err) {
    error.value = err.message
    throw err
  }
}


const requestSwitchToRecognition = async () => {
    if (recognitionModeActive.value) return;

    // 1. Show the unified "Processing..." overlay
    isProcessingSave.value = true;

    try {
        if (modifications.value.length > 0) {
            await saveModifications(); 
        }

        // Silent refresh of data before switching
        await fetchPageData(localManuscriptName.value, localCurrentPage.value, true);
        
        setMode('recognition');

    } catch (e) {
        alert("Error switching mode: " + e.message);
    } finally {
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
        layoutModeActive.value = false;
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
    stroke-width: 0;
}
.polygon-inactive:hover {
    stroke: rgba(255,255,255,0.6);
    stroke-width: 0;
}
.polygon-active {
    pointer-events: none; 
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
.media-container { width: 200px; height: 200px; background: #000; border: 1px solid #444; flex-shrink: 0; position: relative; }
.tutorial-video { width: 100%; height: 100%; object-fit: contain; }
.instructions-container { flex-grow: 1; max-width: 700px; overflow-y: auto; color: #ccc; }
.instructions-container h3 { color: #fff; margin-top: 0; }
.instructions-container h4 { color: #ddd; margin-bottom: 5px; margin-top: 0; }
code { background: #424242; color: #ffb74d; padding: 2px 4px; border-radius: 3px; font-family: monospace; }
.webm-placeholder { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #777; background: #3a3a3a; }

/* Sidebar Log */
.log-sidebar { width: 200px; background: #222; border: 1px solid #444; display: flex; flex-direction: column; }
.log-header { padding: 8px 10px; background: #333; border-bottom: 1px solid #444; display: flex; justify-content: space-between; }
.log-list { list-style: none; padding: 0; margin: 0; overflow-y: auto; max-height: 120px; }
.log-list li { padding: 6px 10px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; color: #aaa; }
.undo-icon { background: none; color: #777; font-size: 1.1rem; }
.undo-icon:hover { color: #fff; }

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

.confidence-strip {
    background: rgba(0,0,0,0.6);
    padding: 4px 12px;
    border-radius: 4px;
    white-space: pre; 
    pointer-events: none; 
    display: flex;
    flex-wrap: wrap;
    margin-top: -2px; 
    border: 1px solid #333;
}

.conf-char {
    display: inline-block;
    font-family: monospace; 
}

/* New Help Grid Styles */
.help-section.full-width {
  width: 100%;
}

.help-grid {
  display: flex;
  width: 100%;
  height: 100%;
  gap: 20px;
  justify-content: space-evenly;
  align-items: center;
}

.help-card {
  flex: 1;
  max-width: 300px;
  height: 100%;
  background: #252525;
  border: 1px solid #3d3d3d;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition: transform 0.2s;
}

.help-card:hover {
  border-color: #555;
  background: #2a2a2a;
}

.media-container-small {
  width: 100%;
  height: 110px; /* Fixed height for video area */
  background: #000;
  border-bottom: 1px solid #333;
  display: flex;
  align-items: center;
  justify-content: center;
}

.card-text {
  padding: 12px;
  text-align: center;
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.card-text h4 {
  margin: 0 0 8px 0;
  color: #fff;
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.card-text p {
  margin: 4px 0;
  font-size: 0.85rem;
  color: #ccc;
}

.key-badge {
  background: #424242;
  color: #ffb74d;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: monospace;
  font-weight: bold;
  border: 1px solid #555;
}
</style>




# app.py

# app.py
import threading  # <--- CRITICAL IMPORT
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import shutil
from pathlib import Path
import base64
import json
import zipfile
import io
import google.generativeai as genai
import glob
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
from os.path import isdir, join
import collections
import math
import difflib
from dotenv import load_dotenv
import concurrent.futures
load_dotenv() 

# from recognition.recognize_manuscript_text import recognize_manuscript_text
# cd recognition
# python recognize_manuscript_text.py complex_layout



# Import your existing pipelines
from inference import process_new_manuscript
from gnn_inference import run_gnn_prediction_for_page, generate_xml_and_images_for_page
from segmentation.utils import load_images_from_folder

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './input_manuscripts'
MODEL_CHECKPOINT = "./pretrained_gnn/v2.pt"
DATASET_CONFIG = "./pretrained_gnn/gnn_preprocessing_v2.yaml"

def parse_page_xml_polygons(xml_path):
    """
    Parses PAGE-XML to extract polygon coordinates for each textline.
    Returns: Dict { structure_line_id: [[x,y], [x,y], ...] }
    """
    polygons = {}
    if not os.path.exists(xml_path):
        return polygons

    try:
        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' not in custom_attr:
                continue
            
            try:
                line_id = str(custom_attr.split('structure_line_id_')[1])
            except IndexError:
                continue

            coords_elem = textline.find('p:Coords', ns)
            if coords_elem is not None:
                points_str = coords_elem.get('points', '')
                if points_str:
                    # Convert "x,y x,y" -> [[x,y], [x,y]]
                    points = [list(map(int, p.split(','))) for p in points_str.strip().split(' ')]
                    polygons[line_id] = points
                    
            # Also try to grab existing text if any
            text_equiv = textline.find('p:TextEquiv/p:Unicode', ns)
            # We can optionally return this, but the main logic relies on the separate dict
            
    except Exception as e:
        print(f"Error parsing XML polygons: {e}")
        
    return polygons


def get_existing_text_content(xml_path):
    """
    Parses PAGE-XML to extract Unicode text AND confidence scores.
    Returns a dict with 'text' and 'confidences' maps.
    """
    text_content = {}
    confidences = {}
    
    if not os.path.exists(xml_path):
        return {"text": {}, "confidences": {}}
        
    try:
        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' in custom_attr:
                try:
                    line_id = str(custom_attr.split('structure_line_id_')[1])
                except IndexError:
                    continue
                
                # Extract Text
                text_equiv = textline.find('p:TextEquiv', ns)
                if text_equiv is not None:
                    uni = text_equiv.find('p:Unicode', ns)
                    if uni is not None and uni.text:
                        text_content[line_id] = uni.text
                        
                        # Extract Confidence from 'custom' attribute of TextEquiv
                        # Format expected: "confidences:0.9,0.5,..."
                        te_custom = text_equiv.get('custom', '')
                        if 'confidences:' in te_custom:
                            try:
                                # Split by 'confidences:' and take the part after it
                                # Then split by ';' in case there is other data
                                raw_conf = te_custom.split('confidences:')[1].split(';')[0]
                                if raw_conf.strip():
                                    confidences[line_id] = [float(x) for x in raw_conf.split(',')]
                            except Exception:
                                pass
    except Exception as e:
        print(f"Error parsing existing text: {e}")
    
    return {"text": text_content, "confidences": confidences}


@app.route('/upload', methods=['POST'])
def upload_manuscript():
    manuscript_name = request.form.get('manuscriptName', 'default_manuscript')
    longest_side = int(request.form.get('longestSide', 2500))
    min_distance = int(request.form.get('minDistance', 20)) 
    
    manuscript_path = os.path.join(UPLOAD_FOLDER, manuscript_name)
    images_path = os.path.join(manuscript_path, "images")
    
    if os.path.exists(manuscript_path):
        shutil.rmtree(manuscript_path)
    os.makedirs(images_path)

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    for file in files:
        if file.filename:
            file.save(os.path.join(images_path, file.filename))

    try:
        process_new_manuscript(manuscript_path, target_longest_side=longest_side, min_distance=min_distance) 
        processed_pages = []
        for f in sorted(Path(manuscript_path).glob("gnn-dataset/*_dims.txt")):
            processed_pages.append(f.name.replace("_dims.txt", ""))
            
        return jsonify({"message": "Processed successfully", "pages": processed_pages})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/manuscript/<name>/pages', methods=['GET'])
def get_pages(name):
    manuscript_path = Path(UPLOAD_FOLDER) / name / "gnn-dataset"
    if not manuscript_path.exists():
        return jsonify([]), 404
    pages = sorted([f.name.replace("_dims.txt", "") for f in manuscript_path.glob("*_dims.txt")])
    return jsonify(pages)

# ----------------------------------------------------------------
# 2. UPDATE: Endpoint to return the new data fields
# ----------------------------------------------------------------
@app.route('/semi-segment/<manuscript>/<page>', methods=['GET'])
def get_page_prediction(manuscript, page):
    # print("Received request for manuscript:", manuscript, "page:", page)
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    try:
        # 1. Run GNN Inference / Get Graph Data
        graph_data = run_gnn_prediction_for_page(
            str(manuscript_path), 
            page, 
            MODEL_CHECKPOINT, 
            DATASET_CONFIG
        )
        
        # 2. Get Image
        img_path = manuscript_path / "images_resized" / f"{page}.jpg"
        encoded_string = ""
        if img_path.exists():
            with open(img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # 3. CRITICAL: Load Existing XML Data
        xml_path = manuscript_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
        polygons = {}
        existing_data = {"text": {}, "confidences": {}}
        
        if xml_path.exists():
            polygons = parse_page_xml_polygons(str(xml_path))
            existing_data = get_existing_text_content(str(xml_path))

        response = {
            "image": encoded_string,
            "dimensions": graph_data['dimensions'],
            "points": [[n['x'], n['y']] for n in graph_data['nodes']],
            "graph": graph_data,
            "textline_labels": graph_data.get('textline_labels', []),
            "textbox_labels": graph_data.get('textbox_labels', []),
            
            # Data for Recognition Mode
            "polygons": polygons, 
            "textContent": existing_data["text"],           # <--- Must match frontend expectation
            "textConfidences": existing_data["confidences"] # <--- Must match frontend expectation
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def ensemble_text_samples(samples):
    """
    Performs character-level ensemble.
    Returns: (consensus_string, confidence_scores_list)
    """
    valid_samples = [s for s in samples if s and s.strip()]
    if not valid_samples:
        return "", []
    if len(valid_samples) == 1:
        # Single sample = 100% confidence for all chars
        return valid_samples[0], [1.0] * len(valid_samples[0])

    valid_samples.sort(key=len)
    pivot_idx = len(valid_samples) // 2
    pivot = valid_samples[pivot_idx]
    
    GAP_TOKEN = "__GAP__"
    total_samples = len(valid_samples) # N
    
    grid = [collections.Counter({char: 1}) for char in pivot]
    insertions = collections.defaultdict(collections.Counter)
    
    others = valid_samples[:pivot_idx] + valid_samples[pivot_idx+1:]
    
    for sample in others:
        matcher = difflib.SequenceMatcher(None, pivot, sample)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    grid[i1 + k][pivot[i1 + k]] += 1
            elif tag == 'replace':
                len_pivot_seg = i2 - i1
                len_sample_seg = j2 - j1
                min_len = min(len_pivot_seg, len_sample_seg)
                for k in range(min_len):
                    grid[i1 + k][sample[j1 + k]] += 1
                for k in range(min_len, len_pivot_seg):
                    grid[i1 + k][GAP_TOKEN] += 1
                if len_sample_seg > len_pivot_seg:
                    inserted_chunk = sample[j1 + min_len : j2]
                    insertions[i2 - 1][inserted_chunk] += 1
            elif tag == 'delete':
                for k in range(i2 - i1):
                    grid[i1 + k][GAP_TOKEN] += 1
            elif tag == 'insert':
                inserted_chunk = sample[j1:j2]
                target_idx = i1 - 1
                insertions[target_idx][inserted_chunk] += 1

    result_chars = []
    result_confidences = []

    def append_result(char_str, vote_count):
        conf = round(vote_count / total_samples, 2)
        # For multi-char chunks (insertions), append conf for each char
        for c in char_str:
            result_chars.append(c)
            result_confidences.append(conf)

    # Handle start insertions
    if -1 in insertions:
        best_ins, count = insertions[-1].most_common(1)[0]
        # Heuristic: Add +1 to count for the pivot's implied "nothing here" if needed, 
        # but generally insertions are voted by 'others'. Let's trust the count.
        # We assume the pivot voted "nothing", so count is strictly from others.
        append_result(best_ins, count)

    for i in range(len(pivot)):
        # Pivot position
        best_char, count = grid[i].most_common(1)[0]
        if best_char != GAP_TOKEN:
            append_result(best_char, count)
            
        # Insertions after
        if i in insertions:
            best_ins, count = insertions[i].most_common(1)[0]
            append_result(best_ins, count)

    return "".join(result_chars), result_confidences


def _run_gemini_recognition_internal(manuscript, page, api_key, N=5, num_trace_points=4):
    """
    Improved drop-in replacement.
    - Parallelizes N samples.
    - Configurable point density for polylines.
    - Aligned with Gemini's spatial grounding CoT.
    """
    print(f"[{page}] Starting parallel recognition with N={N}, points={num_trace_points}...")
    
    # Setup paths
    base_path = Path(UPLOAD_FOLDER) / manuscript
    xml_path = base_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    img_path = base_path / "images_resized" / f"{page}.jpg"

    if not xml_path.exists() or not img_path.exists():
        return {}

    try:
        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
        
        ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # --- 1. GEOMETRY EXTRACTION & INTERPOLATION ---
        def get_equidistant_points(pts, m):
            if len(pts) < 2: return pts * m
            
            # Cumulative distance along the original polyline
            dists = [0.0]
            for i in range(len(pts)-1):
                dists.append(dists[-1] + ((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2)**0.5)
            
            total_dist = dists[-1]
            if total_dist == 0: return [pts[0]] * m
            
            new_pts = []
            for i in range(m):
                target = (i / (m - 1)) * total_dist
                # Find which segment the target distance falls into
                for j in range(len(dists)-1):
                    if dists[j] <= target <= dists[j+1]:
                        segment_dist = dists[j+1] - dists[j]
                        # Linear interpolation within the segment
                        rat = (target - dists[j]) / segment_dist if segment_dist > 0 else 0
                        
                        # FIX: Use 'j' for segment indexing, not 'i'
                        nx = pts[j][0] + rat * (pts[j+1][0] - pts[j][0])
                        ny = pts[j][1] + rat * (pts[j+1][1] - pts[j][1])
                        new_pts.append([int(nx), int(ny)])
                        break
            return new_pts

        lines_geometry = [] 
        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            if 'structure_line_id_' not in custom_attr: continue
            line_id = str(custom_attr.split('structure_line_id_')[1])

            # Geometry extraction (Baseline is primary for Sanskrit)
            base_elem = textline.find('p:Baseline', ns)
            if base_elem is not None and base_elem.get('points'):
                pts = [list(map(int, p.split(','))) for p in base_elem.get('points').strip().split(' ')]
                pts.sort(key=lambda k: k[0])
            else: continue

            # Orientation & Thickness Logic
            coords_elem = textline.find('p:Coords', ns)
            poly_pts = [list(map(int, p.split(','))) for p in coords_elem.get('points').strip().split(' ')] if coords_elem is not None else []
            
            if poly_pts:
                pxs, pys = [p[0] for p in poly_pts], [p[1] for p in poly_pts]
                width_px, height_px = max(pxs)-min(pxs), max(pys)-min(pys)
                is_vert = height_px > (width_px * 1.2)
                thickness = width_px if is_vert else height_px
            else:
                is_vert, thickness = False, 30

            lines_geometry.append({
                "id": line_id, "baseline": pts, 
                "thickness": max(10, min(thickness, 20)), "is_vertical": is_vert
            })

        if not lines_geometry: return {}

        # --- 2. PARALLEL API EXECUTION ---
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-flash') # Or gemini-3-flash

        def normalize(x, y):
            return max(0, min(1000, int((y / img_h) * 1000))), max(0, min(1000, int((x / img_w) * 1000)))

        def sample_worker(sample_idx):
            # Centered at 0.0, ranging from -0.3 to +0.3 (total 60% of thickness)
            shift_ratios = [0.0] if N == 1 else [(i / (N - 1) - 0.5) * 0.6 for i in range(N)]
            current_shift = shift_ratios[sample_idx]
            
            regions_payload = []
            for line in lines_geometry:
                # Interpolate N points
                trace_raw = get_equidistant_points(line['baseline'], num_trace_points)
                shift_px = int(line['thickness'] * current_shift)
                
                # Apply shift based on orientation
                shifted = [[px + shift_px, py] if line['is_vertical'] else [px, py + shift_px] for px, py in trace_raw]
                
                # Format for Gemini: flat list [y, x, y, x...]
                gemini_trace = []
                for px, py in shifted:
                    ny, nx = normalize(px, py)
                    gemini_trace.extend([ny, nx])
                
                regions_payload.append({"id": line['id'], "trace": gemini_trace, "y": trace_raw[0][1]})

            regions_payload.sort(key=lambda k: k['y'])

            # Improved Prompt: Aligning with Autoregressive Spatial Grounding
            prompt_text = (
                "You are an expert paleographer and OCR engine specialized in historical Sanskrit manuscripts.\n"
                "I have provided an image of a manuscript page. Your task is to perform visual grounding OCR: "
                "transcribe the handwritten Devanagari text found at specific spatial locations defined by 'Path Traces'.\n"
                "The coordinates are normalized on a 0-1000 scale (where [0,0] is top-left and [1000,1000] is bottom-right) "
                "to precisely map the text line locations on the image.\n"
                "For each path trace [y_start, x_start, y_mid1, x_mid1, y_mid2, x_mid2, y_end, x_end], transcribe the text that sits along this curve.\n"
                "Focus strictly on the visual line indicated by the trace; ignore text from lines above or below.\n"
                "Output a JSON array of objects with 'id' and 'text'.\n\n"
                "REGIONS:\n"
            )
            for item in regions_payload:
                prompt_text += f"ID: {item['id']} | Trace: {item['trace']}\n"

            try:
                # Place image before text for better multimodal attention in Flash models
                response = model.generate_content(
                    [pil_img, prompt_text],
                    generation_config={"response_mime_type": "application/json", "temperature": 0.2}
                )
                data = json.loads(response.text)
                # Normalize response format
                if isinstance(data, dict) and "transcriptions" in data: data = data["transcriptions"]
                return {str(i['id']): str(i['text']).strip() for i in data if 'id' in i}
            except Exception as e:
                print(f"Sample {sample_idx} error: {e}")
                return None

        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
            future_to_idx = {executor.submit(sample_worker, i): i for i in range(N)}
            all_samples_results = [f.result() for f in concurrent.futures.as_completed(future_to_idx) if f.result()]

        # --- 3. CHARACTER-LEVEL ENSEMBLE ---
        final_map = {}
        final_confidences = {} # Map: line_id -> list of floats
        
        texts_by_id = collections.defaultdict(list)
        for res_map in all_samples_results:
            for lid, txt in res_map.items():
                texts_by_id[lid].append(txt)

        for lid, candidates in texts_by_id.items():
            # Get text AND scores
            consensus_text, scores = ensemble_text_samples(candidates)
            if consensus_text:
                final_map[lid] = consensus_text
                final_confidences[lid] = scores
                if len(set(candidates)) > 1 and N > 1:
                    print(f"[{page}] Line {lid}: Merged {len(candidates)} samples. " 
                          f"Result: {consensus_text[:15]}... (Variants: {len(set(candidates))})")

        # --- 4. UPDATE XML ---
        if final_map:
            changed = False
            for textline in root.findall(".//p:TextLine", ns):
                custom_attr = textline.get('custom', '')
                if 'structure_line_id_' in custom_attr:
                    lid = str(custom_attr.split('structure_line_id_')[1])
                    if lid in final_map:
                        te = textline.find("p:TextEquiv", ns)
                        if te is None: te = ET.SubElement(textline, "TextEquiv")
                        uni = te.find("p:Unicode", ns)
                        if uni is None: uni = ET.SubElement(te, "Unicode")
                        uni.text = final_map[lid]

                        if lid in final_confidences:
                            conf_str = ",".join(map(str, final_confidences[lid]))
                            current_custom = te.get('custom', '')
                            new_custom = f"confidences:{conf_str}" 
                            te.set('custom', new_custom)
                        changed = True
            
            if changed:
                tree.write(xml_path, encoding='UTF-8', xml_declaration=True)
                print(f"[{page}] XML updated with robust ensemble text.")

        return { "text": final_map, "confidences": final_confidences }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Internal Recognition Error: {e}")
        return {}


@app.route('/existing-manuscripts', methods=['GET'])
def list_existing_manuscripts():
    if not os.path.exists(UPLOAD_FOLDER):
        return jsonify([])
    
    manuscripts = [
        d for d in os.listdir(UPLOAD_FOLDER) 
        if isdir(join(UPLOAD_FOLDER, d)) and not d.startswith('.')
    ]
    return jsonify(sorted(manuscripts))


@app.route('/semi-segment/<manuscript>/<page>', methods=['POST'])
def save_correction(manuscript, page):
    data = request.json
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    
    textline_labels = data.get('textlineLabels')
    graph_data = data.get('graph')
    textbox_labels = data.get('textboxLabels')
    nodes_data = graph_data.get('nodes')
    text_content = data.get('textContent') 
    
    # Flags for background processing
    run_recognition = data.get('runRecognition', False)
    api_key = data.get('apiKey', None) #not used, unsafe

    if not textline_labels or not graph_data:
        return jsonify({"error": "Missing labels or graph data"}), 400

    try:
        # 1. SAVE LAYOUT (Synchronous)
        # We must wait for this to finish so the XML/Images exist for the thread to read.
        # This is usually very fast (<200ms).
        result = generate_xml_and_images_for_page(
            str(manuscript_path),
            page,
            textline_labels,
            graph_data['edges'],
            { 
                'BINARIZE_THRESHOLD': 0.5098,
                'BBOX_PAD_V': 0.7,
                'BBOX_PAD_H': 0.5,
                'CC_SIZE_THRESHOLD_RATIO': 0.4
            },
            textbox_labels=textbox_labels,
            nodes=nodes_data,
            text_content=text_content 
        )

         # 2. TRIGGER RECOGNITION (Asynchronous)
        # FIX: Run if requested, even if apiKey not sent (internal function checks os.getenv)
        if run_recognition: 
            
            # Wrapper to log start/finish in backend console
            def background_task(m, p, k):
                _run_gemini_recognition_internal(m, p, k)

            # Spawn the thread
            # Pass None for api_key to force internal function to use .env
            thread = threading.Thread(target=background_task, args=(manuscript, page, None), daemon=True)
            thread.start()
            
            # Let the frontend know we started it, but don't wait for the result
            result['autoRecognitionStatus'] = "processing_in_background"

        # Return immediately to unblock the UI
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------------
# MODIFIED: Existing Recognize Endpoint (Wrapper)
# ----------------------------------------------------------------
@app.route('/recognize-text', methods=['POST'])
def recognize_text():
    data = request.json
    manuscript = data.get('manuscript')
    page = data.get('page')
    api_key = data.get('apiKey')
    
    if not api_key:
        return jsonify({"error": "API Key required"}), 400

    # Use the shared internal function
    result = _run_gemini_recognition_internal(manuscript, page, api_key)
    return jsonify(result)

@app.route('/save-graph/<manuscript>/<page>', methods=['POST'])
def save_generated_graph(manuscript, page):
    return jsonify({"status": "ok"})




@app.route('/download-results/<manuscript>', methods=['GET'])
def download_results(manuscript):
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript / "layout_analysis_output"
    if not manuscript_path.exists():
         return jsonify({"error": "No output found for this manuscript"}), 404
    xml_dir = manuscript_path / "page-xml-format"
    img_dir = manuscript_path / "image-format"
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        if xml_dir.exists():
            for root, dirs, files in os.walk(xml_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('page-xml-format', os.path.relpath(file_path, xml_dir))
                    zf.write(file_path, arcname)
        if img_dir.exists():
            for root, dirs, files in os.walk(img_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('image-format', os.path.relpath(file_path, img_dir))
                    zf.write(file_path, arcname)

    memory_file.seek(0)
    return send_file(
        memory_file, 
        mimetype='application/zip', 
        as_attachment=True, 
        download_name=f'{manuscript}_results.zip'
    )



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  


# App.vue

<template>
  <div class="app-container">
    <div v-if="!currentManuscript">
      <!-- Upload Screen -->
      <div class="upload-card">
        <h1>Historical Manuscript Segmentation</h1>
        <div class="section-divider">
            <h3>Load Existing Project</h3>
            <div class="load-group">
                <select v-model="selectedExisting" class="dropdown">
                    <option value="" disabled>Select a manuscript...</option>
                    <option v-for="m in existingManuscripts" :key="m" :value="m">{{ m }}</option>
                </select>
                <button @click="loadExisting" :disabled="!selectedExisting" class="load-btn">Load</button>
            </div>
        </div>

        <hr class="separator" />

        <!-- Existing Upload Form -->
        <h3>New Project Upload</h3>
        <div class="form-group">
          <label>Manuscript Name:</label>
          <input v-model="formName" type="text" placeholder="e.g. manuscript_1" />
        </div>



        <div class="form-group">
          <label>Resize Dimension (Longest Side):</label>
          <input v-model.number="formLongestSide" type="number" />
        </div>
        <!-- NEW FIELD -->
        <div class="form-group">
          <label>Min Distance (Peak Detection):</label>
          <input v-model.number="formMinDistance" type="number" title="Distance between char centers" />
        </div>
        <!-- END NEW FIELD -->
        <div class="form-group">
          <label>Images:</label>
          <input type="file" multiple @change="handleFileChange" accept="image/*" />
        </div>
        <button @click="upload" :disabled="uploading">
          {{ uploading ? 'Processing (Step 1-3)...' : 'Start Processing' }}
        </button>
        <div v-if="uploadStatus" class="status">{{ uploadStatus }}</div>
      </div>
    </div>

    <div v-else>
      <!-- Main Workstation -->
      <!-- REMOVED: The floating .back-btn is gone. We handle it via the event below -->
      <ManuscriptViewer 
        :manuscriptName="currentManuscript" 
        :pageName="currentPage"
        @page-changed="handlePageChange"
        @back="currentManuscript = null" 
      />
    </div>
  </div>
</template>

<script setup>

import { ref, onMounted } from 'vue' // Added onMounted
import ManuscriptViewer from './components/ManuscriptViewer.vue'

// Basic State
const currentManuscript = ref(null)
const currentPage = ref(null)
const pageList = ref([])

// Upload Form State
const formName = ref('my_manuscript')
const formLongestSide = ref(2500)
const formMinDistance = ref(20) // NEW STATE
const selectedFiles = ref([])
const uploading = ref(false)
const uploadStatus = ref('')
const existingManuscripts = ref([])
const selectedExisting = ref('')

// NEW: Fetch existing on mount
onMounted(async () => {
    try {
        const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/existing-manuscripts`)
        if(res.ok) {
            existingManuscripts.value = await res.json()
        }
    } catch(e) {
        console.error("Failed to load existing manuscripts", e)
    }
})

const handleFileChange = (e) => {
  selectedFiles.value = Array.from(e.target.files)
}

const upload = async () => {
  if (selectedFiles.value.length === 0) return alert('Select files')
  uploading.value = true
  uploadStatus.value = 'Uploading and generating heatmaps/points. This may take a while...'

  const formData = new FormData()
  formData.append('manuscriptName', formName.value)
  formData.append('longestSide', formLongestSide.value)
  formData.append('minDistance', formMinDistance.value) // NEW
  selectedFiles.value.forEach(file => formData.append('images', file))

  try {
    const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/upload`, {
      method: 'POST',
      body: formData
    })
    if(!res.ok) throw new Error('Upload failed')
    const data = await res.json()
    
    // Success - Switch to viewer
    pageList.value = data.pages
    if (pageList.value.length > 0) {
      currentManuscript.value = formName.value
      currentPage.value = pageList.value[0]
    } else {
      uploadStatus.value = 'No pages processed.'
    }
  } catch (e) {
    uploadStatus.value = 'Error: ' + e.message
  } finally {
    uploading.value = false
  }
}

const handlePageChange = (newPage) => {
  currentPage.value = newPage
}

// NEW: Load Logic
const loadExisting = async () => {
    if(!selectedExisting.value) return
    
    // Fetch pages for this manuscript
    try {
        const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/manuscript/${selectedExisting.value}/pages`)
        if(!res.ok) throw new Error("Could not load pages")
        const pages = await res.json()
        
        if(pages.length > 0) {
            pageList.value = pages
            currentManuscript.value = selectedExisting.value
            currentPage.value = pages[0]
        } else {
            alert("This manuscript has no processed pages.")
        }
    } catch(e) {
        alert(e.message)
    }
}

const resetSelection = () => {
    currentManuscript.value = null
    // Refresh list on back
    onMounted() 
}
</script>

<style>
body { margin: 0; font-family: sans-serif; background: #222; color: white; }
.app-container { display: flex; flex-direction: column; height: 100vh; }
.upload-card { max-width: 500px; margin: 100px auto; padding: 20px; background: #333; border-radius: 8px; }
.form-group { margin-bottom: 15px; display: flex; flex-direction: column; }
input { padding: 8px; background: #444; border: 1px solid #555; color: white; margin-top: 5px; }
button { padding: 10px; background: #4CAF50; color: white; border: none; cursor: pointer; }
button:disabled { background: #555; }

.section-divider { margin-bottom: 20px; }
.separator { margin: 20px 0; border: 0; border-top: 1px solid #555; }
.load-group { display: flex; gap: 10px; }
.dropdown { flex-grow: 1; padding: 10px; background: #444; color: white; border: 1px solid #555; }
.load-btn { background: #2196F3; }
</style>

# inference.py

import os
import argparse
import gc
from PIL import Image
import torch

from segmentation.segment_graph import images2points



def process_new_manuscript(manuscript_path, target_longest_side=2500, min_distance=20):
    source_images_path = os.path.join(manuscript_path, "images")
    # We will save processed (and potentially resized) images here
    # to avoid modifying source files while iterating over them.
    resized_images_path = os.path.join(manuscript_path, "images_resized")

    try:
        # Create the target folder
        os.makedirs(resized_images_path, exist_ok=True)
        
        # Verify source exists
        if not os.path.exists(source_images_path):
            print(f"Error: Source directory {source_images_path} not found.")
            return

    except Exception as e:
        print(f"An error occurred setting up directories: {e}")
        return

    # Valid image extensions to look for
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    # Get list of files in the directory
    files = [f for f in os.listdir(source_images_path) if os.path.isfile(os.path.join(source_images_path, f))]

    print(f"Found {len(files)} files in {source_images_path}...")

    for filename in files:
        # Skip non-image files based on extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue

        base_filename = os.path.splitext(filename)[0]
        file_path = os.path.join(source_images_path, filename)

        try:
            # Open the image from the folder
            with Image.open(file_path) as image:
                
                width, height = image.size
                
                # 1. VALIDATION: Check if image is too small for CV tasks
                # If both dimensions are smaller than 600, we reject the image.
                if width < 600 and height < 600:
                    raise ValueError(f"Image resolution too low ({width}x{height}). Both dimensions are < 600px.")

                
                # Check if the longest side exceeds the target
                if max(width, height) > target_longest_side:
                    
                    # Calculate scaling factor
                    scale_factor = target_longest_side / max(width, height)
                    
                    # Calculate new dimensions
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    # Handle Resampling filter compatibility
                    try:
                        resampling_filter = Image.Resampling.LANCZOS
                    except AttributeError:
                        resampling_filter = Image.LANCZOS

                    print(f"Downscaling '{filename}': ({width}x{height}) -> ({new_width}x{new_height})")
                    image = image.resize((new_width, new_height), resampling_filter)
                    
                else:
                    print(f"Image '{filename}' is within limits ({width}x{height}). Keeping original size.")
                    

                # Standardize Color Mode
                if image.mode in ("RGBA", "P", "LA"):
                    image = image.convert("RGB")

                # Save processed image to the NEW folder
                new_filename = f"{base_filename}.jpg"
                save_path = os.path.join(resized_images_path, new_filename)
                
                image.save(save_path, "JPEG")
                print(f"Processed: {new_filename}")

        except Exception as img_err:
            # This block catches the ValueError raised above and prints the message
            print(f"Failed to process image {filename}: {img_err}")
            continue

    # Point the inference function to the new resized/processed folder
    print("Running images2points on processed folder...")
    # --- MODIFIED: Pass min_distance ---
    images2points(resized_images_path, min_distance=min_distance) 
    
    # Cleanup resources
    torch.cuda.empty_cache()
    gc.collect()

    print("Processing complete.")







# gnn_inference.py
import torch
import numpy as np
import yaml
import logging
import shutil
from pathlib import Path
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data
import cv2
from datetime import datetime

# gnn_inference.py
import os
from collections import defaultdict
from gnn_data_preparation.utils import setup_logging
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import xml.etree.ElementTree as ET

from segment_from_point_clusters import segmentLinesFromPointClusters
from gnn_data_preparation.config_models import DatasetCreationConfig
from gnn_data_preparation.graph_constructor import create_input_graph_edges
from gnn_data_preparation.feature_engineering import get_node_features, get_edge_features

# Global Cache
LOADED_MODEL = None
LOADED_CONFIG = None
DEVICE = None

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_once(model_checkpoint_path, config_path):
    global LOADED_MODEL, LOADED_CONFIG, DEVICE
    if LOADED_MODEL is None:
        DEVICE = get_device()
        print(f"Loading model from {model_checkpoint_path} on {DEVICE}...")
        checkpoint = torch.load(model_checkpoint_path, map_location=DEVICE, weights_only=False)
        LOADED_MODEL = checkpoint['model']
        LOADED_MODEL.to(DEVICE)
        LOADED_MODEL.eval()
        
        with open(config_path, 'r') as f:
            LOADED_CONFIG = DatasetCreationConfig(**yaml.safe_load(f))
    return LOADED_MODEL, LOADED_CONFIG, DEVICE

def generate_xml_and_images_for_page(manuscript_path, page_id, node_labels, graph_edges, args_dict, textbox_labels=None, nodes=None, text_content=None):
    """
    Saves user corrections and regenerates XML.
    Handles coordinate scaling: Frontend (Image Space) -> Storage (Heatmap Space).
    """
    base_path = Path(manuscript_path)
    raw_input_dir = base_path / "gnn-dataset"
    output_dir = base_path / "layout_analysis_output"
    gnn_format_dir = output_dir / "gnn-format"
    gnn_format_dir.mkdir(parents=True, exist_ok=True)
    
    # ... [Loading Heatmap Dimensions and Scaling Logic remains exactly the same] ...
    # ... (lines 66-136 in original code) ...
    raw_dims_path = raw_input_dir / f"{page_id}_dims.txt"
    if not raw_dims_path.exists():
        raw_dims_path = gnn_format_dir / f"{page_id}_dims.txt"
    dims = np.loadtxt(raw_dims_path) 
    heatmap_w, heatmap_h = dims[0], dims[1]
    max_dim_heatmap = max(heatmap_w, heatmap_h)

    points_unnormalized = []
    points_normalized = []

    if nodes is not None:
        scale_factor = 0.5 
        for n in nodes:
            img_x, img_y, img_s = float(n['x']), float(n['y']), float(n['s'])
            hm_x, hm_y, hm_s = img_x * scale_factor, img_y * scale_factor, img_s * scale_factor
            points_unnormalized.append([hm_x, hm_y, hm_s])
            norm_x, norm_y, norm_s = hm_x / max_dim_heatmap, hm_y / max_dim_heatmap, hm_s / max_dim_heatmap
            points_normalized.append([norm_x, norm_y, norm_s])
            
        points_unnormalized = np.array(points_unnormalized)
        points_normalized = np.array(points_normalized)
        np.savetxt(gnn_format_dir / f"{page_id}_inputs_unnormalized.txt", points_unnormalized, fmt='%f')
        np.savetxt(gnn_format_dir / f"{page_id}_inputs_normalized.txt", points_normalized, fmt='%f')
        if raw_dims_path.exists():
            shutil.copy(raw_dims_path, gnn_format_dir / f"{page_id}_dims.txt")
    else:
        if not (gnn_format_dir / f"{page_id}_inputs_unnormalized.txt").exists():
            for suffix in ["_inputs_normalized.txt", "_inputs_unnormalized.txt", "_dims.txt"]:
                src = raw_input_dir / f"{page_id}{suffix}"
                dst = gnn_format_dir / f"{page_id}{suffix}"
                if src.exists(): shutil.copy(src, dst)
        points_unnormalized = np.loadtxt(gnn_format_dir / f"{page_id}_inputs_unnormalized.txt")
        if points_unnormalized.size == 0:
            points_unnormalized = np.empty((0, 3))
        elif points_unnormalized.ndim == 1: 
            points_unnormalized = points_unnormalized.reshape(1, -1)

    unique_edges = set()
    num_nodes = len(points_unnormalized)
    for e in graph_edges:
        if 'source' in e and 'target' in e:
            u, v = sorted((int(e['source']), int(e['target'])))
            if u < num_nodes and v < num_nodes:
                unique_edges.add((u, v))
            
    edges_save_path = gnn_format_dir / f"{page_id}_edges.txt"
    if unique_edges:
        np.savetxt(edges_save_path, list(unique_edges), fmt='%d')
    else:
        open(edges_save_path, 'w').close()

    if unique_edges:
        row, col = zip(*unique_edges)
        data = np.ones(len(row) + len(col))
        adj = csr_matrix((data, (list(row)+list(col), list(col)+list(row))), shape=(num_nodes, num_nodes))
    else:
        adj = csr_matrix((num_nodes, num_nodes))

    n_components, final_structural_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    np.savetxt(gnn_format_dir / f"{page_id}_labels_textline.txt", final_structural_labels, fmt='%d')

    final_textbox_labels = np.zeros(num_nodes, dtype=int)
    if textbox_labels is not None:
        if len(textbox_labels) == num_nodes:
            final_textbox_labels = np.array(textbox_labels, dtype=int)
            np.savetxt(gnn_format_dir / f"{page_id}_labels_textbox.txt", final_textbox_labels, fmt='%d')
        else:
             print(f"Warning: Textbox label count {len(textbox_labels)} != Node count {num_nodes}. Resetting.")
             
    # 5. Run Segmentation (Now returns data with images, does not save to disk)
    polygons_data = segmentLinesFromPointClusters(
        str(output_dir.parent), 
        page_id, 
        BINARIZE_THRESHOLD=args_dict.get('BINARIZE_THRESHOLD', 0.5098), 
        BBOX_PAD_V=args_dict.get('BBOX_PAD_V', 0.7), 
        BBOX_PAD_H=args_dict.get('BBOX_PAD_H', 0.5), 
        CC_SIZE_THRESHOLD_RATIO=args_dict.get('CC_SIZE_THRESHOLD_RATIO', 0.4), 
        GNN_PRED_PATH=str(output_dir)
    )

    xml_output_dir = output_dir / "page-xml-format"
    xml_output_dir.mkdir(exist_ok=True)
    
    # --- NEW: Prepare Images Directory ---
    images_output_dir = output_dir / "image-format" / page_id
    if images_output_dir.exists():
        shutil.rmtree(images_output_dir)
    images_output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Generate XML AND Save Images
    create_page_xml(
        page_id,
        unique_edges,
        points_unnormalized,
        {'width': heatmap_w, 'height': heatmap_h}, 
        xml_output_dir / f"{page_id}.xml",
        final_structural_labels, 
        polygons_data,
        textbox_labels=final_textbox_labels,
        image_path=base_path / "images_resized" / f"{page_id}.jpg",
        images_output_dir=images_output_dir,
        text_content=text_content # <--- PASS THIS DOWN
    )

    resized_images_dst_dir = output_dir / "images_resized"
    resized_images_dst_dir.mkdir(exist_ok=True)
    src_img = base_path / "images_resized" / f"{page_id}.jpg"
    if src_img.exists():
        shutil.copy(src_img, resized_images_dst_dir / f"{page_id}.jpg")

    line_count = len(polygons_data) # 1. Capture count first
    del polygons_data
    import gc
    gc.collect()

    return {"status": "success", "lines": line_count}



# ===================================================================
#           UTILITY, METRIC, AND VISUALIZATION FUNCTIONS
# ===================================================================

def fit_robust_line_and_extend(points: np.ndarray, extend_percentage: float = 0.05, robust_method: str = 'huber'):
    """
    Fits a robust line to a set of 2D points, extends it, and returns the new endpoints.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 2) with the [x, y] coordinates.
        extend_percentage (float): The percentage to extend the line by on each end.
        robust_method (str): The robust regression method to use ('huber' or 'ransac').

    Returns:
        tuple: A tuple containing two points, ((x1, y1), (x2, y2)), representing the
               start and end of the extended best-fit line.
    """
    if len(points) < 2:
        return None  # Cannot fit a line to less than two points

    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    # 1. Fit a robust regression model
    if robust_method.lower() == 'ransac':
        # RANSAC is excellent for significant outliers but computationally more expensive.
        model = RANSACRegressor(min_samples=2, residual_threshold=5.0, max_trials=100)
    elif robust_method.lower() == 'huber':
        # Huber is a good default, less sensitive to outliers than OLS.
        model = HuberRegressor(epsilon=1.35)
    else:
        raise ValueError("robust_method must be either 'ransac' or 'huber'")

    try:
        model.fit(x, y)
        y_pred = model.predict(x)
    except Exception:
        return None # Could not fit a model

    # 2. Determine the endpoints of the fitted line on the original data range
    x_min, x_max = np.min(x), np.max(x)
    y_min_pred = model.predict([[x_min]])[0]
    y_max_pred = model.predict([[x_max]])[0]

    p1 = np.array([x_min, y_min_pred])
    p2 = np.array([x_max, y_max_pred])

    # 3. Extend the line by the specified percentage
    direction_vector = p2 - p1
    line_length = np.linalg.norm(direction_vector)
    
    if line_length == 0:
      return ( (p1[0], p1[1]), (p2[0],p2[1]) )

    unit_vector = direction_vector / line_length

    # Calculate the new endpoints
    p1_extended = p1 - unit_vector * (line_length * extend_percentage)
    p2_extended = p2 + unit_vector * (line_length * extend_percentage)

    return ((p1_extended[0], p1_extended[1]), (p2_extended[0], p2_extended[1]))

def find_connected_components(positive_edges: set, num_nodes: int) -> list[list[int]]:
    """
    Finds all connected components (groups of nodes) in the graph.
    This version is guaranteed to be stateless and work correctly in a loop.
    """
    # --- THIS IS THE FIX ---
    # All state variables are defined here, inside the function call,
    # ensuring they are brand new for every page.
    adj = defaultdict(list)
    for u, v in positive_edges:
        adj[u].append(v)
        adj[v].append(u)

    components = []
    visited = set()
    # --- END FIX ---

    if not positive_edges:
        return [[i] for i in range(num_nodes)]

    for i in range(num_nodes):
        if i not in visited:
            component = []
            q = [i]
            visited.add(i)
            while q:
                u = q.pop(0)
                component.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            components.append(component)
            
    return components

def trace_component_with_backtracking(component: list[int], adj: defaultdict) -> list[int]:
    """
    Traces a single, continuous path that covers every edge of a component using a
    clean, standard, non-recursive DFS algorithm. This is guaranteed to terminate and is
    robust to any graph structure, including those with cycles.
    """
    if not component:
        return []

    visited_edges = set()
    path = []

    # A good starting point is a leaf node (degree 1) if one exists.
    start_node = component[0]
    for node in component:
        # We need to check if the node is actually in the adjacency list,
        # as a component could be a single isolated node.
        if node in adj and len(adj[node]) == 1:
            start_node = node
            break

    # Handle the edge case of a single, isolated node with no edges.
    if not adj.get(start_node):
        return [start_node]

    stack = [start_node]
    path.append(start_node)

    while stack:
        u = stack[-1]  # Peek at the top of the stack

        # Find the next unvisited neighbor to travel to.
        next_neighbor = None
        # Sort neighbors for a consistent traversal order.
        for v in sorted(adj[u]):
            edge = tuple(sorted((u, v)))
            if edge not in visited_edges:
                next_neighbor = v
                break

        if next_neighbor is not None:
            # If we found an unvisited neighbor, we go down that branch.
            v = next_neighbor
            visited_edges.add(tuple(sorted((u, v))))
            stack.append(v)
            path.append(v)
        else:
            # If there are no unvisited neighbors, we are at a dead end. Backtrack.
            stack.pop()
            if stack:
                # The new top of the stack is the parent, so we add it to the path
                # to represent the pen moving back.
                parent = stack[-1]
                path.append(parent)

    # The final backtrack might add the start node again. Let's clean it up.
    if len(path) > 1 and path[0] == path[-1]:
       return path[:-1]
       
    return path

# def get_node_labels_from_edge_labels(edge_index, pred_edge_labels, num_nodes):
#     """Computes node clusters from predicted edge labels via connected components."""
#     if isinstance(edge_index, torch.Tensor):
#         edge_index = edge_index.cpu().numpy()
#     positive_edges = edge_index[:, pred_edge_labels == 1]
#     pred_edges_undirected = {tuple(sorted(e)) for e in positive_edges.T}
#     if not pred_edges_undirected:
#         return np.arange(num_nodes)
#     row, col = zip(*pred_edges_undirected)
#     adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
#     n_components, node_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
#     return node_labels

def get_node_labels_from_edge_labels(edge_index, pred_edge_labels, num_nodes):
    """Computes node clusters from predicted edge labels via connected components."""
    logging.debug("=== get_node_labels_from_edge_labels called ===")
    logging.debug(f"Input num_nodes: {num_nodes}")
    logging.debug(f"edge_index type: {type(edge_index)}, shape: {edge_index.shape if hasattr(edge_index, 'shape') else 'N/A'}")
    logging.debug(f"pred_edge_labels type: {type(pred_edge_labels)}, shape: {pred_edge_labels.shape if hasattr(pred_edge_labels, 'shape') else 'N/A'}")
    
    # Convert tensors to numpy
    if isinstance(edge_index, torch.Tensor):
        logging.debug("Converting edge_index from torch.Tensor to numpy")
        edge_index = edge_index.cpu().numpy()
    if isinstance(pred_edge_labels, torch.Tensor):
        logging.debug("Converting pred_edge_labels from torch.Tensor to numpy")
        pred_edge_labels = pred_edge_labels.cpu().numpy()

    # Normalize shapes
    edge_index = np.atleast_2d(edge_index)
    logging.debug(f"After atleast_2d, edge_index shape: {edge_index.shape}")
    
    if edge_index.shape[0] != 2:
        logging.debug(f"Reshaping edge_index from {edge_index.shape} to (2, -1)")
        edge_index = edge_index.reshape(2, -1)
    
    pred_edge_labels = np.atleast_1d(pred_edge_labels)
    logging.debug(f"After atleast_1d, pred_edge_labels shape: {pred_edge_labels.shape}")

    # Handle trivial graph
    if edge_index.shape[1] == 0 or pred_edge_labels.size == 0:
        logging.info(f"Trivial graph detected: edge_index.shape[1]={edge_index.shape[1]}, "
                    f"pred_edge_labels.size={pred_edge_labels.size}. Returning isolated nodes.")
        return np.arange(num_nodes)

    # Select only positive edges
    mask = (pred_edge_labels == 1)
    logging.debug(f"Positive edge mask shape: {mask.shape}, sum: {np.sum(mask)}")
    
    if mask.ndim > 1:
        logging.debug(f"Flattening mask from shape {mask.shape}")
        mask = mask.flatten()
    
    positive_edges = edge_index[:, mask]
    logging.debug(f"positive_edges shape after masking: {positive_edges.shape}")

    # Handle case of no positive edges
    if positive_edges.size == 0:
        logging.info(f"No positive edges found. Returning {num_nodes} isolated nodes.")
        return np.arange(num_nodes)

    # Ensure shape is (2, N)
    if positive_edges.ndim == 1:
        logging.debug(f"Reshaping positive_edges from 1D (size={positive_edges.size}) to (2, 1)")
        positive_edges = positive_edges.reshape(2, 1)
    
    logging.debug(f"Final positive_edges shape: {positive_edges.shape} "
                 f"({positive_edges.shape[1]} edge(s))")

    # Convert to undirected edges - iterate by column index to avoid .T issues
    logging.debug("Building undirected edge set...")
    pred_edges_undirected = {
        tuple(sorted(positive_edges[:, i])) 
        for i in range(positive_edges.shape[1])
    }
    logging.debug(f"Created {len(pred_edges_undirected)} undirected edge(s)")
    
    if not pred_edges_undirected:
        logging.warning("pred_edges_undirected is empty after deduplication. Returning isolated nodes.")
        return np.arange(num_nodes)

    # Build adjacency and find connected components
    logging.debug("Building sparse adjacency matrix...")
    row, col = zip(*pred_edges_undirected)
    adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    
    logging.debug(f"Running connected_components on {num_nodes} nodes with {len(row)} edges...")
    n_components, node_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    
    logging.info(f"Found {n_components} connected component(s) for {num_nodes} nodes")
    logging.debug(f"Node label distribution: {np.bincount(node_labels)}")
    logging.debug("=== get_node_labels_from_edge_labels finished ===")
    
    return node_labels

def run_gnn_prediction_for_page(manuscript_path, page_id, model_path, config_path):
    print(f"Fetching data for page: {page_id}")
    
    base_path = Path(manuscript_path)
    raw_input_dir = base_path / "gnn-dataset"               
    history_dir = base_path / "layout_analysis_output" / "gnn-format" 
    
    # --- 1. Load Node Data (Prioritize Modified History) ---
    modified_norm_path = history_dir / f"{page_id}_inputs_normalized.txt"
    modified_dims_path = history_dir / f"{page_id}_dims.txt"
    
    if modified_norm_path.exists() and modified_dims_path.exists():
        print(f"--> Loading USER-MODIFIED node definitions from {history_dir}")
        file_path = modified_norm_path
        dims_path = modified_dims_path
    else:
        print(f"--> Loading RAW CRAFT node definitions from {raw_input_dir}")
        file_path = raw_input_dir / f"{page_id}_inputs_normalized.txt"
        dims_path = raw_input_dir / f"{page_id}_dims.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Data for page {page_id} not found.")

    # Handle empty files (if user deleted all nodes previously)
    try:
        points_normalized = np.loadtxt(file_path)
    except UserWarning:
        points_normalized = np.array([])

    if points_normalized.size == 0:
        points_normalized = np.empty((0, 3))
    elif points_normalized.ndim == 1: 
        points_normalized = points_normalized.reshape(1, -1)
    
    dims = np.loadtxt(dims_path)
    full_width = dims[0] * 2
    full_height = dims[1] * 2
    max_dimension = max(full_width, full_height)
    
    nodes_payload = [
        {
            "x": float(p[0]) * max_dimension, 
            "y": float(p[1]) * max_dimension, 
            "s": float(p[2])
        } 
        for p in points_normalized
    ]
    
    response = {
        "nodes": nodes_payload,
        "edges": [],
        "textline_labels": [-1] * len(points_normalized),
        "textbox_labels": [],
        "dimensions": [full_width, full_height]
    }

    # --- 2. Check for Saved Topology (Edges/Labels) ---
    saved_edges_path = history_dir / f"{page_id}_edges.txt"
    saved_labels_path = history_dir / f"{page_id}_labels_textline.txt"
    saved_textbox_path = history_dir / f"{page_id}_labels_textbox.txt"
    
    if saved_edges_path.exists():
        print(f"Found saved edge topology...")
        saved_edges = []
        try:
            if saved_edges_path.stat().st_size > 0:
                raw_edges = np.loadtxt(saved_edges_path, dtype=int, ndmin=2)
                if raw_edges.ndim == 1 and raw_edges.size >= 2:
                    raw_edges = raw_edges.reshape(1, -1)
                
                for row in raw_edges:
                    if len(row) >= 2:
                        saved_edges.append({
                            "source": int(row[0]),
                            "target": int(row[1]),
                            "label": 1
                        })
        except Exception as e:
            print(f"Warning reading edges: {e}")
            
        response["edges"] = saved_edges
        
        if saved_labels_path.exists():
            try:
                labels = np.loadtxt(saved_labels_path, dtype=int)
                if labels.size == len(points_normalized):
                     response["textline_labels"] = labels.tolist()
            except Exception: pass 
        
        if saved_textbox_path.exists():
            try:
                tb_labels = np.loadtxt(saved_textbox_path, dtype=int)
                if tb_labels.size == len(points_normalized):
                    response["textbox_labels"] = tb_labels.tolist()
            except Exception: pass

        return response

    # --- 3. Run GNN (Only if no history exists) ---
    if len(points_normalized) == 0:
        return response

    print(f"Running GNN Inference...")
    model, d_config, device = load_model_once(model_path, config_path)
    
    page_dims_norm = {'width': 1.0, 'height': 1.0}
    input_graph_data = create_input_graph_edges(points_normalized, page_dims_norm, d_config.input_graph)
    input_edges_set = input_graph_data["edges"]

    if not input_edges_set:
        return response

    edge_index_undirected = torch.tensor(list(input_edges_set), dtype=torch.long).t().contiguous()
    if d_config.input_graph.directionality == "bidirectional":
        edge_index = torch.cat([edge_index_undirected, edge_index_undirected.flip(0)], dim=1)
    else:
        edge_index = edge_index_undirected

    node_features = get_node_features(points_normalized, input_graph_data["heuristic_degrees"], d_config.features)
    edge_features = get_edge_features(edge_index, node_features, input_graph_data["heuristic_edge_counts"], d_config.features)
    
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features).to(device)

    threshold = 0.5
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = F.softmax(logits, dim=1)
        pred_edge_labels = (probs[:, 1] > threshold).cpu().numpy()

    model_positive_edges = set()
    edge_index_cpu = data.edge_index.cpu().numpy()
    
    for idx, is_pos in enumerate(pred_edge_labels):
        if is_pos:
            u, v = edge_index_cpu[:, idx]
            model_positive_edges.add(tuple(sorted((u, v))))

    final_edges = []
    for u, v in input_edges_set:
        if tuple(sorted((u, v))) in model_positive_edges:
            final_edges.append({"source": int(u), "target": int(v), "label": 1})

    response["edges"] = final_edges
    return response


def create_page_xml(
    page_id,
    model_positive_edges,
    points_unnormalized,
    page_dims,
    output_path: Path,
    pred_node_labels: np.ndarray,
    polygons_data: dict,
    textbox_labels: np.ndarray = None,
    use_best_fit_line: bool = False,
    extend_percentage: float = 0.01,
    image_path: Path = None, 
    save_vis: bool = True,
    images_output_dir: Path = None,
    text_content: dict = None # <--- NEW ARGUMENT
):
    """
    Generates a PAGE XML file with reading order and textregions (textboxes).
    Also saves line images organized by textbox folder.
    """
    PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    ET.register_namespace('', PAGE_XML_NAMESPACE)

    num_nodes = len(points_unnormalized)

    # Build Adjacency List
    adj = defaultdict(list)
    for u, v in model_positive_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Find Connected Components (Text Lines)
    components = find_connected_components(model_positive_edges, num_nodes)
    
    # -- Data Structure Preparation --
    regions = defaultdict(list)
    
    for i, component in enumerate(components):
        if not component: continue
        
        comp_tb_labels = []
        if textbox_labels is not None:
             for node_idx in component:
                 comp_tb_labels.append(textbox_labels[node_idx])
        
        if comp_tb_labels:
            tb_id = np.bincount(comp_tb_labels).argmax()
        else:
            tb_id = 0 
            
        regions[tb_id].append(component)

    # -- PAGE XML Setup --
    pc_gts = ET.Element(f"{{{PAGE_XML_NAMESPACE}}}PcGts")
    metadata = ET.SubElement(pc_gts, "Metadata")
    ET.SubElement(metadata, "Creator").text = "GNN-Layout-Analysis"
    ET.SubElement(metadata, "Created").text = datetime.now().isoformat()
    

    final_w = int(page_dims['width'] * 2)
    final_h = int(page_dims['height'] * 2)

    page = ET.SubElement(pc_gts, "Page", attrib={
        "imageFilename": f"{page_id}.jpg",
        "imageWidth": str(final_w),
        "imageHeight": str(final_h)
    })

    # -- Visualization Setup --
    vis_img = None
    if save_vis:
        if image_path and image_path.exists():
            vis_img = cv2.imread(str(image_path))
            if vis_img is not None and (vis_img.shape[0] != final_h or vis_img.shape[1] != final_w):
                vis_img = cv2.resize(vis_img, (final_w, final_h))
        if vis_img is None:
            vis_img = np.zeros((final_h, final_w, 3), dtype=np.uint8)

    def get_centroid(comp_nodes):
        xs = [points_unnormalized[n][0] * 2 for n in comp_nodes]
        ys = [points_unnormalized[n][1] * 2 for n in comp_nodes]
        return np.mean(xs), np.mean(ys)

    # 1. Sort Regions
    region_centroids = []
    for tb_id, comps in regions.items():
        all_nodes = [n for comp in comps for n in comp]
        if not all_nodes: continue
        cx, cy = get_centroid(all_nodes)
        region_centroids.append({'id': tb_id, 'cx': cx, 'cy': cy})
    
    region_centroids.sort(key=lambda r: (r['cy'], r['cx']))

    # -- Construct XML Hierarchy --
    for r_idx, region_info in enumerate(region_centroids):
        tb_id = region_info['id']
        comps = regions[tb_id]
        
        # --- NEW: Create Directory for this Textbox ---
        current_tb_dir = None
        if images_output_dir:
            current_tb_dir = images_output_dir / f"textbox_label_{tb_id}"
            current_tb_dir.mkdir(exist_ok=True)
        # ---------------------------------------------

        # --- FIXED AREA CALCULATION ---
        region_xs = []
        region_ys = []
        
        for comp in comps:
            line_label = pred_node_labels[comp[0]]
            
            # --- MODIFIED: Access Logic for new Polygons Data Structure ---
            if line_label in polygons_data:
                # === RED TEAM FIX START ===
                # We need to extract the raw list of points from the new dictionary structure
                data_obj = polygons_data[line_label]
                poly_pts = []
                
                # Check: Is this the new format (Dict) or old format (List)?
                if isinstance(data_obj, dict) and 'points' in data_obj:
                    poly_pts = data_obj['points'] # Extract the list
                else:
                    poly_pts = data_obj # Fallback for safety
                # === RED TEAM FIX END ===
                
                if len(poly_pts) > 0:
                    for p in poly_pts:
                        region_xs.append(p[0])
                        region_ys.append(p[1])
            else:
                # Fallback: If no polygon exists, use node centers
                for n in comp:
                    region_xs.append(points_unnormalized[n][0] * 2)
                    region_ys.append(points_unnormalized[n][1] * 2)
        
        if not region_xs: 
            continue 

        min_x, max_x = min(region_xs), max(region_xs)
        min_y, max_y = min(region_ys), max(region_ys)
        
        region_elem = ET.SubElement(page, "TextRegion", id=f"region_{r_idx}", custom=f"textbox_label_{tb_id}")
        region_coords_str = f"{int(min_x)},{int(min_y)} {int(max_x)},{int(min_y)} {int(max_x)},{int(max_y)} {int(min_x)},{int(max_y)}"
        ET.SubElement(region_elem, "Coords", points=region_coords_str)

        if save_vis and vis_img is not None:
            cv2.rectangle(vis_img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 255), 2)
            cv2.putText(vis_img, f"R{r_idx}", (int(min_x), int(min_y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 2. Sort Lines within Region
        comp_centroids = []
        for comp in comps:
            cx, cy = get_centroid(comp)
            comp_centroids.append({'comp': comp, 'cx': cx, 'cy': cy})
        
        comp_centroids.sort(key=lambda c: c['cy'])

        for l_idx, line_info in enumerate(comp_centroids):
            component = line_info['comp']
            line_label = pred_node_labels[component[0]] 
            
            # --- MODIFIED: Add 'custom' attribute to store the integer ID ---
            text_line = ET.SubElement(
                region_elem, 
                "TextLine", 
                id=f"region_{r_idx}_line_{l_idx}",
                custom=f"structure_line_id_{line_label}" # <--- CRITICAL ADDITION
            )

            # === VERIFY THIS BLOCK EXISTS ===
            if text_content and str(line_label) in text_content:
                rec_text = text_content[str(line_label)]
                # Ensure we don't write "None" or null
                if rec_text: 
                    text_equiv = ET.SubElement(text_line, "TextEquiv")
                    unicode_elem = ET.SubElement(text_equiv, "Unicode")
                    unicode_elem.text = str(rec_text)
            # ================================
            
            # --- Baseline Calculation ---
            baseline_points_str = ""
            baseline_vis = []
            
            path_indices = trace_component_with_backtracking(component, adj)
            if len(path_indices) >= 1:
                ordered_points = [points_unnormalized[idx] for idx in path_indices]
                baseline_vis = [[int(p[0]*2), int((p[1]+(p[2]/2))*2)] for p in ordered_points]
                baseline_points_str = " ".join([f"{p[0]},{p[1]}" for p in baseline_vis])
            
            ET.SubElement(text_line, "Baseline", points=baseline_points_str)

            # --- Polygon Coords AND Image Saving ---
            polygon_vis = []
            if line_label in polygons_data:
                data = polygons_data[line_label]
                polygon_points = []
                
                # Check format and extract Image/Points
                if isinstance(data, dict):
                    polygon_points = data.get('points', [])
                    line_img = data.get('image', None)
                    
                    # --- NEW: Save Image to Textbox Folder ---
                    if current_tb_dir is not None and line_img is not None:
                        # e.g., line_005.jpg
                        # Note: line_label is an integer, typically 0-indexed relative to graph
                        img_save_path = current_tb_dir / f"line_{line_label}.jpg"
                        cv2.imwrite(str(img_save_path), line_img)
                    # ------------------------------------------
                else:
                    polygon_points = data # Old format fallback
                
                if polygon_points:
                    coords_str = " ".join([f"{p[0]},{p[1]}" for p in polygon_points])
                    ET.SubElement(text_line, "Coords", points=coords_str)
                    polygon_vis = polygon_points

            # Visualize Line
            if save_vis and vis_img is not None:
                if len(polygon_vis) > 0:
                    pts = np.array(polygon_vis, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
                if len(baseline_vis) > 0:
                    pts = np.array(baseline_vis, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_img, [pts], False, (0, 0, 255), 2)

    # Save XML
    tree = ET.ElementTree(pc_gts)
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="\t", level=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)

    # Save Visualization
    if save_vis and vis_img is not None:
        vis_output_path = output_path.parent / f"{output_path.stem}_viz.jpg"
        cv2.imwrite(str(vis_output_path), vis_img)


# sample PAGE-XML file:
<?xml version='1.0' encoding='UTF-8'?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
	<Metadata>
		<Creator>GNN-Layout-Analysis</Creator>
		<Created>2026-01-29T16:52:26.641551</Created>
	</Metadata>
	<Page imageFilename="233_0002.jpg" imageWidth="2500" imageHeight="940">
		<TextRegion id="region_0" custom="textbox_label_6">
			<Coords points="489,41 1564,41 1564,111 489,111" />
			<TextLine id="region_0_line_0" custom="structure_line_id_0">
				<TextEquiv>
					<Unicode>शित्ताकल्योव्याकरणनिरूक्तजोनिइंदांसि।।४=कत्मसारोइस्वेछयाविहरति।।इ=</Unicode>
				</TextEquiv>
				<Baseline points="516,68 570,72 608,70 638,70 660,70 688,70 709,70 742,72 772,74 796,74 832,74 862,78 893,76 932,74 970,74 1012,82 1036,78 1068,74 1118,74 1160,74 1200,72 1234,74 1266,70 1298,70 1330,72 1368,74 1394,70 1428,70 1458,74 1504,70 1550,72 1504,70 1458,74 1428,70 1394,70 1368,74 1330,72 1298,70 1266,70 1234,74 1200,72 1160,74 1118,74 1068,74 1036,78 1012,82 970,74 932,74 893,76 862,78 832,74 796,74 772,74 742,72 709,70 688,70 660,70 638,70 608,70 570,72" />
				<Coords points="1277,41 1277,46 1276,47 1248,47 1248,53 1247,54 1207,54 1207,55 1206,56 1138,56 1137,55 1137,51 1131,51 1131,58 1130,59 1107,59 1106,58 1106,51 1095,51 1094,50 1094,47 1049,47 1049,51 1048,52 1035,52 1034,51 1034,49 949,49 949,50 948,51 905,51 905,53 904,54 857,54 856,53 856,49 808,49 807,50 774,50 774,54 773,55 770,55 769,54 769,53 743,53 742,52 685,52 684,51 684,49 655,49 654,48 621,48 621,51 620,52 616,52 615,51 615,50 566,50 565,49 489,49 489,91 525,91 526,92 526,95 615,95 615,92 616,91 620,91 621,92 621,96 654,96 654,92 655,91 684,91 685,90 716,90 717,91 717,95 769,95 769,94 770,93 773,93 774,94 774,99 807,99 808,100 808,104 856,104 857,103 884,103 885,102 925,102 925,101 926,100 948,100 949,101 949,104 977,104 978,105 978,111 1034,111 1034,108 1035,107 1062,107 1062,103 1063,102 1094,102 1094,98 1095,97 1106,97 1106,92 1107,91 1130,91 1131,92 1131,98 1137,98 1137,95 1138,94 1186,94 1186,92 1187,91 1206,91 1207,92 1207,96 1276,96 1277,97 1277,100 1322,100 1323,99 1357,99 1357,98 1358,97 1389,97 1389,92 1390,91 1407,91 1408,92 1408,96 1439,96 1440,97 1440,99 1476,99 1476,97 1477,96 1524,96 1525,95 1536,95 1536,93 1537,92 1564,92 1564,53 1537,53 1536,52 1536,49 1525,49 1524,48 1483,48 1483,49 1482,50 1476,50 1475,51 1453,51 1452,50 1452,47 1408,47 1408,51 1407,52 1358,52 1357,51 1357,47 1323,47 1322,46 1322,41" />
			</TextLine>
		</TextRegion>

and so on..


