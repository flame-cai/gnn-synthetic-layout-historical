<template>
  <div class="manuscript-viewer">
    
    <!-- TOP RAIL: Navigation & Global Actions -->
    <div class="top-bar">
      <div class="top-bar-left">
        <button class="nav-btn secondary" @click="$emit('back')">Back</button>
        <span class="page-title">{{ manuscriptNameForDisplay }} <span class="divider">/</span></span>
        
        <!-- NEW: Page Dropdown -->
        <select class="page-select" :value="currentPageForDisplay" @change="handlePageSelect">
           <option v-for="pg in localPageList" :key="pg" :value="pg">Page {{ pg }}</option>
        </select>
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
           <!-- NEW: Simple Save Button -->
           <button class="action-btn" @click="saveCurrentPage" :disabled="loading || isProcessingSave">
             Save (S)
           </button>

           <button class="action-btn primary" @click="saveAndGoNext" :disabled="loading || isProcessingSave">
            {{ autoRecogEnabled ? 'Save, Recog & Next' : 'Save & Next' }}
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
      
      <!-- 1. Unified Overlay for Saving OR Mode Switching (Foreground) -->
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

        <!-- NEW: Wrapper to hide everything when 'v' is pressed -->
        <div :style="{ opacity: isVKeyPressed ? 0 : 1, transition: 'opacity 0.1s' }">
            
            <!-- SVG Graph Layer (Visible in Layout Mode) -->
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
        </div> <!-- End of Visibility Wrapper -->

      </div>
    </div>

    <!-- BOTTOM RAIL: Controls & Help Center -->
    <div class="bottom-panel" :class="{ 'is-collapsed': isPanelCollapsed }">
      
      <!-- Mode Tabs (Always Visible) -->
      <div class="mode-tabs">
          <!-- REMOVED: View Mode Button -->
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
        
        <!-- Layout Mode Help -->
        <div v-if="layoutModeActive || (!layoutModeActive && !recognitionModeActive)" class="help-section full-width" style="flex-direction: column;">
          
          <div class="help-grid" style="height: auto; flex: 1; min-height: 0;">
            
            <!-- Nodes Card -->
            <div class="help-card horizontal-layout">
              <div class="media-container-square">
                <video :src="nodeWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
              </div>
              <div class="card-text">
                <h4>Nodes</h4>
                <p><span class="key-badge">L-Click</span> Add Node</p>
                <p><span class="key-badge">R-Click</span> Delete Node</p>
              </div>
            </div>

            <!-- Edges Card -->
            <div class="help-card horizontal-layout">
              <div class="media-container-square">
                <video :src="edgeWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
              </div>
              <div class="card-text">
                <h4>Edges</h4>
                <p>Hold <span class="key-badge">A</span> + Hover to Connect</p>
                <p>Hold <span class="key-badge">D</span> + Hover to Delete</p>
              </div>
            </div>

            <!-- Regions Card -->
            <div class="help-card horizontal-layout">
              <div class="media-container-square">
                <video :src="regionWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
              </div>
              <div class="card-text">
                <h4>Regions</h4>
                <p>Hold <span class="key-badge">E</span> + Hover to Label</p>
                <p>Release & Repeat for New Box</p>
              </div>
            </div>

          </div>

          <!-- Hotkey Footer -->
          <div class="hotkey-footer">
            <span class="key-hint"><span class="key-badge">V</span> Hold to Hide Graph</span>
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
             <p>Transcribe line-by-line. Auto-save is active (every 20s).</p>
             <ul>
               <li><strong>Navigate:</strong> Press <code>Tab</code> to move to the next line.</li>
               <li><strong>Visibility:</strong> Hold <code>V</code> to hide polygons (if not typing).</li>
               <li v-if="devanagariModeEnabled"><strong>Keys:</strong> Type phonetically (e.g., 'k' -> 'क'). Use '`' for Halant.</li>
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
// Assuming these imports exist in your project structure
import edgeWebm from '../tutorial/_edge.webm'
import regionWebm from '../tutorial/_textbox.webm'
import nodeWebm from '../tutorial/_node.webm'
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
const activeInput = ref(null) 

const setMode = (mode) => {
  layoutModeActive.value = false
  recognitionModeActive.value = false
  
  isAKeyPressed.value = false
  isDKeyPressed.value = false
  isEKeyPressed.value = false
  resetSelection()

  if (mode === 'layout') {
    layoutModeActive.value = true
  } else if (mode === 'recognition') {
    recognitionModeActive.value = true
    sortLinesTopToBottom()
    if(sortedLineIds.value.length > 0 && !focusedLineId.value) {
        activateInput(sortedLineIds.value[0])
    }
  }
  isPanelCollapsed.value = false
}


const isEditModeFlow = computed(() => !!props.manuscriptName && !!props.pageName)

// --- DATA ---
const layoutModeActive = ref(true) // Default to true now
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

// Key states
const isDKeyPressed = ref(false)
const isAKeyPressed = ref(false)
const isEKeyPressed = ref(false) 
const isVKeyPressed = ref(false) // NEW for Visibility

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
const localTextContent = reactive({}) 
const pagePolygons = ref({}) 
const focusedLineId = ref(null)
const sortedLineIds = ref([])
const autoRecogEnabled = ref(true)
const devanagariModeEnabled = ref(true) 
const localTextConfidence = reactive({}) 
const autoSaveInterval = ref(null) // NEW

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

const pointsToSvgString = (pts) => {
    if(!pts) return "";
    return pts.map(p => `${scaleX(p[0])},${scaleY(p[1])}`).join(" ");
}

const sortLinesTopToBottom = () => {
    const ids = Object.keys(pagePolygons.value);
    if(ids.length === 0) {
        sortedLineIds.value = [];
        return;
    }
    
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
    
    stats.sort((a,b) => {
        const diffY = a.minY - b.minY;
        if(Math.abs(diffY) > 20) return diffY; 
        return a.minX - b.minX;
    });
    
    sortedLineIds.value = stats.map(s => s.id);
}

const getActiveInputStyle = () => {
    if(!focusedLineId.value || !pagePolygons.value[focusedLineId.value]) return { display: 'none' };
    
    const pts = pagePolygons.value[focusedLineId.value];
    const xs = pts.map(p => p[0]);
    const ys = pts.map(p => p[1]);
    
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    
    const rawWidth = maxX - minX;
    const rawHeight = maxY - minY;

    const isVertical = rawHeight > (rawWidth * 1.2); 

    const style = {
        position: 'absolute',
        height: 'auto',
        zIndex: 100
    };

    if (isVertical) {
        const pageCenterX = dimensions.value[0] / 2;
        const polyCenterX = minX + (rawWidth / 2);
        
        const INPUT_WIDTH_PX = 250; 
        
        style.top = `${scaleY(minY)}px`; 
        style.width = `${INPUT_WIDTH_PX}px`;

        if (polyCenterX > pageCenterX) {
            style.left = `${scaleX(minX) - INPUT_WIDTH_PX - 10}px`;
        } else {
            style.left = `${scaleX(maxX) + 10}px`;
        }
    } else {
        style.top = `${scaleY(maxY) + 5}px`;
        style.left = `${scaleX(minX)}px`;
        style.width = `${scaleX(rawWidth)}px`;
    }

    return style;
}

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
  if (isEKeyPressed.value) return 'crosshair' 
  if (isAKeyPressed.value) return 'crosshair' 
  if (isDKeyPressed.value) return 'not-allowed' 
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
      // Save generated graph silently
      await fetch(`${import.meta.env.VITE_BACKEND_URL}/save-graph/${manuscript}/${page}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ graph: graph.value }),
      }).catch(e => console.error(e))
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
    
    const data = await response.json()
    // Backend returns { pages: [], last_edited: "..." }
    localPageList.value = data.pages
    return data.last_edited
  } catch (err) {
    localPageList.value = []
    return null
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
  if (layoutModeActive.value && isEKeyPressed.value) {
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
    
    if (layoutModeActive.value && !isAKeyPressed.value && !isDKeyPressed.value && !isEKeyPressed.value) {
        addNode(event.clientX, event.clientY);
        return;
    }
    
    resetSelection();
}

const onNodeClick = (nodeIndex, event) => {
    event.stopPropagation(); 
    if (!layoutModeActive.value || recognitionModeActive.value) return;
    if (isAKeyPressed.value || isDKeyPressed.value || isEKeyPressed.value) return;
    
    const existingIndex = selectedNodes.value.indexOf(nodeIndex);
    if (existingIndex !== -1) selectedNodes.value.splice(existingIndex, 1);
    else selectedNodes.value.length < 2 ? selectedNodes.value.push(nodeIndex) : (selectedNodes.value = [nodeIndex]);
}

const onNodeRightClick = (nodeIndex, event) => {
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

  if (isDKeyPressed.value) {
      handleEdgeHoverDelete(mouseX, mouseY)
      return
  }

  if (isAKeyPressed.value) {
      handleNodeHoverCollect(mouseX, mouseY)
      return
  }

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
  const isInput = tagName === 'input' || tagName === 'textarea';

  const key = e.key.toLowerCase()
  if (key === 's' && !e.repeat && !isInput) {
    e.preventDefault()
    saveCurrentPage()
    return
  }
  
  if (key === 'w' && !e.repeat && !isInput) { e.preventDefault(); setMode('layout'); return }
  if (key === 't' && !e.repeat && !isInput) { e.preventDefault(); requestSwitchToRecognition(); return }
  
  // NEW: Visibility Hotkey 'v'
  if (key === 'v' && !isInput) {
      isVKeyPressed.value = true
      return
  }

  if (layoutModeActive.value && !e.repeat && !isInput) {
      if (key === 'e') { e.preventDefault(); isEKeyPressed.value = true; return }
      if (key === 'd') { e.preventDefault(); isDKeyPressed.value = true; resetSelection(); return }
      if (key === 'a') { e.preventDefault(); isAKeyPressed.value = true; hoveredNodesForMST.clear(); resetSelection(); return }
  }
}

const handleGlobalKeyUp = (e) => {
  const key = e.key.toLowerCase()
  if (key === 'v') { isVKeyPressed.value = false }

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
  for (let i = 0; i < points.length; i++)
    for (let j = i + 1; j < points.length; j++) {
      edges.push({
        source: points[i].originalIndex,
        target: points[j].originalIndex,
        weight: Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y),
      })
    }
  edges.sort((a, b) => a.weight - b.weight)
  
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

const saveModifications = async (background = false) => {
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
    runRecognition: autoRecogEnabled.value && !background, // Don't run GNN/AI on auto-save
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

    // If auto-recog was run, update text
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

    isProcessingSave.value = true;
    try {
        if (modifications.value.length > 0) {
            await saveModifications(); 
        }
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

const handlePageSelect = (event) => {
    const selectedPage = event.target.value;
    if (selectedPage === localCurrentPage.value) return;
    
    confirmAndNavigate(() => {
        navigateToPage(selectedPage);
    });
}

// NEW: Save current page logic (no nav)
const saveCurrentPage = async () => {
  if (loading.value || isProcessingSave.value) return
  isProcessingSave.value = true
  try {
    await saveModifications()
    // Optional: Flash a small 'Saved' toast
  } catch (err) { alert(`Save failed: ${err.message}`) } 
  finally { isProcessingSave.value = false }
}

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

// Auto-Save Logic
watch(recognitionModeActive, (active) => {
    if (active) {
        if(autoSaveInterval.value) clearInterval(autoSaveInterval.value);
        autoSaveInterval.value = setInterval(async () => {
            // Background save only
            try {
                await saveModifications(true);
                console.log("Auto-save completed");
            } catch(e) {
                console.warn("Auto-save failed silently", e);
            }
        }, 20000); // 20 seconds
    } else {
        if(autoSaveInterval.value) {
            clearInterval(autoSaveInterval.value);
            autoSaveInterval.value = null;
        }
    }
})

onMounted(async () => {
  if (props.manuscriptName && props.pageName) {
    localManuscriptName.value = props.manuscriptName
    localCurrentPage.value = props.pageName
    
    // Fetch pages AND the last edited page
    const lastEdited = await fetchPageList(props.manuscriptName)
    
    // Logic: If props.pageName is default (1st page) but a lastEdited exists, 
    // we might want to jump there? The prompt says "When user loads a manuscript... load the page which has been most recently edited".
    // Since App.vue usually passes pageName=pages[0], we override it here if available.
    
    if (lastEdited && lastEdited !== props.pageName) {
         localCurrentPage.value = lastEdited
         emit('page-changed', lastEdited) // Sync with parent
    }

    await fetchPageData(props.manuscriptName, localCurrentPage.value)
  }
  window.addEventListener('keydown', handleGlobalKeyDown)
  window.addEventListener('keyup', handleGlobalKeyUp)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleGlobalKeyDown)
  window.removeEventListener('keyup', handleGlobalKeyUp)
  if(autoSaveInterval.value) clearInterval(autoSaveInterval.value);
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

/* Page Select Dropdown */
.page-select {
    background: #333;
    color: #fff;
    border: 1px solid #444;
    padding: 6px 12px;
    border-radius: 4px;
    outline: none;
    font-size: 0.9rem;
    cursor: pointer;
}
.page-select:hover { border-color: #666; }

/* Main Visualization */
.visualization-container {
  position: relative; overflow: auto; flex-grow: 1; display: flex;
  justify-content: center; align-items: flex-start; padding: 2rem; background-color: #121212;
}
.image-container { position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.6); }
.manuscript-image { display: block; user-select: none; opacity: 0.7; }
.graph-overlay { position: absolute; top: 0; left: 0; opacity: 0; pointer-events: none; transition: opacity 0.2s; }
.graph-overlay.is-visible { opacity: 1; pointer-events: auto; }

/* Input Floater */
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
  height: 110px; 
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
/* Horizontal Card Layout for Square Videos */
.help-card.horizontal-layout {
  flex-direction: row;
  align-items: center;
  height: 100%;
  max-height: 140px; /* Prevent cards from getting too tall */
  width: 32%; /* Ensure 3 cards fit side-by-side */
}

.media-container-square {
  height: 100%;
  aspect-ratio: 1 / 1; /* Forces square shape based on container height */
  background: #000;
  border-right: 1px solid #333;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Adjust text padding for horizontal layout */
.help-card.horizontal-layout .card-text {
  text-align: left;
  padding: 0 16px;
}

/* Hotkey Footer Strip */
.hotkey-footer {
  height: 40px; /* Fixed height for footer */
  border-top: 1px solid #3d3d3d;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background: rgba(0,0,0,0.2);
  border-radius: 4px;
  margin-top: 8px;
}

.key-hint {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.9rem;
  color: #ccc;
}
</style>