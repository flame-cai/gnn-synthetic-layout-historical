<template>
  <div class="manuscript-viewer">
    
    <!-- TOP RAIL: Navigation & Global Actions -->
    <div class="top-bar">
      <div class="top-bar-left">
        <button class="nav-btn secondary" @click="$emit('back')">Back</button>
        <span class="page-title">{{ manuscriptNameForDisplay }} <span class="divider">/</span> Page {{ currentPageForDisplay }}</span>
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
            Save & Next (S)
          </button>
          <button class="action-btn" @click="downloadResults" :disabled="loading || isProcessingSave">
            Download PAGE-XMLs
          </button>
          <button class="action-btn" @click="runHeuristic" :disabled="loading">
            Auto-Link
          </button>
        </div>
      </div>
    </div>

    <!-- MAIN CONTENT: Visualization Area -->
    <div class="visualization-container" ref="container">
      <div v-if="isProcessingSave" class="processing-save-notice">
        Saving graph and processing... Please wait.
      </div>
      <div v-if="error" class="error-message">
        {{ error }}
      </div>
      <div v-if="loading" class="loading">Loading Page Data...</div>
      <div
        v-else
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

        <!-- SVG Graph Layer -->
        <svg
          v-if="graphIsLoaded"
          class="graph-overlay"
          :width="scaledWidth"
          :height="scaledHeight"
          :viewBox="`0 0 ${scaledWidth} ${scaledHeight}`"
          :class="{ 'is-visible': textlineModeActive || textboxModeActive || nodeModeActive }"
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

        <!-- Recognition Input Overlay Layer -->
        <div
            v-if="recognitionModeActive && graphIsLoaded"
            class="input-overlay-container"
            :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }"
        >
            <div
                v-for="(nodeIndices, lineId) in textlines"
                :key="`input-${lineId}`"
                class="line-input-wrapper"
                :style="getLineInputStyle(nodeIndices)"
            >
                <!-- Editable Input -->
                <input 
                    v-if="focusedLineId === lineId"
                    ref="activeInput"
                    v-model="localTextContent[lineId]" 
                    class="line-input active"
                    @blur="focusedLineId = null"
                    @keydown.tab.prevent="focusNextLine(lineId)"
                    placeholder="Transcribe..." 
                />
                <!-- Read-only Display (Click to Edit) -->
                <div 
                    v-else 
                    class="line-input-display"
                    @click="activateInput(lineId)"
                    :class="{ 'has-text': !!localTextContent[lineId] }"
                    :title="localTextContent[lineId] || 'Click to transcribe'"
                >
                    {{ localTextContent[lineId] || '' }}
                </div>
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
           @click="setMode('recognition')"
           :disabled="isProcessingSave || !graphIsLoaded">
           Recognize (T)
         </button>

         
         <!-- Spacer to push toggle button to right -->
         <div class="tab-spacer"></div>

         <!-- Toggle Collapse Button -->
         <button class="panel-toggle-btn" @click="isPanelCollapsed = !isPanelCollapsed" title="Toggle Help Panel">
            <span v-if="isPanelCollapsed">▲ Show Help</span>
            <span v-else>▼ Hide</span>
         </button>
      </div>

      <!-- Help & Actions Content Area (Collapsible) -->
      <div class="help-content-area" v-show="!isPanelCollapsed">
        
        <!-- Section: View Mode (Default) -->
        <div v-if="!textlineModeActive && !textboxModeActive && !nodeModeActive && !recognitionModeActive" class="help-section">
          <div class="instructions-container">
            <h3>View Mode</h3>
            <p>Pan and zoom to inspect the manuscript. No edits can be made in this mode. Select a mode above or use hotkeys to start annotating.</p>
          </div>
        </div>

        <!-- Section: Edge Edit Mode -->
        <div v-if="textlineModeActive" class="help-section">
          <div class="media-container">
            <video :src="edgeWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
          </div>
          <div class="instructions-container">
            <h3>Text-Line Mode</h3>
            <ul>
              <li><strong>Connect:</strong> Hold <code>'a'</code> and hover over nodes to connect them.</li>
              <li><strong>Delete:</strong> Hold <code>'d'</code> and hover over edges to delete them.</li>
              <li><strong>Save:</strong> Press <code>'s'</code> to save changes and move to the next page.</li>
            </ul>
          </div>
        </div>

        <!-- Section: Region Labeling Mode -->
        <div v-if="textboxModeActive" class="help-section">
           <div class="media-container">
            <video :src="regionWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
          </div>
          <div class="instructions-container">
            <h3>Text-Box Mode</h3>
            <p>
              Hold <code>'e'</code> and hover over lines to label them as being in the same text-box. 
              Release <code>'e'</code> and press it again to label a new text-box.
            </p>
          </div>
        </div>

        <!-- Section: Node Mode -->
        <div v-if="nodeModeActive" class="help-section">
           <div class="media-container">
            <video :src="nodeWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
          </div>
          <div class="instructions-container">
            <h3>Node Mode</h3>
            <ul>
              <li><strong>Add Node:</strong> Left-click anywhere on the image to add a new node.</li>
              <li><strong>Delete Node:</strong> Right-click on an existing node to remove it (and its connections).</li>
            </ul>
          </div>
        </div>

        <!-- Section: Recognition Mode (NEW) -->
        <div v-if="recognitionModeActive" class="help-section">
           <div class="media-container">
             <div class="webm-placeholder">
              <span>Recognition Mode</span>
            </div>
           </div>
           <div class="instructions-container">
             <h3>Recognition Mode</h3>
             <p>Use Gemini AI to transcribe text lines automatically, then correct them manually.</p>
             <div class="form-group-inline">
                <input v-model="geminiKey" type="password" placeholder="Enter Gemini API Key" class="api-input" />
                <button class="action-btn primary" @click="triggerRecognition" :disabled="isRecognizing || !geminiKey">
                    {{ isRecognizing ? 'Recognizing...' : 'Auto-Recognize' }}
                </button>
             </div>
             <ul>
               <li><strong>Edit:</strong> Click any text line box on the image to type.</li>
               <li><strong>Navigate:</strong> Press <code>Tab</code> to jump to the next line.</li>
               <li><strong>Save:</strong> Press <code>'s'</code> to save the text into the PAGE-XML.</li>
             </ul>
           </div>
        </div>
        
        <!-- Shared: Modification Log -->
        <div v-if="modifications.length > 0" class="log-sidebar">
            <div class="log-header">
              <span>Changes: {{ modifications.length }}</span>
              <button class="text-btn" @click="resetModifications" :disabled="loading">Reset All</button>
            </div>
            <ul class="log-list">
              <li v-for="(mod, index) in modifications.slice().reverse()" :key="index">
                <small>{{ mod.type === 'add' ? 'Added' : 'Removed' }} edge</small>
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

// --- Helper for Mode Switching via Buttons ---
const setMode = (mode) => {
  if (mode === 'view') {
    textlineModeActive.value = false
    textboxModeActive.value = false
    nodeModeActive.value = false
    recognitionModeActive.value = false
  } else if (mode === 'edge') {
    textlineModeActive.value = true
  } else if (mode === 'region') {
    textboxModeActive.value = true
  } else if (mode === 'node') {
    nodeModeActive.value = true
  } else if (mode === 'recognition') {
    // Enable recognition mode
    textlineModeActive.value = false
    textboxModeActive.value = false
    nodeModeActive.value = false
    recognitionModeActive.value = true
    initializeTextContent()
  }
  isPanelCollapsed.value = false
}

const isEditModeFlow = computed(() => !!props.manuscriptName && !!props.pageName)

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
// NEW: Recognition Mode State
const recognitionModeActive = ref(false)
const geminiKey = ref(localStorage.getItem('gemini_key') || '')
const isRecognizing = ref(false)
const localTextContent = reactive({}) // Map: lineId -> string
const focusedLineId = ref(null)

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

// --- State for region labeling ---
const textlineLabels = reactive({}) 
const textlines = ref({}) 
const nodeToTextlineMap = ref({}) 
const hoveredTextlineId = ref(null)
const textboxLabels = ref(0) 
const labelColors = ['#448aff', '#ffeb3b', '#4CAF50', '#f44336', '#9c27b0', '#ff9800'] 

const scaleFactor = 0.7
const NODE_HOVER_RADIUS = 7
const EDGE_HOVER_THRESHOLD = 5

const manuscriptNameForDisplay = computed(() => localManuscriptName.value)
const currentPageForDisplay = computed(() => localCurrentPage.value)
const isFirstPage = computed(() => localPageList.value.indexOf(localCurrentPage.value) === 0)
const isLastPage = computed(
  () => localPageList.value.indexOf(localCurrentPage.value) === localPageList.value.length - 1
)

const scaledWidth = computed(() => Math.floor(dimensions.value[0] * scaleFactor))
const scaledHeight = computed(() => Math.floor(dimensions.value[1] * scaleFactor))
const scaleX = (x) => x * scaleFactor
const scaleY = (y) => y * scaleFactor
const graphIsLoaded = computed(() => workingGraph.nodes && workingGraph.nodes.length > 0)

// --- RECOGNITION MODE UTILS ---

// Initialize map keys based on current textlines structure
const initializeTextContent = () => {
    Object.keys(textlines.value).forEach(id => {
        if(!(id in localTextContent)) {
            localTextContent[id] = ""
        }
    })
}

// Calculate position for input overlay
const getLineInputStyle = (nodeIndices) => {
    if(!nodeIndices || nodeIndices.length === 0) return { display: 'none' };

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    
    nodeIndices.forEach(idx => {
        const n = workingGraph.nodes[idx];
        if (!n) return;
        if(n.x < minX) minX = n.x;
        if(n.y < minY) minY = n.y;
        if(n.x > maxX) maxX = n.x;
        if(n.y > maxY) maxY = n.y;
    });

    if (minX === Infinity) return { display: 'none' };

    const pad = 5;
    const width = (maxX - minX) + (pad * 2);
    const height = (maxY - minY) + (pad * 2); 
    
    return {
        left: `${scaleX(minX - pad)}px`,
        top: `${scaleY(maxY - pad)}px`,
        width: `${scaleX(width)}px`,
        height: `${scaleY(height)}px`, 
        position: 'absolute'
    }
}

// Trigger Gemini API
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
        if(data.error) throw new Error(data.error);
        
        // Update local state with recognized text
        if(data.transcriptions) {
            Object.entries(data.transcriptions).forEach(([id, text]) => {
                localTextContent[id] = text;
            });
        }
    } catch(e) {
        alert("Recognition failed: " + e.message);
    } finally {
        isRecognizing.value = false;
    }
}

const activateInput = (lineId) => {
    focusedLineId.value = lineId;
    nextTick(() => {
        const el = document.querySelector('.line-input.active');
        if(el) el.focus();
    });
}

const focusNextLine = (currentId) => {
    const ids = Object.keys(textlines.value).map(Number).sort((a,b) => a - b); 
    const currIdx = ids.indexOf(parseInt(currentId));
    if(currIdx !== -1 && currIdx < ids.length - 1) {
        activateInput(ids[currIdx + 1]);
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
  if (textboxModeActive.value) {
    if (isEKeyPressed.value) return 'crosshair'
    return 'pointer'
  }
  if (!textlineModeActive.value && !recognitionModeActive.value && !nodeModeActive.value) return 'default'
  if (nodeModeActive.value) return 'cell'; 
  if (isAKeyPressed.value) return 'crosshair'
  if (isDKeyPressed.value) return 'not-allowed'
  return 'default'
})

const downloadResults = async () => {
    try {
        const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/download-results/${localManuscriptName.value}`, {
            method: 'GET',
        });
        if (!response.ok) throw new Error('Download failed');
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${localManuscriptName.value}_results.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
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

const fetchPageData = async (manuscript, page) => {
  if (!manuscript || !page) {
    error.value = 'Manuscript or page not specified.'
    loading.value = false
    return
  }
  loading.value = true
  error.value = null
  modifications.value = []
  Object.keys(textlineLabels).forEach((key) => delete textlineLabels[key])
  
  // Reset text content on page load
  Object.keys(localTextContent).forEach(key => delete localTextContent[key])

  try {
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscript}/${page}`
    )
    if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch page data')
    const data = await response.json()

    dimensions.value = data.dimensions
    imageData.value = data.image || ''
    points.value = data.points.map((p) => ({ coordinates: [p[0], p[1]], segment: null }))

    if (data.graph) {
      graph.value = data.graph
    } else if (data.points?.length > 0) {
      graph.value = generateLayoutGraph(data.points)
      if (!isEditModeFlow.value) {
        await saveGeneratedGraph(manuscript, page, graph.value)
      }
    }
    if (data.textline_labels) {
      data.textline_labels.forEach((label, index) => {
        if (label !== -1) textlineLabels[index] = label
      })
    }
    if (data.textbox_labels && data.textbox_labels.length > 0) {
       data.textbox_labels.forEach((label, index) => {
           textlineLabels[index] = label
       })
       const maxLabel = Math.max(...data.textbox_labels);
       textboxLabels.value = maxLabel + 1; 
    }
    resetWorkingGraph()
  } catch (err) {
    console.error('Error fetching page data:', err)
    error.value = err.message
  } finally {
    loading.value = false
  }
}

const fetchPageList = async (manuscript) => {
  if (!manuscript) return
  try {
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/manuscript/${manuscript}/pages`
    )
    if (!response.ok) throw new Error('Failed to fetch page list')
    localPageList.value = await response.json()
  } catch (err) {
    console.error('Failed to fetch page list:', err)
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

const getNodeColor = (nodeIndex) => {
  const textlineId = nodeToTextlineMap.value[nodeIndex]
  if (textboxModeActive.value) {
    if (hoveredTextlineId.value !== null && hoveredTextlineId.value === textlineId) {
      return '#ff4081' 
    }
    const label = textlineLabels[nodeIndex]
    if (label !== undefined && label > -1) {
      return labelColors[label % labelColors.length]
    }
    return '#9e9e9e' 
  }
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return '#00bcd4'
  if (isNodeSelected(nodeIndex)) return '#ff9500'
  const edgeCount = nodeEdgeCounts.value[nodeIndex]
  if (edgeCount < 2) return '#f44336'
  if (edgeCount === 2) return '#4CAF50'
  if (edgeCount > 2) return '#2196F3'
  return '#cccccc'
}

const getNodeRadius = (nodeIndex) => {
  if (textboxModeActive.value) {
    const textlineId = nodeToTextlineMap.value[nodeIndex]
    if (hoveredTextlineId.value !== null && hoveredTextlineId.value === textlineId) {
      return 7
    }
    return 5
  }
  const edgeCount = nodeEdgeCounts.value[nodeIndex]
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return 7
  if (isNodeSelected(nodeIndex)) return 6
  return edgeCount < 2 ? 5 : 3
}
const getEdgeColor = (edge) => (edge.modified ? '#f44336' : '#ffffff')
const isNodeSelected = (nodeIndex) => selectedNodes.value.includes(nodeIndex)
const isEdgeSelected = (edge) => {
  return (
    selectedNodes.value.length === 2 &&
    ((selectedNodes.value[0] === edge.source && selectedNodes.value[1] === edge.target) ||
      (selectedNodes.value[0] === edge.target && selectedNodes.value[1] === edge.source))
  )
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
    if (nodeModeActive.value) {
        addNode(event.clientX, event.clientY);
        return;
    }
    if (!isAKeyPressed.value && !isDKeyPressed.value) resetSelection();
}

const onNodeClick = (nodeIndex, event) => {
    event.stopPropagation(); 
    if (nodeModeActive.value) return;
    if (isAKeyPressed.value || isDKeyPressed.value || textboxModeActive.value || recognitionModeActive.value) return;
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
    let newHoveredTextlineId = null
    for (let i = 0; i < workingGraph.nodes.length; i++) {
      const node = workingGraph.nodes[i]
      if (Math.hypot(mouseX - scaleX(node.x), mouseY - scaleY(node.y)) < NODE_HOVER_RADIUS) {
        newHoveredTextlineId = nodeToTextlineMap.value[i]
        break 
      }
    }
    if (newHoveredTextlineId === null) {
      for (const edge of workingGraph.edges) {
        const n1 = workingGraph.nodes[edge.source]
        const n2 = workingGraph.nodes[edge.target]
        if (n1 && n2 && distanceToLineSegment(mouseX, mouseY, scaleX(n1.x), scaleY(n1.y), scaleX(n2.x), scaleY(n2.y)) < EDGE_HOVER_THRESHOLD) {
          newHoveredTextlineId = nodeToTextlineMap.value[edge.source]
          break 
        }
      }
    }
    hoveredTextlineId.value = newHoveredTextlineId
    if (hoveredTextlineId.value !== null && isEKeyPressed.value) {
      labelTextline()
    }
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
    nodesToLabel.forEach((nodeIndex) => {
      textlineLabels[nodeIndex] = textboxLabels.value
    })
  }
}

const handleGlobalKeyDown = (e) => {
  const key = e.key.toLowerCase()
  if (key === 's' && !e.repeat) {
    if ((textlineModeActive.value || textboxModeActive.value || nodeModeActive.value || recognitionModeActive.value) && !loading.value && !isProcessingSave.value) {
      e.preventDefault()
      saveAndGoNext()
    }
    return
  }
  if (key === 'w' && !e.repeat) {
    e.preventDefault()
    setMode('edge')
    return
  }
  if (key === 'r' && !e.repeat) {
    e.preventDefault()
    setMode('region')
    return
  }
  if (key === 'n' && !e.repeat) {
    e.preventDefault()
    setMode('node')
    return
  }
  if (key === 't' && !e.repeat) { // T for Text recognition
    e.preventDefault()
    setMode('recognition')
    return
  }
  if (textboxModeActive.value && key === 'e' && !e.repeat) {
    e.preventDefault()
    isEKeyPressed.value = true
    return
  }
  if (textlineModeActive.value && !e.repeat) {
    if (key === 'd') {
      e.preventDefault()
      isDKeyPressed.value = true
      resetSelection()
    }
    if (key === 'a') {
      e.preventDefault()
      isAKeyPressed.value = true
      hoveredNodesForMST.clear()
      resetSelection()
    }
  }
}

const handleGlobalKeyUp = (e) => {
  const key = e.key.toLowerCase()
  if (textboxModeActive.value && key === 'e') {
    isEKeyPressed.value = false
    textboxLabels.value++ 
  }
  if (textlineModeActive.value) {
    if (key === 'd') {
      isDKeyPressed.value = false
    }
    if (key === 'a') {
      isAKeyPressed.value = false
      if (hoveredNodesForMST.size >= 2) {
        addMSTEdges()
      }
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
  calculateMST(Array.from(hoveredNodesForMST), workingGraph.nodes).forEach((edge) => {
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
  } catch (e) {
    console.error('Error saving generated graph:', e)
  }
}

const saveModifications = async () => {
  const numNodes = workingGraph.nodes.length
  const labelsToSend = new Array(numNodes).fill(0) 
  for (const nodeIndex in textlineLabels) {
    if (nodeIndex < numNodes) {
        labelsToSend[nodeIndex] = textlineLabels[nodeIndex]
    }
  }
  const dummyTextlineLabels = new Array(numNodes).fill(-1);
  const requestBody = {
    graph: workingGraph, 
    modifications: modifications.value,
    textlineLabels: dummyTextlineLabels, 
    textboxLabels: labelsToSend,
    textContent: localTextContent // Send recognized/edited text to backend
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
    const data = await res.json()
    graph.value = JSON.parse(JSON.stringify(workingGraph))
    modifications.value = []
    error.value = null
  } catch (err) {
    error.value = err.message
    throw err
  }
}

const saveCurrentGraph = async () => {
  if (isProcessingSave.value) return
  isProcessingSave.value = true
  try {
    await saveModifications()
  } catch (err) {
    alert(`Save failed: ${err.message}`)
  } finally {
    isProcessingSave.value = false
  }
}

const confirmAndNavigate = async (navAction) => {
  if (isProcessingSave.value) return
  if (modifications.value.length > 0 || (recognitionModeActive.value && Object.keys(localTextContent).some(k => localTextContent[k]))) {
    // Note: Checking for text changes specifically is complex without dirty tracking, 
    // but saving harmlessly re-writes XML.
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

const navigateToPage = (page) => {
  emit('page-changed', page)
}

const previousPage = () =>
  confirmAndNavigate(() => {
    const currentIndex = localPageList.value.indexOf(localCurrentPage.value)
    if (currentIndex > 0) {
      navigateToPage(localPageList.value[currentIndex - 1])
    }
  })

const nextPage = () =>
  confirmAndNavigate(() => {
    const currentIndex = localPageList.value.indexOf(localCurrentPage.value)
    if (currentIndex < localPageList.value.length - 1) {
      navigateToPage(localPageList.value[currentIndex + 1])
    }
  })

const saveAndGoNext = async () => {
  if (loading.value || isProcessingSave.value) return
  isProcessingSave.value = true
  try {
    await saveModifications()
    const currentIndex = localPageList.value.indexOf(localCurrentPage.value)
    if (currentIndex < localPageList.value.length - 1) {
      navigateToPage(localPageList.value[currentIndex + 1])
    } else {
      alert('This was the Last page. Saved successfully!')
    }
  } catch (err) {
    alert(`Save failed: ${err.message}`)
  } finally {
    isProcessingSave.value = false
  }
}

const runHeuristic = () => {
  if(!points.value.length) return;
  const rawPoints = points.value.map(p => [p.coordinates[0], p.coordinates[1], 10]); 
  const heuristicGraph = generateLayoutGraph(rawPoints);
  workingGraph.edges = heuristicGraph.edges.map(e => ({
     source: e.source, 
     target: e.target, 
     label: e.label, 
     modified: true 
  }));
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

watch(
  () => props.pageName,
  (newPageName) => {
    if (newPageName && newPageName !== localCurrentPage.value) {
      localCurrentPage.value = newPageName
      fetchPageData(localManuscriptName.value, newPageName)
    }
  }
)

watch(textlineModeActive, (val) => {
  if (val) {
    textboxModeActive.value = false
    nodeModeActive.value = false
    recognitionModeActive.value = false
  } else {
    resetSelection()
    isAKeyPressed.value = false
    isDKeyPressed.value = false
    hoveredNodesForMST.clear()
  }
})

watch(textboxModeActive, (val) => {
  if (val) {
    textlineModeActive.value = false
    nodeModeActive.value = false
    recognitionModeActive.value = false
    resetSelection()
    const existingLabels = Object.values(textlineLabels)
    if (existingLabels.length > 0) {
      const maxLabel = Math.max(...existingLabels)
      textboxLabels.value = maxLabel + 1
    } else {
      textboxLabels.value = 0
    }
  } else {
    hoveredTextlineId.value = null
  }
})

watch(nodeModeActive, (val) => {
  if (val) {
    textlineModeActive.value = false
    textboxModeActive.value = false
    recognitionModeActive.value = false
    resetSelection()
  }
})

watch(recognitionModeActive, (val) => {
    if(val) {
        textlineModeActive.value = false
        textboxModeActive.value = false
        nodeModeActive.value = false
        resetSelection()
    }
})
</script>

<style scoped>
.manuscript-viewer {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  overflow: hidden;
  background-color: #1e1e1e; /* Darker overall background */
  color: #e0e0e0;
  font-family: 'Roboto', sans-serif;
}

/* --- TOP RAIL --- */
.top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 16px;
  height: 60px;
  background-color: #2c2c2c;
  border-bottom: 1px solid #3d3d3d;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  flex-shrink: 0;
  z-index: 10;
}

.top-bar-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.top-bar-left {
  display: flex;
  align-items: center;
  gap: 16px;
  flex: 1;        /* Takes available space to the left of the actions */
  min-width: 0;   /* Essential for text truncation to work in flexbox */
  margin-right: 20px; /* Safety spacing from the right side controls */
}

.page-title {
  font-size: 1.1rem;
  font-weight: 500;
  color: #fff;
  white-space: nowrap;     /* Keeps text on one line */
  overflow: hidden;        /* Hides overflow */
  text-overflow: ellipsis; /* Adds '...' if the name is too long */
}
.page-title .divider {
  color: #777;
  margin: 0 8px;
}

.action-group {
  display: flex;
  gap: 8px;
}

.separator {
  width: 1px;
  height: 24px;
  background-color: #555;
  margin: 0 4px;
}

/* Button Styling */
button {
  border: none;
  cursor: pointer;
  border-radius: 4px;
  font-size: 0.9rem;
  transition: all 0.2s;
}

.nav-btn {
  background-color: transparent;
  color: #aaa;
  padding: 8px 12px;
  flex-shrink: 0; /* Prevents the button from shrinking */
  display: flex;  /* Ensures internal icon/text alignment */
  align-items: center;
}
.nav-btn:hover:not(:disabled) {
  background-color: rgba(255, 255, 255, 0.1);
  color: #fff;
}
.nav-btn.secondary {
  border: 1px solid #555;
}

.action-btn {
  background-color: #424242;
  color: #fff;
  padding: 8px 16px;
  border: 1px solid #555;
}
.action-btn:hover:not(:disabled) {
  background-color: #505050;
}
.action-btn.primary {
  background-color: #4CAF50; /* Green Save Button */
  border-color: #43a047;
  font-weight: 500;
}
.action-btn.primary:hover:not(:disabled) {
  background-color: #5cb860;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* --- MAIN VIEW --- */
.visualization-container {
  position: relative;
  overflow: auto;
  flex-grow: 1;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 2rem;
  background-color: #121212;
}
.image-container {
  position: relative;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
}
.manuscript-image {
  display: block;
  user-select: none;
  opacity: 0.7;
}
.placeholder-image {
  background-color: #333;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #777;
}
.graph-overlay {
  position: absolute;
  top: 0;
  left: 0;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s ease-in-out;
}
.graph-overlay.is-visible {
  opacity: 1;
  pointer-events: auto;
}

/* Input Overlay */
.input-overlay-container {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none; 
    z-index: 50;
}

.line-input-wrapper {
    pointer-events: auto; 
    background: rgba(0, 0, 0, 0.3);
    border: 1px dashed rgba(255, 255, 255, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
}

.line-input {
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    color: #fff;
    border: 1px solid #4CAF50;
    padding: 2px 5px;
    font-size: 12px;
}

.line-input-display {
    width: 100%;
    height: 100%;
    cursor: text;
    color: rgba(255,255,255,0.5);
    font-size: 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding: 2px;
    display: flex;
    align-items: center;
}
.line-input-display:hover {
    background: rgba(255,255,255,0.1);
}
.line-input-display.has-text {
    color: #fff;
    font-weight: bold;
    background: rgba(0, 0, 0, 0.5);
}


/* Loading/Error States */
.processing-save-notice,
.loading,
.error-message {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  padding: 20px 30px;
  border-radius: 8px;
  z-index: 10000;
  text-align: center;
  box-shadow: 0 4px 15px rgba(0,0,0,0.5);
}
.processing-save-notice { background-color: rgba(33, 33, 33, 0.95); border: 1px solid #444; color: #fff; }
.error-message { background-color: #c62828; color: white; }
.loading { font-size: 1.2rem; color: #aaa; background: rgba(0,0,0,0.5); }


/* --- BOTTOM RAIL --- */
.bottom-panel {
  background-color: #2c2c2c;
  border-top: 1px solid #3d3d3d;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  
  /* FIXED HEIGHT for smoothness */
  height: 280px; 
  transition: height 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); /* Smoother bezier curve */
  
  box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
  overflow: hidden; /* Prevents scrollbars appearing during transition */
  will-change: height; /* Optimization for animation performance */
}

.bottom-panel.is-collapsed {
  height: 45px; /* Exact height of the tab bar */
}

/* Mode Tabs */
.mode-tabs {
  display: flex;
  background-color: #212121;
  height: 45px;
  flex-shrink: 0; /* Prevents tabs from shrinking if panel acts up */
  align-items: stretch;
}

.mode-tab {
  flex: 1;
  padding: 0 12px;
  background: transparent;
  border: none;
  border-bottom: 3px solid transparent;
  color: #888;
  border-radius: 0;
  text-transform: uppercase;
  font-size: 0.85rem;
  letter-spacing: 0.5px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.mode-tab:hover:not(:disabled) {
  background-color: #2a2a2a;
  color: #bbb;
}
.mode-tab.active {
  background-color: #2c2c2c;
  color: #448aff;
  border-bottom-color: #448aff;
  font-weight: 500;
}

.tab-spacer {
  flex-grow: 1;
  background-color: #212121;
}

.panel-toggle-btn {
  background-color: #333;
  color: #aaa;
  border: none;
  border-left: 1px solid #444;
  padding: 0 16px;
  font-size: 0.8rem;
  border-radius: 0;
  min-width: 100px;
}
.panel-toggle-btn:hover {
  background-color: #444;
  color: #fff;
}


/* Help Content */
.help-content-area {
  padding: 16px 24px;
  display: flex;
  flex-grow: 1;
  gap: 24px;
  height: 100%; /* Fill remaining space */
  overflow: hidden; 
}

.help-section {
  display: flex;
  gap: 24px;
  flex-grow: 1;
  align-items: flex-start;
  height: 100%;
}

.media-container {
  /* CHANGED: Square dimensions */
  width: 200px;
  height: 200px;
  
  flex-shrink: 0;
  background-color: #000;
  border-radius: 6px;
  border: 1px solid #444;
  overflow: hidden;
}

.tutorial-video {
  width: 100%;
  height: 100%;
  object-fit: contain; /* Keeps aspect ratio inside the fixed box */
  display: block;
}

.webm-placeholder {
  width: 100%;
  height: 100%;
  background-color: #3a3a3a;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #777;
  font-size: 0.8rem;
}

.instructions-container {
  flex-grow: 1;
  max-width: 600px;
  overflow-y: auto; /* Allows text to scroll if it exceeds fixed height */
  height: 100%;
  padding-right: 10px; /* Space for scrollbar */
}

/* Scrollbar styling for instructions */
.instructions-container::-webkit-scrollbar {
  width: 6px;
}
.instructions-container::-webkit-scrollbar-track {
  background: #2c2c2c;
}
.instructions-container::-webkit-scrollbar-thumb {
  background: #555;
  border-radius: 3px;
}

.instructions-container h3 {
  margin: 0 0 10px 0;
  font-size: 1.1rem;
  color: #fff;
}
.instructions-container p {
  color: #ccc;
  font-size: 0.95rem;
  line-height: 1.5;
  margin-bottom: 12px;
}
.instructions-container ul {
  padding-left: 20px;
  color: #ccc;
  font-size: 0.95rem;
  margin-bottom: 16px;
}
.instructions-container li {
  margin-bottom: 6px;
}
code {
  background-color: #424242;
  padding: 2px 4px;
  border-radius: 3px;
  font-family: monospace;
  color: #ffb74d;
}

/* Context Buttons */
.context-actions {
  display: flex;
  gap: 10px;
  margin-top: 10px;
}
.primary-action {
  background-color: #2196F3;
  border-color: #1976D2;
}
.danger-action {
  background-color: #f44336;
  border-color: #d32f2f;
}

/* API Input & Inline Form */
.form-group-inline {
    display: flex;
    gap: 10px;
    margin-bottom: 10px;
}
.api-input {
    background: #444;
    border: 1px solid #555;
    color: #fff;
    padding: 5px 10px;
    flex-grow: 1;
}

/* Log Sidebar */
.log-sidebar {
  width: 200px;
  background-color: #222;
  border: 1px solid #444;
  border-radius: 4px;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}
.log-header {
  padding: 8px 10px;
  background-color: #333;
  border-bottom: 1px solid #444;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.85rem;
}
.text-btn {
  background: none;
  border: none;
  color: #f44336;
  padding: 0;
  font-size: 0.8rem;
}
.log-list {
  list-style: none;
  padding: 0;
  margin: 0;
  overflow-y: auto;
  max-height: 120px;
}
.log-list li {
  padding: 6px 10px;
  border-bottom: 1px solid #333;
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: #aaa;
}
.undo-icon {
  background: none;
  border: none;
  padding: 0;
  font-size: 1.1rem;
  color: #777;
}
.undo-icon:hover { color: #fff; }

</style>