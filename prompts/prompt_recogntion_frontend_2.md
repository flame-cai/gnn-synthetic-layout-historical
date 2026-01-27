As an expert software development, specializing front-end sofware development using .vue, please assist me in making precise improvements to the below code.

This code is part of a larger system which performs layout analysis on manuscript images using a Graph Neural Network (GNN). The layout analysis problem is formulated in a graph based manner, where characters are treated as nodes and characters of the same text lines are connected with edges. Thus nodes containing the same textline have the same text line label. The user can also label nodes with textbox labels, marking nodes of each text box with the same integer label. Once labelled (using gnn layout inference + manual corrections), the system generates PAGE XML files containing textbox and text line bounding polygons, along with visualizations. The system also saves textline images, for each textbox.
After detecting bounding polygons using the graph neural network, we use Gemini API to recognize text context from each textline in the "Recognition Mode".

I want your help in updating the User interface and User Experience, more specifically making adjustments to the the various "modes" of annotations, and loading manuscripts which have been worked on and saved in a previous session. More precisely we want to make the following changes: 


1) right now, when typing in the text input field in the recognition mode sometimes causes the bug. I cannot type hotkeys such as "s" or "n". So please disable hotkeys when any text input field is selected.



2) right now, to recognize the text content from the manuscript, i first need to save the manuscript which will create graph based data and PAGE-XML files (without the text). Then in the recognition mode, I enter API key, and Gemini recognizes the text and populates the relevant text in the PAGE-XML files, and then I save it again. This is workflow is a bit tedious. Instead we want to do the following: 
- In the main top panel, near the save and next button, we should allow the user to enter their API key, and toggle auto-recognition model ON or OFF. Turning the auto-recogntion mode ON, will not require the user to go the recognition mode to recognize the text using Gemini. Instead, saving the page in any other mode, will automatically make the API call to Gemini in the backend, while the frontend will go to next page. So in the backend, we will save PAGE-XML file without text, and if auto-recognition mode is ON, we will use update this PAGE-XML automatically with the text in the backend. Please understand this carefully before making the changes.

3) right now the tool does not allow the user to load previously worked on manuscript. We need to implement this functionality carefully, taking into account the different types of annotations and other miscellenious data saved in graph based .txt format and the PAGE-XML format. 

Please first understand the code below, play a red team application tester and think carefully to write robust code, with good logging for easy debugging, with assert statements where required.
Do not make unnecessary changes to other parts of the code, make precise changes.


#ManuscriptViewer.vue
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


const triggerRecognition = async () => {
    if(!geminiKey.value) return alert("Please enter an API Key");
    
    // Save key for convenience
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
        
        if (!res.ok) {
            throw new Error(data.error || "Unknown server error");
        }
        
        if (data.transcriptions) {
            let count = 0;
            // The backend now guarantees a clean dictionary: { "0": "text", "1": "text" }
            Object.entries(data.transcriptions).forEach(([id, text]) => {
                // Only update if we got actual text back
                if (text !== null && text !== undefined) {
                    localTextContent[id] = text;
                    count++;
                }
            });
            
            // Optional: Provide feedback
            if (count === 0) {
                alert("Gemini finished but returned no text. Check if the image is clear.");
            }
        }
    } catch(e) {
        console.error(e);
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


# app.py
# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import shutil
from pathlib import Path
import base64
import json
import zipfile
import io
import google.generativeai as genai # NEW IMPORT
import glob # NEW IMPORT
from PIL import Image

# Import your existing pipelines
from inference import process_new_manuscript
from gnn_inference import run_gnn_prediction_for_page, generate_xml_and_images_for_page
from segmentation.utils import load_images_from_folder
import xml.etree.ElementTree as ET # Ensure this is imported

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = './input_manuscripts'
MODEL_CHECKPOINT = "./pretrained_gnn/v2.pt"
DATASET_CONFIG = "./pretrained_gnn/gnn_preprocessing_v2.yaml"

@app.route('/upload', methods=['POST'])
def upload_manuscript():
    """
    Step 1 & 2: Upload images, resize (inference.py), Generate Heatmaps & GNN inputs.
    """
    manuscript_name = request.form.get('manuscriptName', 'default_manuscript')
    longest_side = int(request.form.get('longestSide', 2500))
    # --- MODIFIED: Parse min_distance ---
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
        # Run Step 1-3: Resize and Generate Heatmaps/Points
        # --- MODIFIED: Pass min_distance ---
        process_new_manuscript(manuscript_path, target_longest_side=longest_side, min_distance=min_distance) 
        
        # Get list of processed pages
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

@app.route('/semi-segment/<manuscript>/<page>', methods=['GET'])
def get_page_prediction(manuscript, page):
    """
    Step 4 Inference: Run GNN, get graph, return to frontend.
    """
    print("Received request for manuscript:", manuscript, "page:", page)
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    try:
        # Run GNN Inference
        graph_data = run_gnn_prediction_for_page(
            str(manuscript_path), 
            page, 
            MODEL_CHECKPOINT, 
            DATASET_CONFIG
        )
        
        # Load Image to send to frontend
        img_path = manuscript_path / "images_resized" / f"{page}.jpg"
        
        if not img_path.exists():
            return jsonify({"error": "Image not found"}), 404

        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        response = {
            "image": encoded_string,
            "dimensions": graph_data['dimensions'],
            "points": [[n['x'], n['y']] for n in graph_data['nodes']],
            "graph": graph_data,
            "textline_labels": graph_data.get('textline_labels', []),
            "textbox_labels": graph_data.get('textbox_labels', []) # Return textbox labels if they exist
        }
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# 1. Update save_correction to receive text content
@app.route('/semi-segment/<manuscript>/<page>', methods=['POST'])
def save_correction(manuscript, page):
    data = request.json
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript
    
    textline_labels = data.get('textlineLabels')
    graph_data = data.get('graph')
    textbox_labels = data.get('textboxLabels')
    nodes_data = graph_data.get('nodes')
    text_content = data.get('textContent') # <--- NEW: Get text from frontend
    
    if not textline_labels or not graph_data:
        return jsonify({"error": "Missing labels or graph data"}), 400

    try:
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
            text_content=text_content # <--- PASS TO LOGIC
        )
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/recognize-text', methods=['POST'])
def recognize_text():
    """
    Refined based on Google AI docs:
    1. Uses gemini-1.5-flash (optimized for multimodal speed/cost).
    2. Uses native JSON Mode for robust output.
    3. Normalizes coordinates to 0-1000 (Gemini native scale).
    """
    data = request.json
    manuscript = data.get('manuscript')
    page = data.get('page')
    api_key = data.get('apiKey')
    
    if not api_key:
        return jsonify({"error": "API Key required"}), 400

    # 1. Setup Paths
    base_path = Path(UPLOAD_FOLDER) / manuscript
    xml_path = base_path / "layout_analysis_output" / "page-xml-format" / f"{page}.xml"
    img_path = base_path / "images_resized" / f"{page}.jpg"
    
    if not xml_path.exists() or not img_path.exists():
        return jsonify({"error": "Page XML or Image not found. Please save layout first."}), 404

    # 2. Load Image & Dimensions
    try:
        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
    except Exception as e:
        return jsonify({"error": f"Failed to load image: {str(e)}"}), 500

    # 3. Parse XML & Prepare Regions
    # We map "structure_line_id" -> [ymin, xmin, ymax, xmax]
    regions_to_process = []
    
    ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for textline in root.findall(".//p:TextLine", ns):
            custom_attr = textline.get('custom', '')
            
            # Extract ID
            if 'structure_line_id_' not in custom_attr:
                continue
            try:
                line_id = str(custom_attr.split('structure_line_id_')[1])
            except IndexError:
                continue

            # Extract Coords
            coords_elem = textline.find('p:Coords', ns)
            if coords_elem is None: continue
            points_str = coords_elem.get('points', '')
            if not points_str: continue

            try:
                points = [list(map(int, p.split(','))) for p in points_str.strip().split(' ')]
            except ValueError: continue
            
            if not points: continue

            # Convert Polygon -> Bounding Box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            # Normalize to 0-1000 (Gemini Native Scale)
            # Formula: int(val / dimension * 1000)
            n_ymin = int((min(ys) / img_h) * 1000)
            n_xmin = int((min(xs) / img_w) * 1000)
            n_ymax = int((max(ys) / img_h) * 1000)
            n_xmax = int((max(xs) / img_w) * 1000)

            # Clamp & Sort (Safety)
            n_ymin, n_ymax = sorted([max(0, min(1000, n_ymin)), max(0, min(1000, n_ymax))])
            n_xmin, n_xmax = sorted([max(0, min(1000, n_xmin)), max(0, min(1000, n_xmax))])

            regions_to_process.append({
                "id": line_id,
                "box_2d": [n_ymin, n_xmin, n_ymax, n_xmax]
            })

    except Exception as e:
        return jsonify({"error": f"XML Parsing Error: {str(e)}"}), 500

    if not regions_to_process:
         return jsonify({"transcriptions": {}})

    # 4. Construct Prompt
    # We ask for a list of objects, which is more robust for JSON mode than dynamic keys.
    prompt_text = (
        "You are an expert paleographer analyzing a historical manuscript.\n"
        "Your task is to transcribe the handwritten text found inside specific bounding boxes.\n\n"
        "INPUT CONTEXT:\n"
        "The coordinates are in the format [ymin, xmin, ymax, xmax] on a scale of 0 to 1000.\n\n"
        "REGIONS TO TRANSCRIBE:\n"
    )
    
    for item in regions_to_process:
        prompt_text += f"- Region ID '{item['id']}' at Box: {item['box_2d']}\n"

    prompt_text += (
        "\nOUTPUT INSTRUCTIONS:\n"
        "1. Return a JSON List of objects.\n"
        "2. Each object must have two keys: 'id' (string) and 'text' (string).\n"
        "3. Do not modify the Region ID.\n"
        "4. If the text is illegible, set 'text' to an empty string.\n"
    )

    # 5. Call Gemini API
    try:
        genai.configure(api_key=api_key)
        
        # Use 1.5-flash (Best for OCR speed/cost)
        model = genai.GenerativeModel('gemini-2.5-flash')

        response = model.generate_content(
            [pil_img, prompt_text],
            generation_config={
                "response_mime_type": "application/json",
                # We expect a structure like: [{"id": "1", "text": "abc"}, ...]
            }
        )
        
        # 6. Process Response
        # Because we used response_mime_type, .text is guaranteed to be JSON (no markdown backticks)
        raw_result = json.loads(response.text)
        
        # Convert List back to Map for Frontend: { "1": "abc", "2": "def" }
        # Handle cases where Gemini might wrap the list in a root key like {"result": [...]}
        result_list = []
        if isinstance(raw_result, list):
            result_list = raw_result
        elif isinstance(raw_result, dict):
            # Try to find the first list value
            for val in raw_result.values():
                if isinstance(val, list):
                    result_list = val
                    break

        final_map = {}
        for item in result_list:
            if 'id' in item and 'text' in item:
                final_map[str(item['id'])] = item['text']

        return jsonify({"transcriptions": final_map})

    except Exception as e:
        print(f"Gemini Error: {e}")
        # Detailed error for debugging
        return jsonify({"error": str(e)}), 500


@app.route('/save-graph/<manuscript>/<page>', methods=['POST'])
def save_generated_graph(manuscript, page):
    return jsonify({"status": "ok"})

# --- NEW: Endpoint to download results ---
@app.route('/download-results/<manuscript>', methods=['GET'])
def download_results(manuscript):
    manuscript_path = Path(UPLOAD_FOLDER) / manuscript / "layout_analysis_output"
    
    if not manuscript_path.exists():
         return jsonify({"error": "No output found for this manuscript"}), 404
         
    # Directories to zip
    xml_dir = manuscript_path / "page-xml-format"
    img_dir = manuscript_path / "image-format"
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add XMLs
        if xml_dir.exists():
            for root, dirs, files in os.walk(xml_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.join('page-xml-format', os.path.relpath(file_path, xml_dir))
                    zf.write(file_path, arcname)
                    
        # Add Line Images
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

# backend
# ssh -N -L 5001:localhost:5000 kartik@192.168.8.12

# frontend
# ssh -L 8000:localhost:5173 kartik@192.168.8.12




# inference.py

import os
import argparse
import gc
from PIL import Image
import torch

from segmentation.segment_graph import images2points
from gnn_inference import run_gnn_inference



def process_new_manuscript(manuscript_path="./input_manuscripts/sample_manuscript_1"):
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

                # 2. RESIZING: Downscale only if too large
                target_longest_side = 2500
                
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
    images2points(resized_images_path) 
    
    # Cleanup resources
    torch.cuda.empty_cache()
    gc.collect()

    print("Processing complete.")






if __name__ == "__main__":
    # 1. Parse standard CLI arguments4
    parser = argparse.ArgumentParser(description="GNN Layout Analysis Inference")
    parser.add_argument("--manuscript_path", type=str, default="./input_manuscripts/sample_manuscript_1", help="Path to the manuscript directory")
    args = parser.parse_args()

    # the data preparation.yaml is tied to the model_checkpoint used.
    args.model_checkpoint = "./pretrained_gnn/v2.pt"
    args.dataset_config_path = "./pretrained_gnn/gnn_preprocessing_v2.yaml"

    # -- Hyperparameters
    args.visualize = True
    args.BINARIZE_THRESHOLD = 0.5098
    args.BBOX_PAD_V = 0.7
    args.BBOX_PAD_H = 0.5
    args.CC_SIZE_THRESHOLD_RATIO = 0.4

    process_new_manuscript(args.manuscript_path)
    run_gnn_inference(args)



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
