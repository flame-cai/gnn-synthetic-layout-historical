As an expert in frontend development with javascript, vue.js please help me convert my fully autonomous python application into a semi-autonomous applications which allows humans to be in the loop.

At present the application uses a Graph Neural Network to segment text-line images from historical manuscript images in a fully automatic way as follows:

1) Resize the original images
2) Convert the images to heatmaps
3) Convert the heatmap into a GNN friendly format with nodes as character locations (x,y) and font_size.
	Hence the gnn-dataset at his stage looks like: looks like:
	
	{page_id}_dims.txt containing:
	1250.0 642.0
	
	{page_id}_inputs_normalized.txt containing:
	0.455200 0.158400 0.017600
	0.592000 0.158400 0.011200
	0.392800 0.159200 0.008800
	0.430400 0.159200 0.010400
	0.447200 0.159200 0.013600
	
	{page_id}_inputs_unnormalized.txt containing:
	569.000000 198.000000 22.000000
	740.000000 198.000000 14.000000
	491.000000 199.000000 11.000000
	538.000000 199.000000 13.000000
	559.000000 199.000000 17.000000
4) Then a GNN pipeline is used label points (nodes) belonging to the same text-line with the same label. Hence the GNN pipeline creates a {page_id}_labels_textline.txt file.

	{page_id}_labels_textline.txt containing:
	0
	0
	0
	0
	0

	In this example, all 5 points belong to the same line hence have the same label 0.
5) In the last step, the predicted labels in gnn format are used to generate a PAGE XML file and segmented line images for that page.


However a problem with this fully automatic pipelines is that some times the predicted labels in {page_id}_labels_textline.txt are incorrect. Hence we would like to have human supervision at that step, such that a human a manually verify if the predictions are correct, and if they are not, the human should be able to correct them.

Hence, I want your help converting this fully automatic pipeline to an application with a fronted and backend, with the frontend being "in between" step 4 and step 5, allowing the user to verify and make corrections to the node labels before proceeding to segment line images and create PAGE XML files.

Do not make unnecessary changes to the gnn inference pipeline. Please also update the readme.md to reflect these changes to help users understand how to install and use this new semi-autonomous application.

Hence the user flow should be as follows:
1) On the main page, the user uploads the manuscript images to be processed. The main page also allows the user to modify the default hyperparameters: min_distance=20 used in step 3 above for feature engineering, and the dimensions used to resize the original images in step 1 above (default longest side=2500).
2) After uploading, the backend processes the images till step 4 above, and then serves the frontend a page where the user can see the manuscript image overlayed with the detected character points (nodes) and edges between them. The frontend for page 1 should be displayed once the GNN inference (till step 4) is complete for that page, while the GNN inference for other pages can continue in the background. Handle this carefully to ensure smooth user experience.
3) Using the frontend, the user can then verify if the node labels are correct. If they are not correct, the user can correct them manually. In the frontend the user can also optionally choose to use the heuristic algorithm (same as used in LayoutGraphGenerator.js) to automatically connect points of the same text-line together, and then the user can make minor corrections to that if needed. The heuristic algorithm does the same thing the GNN inference in the backend does, but is less accurate.
4) Then the user can click a "Save and Proceed" button, which saves the corrected node labels to {page_id}_labels_textline.txt and then the backend code proceeds to step 5 above, generating the PAGE XML files and segmented line images for that page. The frontend then displays the next page for verification, while the backend continues processing other pages in the background.
5) This continues till all pages are processed.

Please take care to ensure that the core GNN inference pipeline code is not changed unnecessarily, and that the new frontend and backend code is modular and cleanly integrated into the existing codebase.
Think carefully about the best way to implement this new frontend and backend to achieve the above user flow. 
IMPORTANT: Please rewrite entire files where required, or please tell me what to replace with what in which files to achieve this. 

Please take inspiration from the below vue.js code and js code for building the frontend, but feel free to modify as needed to suit the new requirements.

vue.js code:

<template>
  <div class="manuscript-viewer">
    <!-- Top Toolbar: Collapsible -->
    <div class="toolbar">
      <h10>{{ manuscriptNameForDisplay }} - Page {{ currentPageForDisplay }}</h10>
      <div v-show="!isToolbarCollapsed" class="toolbar-controls">
        <button @click="previousPage" :disabled="loading || isProcessingSave || isFirstPage">
          Previous
        </button>
        <button @click="nextPage" :disabled="loading || isProcessingSave || isLastPage">Next</button>
        <button @click="saveAndGoNext" :disabled="loading || isProcessingSave">
          Save & Next (S)
        </button>
        <button @click="goToIMG2TXTPage" :disabled="loading || isProcessingSave">
          Annotate Text
        </button>
        <div class="toggle-container">
          <label>
            <input type="checkbox" v-model="editModeActive" :disabled="isProcessingSave" />
            Edge Edit (W)
          </label>
        </div>
        <div class="toggle-container">
          <label>
            <input
              type="checkbox"
              v-model="regionLabelingModeActive"
              :disabled="isProcessingSave || !graphIsLoaded"
            />
            Region Labeling (R)
          </label>
        </div>
      </div>
      <button class="panel-toggle-btn" @click="isToolbarCollapsed = !isToolbarCollapsed">
        {{ isToolbarCollapsed ? 'Show Toolbar' : 'Hide' }}
      </button>
    </div>

    <!-- Main Content: Visualization Area -->
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

        <svg
          v-if="graphIsLoaded"
          class="graph-overlay"
          :class="{ 'is-visible': editModeActive || regionLabelingModeActive }"
          :width="scaledWidth"
          :height="scaledHeight"
          :style="{ cursor: svgCursor }"
          @click="editModeActive && onBackgroundClick($event)"
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
            @click.stop="editModeActive && onEdgeClick(edge, $event)"
          />

          <circle
            v-for="(node, nodeIndex) in workingGraph.nodes"
            :key="`node-${nodeIndex}`"
            :cx="scaleX(node.x)"
            :cy="scaleY(node.y)"
            :r="getNodeRadius(nodeIndex)"
            :fill="getNodeColor(nodeIndex)"
            @click.stop="editModeActive && onNodeClick(nodeIndex, $event)"
          />

          <line
            v-if="
              editModeActive &&
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
      </div>
    </div>

    <!-- Bottom Panel: Collapsible -->
    <div class="bottom-panel">
      <div class="panel-toggle-bar" @click="isControlsCollapsed = !isControlsCollapsed">
        <div class="edit-instructions">
          <p v-if="isControlsCollapsed && regionLabelingModeActive">
            Hold 'e' and hover over lines to label them. Release 'e' and press again for the next
            label. 's' to save.
          </p>
          <p v-else-if="isControlsCollapsed && editModeActive">
            Hold 'a' to connect, 'd' to delete. Press 's' to save & next. Toggle modes with 'w'/'r'.
          </p>
          <p v-else-if="isControlsCollapsed && !editModeActive && !regionLabelingModeActive">
            Press 'w' to edit edges, 'r' to label regions.
          </p>
          <p v-else-if="regionLabelingModeActive">
            Hold 'e' to label textlines with the current label. Release and press 'e' again to move
            to the next label.
          </p>
          <p v-else-if="editModeActive && !isAKeyPressed && !isDKeyPressed">
            Select nodes to manage edges, or use hotkeys.
          </p>
          <p v-else-if="editModeActive && isAKeyPressed">Release 'A' to connect nodes.</p>
          <p v-else-if="editModeActive && isDKeyPressed">Release 'D' to stop deleting.</p>
        </div>
        <button class="panel-toggle-btn">
          {{ isControlsCollapsed ? 'Show Controls' : 'Hide Controls' }}
        </button>
      </div>

      <div v-show="!isControlsCollapsed" class="bottom-panel-content">
        <div v-if="editModeActive && !isAKeyPressed && !isDKeyPressed" class="edit-controls">
          <div class="edit-actions">
            <button @click="resetSelection">Cancel Selection</button>
            <button
              @click="addEdge"
              :disabled="selectedNodes.length !== 2 || edgeExists(selectedNodes[0], selectedNodes[1])"
            >
              Add Edge
            </button>
            <button
              @click="deleteEdge"
              :disabled="selectedNodes.length !== 2 || !edgeExists(selectedNodes[0], selectedNodes[1])"
            >
              Delete Edge
            </button>
          </div>
        </div>

        <div
          v-if="(editModeActive || regionLabelingModeActive) && graphIsLoaded"
          class="modifications-log-container"
        >
          <button @click="saveCurrentGraph" :disabled="loading || isProcessingSave">
            Save Graph & Labels
          </button>
          <div v-if="modifications.length > 0" class="modifications-details">
            <h3>Modifications ({{ modifications.length }})</h3>
            <button @click="resetModifications" :disabled="loading">Reset All Changes</button>
            <ul>
              <li
                v-for="(mod, index) in modifications"
                :key="index"
                class="modification-item"
              >
                {{ mod.type === 'add' ? 'Added' : 'Removed' }} edge: {{ mod.source }} ↔
                {{ mod.target }}
                <button @click="undoModification(index)" class="undo-button">Undo</button>
              </li>
            </ul>
          </div>
          <p v-else-if="!loading">No edge modifications in this session.</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, computed, watch, reactive } from 'vue'
import { useAnnotationStore } from '@/stores/annotationStore'
import { generateLayoutGraph } from './layout-analysis-utils/LayoutGraphGenerator.js'
import { useRouter } from 'vue-router'

const props = defineProps({
  manuscriptName: {
    type: String,
    default: null,
  },
  pageName: {
    type: String,
    default: null,
  },
})

const router = useRouter()
const annotationStore = useAnnotationStore()

const isEditModeFlow = computed(() => !!props.manuscriptName && !!props.pageName)

const localManuscriptName = ref('')
const localCurrentPage = ref('')
const localPageList = ref([])

const loading = ref(true)
const isProcessingSave = ref(false)
const error = ref(null)
const imageData = ref('')
const imageLoaded = ref(false)

const isToolbarCollapsed = ref(true)
const isControlsCollapsed = ref(true)
const editModeActive = ref(false)
const regionLabelingModeActive = ref(false)

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
const isEKeyPressed = ref(false) // For holding E
const hoveredNodesForMST = reactive(new Set())
const container = ref(null)
const svgOverlayRef = ref(null)

// --- State for region labeling ---
const regionLabels = reactive({}) // Maps node index to a region label (0, 1, 2...)
const textlines = ref({}) // Maps textline ID to a list of node indices
const nodeToTextlineMap = ref({}) // Maps node index to its textline ID
const hoveredTextlineId = ref(null)
const currentLabelIndex = ref(0) // The current label to apply (0, 1, 2, ...)
const labelColors = ['#448aff', '#ffeb3b', '#4CAF50', '#f44336', '#9c27b0', '#ff9800'] // Colors for different labels

const scaleFactor = 1.0
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

const svgCursor = computed(() => {
  if (regionLabelingModeActive.value) {
    if (isEKeyPressed.value) return 'crosshair'
    return 'pointer'
  }
  if (!editModeActive.value) return 'default'
  if (isAKeyPressed.value) return 'crosshair'
  if (isDKeyPressed.value) return 'not-allowed'
  return 'default'
})

const computeTextlines = () => {
  if (!graphIsLoaded.value) return
  const numNodes = workingGraph.nodes.length
  const adj = Array(numNodes)
    .fill(0)
    .map(() => [])
  for (const edge of workingGraph.edges) {
    adj[edge.source].push(edge.target)
    adj[edge.target].push(edge.source)
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
  Object.keys(regionLabels).forEach((key) => delete regionLabels[key])

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
    if (data.region_labels) {
      data.region_labels.forEach((label, index) => {
        if (label !== -1) {
          regionLabels[index] = label
        }
      })
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
  workingGraph.nodes.forEach((_, index) => {
    counts[index] = 0
  })

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

watch(
  () => workingGraph.edges,
  () => {
    updateUniqueNodeEdgeCounts()
    computeTextlines()
  },
  { deep: true, immediate: true }
)

const resetWorkingGraph = () => {
  workingGraph.nodes = JSON.parse(JSON.stringify(graph.value.nodes || []))
  workingGraph.edges = JSON.parse(JSON.stringify(graph.value.edges || []))
  resetSelection()
  computeTextlines()
}

const getNodeColor = (nodeIndex) => {
  const textlineId = nodeToTextlineMap.value[nodeIndex]

  if (regionLabelingModeActive.value) {
    if (hoveredTextlineId.value !== null && hoveredTextlineId.value === textlineId) {
      return '#ff4081' // Hot pink for hovered textline
    }
    const label = regionLabels[nodeIndex]
    if (label !== undefined && label > -1) {
      return labelColors[label % labelColors.length]
    }
    return '#9e9e9e' // Grey for unlabeled nodes in this mode
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
  if (regionLabelingModeActive.value) {
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
const onNodeClick = (nodeIndex, event) => {
  if (isAKeyPressed.value || isDKeyPressed.value || regionLabelingModeActive.value) return
  event.stopPropagation()
  const existingIndex = selectedNodes.value.indexOf(nodeIndex)
  if (existingIndex !== -1) selectedNodes.value.splice(existingIndex, 1)
  else
    selectedNodes.value.length < 2
      ? selectedNodes.value.push(nodeIndex)
      : (selectedNodes.value = [nodeIndex])
}
const onEdgeClick = (edge, event) => {
  if (isAKeyPressed.value || isDKeyPressed.value || regionLabelingModeActive.value) return
  event.stopPropagation()
  selectedNodes.value = [edge.source, edge.target]
}
const onBackgroundClick = () => {
  if (!isAKeyPressed.value && !isDKeyPressed.value) resetSelection()
}

const handleSvgMouseMove = (event) => {
  if (!svgOverlayRef.value) return
  const { left, top } = svgOverlayRef.value.getBoundingClientRect()
  const mouseX = event.clientX - left
  const mouseY = event.clientY - top

  if (regionLabelingModeActive.value) {
    let newHoveredTextlineId = null

    // 1. Check for node hover first (more precise)
    for (let i = 0; i < workingGraph.nodes.length; i++) {
      const node = workingGraph.nodes[i]
      if (Math.hypot(mouseX - scaleX(node.x), mouseY - scaleY(node.y)) < NODE_HOVER_RADIUS) {
        newHoveredTextlineId = nodeToTextlineMap.value[i]
        break // Exit loop once found
      }
    }

    // 2. If no node hovered, check for edge hover
    if (newHoveredTextlineId === null) {
      for (const edge of workingGraph.edges) {
        const n1 = workingGraph.nodes[edge.source]
        const n2 = workingGraph.nodes[edge.target]
        if (
          n1 &&
          n2 &&
          distanceToLineSegment(
            mouseX,
            mouseY,
            scaleX(n1.x),
            scaleY(n1.y),
            scaleX(n2.x),
            scaleY(n2.y)
          ) < EDGE_HOVER_THRESHOLD
        ) {
          // An edge connects two nodes of the same textline, so we can use either.
          newHoveredTextlineId = nodeToTextlineMap.value[edge.source]
          break // Exit loop once found
        }
      }
    }

    // 3. Update the hovered textline ID
    hoveredTextlineId.value = newHoveredTextlineId

    // 4. Apply label if key is pressed
    if (hoveredTextlineId.value !== null && isEKeyPressed.value) {
      labelTextline()
    }
    return
  }

  if (!editModeActive.value) return
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
      regionLabels[nodeIndex] = currentLabelIndex.value
    })
  }
}

const handleGlobalKeyDown = (e) => {
  const key = e.key.toLowerCase()

  // General hotkeys that work in multiple modes
  if (key === 's' && !e.repeat) {
    if (
      (editModeActive.value || regionLabelingModeActive.value) &&
      !loading.value &&
      !isProcessingSave.value
    ) {
      e.preventDefault()
      saveAndGoNext()
    }
    return
  }
  if (key === 'w' && !e.repeat) {
    e.preventDefault()
    editModeActive.value = !editModeActive.value
    return
  }
  if (key === 'r' && !e.repeat) {
    e.preventDefault()
    regionLabelingModeActive.value = !regionLabelingModeActive.value
    return
  }

  // Region labeling specific hotkeys
  if (regionLabelingModeActive.value && !e.repeat) {
    if (key === 'e') {
      e.preventDefault()
      isEKeyPressed.value = true
    }
    return
  }

  // Edge editing specific hotkeys
  if (!editModeActive.value || e.repeat) return

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

const handleGlobalKeyUp = (e) => {
  const key = e.key.toLowerCase()

  if (regionLabelingModeActive.value && key === 'e') {
    isEKeyPressed.value = false
    currentLabelIndex.value++ // Increment label for the next group
  }

  if (!editModeActive.value) return

  if (key === 'd') isDKeyPressed.value = false
  if (key === 'a') {
    isAKeyPressed.value = false
    if (hoveredNodesForMST.size >= 2) addMSTEdges()
    hoveredNodesForMST.clear()
  }
}

const edgeExists = (nodeA, nodeB) =>
  workingGraph.edges.some(
    (e) =>
      (e.source === nodeA && e.target === nodeB) || (e.source === nodeB && e.target === nodeA)
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
    (e) =>
      (e.source === source && e.target === target) || (e.source === target && e.target === source)
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
    px -
      (x1 +
        Math.max(
          0,
          Math.min(
            1,
            ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) /
              (Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1)
          )
        ) *
          (x2 - x1)),
    py -
      (y1 +
        Math.max(
          0,
          Math.min(
            1,
            ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) /
              (Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1)
          )
        ) *
          (y2 - y1))
  )
const handleEdgeHoverDelete = (mouseX, mouseY) => {
  for (let i = workingGraph.edges.length - 1; i >= 0; i--) {
    const edge = workingGraph.edges[i]
    const n1 = workingGraph.nodes[edge.source],
      n2 = workingGraph.nodes[edge.target]
    if (
      n1 &&
      n2 &&
      distanceToLineSegment(mouseX, mouseY, scaleX(n1.x), scaleY(n1.y), scaleX(n2.x), scaleY(n2.y)) <
        EDGE_HOVER_THRESHOLD
    ) {
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
    const rootI = find(i),
      rootJ = find(j)
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
  const labelsToSend = new Array(numNodes).fill(-1)
  for (const nodeIndex in regionLabels) {
    labelsToSend[nodeIndex] = regionLabels[nodeIndex]
  }

  const requestBody = {
    graph: workingGraph,
    modifications: modifications.value,
    regionLabels: labelsToSend,
  }

  if (annotationStore.modelName) {
    requestBody.modelName = annotationStore.modelName
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
    if (!isEditModeFlow.value && data.lines) {
      annotationStore.recognitions[localManuscriptName.value][localCurrentPage.value] = data.lines
    }
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
    // alert('Graph and labels saved!')
  } catch (err) {
    alert(`Save failed: ${err.message}`)
  } finally {
    isProcessingSave.value = false
  }
}

const confirmAndNavigate = async (navAction) => {
  if (isProcessingSave.value) return
  if (modifications.value.length > 0) {
    if (confirm('You have unsaved changes. Do you want to save them before navigating?')) {
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
  if (isEditModeFlow.value) {
    router.push({
      name: 'edit-manuscript-layout',
      params: { manuscriptName: localManuscriptName.value, pageName: page },
    })
  } else {
    annotationStore.setCurrentPage(page)
  }
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
const goToIMG2TXTPage = () => {
  if (isEditModeFlow.value) {
    alert(
      "Text annotation is part of the 'New Manuscript' flow. This action is disabled in edit mode."
    )
    return
  }
  confirmAndNavigate(() => router.push({ name: 'img-2-txt' }))
}

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

onMounted(async () => {
  if (isEditModeFlow.value) {
    localManuscriptName.value = props.manuscriptName
    localCurrentPage.value = props.pageName
    await fetchPageList(props.manuscriptName)
    await fetchPageData(props.manuscriptName, props.pageName)
  } else {
    localManuscriptName.value = Object.keys(annotationStore.recognitions)[0] || ''
    localPageList.value = annotationStore.sortedPageIds
    localCurrentPage.value = annotationStore.currentPage
    if (localManuscriptName.value && localCurrentPage.value) {
      await fetchPageData(localManuscriptName.value, localCurrentPage.value)
    } else {
      loading.value = false
      error.value = "No manuscript data found. Please start from the 'New Manuscript' page."
    }
  }

  window.addEventListener('keydown', handleGlobalKeyDown)
  window.addEventListener('keyup', handleGlobalKeyUp)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleGlobalKeyDown)
  window.removeEventListener('keyup', handleGlobalKeyUp)
})

watch(
  () => annotationStore.currentPage,
  (newPage) => {
    if (!isEditModeFlow.value && newPage && newPage !== localCurrentPage.value) {
      localCurrentPage.value = newPage
      fetchPageData(localManuscriptName.value, newPage)
    }
  }
)

watch(
  () => props.pageName,
  (newPageName) => {
    if (isEditModeFlow.value && newPageName && newPageName !== localCurrentPage.value) {
      localCurrentPage.value = newPageName
      fetchPageData(localManuscriptName.value, newPageName)
    }
  }
)

watch(editModeActive, (isEditing) => {
  if (isEditing) regionLabelingModeActive.value = false
  if (!isEditing) {
    resetSelection()
    isAKeyPressed.value = false
    isDKeyPressed.value = false
    hoveredNodesForMST.clear()
  }
})

watch(regionLabelingModeActive, (isLabeling) => {
  if (isLabeling) {
    console.log('Entering Region Labeling mode.')
    editModeActive.value = false
    resetSelection()

    // Ensure the next label index is unique by checking existing labels
    const existingLabels = Object.values(regionLabels)
    if (existingLabels.length > 0) {
      // Find the maximum label value currently in use and add 1
      const maxLabel = Math.max(...existingLabels)
      currentLabelIndex.value = maxLabel + 1
      console.log(`Resuming labeling. Next available label index: ${currentLabelIndex.value}`)
    } else {
      // No labels exist yet, start from 0
      currentLabelIndex.value = 0
      console.log('No existing labels. Starting new labeling at index: 0')
    }
  } else {
    console.log('Exiting Region Labeling mode.')
  }
  hoveredTextlineId.value = null
})
</script>
<style scoped>
.manuscript-viewer {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  overflow: hidden;
  background-color: #333;
  color: #fff;
}

/* --- Toolbar --- */
.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  background-color: #424242;
  border-bottom: 1px solid #555;
  flex-shrink: 0;
  gap: 16px;
}
.toolbar-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.toggle-container {
  display: flex;
  align-items: center;
  background-color: #3a3a3a;
  padding: 4px 8px;
  border-radius: 4px;
}

/* --- Main Visualization Area --- */
.visualization-container {
  position: relative;
  overflow: auto;
  flex-grow: 1;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 1rem;
}
.image-container {
  position: relative;
  box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
}
.manuscript-image {
  display: block;
  user-select: none;
  opacity: 0.7;
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

/* --- Bottom Panel --- */
.bottom-panel {
  background-color: #4f4f4f;
  border-top: 1px solid #555;
  flex-shrink: 0;
  transition: all 0.3s ease;
}
.panel-toggle-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 16px;
  cursor: pointer;
}
.edit-instructions p {
  margin: 0;
  font-size: 0.9em;
  color: #ccc;
  font-style: italic;
}
.bottom-panel-content {
  padding: 10px 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.edit-controls,
.modifications-log-container {
  display: flex;
  align-items: flex-start;
  gap: 20px;
}
.edit-actions {
  display: flex;
  gap: 8px;
}

/* --- UI Elements & States --- */
.panel-toggle-btn {
  padding: 4px 10px;
  font-size: 0.8em;
  background-color: #616161;
  border: 1px solid #757575;
}
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
}
.processing-save-notice {
  background-color: rgba(0, 0, 0, 0.8);
}
.error-message {
  background-color: #c62828;
}
.loading {
  font-size: 1.2rem;
  color: #aaa;
  background: none;
}
button {
  padding: 6px 14px;
  border-radius: 4px;
  border: 1px solid #666;
  background-color: #555;
  color: #fff;
  cursor: pointer;
  transition: background-color 0.2s ease;
}
button:hover:not(:disabled) {
  background-color: #6a6a6a;
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* --- Modifications Log --- */
.modifications-details {
  flex-grow: 1;
}
.modifications-details h3 {
  margin: 0 0 8px 0;
  font-size: 1.1em;
  color: #eee;
}
.modifications-details ul {
  list-style-type: none;
  padding: 0;
  max-height: 120px;
  overflow-y: auto;
  border: 1px solid #666;
  background-color: #3e3e3e;
  border-radius: 3px;
}
.modification-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 10px;
  border-bottom: 1px solid #555;
  font-size: 0.9em;
}
.modification-item:last-child {
  border-bottom: none;
}
.undo-button {
  background-color: #6d6d3d;
  border-color: #888855;
}
.undo-button:hover:not(:disabled) {
  background-color: #7a7a4a;
}
</style>




LayoutGraphGenerator.js: (this uses a heuristic algorithm to connect points of the same text-line together. This algorithm does the same thing the GNN would do, but in a faster, more lightweight manner.)

// layoutGraphGenerator.js
/**
 * Build a KD-Tree for fast neighbor lookup
 */
class KDTree {
  constructor(points) {
    this.points = points;
    this.tree = this.buildTree(points.map((p, i) => ({ point: p, index: i })), 0);
  }

  buildTree(points, depth) {
    if (points.length === 0) return null;
    if (points.length === 1) return points[0];

    const k = 2; // 2D points
    const axis = depth % k;
    
    points.sort((a, b) => a.point[axis] - b.point[axis]);
    const median = Math.floor(points.length / 2);
    
    return {
      point: points[median].point,
      index: points[median].index,
      left: this.buildTree(points.slice(0, median), depth + 1),
      right: this.buildTree(points.slice(median + 1), depth + 1),
      axis: axis
    };
  }

  query(queryPoint, k) {
    const best = [];
    
    const search = (node, depth) => {
      if (!node) return;
      
      const distance = this.euclideanDistance(queryPoint, node.point);
      
      if (best.length < k) {
        best.push({ distance, index: node.index });
        best.sort((a, b) => a.distance - b.distance);
      } else if (distance < best[best.length - 1].distance) {
        best[best.length - 1] = { distance, index: node.index };
        best.sort((a, b) => a.distance - b.distance);
      }
      
      const axis = depth % 2;
      const diff = queryPoint[axis] - node.point[axis];
      
      const closer = diff < 0 ? node.left : node.right;
      const farther = diff < 0 ? node.right : node.left;
      
      search(closer, depth + 1);
      
      if (best.length < k || Math.abs(diff) < best[best.length - 1].distance) {
        search(farther, depth + 1);
      }
    };
    
    search(this.tree, 0);
    return best.map(b => b.index);
  }

  euclideanDistance(p1, p2) {
    return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
  }
}

/**
 * DBSCAN clustering implementation to identify majority cluster and outliers
 */
function clusterWithSingleMajority(toCluster, eps = 10, minSamples = 2) {
  if (toCluster.length === 0) return [];
  
  // DBSCAN implementation
  const labels = dbscan(toCluster, eps, minSamples);
  
  // Count the occurrences of each label
  const labelCounts = {};
  labels.forEach(label => {
    labelCounts[label] = (labelCounts[label] || 0) + 1;
  });
  
  // Find the majority cluster label (excluding -1 outliers)
  let majorityLabel = null;
  let maxCount = 0;
  
  for (const [label, count] of Object.entries(labelCounts)) {
    const labelNum = parseInt(label);
    if (labelNum !== -1 && count > maxCount) {
      majorityLabel = labelNum;
      maxCount = count;
    }
  }
  
  // Create a new label array where the majority cluster is 0 and all others are -1
  const newLabels = new Array(labels.length).fill(-1); // Initialize all as outliers
  
  if (majorityLabel !== null) {
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] === majorityLabel) {
        newLabels[i] = 0; // Assign 0 to the majority cluster
      }
    }
  }
  
  return newLabels;
}

/**
 * DBSCAN clustering algorithm implementation
 */
function dbscan(points, eps, minSamples) {
  const labels = new Array(points.length).fill(-1); // -1 means unclassified
  let clusterId = 0;
  
  for (let i = 0; i < points.length; i++) {
    if (labels[i] !== -1) continue; // Already processed
    
    const neighbors = getNeighbors(points, i, eps);
    
    if (neighbors.length < minSamples) {
      labels[i] = -1; // Mark as noise/outlier
    } else {
      // Start a new cluster
      expandCluster(points, labels, i, neighbors, clusterId, eps, minSamples);
      clusterId++;
    }
  }
  
  return labels;
}

/**
 * Get neighbors within eps distance
 */
function getNeighbors(points, pointIndex, eps) {
  const neighbors = [];
  const point = points[pointIndex];
  
  for (let i = 0; i < points.length; i++) {
    if (euclideanDistance(point, points[i]) <= eps) {
      neighbors.push(i);
    }
  }
  
  return neighbors;
}

/**
 * Expand cluster by adding density-reachable points
 */
function expandCluster(points, labels, pointIndex, neighbors, clusterId, eps, minSamples) {
  labels[pointIndex] = clusterId;
  
  let i = 0;
  while (i < neighbors.length) {
    const neighborIndex = neighbors[i];
    
    if (labels[neighborIndex] === -1) {
      labels[neighborIndex] = clusterId;
      
      const neighborNeighbors = getNeighbors(points, neighborIndex, eps);
      if (neighborNeighbors.length >= minSamples) {
        // Add new neighbors to the list (union operation)
        for (const newNeighbor of neighborNeighbors) {
          if (!neighbors.includes(newNeighbor)) {
            neighbors.push(newNeighbor);
          }
        }
      }
    }
    
    i++;
  }
}

function euclideanDistance(p1, p2) {
  return Math.sqrt(p1.reduce((sum, val, i) => sum + (val - p2[i]) ** 2, 0));
}

/**
 * Generate a graph representation of text layout based on points.
 * This function implements the core layout analysis logic.
 */
export function generateLayoutGraph(points) { // TODO ADD FEATURES
  const NUM_NEIGHBOURS = 6;
  const cos_similarity_less_than = -0.8;
  
  // Build a KD-tree for fast neighbor lookup
  const tree = new KDTree(points);
  const indices = points.map((point, i) => tree.query(point, NUM_NEIGHBOURS));
  
  // Store graph edges and their properties
  const edges = [];
  const edgeProperties = [];
  
  // Process nearest neighbors
  for (let currentPointIndex = 0; currentPointIndex < indices.length; currentPointIndex++) {
    const nbrIndices = indices[currentPointIndex];
    const currentPoint = points[currentPointIndex];
    
    const normalizedPoints = nbrIndices.map(idx => [
      points[idx][0] - currentPoint[0],
      points[idx][1] - currentPoint[1]
    ]);
    
    const scalingFactor = Math.max(...normalizedPoints.flat().map(Math.abs)) || 1;
    const scaledPoints = normalizedPoints.map(np => [np[0] / scalingFactor, np[1] / scalingFactor]);
    
    // Create a list of relative neighbors with their global indices
    const relativeNeighbours = nbrIndices.map((globalIdx, i) => ({
      globalIdx,
      scaledPoint: scaledPoints[i],
      normalizedPoint: normalizedPoints[i]
    }));
    
    const filteredNeighbours = [];
    
    for (let i = 0; i < relativeNeighbours.length; i++) {
      for (let j = i + 1; j < relativeNeighbours.length; j++) {
        const neighbor1 = relativeNeighbours[i];
        const neighbor2 = relativeNeighbours[j];
        
        const norm1 = Math.sqrt(neighbor1.scaledPoint[0] ** 2 + neighbor1.scaledPoint[1] ** 2);
        const norm2 = Math.sqrt(neighbor2.scaledPoint[0] ** 2 + neighbor2.scaledPoint[1] ** 2);
        
        let cosSimilarity = 0.0;
        if (norm1 * norm2 !== 0) {
          const dotProduct = neighbor1.scaledPoint[0] * neighbor2.scaledPoint[0] + 
                           neighbor1.scaledPoint[1] * neighbor2.scaledPoint[1];
          cosSimilarity = dotProduct / (norm1 * norm2);
        }
        
        // Calculate non-normalized distances
        const norm1Real = Math.sqrt(neighbor1.normalizedPoint[0] ** 2 + neighbor1.normalizedPoint[1] ** 2);
        const norm2Real = Math.sqrt(neighbor2.normalizedPoint[0] ** 2 + neighbor2.normalizedPoint[1] ** 2);
        const totalLength = norm1Real + norm2Real;
        
        // Select pairs with angles close to 180 degrees (opposite directions)
        if (cosSimilarity < cos_similarity_less_than) {
          filteredNeighbours.push({
            neighbor1,
            neighbor2,
            totalLength,
            cosSimilarity
          });
        }
      }
    }
    
    if (filteredNeighbours.length > 0) {
      // Find the shortest total length pair
      const shortestPair = filteredNeighbours.reduce((min, curr) => 
        curr.totalLength < min.totalLength ? curr : min
      );
      
      const { neighbor1: connection1, neighbor2: connection2, totalLength, cosSimilarity } = shortestPair;
      
      // Calculate angles with x-axis
      const thetaA = Math.atan2(connection1.normalizedPoint[1], connection1.normalizedPoint[0]) * 180 / Math.PI;
      const thetaB = Math.atan2(connection2.normalizedPoint[1], connection2.normalizedPoint[0]) * 180 / Math.PI;
      
      // Add edges to the graph
      edges.push([currentPointIndex, connection1.globalIdx]);
      edges.push([currentPointIndex, connection2.globalIdx]);
      
      // Calculate feature values for clustering
      const yDiff1 = Math.abs(connection1.normalizedPoint[1]);
      const yDiff2 = Math.abs(connection2.normalizedPoint[1]);
      const avgYDiff = (yDiff1 + yDiff2) / 2;
      
      const xDiff1 = Math.abs(connection1.normalizedPoint[0]);
      const xDiff2 = Math.abs(connection2.normalizedPoint[0]);
      const avgXDiff = (xDiff1 + xDiff2) / 2;
      
      // Calculate aspect ratio (height/width)
      const aspectRatio = avgYDiff / Math.max(avgXDiff, 0.001);
      
      // Calculate vertical alignment consistency
      const vertConsistency = Math.abs(yDiff1 - yDiff2);
      
      // Store edge properties for clustering
      edgeProperties.push([
        totalLength,
        Math.abs(thetaA + thetaB),
        // aspectRatio,
        // vertConsistency,
        // avgYDiff
      ]);
    }
  }
  
  // Cluster the edges based on their properties
  const edgeLabels = clusterWithSingleMajority(edgeProperties);
  
  // Create a mask for edges that are not outliers (label != -1)
  const nonOutlierMask = edgeLabels.map(label => label !== -1);
  
  // Prepare the final graph structure
  const graphData = {
    nodes: points.map((point, i) => ({
      id: i,
      x: parseFloat(point[0]),
      y: parseFloat(point[1]),
      s: parseFloat(point[2]),
    })),
    edges: []
  };
  
  // Add edges with their labels, filtering out outliers
  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];
    // Determine the corresponding edge label using division by 2 (each edge appears twice)
    const labelIndex = Math.floor(i / 2);
    const edgeLabel = edgeLabels[labelIndex];
    
    // Only add the edge if it is not an outlier
    if (nonOutlierMask[labelIndex]) {
      graphData.edges.push({
        source: parseInt(edge[0]),
        target: parseInt(edge[1]),
        label: parseInt(edgeLabel)
      });
    }
  }
  
  return graphData;
}



Here is the full code base as it is now:
app
Sat Jan 10 10:40:17 AM IST 2026

# Complete Repository Structure:
# (showing all directories and files with token counts)
#/ (~14176 tokens)
#  └── environment.yaml (~140 tokens)
#  └── gnn_inference.py (~5920 tokens)
#  └── inference.py (~1363 tokens)
#  └── README.md (~577 tokens)
#  └── segment_from_point_clusters.py (~6176 tokens)
#  /demo_manuscripts/ (~0 tokens)
#    /demo_manuscripts/sample_manuscript_1/ (~0 tokens)
#      /demo_manuscripts/sample_manuscript_1/images/ (~0 tokens)
#    /demo_manuscripts/sample_manuscript_2/ (~0 tokens)
#      /demo_manuscripts/sample_manuscript_2/images/ (~0 tokens)
#    /demo_manuscripts/sample_manuscript_3/ (~0 tokens)
#      /demo_manuscripts/sample_manuscript_3/images/ (~0 tokens)
#    /demo_manuscripts/sample_manuscript_4/ (~0 tokens)
#      /demo_manuscripts/sample_manuscript_4/images/ (~0 tokens)
#  /gnn_data_preparation/ (~10001 tokens)
#    └── config_models.py (~1087 tokens)
#    └── dataset_generator.py (~1011 tokens)
#    └── feature_engineering.py (~856 tokens)
#    └── graph_constructor.py (~4263 tokens)
#    └── __init__.py (~0 tokens)
#    └── main_create_dataset.py (~2553 tokens)
#    └── utils.py (~231 tokens)
#  /pretrained_gnn/ (~960 tokens)
#    └── gnn_preprocessing_v2.yaml (~960 tokens)
#  /segmentation/ (~4979 tokens)
#    └── craft.py (~2654 tokens)
#    └── segment_graph.py (~2018 tokens)
#    └── utils.py (~307 tokens)
#    /segmentation/pretrained_unet_craft/ (~47 tokens)
#      └── README.md (~47 tokens)
#
---
---
environment.yaml
---
name: gnn_layout

channels:
  - conda-forge
  - pytorch
  - pyg
  - nvidia
  - defaults

dependencies:
  - python=3.11
  - pip
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - scikit-image
  - matplotlib
  - pytorch
  - torchvision
  - pytorch-cuda=12.1 
  - pyg
  - flask
  - flask-cors
  - flask-sqlalchemy
  - werkzeug
  - pillow
  - opencv
  - python-dotenv
  - packaging
  - six
  - natsort
  - pyyaml
  - conda-forge::pytorch_spline_conv
  - pip:
    - shapely
    - lmdb
    - nltk
    - python-json-logger
    - regex
    - pydantic
    - wandb
    
---
gnn_data_preparation/config_models.py
---
# data_creation/config_models.py
from pydantic import BaseModel, Field, conint, confloat
from typing import List, Literal, Optional, Annotated


# Feature Engineering Models
class HeuristicDegreeEncoding(BaseModel):
    linear_map_factor: float = 0.1
    one_hot_max_degree: int = 10

class OverlapEncoding(BaseModel):
    linear_map_factor: float = 0.1
    one_hot_max_overlap: int = 10

class FeaturesConfig(BaseModel):
    use_node_coordinates: bool = True
    use_node_font_size: bool = True
    use_heuristic_degree: bool = True
    heuristic_degree_encoding: Literal["linear_map", "one_hot"] = "linear_map"
    heuristic_degree_encoding_params: HeuristicDegreeEncoding = Field(default_factory=HeuristicDegreeEncoding)
    use_relative_distance: bool = True
    use_euclidean_distance: bool = True
    use_aspect_ratio_rel: bool = True
    use_overlap: bool = True
    overlap_encoding: Literal["linear_map", "one_hot"] = "linear_map"
    overlap_encoding_params: OverlapEncoding = Field(default_factory=OverlapEncoding)
    use_page_aspect_ratio: bool = True


# Graph Construction Models
class HeuristicParams(BaseModel):
    k: int = 10
    cosine_sim_threshold: float = -0.8

# Define parameter models for each connectivity strategy
class KnnParams(BaseModel):
    k: int = 10

class SecondShortestHeuristicParams(BaseModel):
    k: int = 10
    cosine_sim_threshold: float = -0.8
    min_angle_degrees: float = 45.0 # Add this new parameter with a default value

class AngularKnnParams(BaseModel):
    k: int = 50  # K for angular KNN
    sector_angle_degrees: float = 20.0  # Minimum angle between edges to consider them connected

# Update ConnectivityConfig to handle a list of strategies and their params
class ConnectivityConfig(BaseModel):
    strategies: List[Literal["knn", "second_shortest_heuristic", "angular_knn"]] = []
    knn_params: Optional[KnnParams] = Field(default_factory=KnnParams)
    second_shortest_params: Optional[SecondShortestHeuristicParams] = Field(default_factory=SecondShortestHeuristicParams)
    angular_knn_params: Optional[AngularKnnParams] = Field(default_factory=AngularKnnParams)

class InputGraphConfig(BaseModel):
    use_heuristic_graph: bool = True
    heuristic_params: HeuristicParams = Field(default_factory=HeuristicParams)
    connectivity: ConnectivityConfig = Field(default_factory=ConnectivityConfig)
    directionality: Literal["bidirectional", "unidirectional"] = "bidirectional"



# Ground Truth Model
class GroundTruthConfig(BaseModel):
    algorithm: Literal["mst", "greedy_path"] = "mst"

# MODIFIED: Simplified SplittingConfig for the new fixed-split strategy
class SplittingConfig(BaseModel):
    """Configuration for splitting the validation/test dataset."""
    random_seed: int = 49
    val_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.75 # Ratio of the val/test data to be used for validation

# Sklearn Format Model
class NHopConfig(BaseModel):
    hops: int = 1
    aggregations: List[str] = ["mean", "std"]
    
class SklearnFormatConfig(BaseModel):
    enabled: bool = True
    features: List[str] = ["source_node_features", "target_node_features", "edge_features", "page_features"]
    use_n_hop_features: bool = False
    n_hop_config: Optional[NHopConfig] = Field(default_factory=NHopConfig)

# Top-level Configuration Model
class DatasetCreationConfig(BaseModel):
    """
    Top-level configuration model for the entire dataset creation process.
    """
    # make these optional in pydantic..
    version: Optional[str] = None
    manuscript_name: Optional[str] = None
    train_data_dir: Optional[str] = None
    val_test_data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    
    min_nodes_per_page: Annotated[int, Field(ge=1)] = 10
    
    # This field now uses the new, simplified SplittingConfig
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    
    # The rest of the configuration remains unchanged
    input_graph: InputGraphConfig = Field(default_factory=InputGraphConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    ground_truth: GroundTruthConfig = Field(default_factory=GroundTruthConfig)
    sklearn_format: SklearnFormatConfig = Field(default_factory=SklearnFormatConfig)
---
gnn_data_preparation/dataset_generator.py
---
# data_creation/dataset_generator.py
import torch
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from typing import List, Dict

class HistoricalLayoutGNNDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for the historical layout analysis task.
    """
    def __init__(self, root, transform=None, pre_transform=None, data_list=None):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # We process manually, so this can be empty
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Data is assumed to be present locally in the specified raw dir
        pass

    def process(self):
        # If data is provided directly, just save it
        if self.data_list is not None:
            data, slices = self.collate(self.data_list)
            torch.save((data, slices), self.processed_paths[0])
        else:
            # This would be used if we were processing from raw files inside this class
            # But we are doing it in the main script for better control over folds.
            print("Processing from raw files is not implemented directly in the class. "
                  "Please provide a data_list during initialization.")

def create_sklearn_dataframe(data_list: List[Data], page_map: Dict[int, str], config) -> pd.DataFrame:
    """
    Converts a list of PyG Data objects into a single pandas DataFrame
    suitable for training with scikit-learn models.
    """
    if not config.sklearn_format.enabled:
        return None

    all_edges_data = []
    
    node_feature_names = []
    if config.features.use_node_coordinates: node_feature_names.extend(['x', 'y'])
    if config.features.use_node_font_size: node_feature_names.append('font_size')
    # Add other node features names here if they are simple values
    
    for i, data in enumerate(data_list):
        page_id = page_map[i]
        edge_index = data.edge_index.T.numpy()
        edge_y = data.edge_y.numpy()
        
        for edge_idx, (u, v) in enumerate(edge_index):
            row = {'page_id': page_id, 'source_node_id': u, 'target_node_id': v}
            
            # Source and Target Node Features
            if "source_node_features" in config.sklearn_format.features:
                for feat_idx, name in enumerate(node_feature_names):
                    row[f'source_{name}'] = data.x[u, feat_idx].item()
            
            if "target_node_features" in config.sklearn_format.features:
                for feat_idx, name in enumerate(node_feature_names):
                    row[f'target_{name}'] = data.x[v, feat_idx].item()
            
            # Edge Features
            if "edge_features" in config.sklearn_format.features and data.edge_attr is not None:
                # This needs to be more specific based on config, but for now we dump them
                for feat_idx in range(data.edge_attr.shape[1]):
                    row[f'edge_attr_{feat_idx}'] = data.edge_attr[edge_idx, feat_idx].item()
                    
            # Page Features
            if "page_features" in config.sklearn_format.features and hasattr(data, 'page_aspect_ratio'):
                row['page_aspect_ratio'] = data.page_aspect_ratio.item()

            # Target Label
            row['label'] = edge_y[edge_idx]
            
            all_edges_data.append(row)

    df = pd.DataFrame(all_edges_data)
    # Note: N-hop features are complex and would require a separate graph traversal step here.
    # This is a placeholder for that advanced functionality.
    if config.sklearn_format.use_n_hop_features:
        print("Warning: N-hop feature generation for sklearn is an advanced feature and not yet implemented.")
        
    return df
---
gnn_data_preparation/feature_engineering.py
---
# data_creation/feature_engineering.py
import torch
import numpy as np

try:
    from .config_models import FeaturesConfig
except ImportError:
    from config_models import FeaturesConfig


def get_node_features(points: np.ndarray, heuristic_degrees: np.ndarray, config: FeaturesConfig) -> torch.Tensor:
    """Combines all enabled node features into a single tensor."""
    features_list = []
    
    if config.use_node_coordinates:
        features_list.append(torch.from_numpy(points[:, :2]).float())
    
    if config.use_node_font_size:
        features_list.append(torch.from_numpy(points[:, 2]).float().unsqueeze(1))
        
    if config.use_heuristic_degree:
        deg_feat = encode_categorical_feature(
            heuristic_degrees,
            config.heuristic_degree_encoding,
            config.heuristic_degree_encoding_params.linear_map_factor,
            config.heuristic_degree_encoding_params.one_hot_max_degree
        )
        features_list.append(deg_feat)
        
    return torch.cat(features_list, dim=1)


def get_edge_features(edge_index: torch.Tensor, node_features: torch.Tensor, heuristic_edge_counts: dict, config: FeaturesConfig) -> torch.Tensor:
    """Computes all enabled edge features for the given edges."""
    features_list = []
    source_nodes, target_nodes = edge_index[0], edge_index[1]
    
    # Use original node coordinates for distance calculations
    # Assuming first 2 features are x, y
    source_pos = node_features[source_nodes, :2]
    target_pos = node_features[target_nodes, :2]
    
    relative_pos = target_pos - source_pos
    
    if config.use_relative_distance:
        features_list.append(relative_pos) # rel_x, rel_y
        
    if config.use_euclidean_distance:
        dist = torch.linalg.norm(relative_pos, dim=1).unsqueeze(1)
        features_list.append(dist)
        
    if config.use_aspect_ratio_rel:
        # Add epsilon for stability
        aspect = torch.abs(relative_pos[:, 1]) / (torch.abs(relative_pos[:, 0]) + 1e-6)
        features_list.append(aspect.unsqueeze(1))
        
    if config.use_overlap:
        overlaps = []
        for u, v in edge_index.T.numpy():
            edge_key = tuple(sorted((u, v)))
            overlaps.append(heuristic_edge_counts.get(edge_key, 0))
        overlaps = np.array(overlaps)
        
        overlap_feat = encode_categorical_feature(
            overlaps,
            config.overlap_encoding,
            config.overlap_encoding_params.linear_map_factor,
            config.overlap_encoding_params.one_hot_max_overlap
        )
        features_list.append(overlap_feat)

    if not features_list:
        return None

    return torch.cat(features_list, dim=1)

def encode_categorical_feature(values: np.ndarray, method: str, factor: float, max_val: int) -> torch.Tensor:
    """Encodes a categorical feature using the specified method."""
    if method == "linear_map":
        return torch.from_numpy(values).float().unsqueeze(1) * factor
    elif method == "one_hot":
        # Clamp values to the max to avoid oversized tensors
        values_clamped = np.clip(values, 0, max_val)
        one_hot = torch.nn.functional.one_hot(torch.from_numpy(values_clamped).long(), num_classes=max_val + 1)
        return one_hot.float()
    else:
        raise ValueError(f"Unknown encoding method: {method}")
---
gnn_data_preparation/graph_constructor.py
---
# data_creation/graph_constructor.py
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.neighbors import KDTree, NearestNeighbors
from itertools import combinations
from collections import Counter, defaultdict
import logging
from dataclasses import dataclass
from scipy.spatial import KDTree

try:
    # Case 1: run as part of a package
    from .config_models import InputGraphConfig, GroundTruthConfig, KnnParams, SecondShortestHeuristicParams
except ImportError:
    # Case 2: run directly (standalone script)
    from config_models import InputGraphConfig, GroundTruthConfig, KnnParams, SecondShortestHeuristicParams



@dataclass
class AngularKnnParams:
    """Parameters for the Angular K-Nearest Neighbor function."""
    # The width of each angular sector in degrees.
    sector_angle_degrees: float = 10.0
    
    # The number of candidate neighbors to check for each point.
    # This is a crucial performance parameter. A higher value is more accurate
    # for sparse sectors but slower.
    k: int = 50


def create_heuristic_graph(points: np.ndarray, page_dims: dict, config: InputGraphConfig) -> dict:
    """Creates a heuristic graph based on proximity and collinearity."""
    n_points = len(points)
    params = config.heuristic_params
    if n_points < 3: #params.k:
        logging.warning(f"Not enough points ({n_points}) to build heuristic graph with k={params.k}. Skipping.")
        return {"edges": set(), "degrees": np.zeros(n_points, dtype=int), "edge_counts": Counter()}

    # Normalize points for stable KDTree and cosine similarity
    normalized_points = np.copy(points)
    max_dim = max(page_dims['width'], page_dims['height'])
    normalized_points[:, :2] /= max_dim
    
    kdtree = KDTree(normalized_points[:, :2])
    heuristic_directed_edges = []
    
    for i in range(n_points):
        num_potential_neighbors = n_points - 1
        # We need at least 2 neighbors to form a pair.
        if num_potential_neighbors < 2:
            continue
        # The k for the query must be less than or equal to n_points.
        # We query for min(config_k, potential_neighbors) + 1 (for the point itself)
        k_for_query = min(params.k, num_potential_neighbors)
        _, neighbor_indices = kdtree.query(normalized_points[i, :2].reshape(1, -1), k=k_for_query + 1)
        
        # # Query k+1 to exclude the point itself
        # _, neighbor_indices = kdtree.query(normalized_points[i, :2].reshape(1, -1), k=params.k + 1)
        neighbor_indices = neighbor_indices[0][1:]
        
        best_pair, min_dist_sum = None, float('inf')
        
        # Find two neighbors that are most collinear and opposite to the current point
        for n1_idx, n2_idx in combinations(neighbor_indices, 2):
            vec1 = normalized_points[n1_idx, :2] - normalized_points[i, :2]
            vec2 = normalized_points[n2_idx, :2] - normalized_points[i, :2]
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0: continue
            
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            
            if cosine_sim < params.cosine_sim_threshold:
                dist_sum = norm1 + norm2
                if dist_sum < min_dist_sum:
                    min_dist_sum, best_pair = dist_sum, (n1_idx, n2_idx)
    
        if best_pair:
            heuristic_directed_edges.extend([(i, best_pair[0]), (i, best_pair[1])])

    # Convert to undirected edges for initial set and calculate degrees/overlaps
    heuristic_edges = {tuple(sorted(edge)) for edge in heuristic_directed_edges}
    degrees = np.zeros(n_points, dtype=int)
    for u, v in heuristic_edges:
        degrees[u] += 1
        degrees[v] += 1
        
    edge_counts = Counter(tuple(sorted(edge)) for edge in heuristic_directed_edges)

    return {"edges": heuristic_edges, "degrees": degrees, "edge_counts": edge_counts}


# Modify the function signature to accept its own specific parameters for better modularity
def add_knn_edges(points: np.ndarray, existing_edges: set, params: KnnParams) -> set:
    """Adds K-Nearest Neighbor edges to the graph."""
    n_points = len(points)
    k = params.k
    if n_points <= k:
        k = n_points - 1

    if k <= 0:
        return set()

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(points[:, :2])
    _, indices = nn.kneighbors(points[:, :2])

    knn_edges = set()
    for i in range(n_points):
        for j_idx in indices[i, 1:]:  # Skip self
            edge = tuple(sorted((i, j_idx)))
            if edge not in existing_edges:
                knn_edges.add(edge)
    return knn_edges

def add_angular_knn_edges(points: np.ndarray, existing_edges: set, params: AngularKnnParams) -> set:
    """
    Adds edges to the nearest neighbor in distinct angular sectors around each point.

    Instead of finding the k absolute nearest neighbors, this method divides the
    360-degree space around each point into sectors (e.g., 10-degree cones)
    and finds the single closest neighbor within each sector. This ensures a
    more directionally uniform set of connections.

    Args:
        points: A NumPy array of shape (n_points, D) where D >= 2.
        existing_edges: A set of existing edges (as sorted tuples) to avoid duplication.
        params: An AngularKnnParams object with configuration.

    Returns:
        A set of new, undirected edges represented as sorted tuples.
    """
    n_points = len(points)
    if n_points < 2:
        return set()

    # --- Parameter Validation and Setup ---
    if not (0 < params.sector_angle_degrees <= 180):
        logging.error("sector_angle_degrees must be between 0 and 180.")
        return set()
    
    num_sectors = int(360 / params.sector_angle_degrees)
    
    # To avoid missing neighbors in sparse regions, we must check a reasonable
    # number of candidate points. We cap this at the total number of other points.
    k_for_query = min(params.k, n_points - 1)
    if k_for_query <= 0:
        return set()

    # --- Algorithm Implementation ---
    
    # 1. Build a spatial index (KDTree) for all points. This is the crucial
    #    first step for achieving high performance.
    kdtree = KDTree(points[:, :2])
    
    new_edges = set()

    # 2. Iterate through each point to find its angular neighbors.
    for i in range(n_points):
        # 3. Fast Pre-selection: Query the KDTree for a set of candidate neighbors.
        #    This is vastly more efficient than checking all N-1 other points.
        #    We query for k+1 because the point itself is always the first neighbor.
        _, candidate_indices = kdtree.query(points[i, :2], k=k_for_query + 1)
        
        # Exclude the point itself (which is always at index 0)
        candidate_indices = candidate_indices[1:]
        
        # --- Vectorized Calculations on Candidates ---
        
        # 4. Calculate vectors from the current point 'i' to all its candidates.
        vectors = points[candidate_indices, :2] - points[i, :2]
        
        # 5. Calculate the distances (norms) of these vectors.
        distances = np.linalg.norm(vectors, axis=1)
        
        # 6. Calculate the angles of these vectors in degrees [0, 360).
        #    np.arctan2 is used for quadrant-aware angle calculation.
        angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
        angles_deg = np.rad2deg(angles_rad) % 360
        
        # 7. Determine the angular sector index for each candidate.
        sector_indices = np.floor(angles_deg / params.sector_angle_degrees).astype(int)
        
        # --- Find the Best Neighbor in Each Sector ---
        
        # 8. We now efficiently find the closest point within each sector.
        #    Initialize arrays to store the minimum distance and corresponding neighbor
        #    index found so far for each sector.
        min_dists_in_sector = np.full(num_sectors, np.inf)
        best_neighbor_in_sector = np.full(num_sectors, -1, dtype=int)
        
        # This loop is fast as it only iterates over the small set of 'k candidates'.
        for j in range(len(candidate_indices)):
            sector_idx = sector_indices[j]
            dist = distances[j]
            
            if dist < min_dists_in_sector[sector_idx]:
                min_dists_in_sector[sector_idx] = dist
                best_neighbor_in_sector[sector_idx] = candidate_indices[j]

        # 9. Create new edges from the results.
        for neighbor_idx in best_neighbor_in_sector:
            if neighbor_idx != -1:  # A neighbor was found in this sector
                edge = tuple(sorted((i, neighbor_idx)))
                if edge not in existing_edges:
                    new_edges.add(edge)
                    
    return new_edges

def add_second_shortest_heuristic_edges(points: np.ndarray, page_dims: dict, existing_edges: set, params: SecondShortestHeuristicParams) -> set:
    """
    Creates edges based on a secondary neighbor pair that is angularly separated
    from the primary (shortest) pair.
    """
    n_points = len(points)
    if n_points < 4:  # Need at least 3 neighbors for a chance at two pairs
        logging.debug(f"Not enough points ({n_points}) for second_shortest_heuristic. Skipping.")
        return set()
    
    # Pre-calculate the cosine of the minimum angle for efficiency.
    # We use the absolute value of the cosine for the check.
    cos_angle_threshold = np.cos(np.deg2rad(params.min_angle_degrees))

    # Normalize points for stable KDTree and cosine similarity
    normalized_points = np.copy(points)
    max_dim = max(page_dims['width'], page_dims['height'])
    normalized_points[:, :2] /= max_dim

    kdtree = KDTree(normalized_points[:, :2])
    new_directed_edges = []

    for i in range(n_points):
        num_potential_neighbors = n_points - 1
        if num_potential_neighbors < 3:
            continue

        k_for_query = min(params.k, num_potential_neighbors)
        _, neighbor_indices = kdtree.query(normalized_points[i, :2].reshape(1, -1), k=k_for_query + 1)
        neighbor_indices = neighbor_indices[0][1:]

        valid_pairs = []
        for n1_idx, n2_idx in combinations(neighbor_indices, 2):
            vec1 = normalized_points[n1_idx, :2] - normalized_points[i, :2]
            vec2 = normalized_points[n2_idx, :2] - normalized_points[i, :2]
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0: continue

            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)

            if cosine_sim < params.cosine_sim_threshold:
                dist_sum = norm1 + norm2
                valid_pairs.append({'dist': dist_sum, 'pair': (n1_idx, n2_idx)})

        # We need at least two valid pairs to proceed
        if len(valid_pairs) < 2:
            continue

        # Sort pairs by distance to find the best and subsequent candidates
        valid_pairs.sort(key=lambda x: x['dist'])
        
        best_pair_nodes = valid_pairs[0]['pair']
        
        # Define the direction vector for the best pair
        # This vector points from the center node 'i' to the midpoint of its pair
        best_pair_midpoint = (normalized_points[best_pair_nodes[0], :2] + normalized_points[best_pair_nodes[1], :2]) / 2
        vec_best = best_pair_midpoint - normalized_points[i, :2]
        norm_vec_best = np.linalg.norm(vec_best)

        if norm_vec_best == 0: continue

        # Find the first subsequent pair that meets the angle criteria
        for candidate in valid_pairs[1:]:
            candidate_nodes = candidate['pair']
            
            # Define the direction vector for the candidate pair
            candidate_midpoint = (normalized_points[candidate_nodes[0], :2] + normalized_points[candidate_nodes[1], :2]) / 2
            vec_candidate = candidate_midpoint - normalized_points[i, :2]
            norm_vec_candidate = np.linalg.norm(vec_candidate)

            if norm_vec_candidate == 0: continue

            # Calculate the cosine of the angle between the two direction vectors
            cos_angle_between_pairs = np.dot(vec_best, vec_candidate) / (norm_vec_best * norm_vec_candidate)

            # Check if the angle is large enough (i.e., vectors are not aligned)
            # abs(cos(theta)) < cos(45) means theta is between 45 and 135 degrees.
            if abs(cos_angle_between_pairs) < cos_angle_threshold:
                # Found a suitable "cross" pair, add its edges and stop searching for this point `i`
                second_best_pair = candidate_nodes
                new_directed_edges.extend([(i, second_best_pair[0]), (i, second_best_pair[1])])
                break # Move to the next point i

    # Convert to undirected edges, ensuring no duplicates are added
    new_edges = set()
    for u, v in new_directed_edges:
        edge = tuple(sorted((u, v)))
        if edge not in existing_edges:
            new_edges.add(edge)
            
    return new_edges



# Update the main graph creation function to iterate through the chosen strategies
def create_input_graph_edges(points: np.ndarray, page_dims: dict, config: InputGraphConfig) -> dict:
    """Constructs the full input graph by combining heuristic and connectivity strategies."""
    heuristic_result = {"edges": set(), "degrees": np.zeros(len(points), dtype=int), "edge_counts": Counter()}
    if config.use_heuristic_graph:
        heuristic_result = create_heuristic_graph(points, page_dims, config)

    # Union of all edges from different strategies
    all_edges = heuristic_result["edges"].copy()

    # Iterate through the list of selected strategies and add their edges
    for strategy in config.connectivity.strategies:
        current_edges = all_edges.copy() # Pass a copy to avoid modifying the set while iterating
        if strategy == "knn" and config.connectivity.knn_params:
            knn_edges = add_knn_edges(points, current_edges, config.connectivity.knn_params)
            all_edges.update(knn_edges)
        elif strategy == "second_shortest_heuristic" and config.connectivity.second_shortest_params:
            second_shortest_edges = add_second_shortest_heuristic_edges(points, page_dims, current_edges, config.connectivity.second_shortest_params)
            all_edges.update(second_shortest_edges)
        elif strategy == "angular_knn" and config.connectivity.angular_knn_params:
            angular_knn_edges = add_angular_knn_edges(points, current_edges, config.connectivity.angular_knn_params)
            all_edges.update(angular_knn_edges)
    
    return {
        "edges": all_edges,
        "heuristic_degrees": heuristic_result["degrees"],
        "heuristic_edge_counts": heuristic_result["edge_counts"]
    }

def create_ground_truth_graph_edges(points: np.ndarray, labels: np.ndarray, config: GroundTruthConfig) -> set:
    """Constructs the ground truth graph by connecting nodes within the same textline."""
    gt_edges = set()
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1: continue # Skip noise/unlabeled points
            
        indices = np.where(labels == label)[0]
        if len(indices) < 2: continue

        line_points = points[indices, :2]
        
        if config.algorithm == "mst":
            # Build a distance matrix and run MST
            dist_matrix = np.linalg.norm(line_points[:, np.newaxis, :] - line_points[np.newaxis, :, :], axis=2)
            csr_dist = csr_matrix(dist_matrix)
            mst = minimum_spanning_tree(csr_dist)
            
            # Convert MST to edges
            rows, cols = mst.nonzero()
            for i, j in zip(rows, cols):
                u, v = indices[i], indices[j]
                gt_edges.add(tuple(sorted((u, v))))

        elif config.algorithm == "greedy_path":
            # A simple greedy path construction
            # This can be implemented as an alternative
            # For now, we focus on MST
            logging.warning("Greedy Path GT construction not implemented, falling back to MST.")
            # Build a distance matrix and run MST
            dist_matrix = np.linalg.norm(line_points[:, np.newaxis, :] - line_points[np.newaxis, :, :], axis=2)
            csr_dist = csr_matrix(dist_matrix)
            mst = minimum_spanning_tree(csr_dist)
            
            # Convert MST to edges
            rows, cols = mst.nonzero()
            for i, j in zip(rows, cols):
                u, v = indices[i], indices[j]
                gt_edges.add(tuple(sorted((u, v))))

    return gt_edges
---
gnn_data_preparation/__init__.py
---

---
gnn_data_preparation/main_create_dataset.py
---
# data_creation/main_create_dataset.py
import argparse
import yaml
import logging
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config_models import DatasetCreationConfig
from graph_constructor import create_input_graph_edges, create_ground_truth_graph_edges
from feature_engineering import get_node_features, get_edge_features
from dataset_generator import HistoricalLayoutGNNDataset, create_sklearn_dataframe
from utils import setup_logging, find_page_ids
from torch_geometric.data import Data

def process_page(page_id: str, data_dir: Path, config: DatasetCreationConfig):
    """Processes a single page from a given directory to create a PyG Data object."""
    try:
        # 1. Load data from .txt files using the provided data_dir
        dims_path = data_dir / f"{page_id}_dims.txt"
        inputs_norm_path = data_dir / f"{page_id}_inputs_normalized.txt"
        labels_path = data_dir / f"{page_id}_labels_textline.txt"

        if not all([dims_path.exists(), inputs_norm_path.exists(), labels_path.exists()]):
            logging.warning(f"Skipping page {page_id}: missing one or more required files in {data_dir}.")
            return None

        if inputs_norm_path.stat().st_size == 0 or labels_path.stat().st_size == 0:
            logging.warning(f"Skipping page {page_id}: input or label file is empty.")
            return None

        page_dims_arr = np.loadtxt(dims_path)
        page_dims = {'width': page_dims_arr[0], 'height': page_dims_arr[1]}
        points_normalized = np.loadtxt(inputs_norm_path)
        
        if points_normalized.ndim == 1:
            points_normalized = points_normalized.reshape(1, -1)
            
        textline_labels = np.loadtxt(labels_path, dtype=int)

        if len(points_normalized) < config.min_nodes_per_page:
            logging.info(f"Skipping page {page_id}: has {len(points_normalized)} nodes, less than min {config.min_nodes_per_page}.")
            return None
            
    except Exception as e:
        logging.error(f"Error loading or processing initial data for page {page_id}: {e}")
        return None

    # 2. Construct Input and Ground Truth Graphs (Unchanged)
    input_graph_data = create_input_graph_edges(points_normalized, page_dims, config.input_graph)
    gt_edges_set = create_ground_truth_graph_edges(points_normalized, textline_labels, config.ground_truth)

    # 3. Create edge_index and edge labels (y) (Unchanged)
    input_edges = list(input_graph_data["edges"])
    if not input_edges:
        logging.warning(f"Skipping page {page_id}: no edges were generated for the input graph.")
        return None
        
    edge_index_undirected = torch.tensor(input_edges, dtype=torch.long).t().contiguous()
    edge_y = torch.tensor([1 if tuple(e) in gt_edges_set else 0 for e in input_edges], dtype=torch.long)
    
    if config.input_graph.directionality == "bidirectional":
        edge_index = torch.cat([edge_index_undirected, edge_index_undirected.flip(0)], dim=1)
        edge_y = torch.cat([edge_y, edge_y], dim=0)
    else: # unidirectional
        edge_index = edge_index_undirected

    # 4. Feature Engineering (Unchanged)
    node_features = get_node_features(points_normalized, input_graph_data["heuristic_degrees"], config.features)
    edge_features = get_edge_features(edge_index, node_features, input_graph_data["heuristic_edge_counts"], config.features)
    
    # 5. Assemble PyG Data object (Unchanged)
    data = Data(x=node_features, edge_index=edge_index, edge_y=edge_y)
    if edge_features is not None:
        data.edge_attr = edge_features
        
    if config.features.use_page_aspect_ratio:
        data.page_aspect_ratio = torch.tensor([page_dims['width'] / page_dims['height'] if page_dims['height'] > 0 else 1.0])

    data.page_id = page_id
    data.num_nodes = len(points_normalized)
    
    # Sanity checks (Unchanged)
    assert data.x.shape[0] == data.num_nodes, "Node feature dimension mismatch"
    assert data.edge_index.shape[1] == data.edge_y.shape[0], "Edge index and edge label mismatch"
    if data.edge_attr is not None:
        assert data.edge_index.shape[1] == data.edge_attr.shape[0], "Edge index and edge attribute mismatch"
        
    return data

def main():
    # --- NEW: Setup argument parsing ---
    parser = argparse.ArgumentParser(description="Create a dataset with predefined training and validation/test splits.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/1_dataset_creation_config.yaml',
        help='Path to the YAML configuration file.'
    )
    parser.add_argument('--output_dir', type=str, help='Override the main output directory.')
    parser.add_argument('--train_data_dir', type=str, help='Override the input directory for training data.')
    parser.add_argument('--val_test_data_dir', type=str, help='Override the input directory for validation/test data.')
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    assert config_path.exists(), f"Configuration file not found at {config_path}"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = DatasetCreationConfig(**config_dict)

    # --- MODIFIED: Prioritize command-line arguments for directories ---
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_dir)
    train_data_dir = Path(args.train_data_dir) if args.train_data_dir else Path(config.train_data_dir)
    val_test_data_dir = Path(args.val_test_data_dir) if args.val_test_data_dir else Path(config.val_test_data_dir)

    # Setup output directory and logging
    dataset_version_dir = output_dir #/ f"{config.manuscript_name}-v{config.version}"
    dataset_version_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(dataset_version_dir / "dataset_creation.log")
    
    logging.info("Starting dataset creation process with predefined splits...")
    logging.info(f"Configuration (with potential overrides):\n"
                 f"  output_dir: {output_dir}\n"
                 f"  train_data_dir: {train_data_dir}\n"
                 f"  val_test_data_dir: {val_test_data_dir}")

    # Validate input directories
    logging.info(f"Training data source: {train_data_dir}")
    logging.info(f"Validation/Test data source: {val_test_data_dir}")
    assert train_data_dir.is_dir(), f"Training data directory not found: {train_data_dir}"
    assert val_test_data_dir.is_dir(), f"Validation/Test data directory not found: {val_test_data_dir}"

    # Find page IDs from respective directories
    train_page_ids = find_page_ids(train_data_dir)
    val_test_page_ids = find_page_ids(val_test_data_dir)
    
    logging.info(f"Found {len(train_page_ids)} pages for the training set.")
    logging.info(f"Found {len(val_test_page_ids)} pages to be split into validation and test sets.")
    
    if not train_page_ids or not val_test_page_ids:
        logging.critical("One of the source directories is empty. Cannot proceed.")
        return

    # Split the validation/test data
    val_ratio = config.splitting.val_ratio
    logging.info(f"Splitting validation/test data: {val_ratio*100}% validation, {(1-val_ratio)*100:.0f}% test.")
    
    rng = np.random.default_rng(config.splitting.random_seed)
    val_test_ids_arr = np.array(val_test_page_ids)
    rng.shuffle(val_test_ids_arr)
    
    split_index = int(len(val_test_ids_arr) * val_ratio)
    val_ids = val_test_ids_arr[:split_index]
    test_ids = val_test_ids_arr[split_index:]
    
    # Define the final splits
    splits = {
        "train": train_page_ids,
        "val": list(val_ids),
        "test": list(test_ids),
    }

    logging.info(f"Final split sizes: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    page_id_to_dir_map = {page_id: train_data_dir for page_id in splits['train']}
    page_id_to_dir_map.update({page_id: val_test_data_dir for page_id in splits['val']})
    page_id_to_dir_map.update({page_id: val_test_data_dir for page_id in splits['test']})

    # To maintain a similar output structure, we simulate a single "fold"
    fold_idx = 0 
    fold_dir = dataset_version_dir / "folds" / f"fold_{fold_idx}"
    logging.info(f"===== Processing all splits into '{fold_dir}' =====")

    for split_name, split_page_ids in splits.items():
        if not split_page_ids:
            logging.warning(f"Skipping {split_name} split as it contains no page IDs.")
            continue
            
        logging.info(f"--- Processing {split_name} split with {len(split_page_ids)} pages ---")
        
        data_list = []
        page_map = {} 

        for page_id in tqdm(split_page_ids, desc=f"Processing {split_name}"):
            source_dir = page_id_to_dir_map[page_id]
            graph_data = process_page(page_id, source_dir, config)
            if graph_data:
                page_map[len(data_list)] = page_id
                data_list.append(graph_data)

        if not data_list:
            logging.warning(f"No data generated for {split_name} split. Skipping save steps.")
            continue

        # Save GNN Dataset
        gnn_dir = fold_dir / "gnn" / split_name
        gnn_dir.mkdir(parents=True, exist_ok=True)
        HistoricalLayoutGNNDataset(root=str(gnn_dir), data_list=data_list)
        logging.info(f"Saved GNN {split_name} dataset to '{gnn_dir}'")
        
        # Save Sklearn Dataset
        if config.sklearn_format.enabled:
            sklearn_dir = fold_dir / "sklearn"
            sklearn_dir.mkdir(parents=True, exist_ok=True)
            df = create_sklearn_dataframe(data_list, page_map, config)
            if df is not None:
                csv_path = sklearn_dir / f"{split_name}.csv"
                df.to_csv(csv_path, index=False)
                logging.info(f"Saved Sklearn {split_name} dataset to '{csv_path}'")
    
    logging.info("Dataset creation complete.")
    
if __name__ == "__main__":
    main()
---
gnn_data_preparation/utils.py
---
# data_creation/utils.py
import logging
import sys
from typing import List
from pathlib import Path

def setup_logging(log_path: Path):
    """Sets up a logger that prints to console and saves to a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

def find_page_ids(data_dir: Path) -> List[str]:
    """Finds all unique page identifiers in the raw data directory."""
    page_ids = set()
    for f in data_dir.glob('*_dims.txt'):
        page_id = f.name.replace('_dims.txt', '')
        page_ids.add(page_id)
    
    sorted_ids = sorted(list(page_ids))
    logging.info(f"Found {len(sorted_ids)} pages in '{data_dir}'.")
    return sorted_ids
---
gnn_inference.py
---
# inference_with_eval.py


import torch
import numpy as np
import yaml
import logging
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch.nn.functional as F


import matplotlib
matplotlib.use('Agg')  # MUST be called before importing pyplot
import shutil
from collections import defaultdict
import xml.etree.ElementTree as ET
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from segment_from_point_clusters import segmentLinesFromPointClusters



from gnn_data_preparation.config_models import DatasetCreationConfig
from gnn_data_preparation.graph_constructor import create_input_graph_edges
from gnn_data_preparation.feature_engineering import get_node_features, get_edge_features
from gnn_data_preparation.utils import setup_logging
from torch_geometric.data import Data



def get_device(device_config: str) -> torch.device:
    """Gets the torch device based on config and availability, and logs the choice."""
    if device_config == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    print(f"Using device: {device}") # This line confirms the choice in your logs
    return device



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


def create_page_xml(
    page_id,
    model_positive_edges,
    points_unnormalized,
    page_dims,
    output_path: Path,
    pred_node_labels: np.ndarray,
    polygons_data: dict,
    use_best_fit_line: bool = False,
    extend_percentage: float = 0.01
):
    """
    Generates a PAGE XML file. For each connected component, it creates a <TextLine>
    with a baseline and now also includes the <Coords> for the text line polygon.
    """
    PAGE_XML_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    ET.register_namespace('', PAGE_XML_NAMESPACE)

    num_nodes = len(points_unnormalized)

    adj = defaultdict(list)
    for u, v in model_positive_edges:
        adj[u].append(v)
        adj[v].append(u)

    components = find_connected_components(model_positive_edges, num_nodes)

    pc_gts = ET.Element(f"{{{PAGE_XML_NAMESPACE}}}PcGts")
    metadata = ET.SubElement(pc_gts, "Metadata")
    ET.SubElement(metadata, "Creator").text = "GNN-Prediction-Script"

    page = ET.SubElement(pc_gts, "Page", attrib={
        "imageFilename": f"{page_id}.jpg",
        "imageWidth": str(int(page_dims['width']*2)),
        "imageHeight": str(int(page_dims['height']*2))
    })

    min_x = np.min(points_unnormalized[:, 0])
    min_y = np.min(points_unnormalized[:, 1])
    max_x = np.max(points_unnormalized[:, 0])
    max_y = np.max(points_unnormalized[:, 1])
    region_coords = f"{int(min_x*2)},{int(min_y*2)} {int(max_x*2)},{int(min_y*2)} {int(max_x*2)},{int(max_y*2)} {int(min_x*2)},{int(max_y*2)}"

    text_region = ET.SubElement(page, "TextRegion", id="region_1")
    ET.SubElement(text_region, "Coords", points=region_coords)

    for component in components:
        if not component: continue
        
        # Get the points for the current component
        component_points = np.array([points_unnormalized[idx] for idx in component])
        
        if len(component_points) < 1:
            continue

        # Determine the label for this component to ensure consistent IDs
        line_label = pred_node_labels[component[0]]
            
        text_line = ET.SubElement(text_region, "TextLine", id=f"line_{line_label + 1}")
        
        if use_best_fit_line:
            baseline_points_for_fitting = np.array(
                [[p[0], p[1] + (p[2] / 2)] for p in component_points]
            )
            endpoints = fit_robust_line_and_extend(
                baseline_points_for_fitting, 
                extend_percentage=extend_percentage,
                robust_method='huber'
            )
            if endpoints:
                p1, p2 = endpoints
                baseline_points_str = f"{int(p1[0] * 2)},{int(p1[1] * 2)} {int(p2[0] * 2)},{int(p2[1] * 2)}"
            else:
                continue
        else:
            path_indices = trace_component_with_backtracking(component, adj)
            if len(path_indices) < 1: continue
            ordered_points = [points_unnormalized[idx] for idx in path_indices]
            baseline_points_str = " ".join([f"{int(p[0]*2)},{int((p[1]+(p[2]/2))*2)}" for p in ordered_points])

        ET.SubElement(text_line, "Baseline", points=baseline_points_str)
        
        # Add the corresponding polygon coordinates to the TextLine
        if line_label in polygons_data:
            polygon_points = polygons_data[line_label]
            coords_str = " ".join([f"{p[0]},{p[1]}" for p in polygon_points]) # we do not double the coords here, because we upscale the HEATMAP!
            ET.SubElement(text_line, "Coords", points=coords_str)
        else:
            logging.warning(f"Page {page_id}: No polygon data found for line label {line_label}, Coords tag will be omitted.")

    tree = ET.ElementTree(pc_gts)
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="\t", level=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding='UTF-8', xml_declaration=True)
# ===================================================================
#                       MAIN INFERENCE SCRIPT
# ===================================================================

def run_gnn_inference(args):
    """Main function for system-level evaluation."""
    # 2. Set other arguments
    input_dir = f"{args.manuscript_path}/gnn-dataset"
    output_dir = f"{args.manuscript_path}/segmented_lines"

    setup_logging(Path(output_dir) / 'inference_with_eval.log')
    device = get_device('auto')

    with open(args.dataset_config_path, 'r') as f: 
        d_config = DatasetCreationConfig(**yaml.safe_load(f))
    checkpoint = torch.load(args.model_checkpoint, map_location=device, weights_only=False)
    model = checkpoint['model']
    model.to(device)
    model.eval()


    predictions_dataset_dir = Path(output_dir) / "gnn-format" # TODO use path to join
    predictions_dataset_dir.mkdir(exist_ok=True)
    xml_output_dir = Path(output_dir) / "page-xml-format" #fix syntax
    xml_output_dir.mkdir(exist_ok=True)


    
    input_files = sorted(list(Path(input_dir).glob('*_inputs_normalized.txt')))
    
    for file_path in input_files:
        page_id = file_path.name.replace('_inputs_normalized.txt', '')
        logging.info("--- Processing page: %s ---", page_id)
        try:
            points_normalized = np.loadtxt(file_path)
            if points_normalized.ndim == 1: points_normalized = points_normalized.reshape(1, -1)
            unnormalized_path = file_path.parent / f"{page_id}_inputs_unnormalized.txt"
            dims_path = file_path.parent / f"{page_id}_dims.txt"

            if not unnormalized_path.exists() or not dims_path.exists():
                logging.warning("Skipping XML generation for page %s: Unnormalized or dims file not found.", page_id)
                can_generate_xml = False
            else:
                points_unnormalized = np.loadtxt(unnormalized_path)
                if points_unnormalized.ndim == 1: points_unnormalized = points_unnormalized.reshape(1, -1)
                dims = np.loadtxt(dims_path)
                page_dims = {'width': dims[0], 'height': dims[1]}
                can_generate_xml = True

        except Exception as e:
            logging.error("Could not load data for page %s: %s", page_id, e); continue


        page_dims_norm = {'width': 1.0, 'height': 1.0} # Use normalized dims for graph creation
        input_graph_data = create_input_graph_edges(points_normalized, page_dims_norm, d_config.input_graph)
        input_edges_set = input_graph_data["edges"]

        if not input_edges_set:
            logging.warning("Skipping page %s: No candidate edges generated by input graph constructor.", page_id)

        else:
            edge_index_undirected = torch.tensor(list(input_edges_set), dtype=torch.long).t().contiguous()
            if d_config.input_graph.directionality == "bidirectional":
                edge_index = torch.cat([edge_index_undirected, edge_index_undirected.flip(0)], dim=1)
            else:
                edge_index = edge_index_undirected
            
            node_features = get_node_features(points_normalized, input_graph_data["heuristic_degrees"], d_config.features)
            edge_features = get_edge_features(edge_index, node_features, input_graph_data["heuristic_edge_counts"], d_config.features)
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features).to(device)

            threshold = 0.5  # this can be adjusted to optimize precision/recall trade-off

            with torch.no_grad():
                logits = model(data.x, data.edge_index, data.edge_attr)  # [num_edges, num_classes]
                probs = F.softmax(logits, dim=1)  # convert logits → probabilities
                # Take probability of class 1 (positive edge)
                pos_probs = probs[:, 1]
                # Apply threshold
                pred_edge_labels = (pos_probs > threshold).cpu().numpy().astype(int)

            model_positive_edges = {
                tuple(sorted(e)) 
                for e in data.edge_index[:, pred_edge_labels == 1].cpu().numpy().T
            }

        # For Rand Index and XML, we need the final node clusters from the model's predictions
        # This requires the edge_index tensor that was fed to the model
        pred_edge_index_tensor = torch.tensor(list(input_edges_set), dtype=torch.long).t()
        pred_node_labels_all_edges = torch.zeros(pred_edge_index_tensor.shape[1], dtype=torch.int32)
        # Create a boolean mask to identify positive predictions
        positive_edges_map = {edge: True for edge in model_positive_edges}
        for i in range(pred_edge_index_tensor.shape[1]):
            edge = tuple(sorted(pred_edge_index_tensor[:, i].tolist()))
            if edge in positive_edges_map:
                pred_node_labels_all_edges[i] = 1
    
        # convert binary edge classification predictions to node labels
        pred_node_labels = get_node_labels_from_edge_labels(pred_edge_index_tensor, pred_node_labels_all_edges, len(points_normalized))

        # Save predicted node labels, and input files
        pred_labels_path = predictions_dataset_dir / f"{page_id}_labels_textline.txt"
        np.savetxt(pred_labels_path, pred_node_labels, fmt='%d')
        gnn_input_dir = file_path.parent
        for associated_file in gnn_input_dir.glob(f"{page_id}_*"):
            if associated_file.name == f"{page_id}_labels_textline.txt": continue
            shutil.copy(associated_file, predictions_dataset_dir / associated_file.name)


        # TODO: At this step, the image has been processed, heatmaps generated, GNN inference and predicted labels obtained and saved in gnn-format as {page_id}_labels_textline.txt. {page_id}_labels_textline.txt contains the predicted node labels for text lines, just that nodes (in file {page_id}_inputs_normalized.txt) belonging to the same text line have the same integer label. It is at this step that we want to (optionally) have a UI such that a human can correct the predicted node labels before proceeding to generate the PAGE XML files and segmented line images.



        # Generate line images and polygon data first, now that the files are in place.
        logging.info("Generating line images and polygon data for page %s...", page_id)
        polygons_data = segmentLinesFromPointClusters(Path(input_dir).parent, page_id, BINARIZE_THRESHOLD=args.BINARIZE_THRESHOLD, BBOX_PAD_V=args.BBOX_PAD_V, BBOX_PAD_H=args.BBOX_PAD_H, CC_SIZE_THRESHOLD_RATIO=args.CC_SIZE_THRESHOLD_RATIO, GNN_PRED_PATH=output_dir)

        # Generate the PAGE XML, now including the polygon data
        if can_generate_xml:
            xml_path = xml_output_dir / f"{page_id}.xml"
            create_page_xml(
                page_id,
                model_positive_edges,
                points_unnormalized,
                page_dims,
                xml_path,
                pred_node_labels,   # Pass node labels for matching
                polygons_data,      # Pass the generated polygon data
                use_best_fit_line=False,
                extend_percentage=0.01
            )
            logging.info("Saved PAGE XML with polygon predictions to: %s", xml_path)

        #copy files from images_resized to segmented_lines/images_resized
        resized_images_src = Path(args.manuscript_path) / "images_resized"
        resized_images_dst = Path(output_dir) / "images_resized"
        resized_images_dst.mkdir(exist_ok=True)
        for img_file in resized_images_src.glob("*.jpg"):
            shutil.copy(img_file, resized_images_dst / img_file.name)




---
inference.py
---
import os
import argparse
import gc
from PIL import Image
import torch

from segmentation.segment_graph import images2points
from gnn_inference import run_gnn_inference

import sys
# Get the directory where the current script is located (gnn_inference)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (src)
parent_dir = os.path.dirname(current_dir)
# Add 'src' to the system path so Python can find 'gnn_training'
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
print(f"Added {parent_dir} to sys.path to allow imports from 'gnn_training'")




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




---
pretrained_gnn/gnn_preprocessing_v2.yaml
---
# ===================================================================
#             Dataset Creation Configuration
# ===================================================================

min_nodes_per_page: 10 # Skip pages with fewer nodes than this

# --- MODIFIED: Data splitting configuration for a fixed split ---
splitting:
  random_seed: 49 # Seed for shuffling the val/test data before splitting
  val_ratio: 0.99 # 95% of val_test_data_dir goes to validation, 5% to test

# ===================================================================
#                Input Graph Construction
# ===================================================================
input_graph:
  # --- Heuristic Graph (Optional) ---
  use_heuristic_graph: True
  heuristic_params:
    k: 10 # HEURISTIC_GRAPH_K
    cosine_sim_threshold: -0.8 # OPPOSITE_NEIGHBOR_COS_SIM_THRESHOLD

  # --- Additional Connectivity ---
  # Strategy to add more edges to ensure GT is a subset.
  # Options: "knn", "none" (future: "expander")
  connectivity:
    # strategies: ["knn"] # Example: using both strategies "knn", 
    strategies: ["angular_knn"] #angular_knn

    # Parameters for the knn strategy
    knn_params:
      k: 12 # K for KNN # we can only predict if this is a "true" super set. Make this big!

    # Parameters for the second_shortest_heuristic strategy
    second_shortest_params:
          k: 10 # Number of neighbors to consider for finding pairs
          cosine_sim_threshold: -0.8 # Collinearity threshold
          min_angle_degrees: 45 # The minimum angle between the 1st and 2nd best pairs' edges

    angular_knn_params:
      k: 50 # K for angular KNN
      sector_angle_degrees: 20 # Minimum angle between edges to consider them connected


  # --- Graph Structure ---
  # Options: "bidirectional", "unidirectional"
  directionality: "bidirectional"

# ===================================================================
#                  Feature Engineering
# ===================================================================
features:
  # --- Node Features ---
  use_node_coordinates: True
  use_node_font_size: False
  use_heuristic_degree: True
  heuristic_degree_encoding: "one_hot" # Options: "linear_map", "one_hot"
  heuristic_degree_encoding_params:
    linear_map_factor: 0.1
    one_hot_max_degree: 10 # Cap for one-hot encoding dimension

  # --- Edge Features ---
  use_relative_distance: True # rel_x, rel_y
  use_euclidean_distance: True
  use_aspect_ratio_rel: True
  use_overlap: True
  overlap_encoding: "one_hot" # Options: "linear_map", "one_hot"
  overlap_encoding_params:
    linear_map_factor: 0.1
    one_hot_max_overlap: 10

  # --- Graph-level Features ---
  use_page_aspect_ratio: True

# ===================================================================
#              Ground Truth Graph Construction
# ===================================================================
ground_truth:
  # Algorithm to connect nodes within a textline
  # Options: "mst", "greedy_path"
  algorithm: "mst"

# ===================================================================
#               Alternate (Sklearn) Dataset Format
# ===================================================================
sklearn_format:
  enabled: False
  features:
    # Select which features to include in the CSV
    - "source_node_features" # includes x, y, font_size, heuristic_degree
    - "target_node_features"
    - "edge_features" # includes all enabled edge features
    - "page_features" # includes page_aspect_ratio
  # Advanced: Add aggregated features from N-hop neighbors
  # This is computationally expensive.
  use_n_hop_features: False
  n_hop_config:
    hops: 1 # Number of hops
    aggregations: ["mean", "std"] # Aggregations to apply
---
README.md
---
# Towards Text-Line Segmentation of Historical Documents Using Graph Neural Networks and Synthetic Layout Data


**Version:** 2.0
**Last Updated:** Jan 8, 2026

## **Project Components**
*   **💻 [Out-of-the-box Inference](https://github.com/flame-cai/gnn-synthetic-layout-historical?tab=readme-ov-file#-stand-alone-out-of-the-box-inference):** Run stand-alone inference

## **How to Use**
###  Stand-alone Out-of-the-box Inference
#### 🔵 Install Conda Environment
```bash
cd src
conda env create -f environment.yaml
conda activate gnn_layout
```

```bash
cd src/gnn_inference
python inference.py --manuscript_path "./demo_manuscripts/sample_manuscript_1/"
```

This will process all the manuscript images in sample_manuscript_1 and save the segmented line images in folder `sample_manuscript_1/segmented_lines/` in PAGE_XML format, GNN format, and as individual line images.

> **NOTE 1:**  
> This project is made for Handwritten Sanskrit Manuscripts in Devanagari script, however it will work reasonibly well on other scripts if they fit the following criteria:
> 1) [CRAFT](https://github.com/clovaai/CRAFT-pytorch) successfully detects the script characters  
> 2) Character spacing is less than Line spacing.

> **NOTE 2:**  
> `sample_manuscript_1/` and `sample_manuscript_2` contain high resolution images and will work out of the box. However, `sample_manuscript_3/` contains lower resolution images - for whom the feature engineering parameter `min_distance` in `src/gnn_inference/segmentation/segment_graph.py` will need to be adjusted as follows:  
>  
> `raw_points = heatmap_to_pointcloud(region_score, min_peak_value=0.4, min_distance=10)`

> **NOTE 3:**  
> The inference code resizes very large images to `2500` longest side for processing to reduce the GPU memory requirements and to standardize the feature extraction process. If you wish to change this limit, you can do so in `src/gnn_inference/inference.py` at the following lines:
> ```python
> target_longest_side = 2500
> ```
> However, this is also require adjusting the feature extraction parameter `min_distance` in `src/gnn_inference/segmentation/segment_graph.py` accordingly.


## Acknowledgements
We would like to thank Petar Veličković, Oliver Hellwig, Dhavel Patel for their extermely valuable inputs and discussions.

---
segmentation/craft.py
---
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torchvision import models
# import matplotlib.pyplot as plt
from collections import namedtuple
from packaging import version
from collections import OrderedDict


# #GLOBAL VARIABLES
# lineheight_baseline_percentile = None
# binarize_threshold = None




def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        if version.parse(torchvision.__version__) >= version.parse('0.13'):
            vgg_pretrained_features = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
            ).features
        else: #torchvision.__version__ < 0.13
            models.vgg.model_urls['vgg16_bn'] = models.vgg.model_urls['vgg16_bn'].replace('https://', 'http://')
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):         # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
            
        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad= False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        try: # multi gpu needs this
            self.rnn.flatten_parameters()
        except: # quantization doesn't work with this
            pass
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

    def forward(self, input):
        return self.ConvNet(input)



class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction

"""### CRAFT Model"""

#CRAFT

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def detect(img, detector, device):
    x = [np.transpose(normalizeMeanVariance(img), (2, 0, 1))]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)
    with torch.no_grad():
        y = detector(x)
        
    region_score = y[0,:,:,0].cpu().data.numpy()
    affinity_score = y[0,:,:,1].cpu().data.numpy()

    # clear GPU memory
    del x
    del y
    torch.cuda.empty_cache()

    return region_score,affinity_score
---
segmentation/pretrained_unet_craft/README.md
---
Download craft_mlt_pth from [here](https://huggingface.co/amitesh863/craft/resolve/main/craft_mlt_25k.pth?download=true) into this folder.

Reference paper: https://arxiv.org/pdf/1904.01941
---
segmentation/segment_graph.py
---
import os
import numpy as np
import torch
import cv2
from scipy.ndimage import maximum_filter
from scipy.ndimage import label
from scipy.ndimage import maximum_filter, label
from skimage.draw import circle_perimeter

from .craft import CRAFT, copyStateDict, detect
from .utils import load_images_from_folder


def heatmap_to_pointcloud(heatmap, min_peak_value=0.3, min_distance=5, max_growth_radius=50):
    """
    Convert a 2D heatmap to a point cloud (X, Y, Radius) by identifying local maxima
    and estimating a radius for each by growing a circle as long as the heatmap
    intensity along the circumference is decreasing.
    
    Parameters:
    -----------
    heatmap : numpy.ndarray
        2D array representing the heatmap.
    min_peak_value : float
        Minimum normalized value for a peak to be considered (normalized between 0 and 1).
        Peaks must have an intensity strictly greater than this value.
    min_distance : int
        Minimum distance between peaks in pixels. Used for `maximum_filter`.
    max_growth_radius : int, optional
        Maximum radius the circle is allowed to grow. If None, it defaults to
        half the minimum dimension of the heatmap.
        
    Returns:
    --------
    points_with_radius : numpy.ndarray
        Array of shape (N, 3) where N is the number of detected characters.
        Each row contains [Peak_X, Peak_Y, Estimated_Radius].
        Peak_X, Peak_Y are from the original peak detection.
        Estimated_Radius is the radius of the largest circle around the peak
        for which the average intensity on its circumference was still decreasing.
    """
    if heatmap.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max == h_min: # Handle flat heatmap
        return np.empty((0, 3), dtype=np.float64)
        
    # 1. Normalize heatmap to [0, 1]
    heatmap_norm = (heatmap - h_min) / (h_max - h_min)
    
    # 2. Find local maxima (Original logic)
    local_max_values = maximum_filter(heatmap_norm, size=min_distance)
    peaks_mask = (heatmap_norm == local_max_values) & (heatmap_norm > min_peak_value)
    
    # 3. Label connected components of these peak pixels
    labeled_individual_peaks, num_individual_peaks = label(peaks_mask)
    
    if num_individual_peaks == 0:
        return np.empty((0, 3), dtype=np.float64)

    points_and_radius = []
    
    H, W = heatmap_norm.shape
    if max_growth_radius is None:
        max_r_search = min(H, W) // 2
    else:
        max_r_search = max_growth_radius

    # 4. For each peak, grow a circle to estimate radius
    for peak_idx in range(1, num_individual_peaks + 1):
        peak_loc_y_arr, peak_loc_x_arr = np.where(labeled_individual_peaks == peak_idx)
        
        if peak_loc_y_arr.size == 0:
            continue
            
        peak_y, peak_x = peak_loc_y_arr[0], peak_loc_x_arr[0] # Use the first pixel of the peak area

        current_peak_intensity = heatmap_norm[peak_y, peak_x]
        last_ring_avg_intensity = current_peak_intensity
        estimated_radius = 0 # Radius 0 is the peak itself

        for r_test in range(1, max_r_search + 1):
            # Get coordinates of pixels on the circumference of radius r_test
            # skimage.draw.circle_perimeter ensures coordinates are within `shape` if provided.
            rr, cc = circle_perimeter(peak_y, peak_x, r_test, shape=heatmap_norm.shape)
            
            if rr.size == 0: # No pixels on this circumference (e.g., peak near edge, radius too large)
                break 

            current_ring_intensities = heatmap_norm[rr, cc]
            current_ring_avg_intensity = np.mean(current_ring_intensities)

            # Stop if slope is no longer strictly downward (i.e., current is flat or increasing)
            if current_ring_avg_intensity >= last_ring_avg_intensity:
                break 
            else:
                # Still decreasing, this radius is good. Update for next iteration.
                last_ring_avg_intensity = current_ring_avg_intensity
                estimated_radius = r_test # Update to this successful radius
        
        points_and_radius.append([float(peak_x), float(peak_y), float(estimated_radius)])

    return np.array(points_and_radius, dtype=np.float64)







def images2points(folder_path):
    print(folder_path)
    # how to get manuscript path from folder path - get parent directory
    m_path = os.path.dirname(folder_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Model Loading ---
    _detector = CRAFT()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pth_path = os.path.join(BASE_DIR, "pretrained_unet_craft", "craft_mlt_25k.pth")
    _detector.load_state_dict(copyStateDict(torch.load(pth_path, map_location=device)))
    detector = torch.nn.DataParallel(_detector).to(device)
    detector.eval()

    # --- Data Loading ---
    inp_images, file_names = load_images_from_folder(folder_path)
    print("Current Working Directory:", os.getcwd())

    # --- Processing Loop ---
    out_images = []
    normalized_points_list = [] # List for normalized points
    unnormalized_points_list = [] # NEW: List for raw, unnormalized points
    page_dimensions = []
    
    for image, _filename in zip(inp_images, file_names):
        # 0. Store original page dimensions
        original_height, original_width, _ = image.shape
        page_dimensions.append((original_width, original_height))

        # 1. Get region score (heatmap)
        region_score, affinity_score = detect(image, detector, device)
        assert region_score.shape == affinity_score.shape
        
        # 2. Convert heatmap to raw point coordinates (unnormalized)
        raw_points = heatmap_to_pointcloud(region_score, min_peak_value=0.4, min_distance=20)
        
        # --- NEW: Store the unnormalized points first ---
        unnormalized_points_list.append(raw_points)

        # 3. Normalize the points
        height, width = region_score.shape
        longest_dim = max(height, width)
        
        if longest_dim > 0:
            normalized_points = raw_points / longest_dim
        else:
            normalized_points = raw_points

        # 4. Store the processed data
        normalized_points_list.append(normalized_points)
        out_images.append(np.copy(region_score))

    # --- Saving Results ---
    heatmap_dir = f'{m_path}/heatmaps'
    base_data_dir = f'{m_path}/gnn-dataset'
    # frontend_graph_data_dir = f'{m_path}/frontend-graph-data'

    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(base_data_dir, exist_ok=True)
    # os.makedirs(frontend_graph_data_dir, exist_ok=True)

    # Save heatmaps
    for _img, _filename in zip(out_images, file_names):
        cv2.imwrite(os.path.join(heatmap_dir, _filename), 255 * _img)
    
    # --- Save NORMALIZED node features (for backward compatibility) ---
    for points, _filename in zip(normalized_points_list, file_names):
        output_filename = os.path.splitext(_filename)[0] + '_inputs_normalized.txt'
        output_path = os.path.join(base_data_dir, output_filename)
        np.savetxt(output_path, points, fmt='%f')

    # --- NEW: Save UNNORMALIZED node features to a separate file ---
    for raw_points, _filename in zip(unnormalized_points_list, file_names):
        raw_output_filename = os.path.splitext(_filename)[0] + '_inputs_unnormalized.txt'
        raw_output_path = os.path.join(base_data_dir, raw_output_filename)
        np.savetxt(raw_output_path, raw_points, fmt='%f')

    # Save the page dimensions
    for (width, height), _filename in zip(page_dimensions, file_names):
        dims_filename = os.path.splitext(_filename)[0] + '_dims.txt'
        dims_path = os.path.join(base_data_dir, dims_filename)
        with open(dims_path, 'w') as f:
            f.write(f"{width/2} {height/2}")


    # --- Cleanup ---
    del detector
    del _detector
    torch.cuda.empty_cache()

    print(f"Finished processing. All data saved to: {base_data_dir}")
    



---
segmentation/utils.py
---
import os
import numpy as np
import cv2
from skimage import io

# Function Definitions
def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def load_images_from_folder(folder_path):
    inp_images = []
    file_names = []
    
    # Get all files in the directory
    files = sorted(os.listdir(folder_path))
    
    for file in files:
        # Check if the file is an image (PNG or JPG)
        if file.lower().endswith(('.png', '.jpg', '.jpeg','.tif')):
            try:
                # Construct the full file path
                file_path = os.path.join(folder_path, file)
                
                # Open the image file
                image = loadImage(file_path)
                
                # Append the image and filename to our lists
                inp_images.append(image)
                file_names.append(file)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    
    return inp_images, file_names
---
segment_from_point_clusters.py
---
import os
import shutil
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import UnivariateSpline
import math
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend (fast)
import matplotlib.pyplot as plt
from skimage import io


def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def resize_with_padding(image, target_size, background_color=(0, 0, 0)):
    """
    Resizes an image to a target size while maintaining its aspect ratio by padding.
    """
    target_w, target_h = target_size
    if image is None or image.size == 0:
        return np.full((target_h, target_w, 3), background_color, dtype=np.uint8)
        
    h, w = image.shape[:2]

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded_image = np.full((target_h, target_w, 3), background_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    
    return padded_image

def visualize_projection_profile(profile, crop_shape, orientation='horizontal', color=(255, 255, 255), thickness=1):
    """
    Visualizes a 1D projection profile, creating an image that corresponds to the
    dimensions of the original crop.
    """
    if profile is None or len(profile) == 0:
        return np.zeros((crop_shape[0], crop_shape[1], 3), dtype=np.uint8)

    crop_h, crop_w = crop_shape
    max_val = np.max(profile)
    if max_val == 0:
        max_val = 1

    if orientation == 'horizontal':
        vis_image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        for i, val in enumerate(profile):
            length = int((val / max_val) * crop_w)
            if i < crop_h:
                cv2.line(vis_image, (0, i), (length, i), color, thickness)
    else:  # vertical
        vis_image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        for i, val in enumerate(profile):
            length = int((val / max_val) * crop_h)
            if i < crop_w:
                cv2.line(vis_image, (i, crop_h - 1), (i, crop_h - 1 - length), color, thickness)
            
    return vis_image

def create_debug_collage(original_uncropped, padded_crop, cleaned_blob, component_viz_img, heatmap_crop, config):
    """
    Creates a 2x4 collage of debugging images for a single bounding box.
    """
    TILE_SIZE = (200, 200)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.4
    FONT_COLOR = (255, 255, 255)

    # Prepare heatmap-related visualizations
    _, bin_heat_crop = cv2.threshold(heatmap_crop, config['BINARIZE_THRESHOLD'], 255, cv2.THRESH_BINARY)
    h_prof_heat = np.sum(bin_heat_crop, axis=1) / 255
    v_prof_heat = np.sum(bin_heat_crop, axis=0) / 255
    v_prof_heat_viz = visualize_projection_profile(v_prof_heat, bin_heat_crop.shape, 'vertical', color=(0, 0, 255))
    h_prof_heat_viz = visualize_projection_profile(h_prof_heat, bin_heat_crop.shape, 'horizontal', color=(0, 0, 255))
    heatmap_colorized = cv2.applyColorMap(heatmap_crop, cv2.COLORMAP_JET)

    # Create resized tiles for the collage
    orig_uncropped_tile = resize_with_padding(original_uncropped, TILE_SIZE)
    padded_crop_tile = resize_with_padding(padded_crop, TILE_SIZE)
    cleaned_blob_tile = resize_with_padding(cleaned_blob, TILE_SIZE)
    component_viz_tile = resize_with_padding(component_viz_img, TILE_SIZE)
    heat_crop_tile = resize_with_padding(heatmap_colorized, TILE_SIZE)
    v_prof_heat_tile = resize_with_padding(v_prof_heat_viz, TILE_SIZE)
    h_prof_heat_tile = resize_with_padding(h_prof_heat_viz, TILE_SIZE)
    bin_heat_tile = resize_with_padding(bin_heat_crop, TILE_SIZE)

    # Add labels to each tile
    cv2.putText(orig_uncropped_tile, "Original BBox", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(padded_crop_tile, "Padded BBox", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(cleaned_blob_tile, "Cleaned Blob", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(component_viz_tile, "Analyzed Components", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(heat_crop_tile, "Heatmap", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(v_prof_heat_tile, "V-Profile (Heat)", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(h_prof_heat_tile, "H-Profile (Heat)", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(bin_heat_tile, "Binarized Heatmap", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)

    # Assemble the collage
    row1 = cv2.hconcat([orig_uncropped_tile, padded_crop_tile, cleaned_blob_tile, component_viz_tile])
    row2 = cv2.hconcat([heat_crop_tile, v_prof_heat_tile, h_prof_heat_tile, bin_heat_tile])
    collage = cv2.vconcat([row1, row2])
    
    return collage


def analyze_and_clean_blob(blob, line_type, config):
    """
    Analyzes connected components in a blob, identifies noise touching boundaries,
    and returns a cropped version of the blob along with crop coordinates.
    """
    if blob.size == 0:
        return blob, np.zeros_like(blob, dtype=np.uint8), [0, 0, 0, 0]

    _, bin_blob = cv2.threshold(blob, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_blob, connectivity=8)
    
    component_viz_img = np.zeros((blob.shape[0], blob.shape[1], 3), dtype=np.uint8)
    component_viz_img[labels != 0] = [255, 0, 0] # Default: Blue for valid components

    crop_h, crop_w = blob.shape
    crop_coords = [0, crop_h, 0, crop_w]

    if num_labels > 1:
        for i in range(1, num_labels):
            x_c, y_c, w_c, h_c, _ = stats[i]
            
            is_touching_boundary = (y_c == 0 or y_c + h_c == crop_h) if line_type != 'vertical' else (x_c == 0 or x_c + w_c == crop_w)
            is_size_constrained = (h_c <= config['CC_SIZE_THRESHOLD_RATIO'] * crop_h) if line_type != 'vertical' else (w_c <= config['CC_SIZE_THRESHOLD_RATIO'] * crop_w)

            if is_touching_boundary and is_size_constrained:
                component_viz_img[labels == i] = [0, 0, 255] # Red for noise
                if line_type != 'vertical':
                    if y_c == 0: crop_coords[0] = max(crop_coords[0], y_c + h_c)
                    if y_c + h_c == crop_h: crop_coords[1] = min(crop_coords[1], y_c)
                else:
                    if x_c == 0: crop_coords[2] = max(crop_coords[2], x_c + w_c)
                    if x_c + w_c == crop_w: crop_coords[3] = min(crop_coords[3], x_c)
    
    if crop_coords[0] >= crop_coords[1] or crop_coords[2] >= crop_coords[3]:
        return np.array([]), np.array([]), [0, 0, 0, 0]

    top, bottom, left, right = crop_coords
    cleaned_blob = blob[top:bottom, left:right]
    final_viz_img = component_viz_img[top:bottom, left:right]

    return cleaned_blob, final_viz_img, crop_coords


def gen_bounding_boxes(det, binarize_threshold):
    """
    Generates bounding boxes from a 2D heatmap loaded from an image file.

    This function assumes the input `det` is a NumPy array representing an image
    with pixel values in the [0, 255] range. It converts a normalized
    threshold (0.0 to 1.0) to this scale and then finds contours.

    Args:
        det (np.ndarray): The 2D input heatmap, assumed to be on a [0, 255] scale.
        binarize_threshold (float): The normalized threshold to apply.
                                    This value should be between 0.0 and 1.0.

    Returns:
        list[tuple]: A list of bounding boxes in the format (x, y, w, h).
    """
    # 1. Ensure the input is in the correct data type for OpenCV functions.
    #    We do NOT re-normalize the value range, as it's assumed to be 0-255.
    img = np.uint8(det)
    threshold_val = int(binarize_threshold * 255)
    _, img1 = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]




def load_node_features_and_labels(points_file, labels_file):
    points = np.loadtxt(points_file, dtype=float, ndmin=2).astype(int)
    with open(labels_file, "r") as f: labels = [line.strip() for line in f]
    features, filtered_labels = [], []
    for point, label in zip(points, labels):
        if label.lower() != "none":
            features.append(point)
            filtered_labels.append(int(label))
    return np.array(features), np.array(filtered_labels)

def assign_labels_and_plot(bounding_boxes, points, labels, image, output_path):
    # if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labeled_bboxes = []
    for x_min, y_min, w, h in bounding_boxes:
        x_max, y_max = x_min + w, y_min + h
        pts = [(p[0], p[1], lab) for p, lab in zip(points, labels) if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max]
        if pts and len({lab for _, _, lab in pts}) == 1:
            labeled_bboxes.append((x_min, y_min, w, h, pts[0][2]))
        elif pts:
            pts.sort(key=lambda p: p[1])
            boundaries = [y_min] + [max(y_min, min(y_max, int((pts[i-1][1] + pts[i][1]) / 2))) for i in range(1, len(pts)) if pts[i][2] != pts[i-1][2]] + [y_max]
            for i in range(1, len(boundaries)):
                top, bot = boundaries[i-1], boundaries[i]
                seg_label = next((lab for _, py, lab in pts if top <= py <= bot), None)
                if seg_label: labeled_bboxes.append((x_min, top, w, bot - top, seg_label))
    # for x, y, w, h, label in labeled_bboxes:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     cv2.putText(image, str(label), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.imwrite(output_path, image)
    # print(f"Annotated image saved as: {output_path}")
    return labeled_bboxes

def detect_line_type(boxes):
    if len(boxes) < 2: return 'horizontal', None
    centers = sorted([(x + w//2, y + h//2) for x, y, w, h, _ in boxes], key=lambda p: p[0])
    x_coords, y_coords = [p[0] for p in centers], [p[1] for p in centers]
    x_range, y_range = (max(coords) - min(coords) for coords in (x_coords, y_coords)) if centers else (0, 0)
    if x_range < y_range * 0.3: return 'vertical', None
    if y_range < x_range * 0.3: return 'horizontal', None
    try:
        X, y = np.array(x_coords).reshape(-1, 1), np.array(y_coords)
        ransac = RANSACRegressor(random_state=42).fit(X, y)
        if ransac.score(X, y) > 0.85: return 'slanted', {'slope': ransac.estimator_.coef_[0], 'intercept': ransac.estimator_.intercept_}
        return 'curved', {'spline': UnivariateSpline(x_coords, y_coords, s=len(centers)*2)}
    except: return 'horizontal', None

def transform_boxes_to_horizontal(boxes, line_type, params):
    if line_type == 'horizontal': return boxes
    t_boxes = []
    if line_type == 'vertical':
        for x, y, w, h, label in boxes: t_boxes.append((y, -x - w, h, w, label))
    elif line_type == 'slanted' and params:
        angle = math.atan(params['slope'])
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        for x, y, w, h, label in boxes:
            cx, cy = x + w//2, y + h//2
            t_boxes.append((int(cx*cos_a - cy*sin_a - w/2), int(cx*sin_a + cy*cos_a - h/2), w, h, label))
    else: return boxes
    return t_boxes

def normalize_coordinates(boxes):
    if not boxes: return []
    min_x, min_y = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return [(x - min_x, y - min_y, w, h, label) for x, y, w, h, label in boxes]

def crop_img(img):
    mask = img != int(np.median(img))
    if not np.any(mask): return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def get_bboxes_for_lines(img, unique_labels, bounding_boxes, debug_mode=False, debug_info=None):
    """
    Generates cropped line images. Uses DYNAMIC PADDING for the initial cleaning crop
    and STATIC PADDING from the config for the final canvas border.
    """
    line_images_data = []
    line_bounding_boxes_data = {label: [] for label in unique_labels}
    box_counter = 0
    config = debug_info.get('CONFIG', {}) if debug_info else {}


    for l in unique_labels:
        filtered_boxes = [box for box in bounding_boxes if box[4] == l]
        if not filtered_boxes:
            continue

        line_type, params = detect_line_type(filtered_boxes)

        if line_type == 'horizontal':
            PADDING_RATIO_V = config.get('BBOX_PAD_V', 0.7) # Default to 70% if not in config
            PADDING_RATIO_H = config.get('BBOX_PAD_H', 0.5) # Default to 50% if not in config
        else:
            PADDING_RATIO_V = config.get('BBOX_PAD_H', 0.5) # Default to 50% if not in config
            PADDING_RATIO_H = config.get('BBOX_PAD_V', 0.7) # Default to 70% if not in config

        cleaned_blobs_for_line = []
        final_coords_for_line = []

        for box in filtered_boxes:
            box_counter += 1
            orig_x, orig_y, orig_w, orig_h, _ = box
            try:
                # Get the original, unpadded crop for debugging.
                original_uncropped_blob = img[orig_y:orig_y + orig_h, orig_x:orig_x + orig_w]
                
                # Use dynamic, ratio-based padding for the initial crop to help cleaning.
                dynamic_pad_v = int(orig_h * PADDING_RATIO_V)
                dynamic_pad_h = int(orig_w * PADDING_RATIO_H)

                y1 = max(0, orig_y - dynamic_pad_v)
                y2 = orig_y + orig_h + dynamic_pad_v
                x1 = max(0, orig_x - dynamic_pad_h)
                x2 = orig_x + orig_w + dynamic_pad_h

                blob = img[y1:y2, x1:x2]
                if blob.size == 0:
                    continue

                cleaned_blob, component_viz_img, crop_coords = analyze_and_clean_blob(blob, line_type, config)
                
                if cleaned_blob.size == 0:
                    if debug_mode and debug_info:
                        # Even if the blob is empty, save a debug collage to see why
                        det_resized = debug_info.get('det_resized')
                        if det_resized is not None:
                            heatmap_crop = det_resized[y1:y2, x1:x2]
                            collage = create_debug_collage(original_uncropped_blob, blob, cleaned_blob, 
                                                           component_viz_img, heatmap_crop, config)
                            cv2.imwrite(os.path.join(debug_info['DEBUG_DIR'], f"line_{l:03d}_box_{box_counter:04d}_EMPTY.jpg"), collage)
                    continue

                c_top, _, c_left, _ = crop_coords
                final_box_x = x1 + c_left
                final_box_y = y1 + c_top
                final_box_w = cleaned_blob.shape[1]
                final_box_h = cleaned_blob.shape[0]

                if final_box_w > 0 and final_box_h > 0:
                    cleaned_blobs_for_line.append(cleaned_blob)
                    final_coords_for_line.append([final_box_x, final_box_y, final_box_w, final_box_h])

                if debug_mode and debug_info:
                    det_resized = debug_info.get('det_resized')
                    if det_resized is not None:
                        heatmap_crop = det_resized[y1:y2, x1:x2]
                        if heatmap_crop.size > 0:
                            collage = create_debug_collage(original_uncropped_blob, blob, cleaned_blob,
                                                           component_viz_img, heatmap_crop, config)
                            cv2.imwrite(os.path.join(debug_info['DEBUG_DIR'], f"line_{l:03d}_box_{box_counter:04d}.jpg"), collage)

            except Exception as e:
                print(f"Warning: Skipped box during analysis in line {l}: {e}")
        
        line_bounding_boxes_data[l] = final_coords_for_line
        
        
    return line_bounding_boxes_data


def segmentLinesFromPointClusters(BASE_PATH, page, upscale_heatmap=True, debug_mode=False, BINARIZE_THRESHOLD=0.30, BBOX_PAD_V=0.7, BBOX_PAD_H=0.5, CC_SIZE_THRESHOLD_RATIO=0.4, GNN_PRED_PATH=''):
    IMAGE_FILEPATH = os.path.join(BASE_PATH, "images_resized", f"{page}.jpg")
    HEATMAP_FILEPATH = os.path.join(BASE_PATH, "heatmaps", f"{page}.jpg")
    POINTS_FILEPATH = os.path.join(GNN_PRED_PATH, "gnn-format", f"{page}_inputs_unnormalized.txt")
    LABELS_FILEPATH = os.path.join(GNN_PRED_PATH, "gnn-format", f"{page}_labels_textline.txt")
    LINES_DIR = os.path.join(GNN_PRED_PATH, "image-format", page)
    DEBUG_DIR = os.path.join(GNN_PRED_PATH, "debug", page)
    POLY_VISUALIZATIONS_DIR = os.path.join(DEBUG_DIR, "poly_visualizations")

    # The polygon directory is no longer needed as polygons are now saved in the XML
    if os.path.exists(LINES_DIR): shutil.rmtree(LINES_DIR)
    os.makedirs(LINES_DIR)

    image = loadImage(IMAGE_FILEPATH)
    det = loadImage(HEATMAP_FILEPATH)
    if det.ndim == 3: det = det[:, :, 0]

    h_img, w_img = image.shape[:2]; h_heat, w_heat = det.shape[:2]
    features, labels = load_node_features_and_labels(POINTS_FILEPATH, LABELS_FILEPATH)

    if upscale_heatmap:
        det_resized = cv2.resize(det, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        processing_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if features.size > 0:
            features[:, :2] = (features[:, :2].astype(np.float64) * [w_img / w_heat, h_img / h_heat]).astype(int)
    else:
        image = cv2.resize(image, (w_heat, h_heat))
        h_img, w_img = image.shape[:2]
        det_resized = det
        processing_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    CONFIG = {
        'BINARIZE_THRESHOLD': BINARIZE_THRESHOLD,
        'CC_SIZE_THRESHOLD_RATIO': CC_SIZE_THRESHOLD_RATIO,
        'PAGE_MEDIAN_COLOR': int(np.median(processing_image))
        # Padding ratios for initial cleaning crop
        ,'BBOX_PAD_V': BBOX_PAD_V # 70% vertical padding for horizontal lines
        ,'BBOX_PAD_H': BBOX_PAD_H # 50% horizontal
    }

    bounding_boxes = gen_bounding_boxes(det_resized, CONFIG['BINARIZE_THRESHOLD'])
    labeled_bboxes = assign_labels_and_plot(bounding_boxes, features, labels, image.copy(),
                                            output_path=os.path.join(BASE_PATH, "heatmaps", f"{page}_all_labelled_boxes.jpg"))

    unique_labels = sorted(list(set(b[4] for b in labeled_bboxes)))

    # debug_info = None
    debug_info = {"DEBUG_DIR": DEBUG_DIR, "det_resized": det_resized, "CONFIG": CONFIG}
    if upscale_heatmap and debug_mode:
        print("Debug mode is ON.")
        if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(DEBUG_DIR)
        os.makedirs(POLY_VISUALIZATIONS_DIR)
        # debug_info = {"DEBUG_DIR": DEBUG_DIR, "det_resized": det_resized, "CONFIG": CONFIG}

    line_bounding_boxes_data = get_bboxes_for_lines(processing_image, unique_labels, labeled_bboxes,
                                                    debug_mode=(upscale_heatmap and debug_mode), debug_info=debug_info)

    poly_viz_page_img = image.copy()
    colors = [plt.cm.get_cmap('hsv', len(unique_labels) + 1)(i) for i in range(len(unique_labels))]
    label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}
    line_polygons_data = {}  # To store polygon data for returning

    for line_label, cleaned_boxes in line_bounding_boxes_data.items():
        if not cleaned_boxes: continue

        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for (x, y, w, h) in cleaned_boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            # save the mask for debugging if needed
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 1:
            avg_line_height = np.mean([box[3] for box in cleaned_boxes])
            box_groups = [[] for _ in contours]
            for box in cleaned_boxes:
                center_x, center_y = box[0] + box[2] // 2, box[1] + box[3] // 2
                for i, contour in enumerate(contours):
                    point_to_test = (float(center_x), float(center_y))
                    if cv2.pointPolygonTest(contour, point_to_test, False) >= 0:
                        box_groups[i].append(box)
                        break
            
            box_groups = [group for group in box_groups if group]
            if len(box_groups) <= 1:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                connected_groups = [box_groups[0]]
                unconnected_groups = box_groups[1:]

                while unconnected_groups:
                    best_dist = float('inf')
                    closest_box_pair = (None, None)
                    group_to_add_index = -1

                    for i, u_group in enumerate(unconnected_groups):
                        for c_group in connected_groups:
                            for u_box in u_group:
                                for c_box in c_group:
                                    dist = np.linalg.norm(
                                        np.array([u_box[0] + u_box[2]/2, u_box[1] + u_box[3]/2]) -
                                        np.array([c_box[0] + c_box[2]/2, c_box[1] + c_box[3]/2])
                                    )
                                    if dist < best_dist:
                                        best_dist = dist
                                        closest_box_pair = (u_box, c_box)
                                        group_to_add_index = i
                    
                    if closest_box_pair[0] is not None:
                        box1, box2 = closest_box_pair
                        left_box, right_box = (box1, box2) if box1[0] < box2[0] else (box2, box1)
                        
                        y_center1 = left_box[1] + left_box[3] / 2
                        y_center2 = right_box[1] + right_box[3] / 2
                        bridge_y_center = (y_center1 + y_center2) / 2
                        
                        bridge_y1 = int(bridge_y_center - avg_line_height / 2)
                        bridge_y2 = int(bridge_y_center + avg_line_height / 2)
                        bridge_x1 = left_box[0] + left_box[2]
                        bridge_x2 = right_box[0]
                        
                        cv2.rectangle(mask, (bridge_x1, bridge_y1), (bridge_x2, bridge_y2), 255, -1)

                    connected_groups.append(unconnected_groups.pop(group_to_add_index))

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            polygon = max(contours, key=cv2.contourArea)
            line_filename_base = f"line{line_label+1:03d}"
            
            # Instead of saving to JSON, store the polygon points in the dictionary
            polygon_points_xy = [point[0].tolist() for point in polygon]
            line_polygons_data[line_label] = polygon_points_xy

            if upscale_heatmap and debug_mode:
                color_idx = label_to_color_idx.get(line_label, 0)
                color = tuple(c * 255 for c in colors[color_idx][:3])
                cv2.drawContours(poly_viz_page_img, [polygon], -1, color, 2)
            
            # Save the cropped polygon area as the line image (this functionality remains)
            x, y, w, h = cv2.boundingRect(polygon)
            cropped_line_image = processing_image[y:y+h, x:x+w]
            new_img = np.ones(cropped_line_image.shape, dtype=np.uint8) * CONFIG['PAGE_MEDIAN_COLOR']
            mask_polygon = np.zeros(cropped_line_image.shape[:2], dtype=np.uint8)
            polygon_shifted = polygon - [x, y]
            cv2.drawContours(mask_polygon, [polygon_shifted], -1, 255, -1)
            new_img[mask_polygon == 255] = cropped_line_image[mask_polygon == 255]
            cv2.imwrite(os.path.join(LINES_DIR, f"{line_filename_base}.jpg"), new_img)

    if upscale_heatmap and debug_mode:
        viz_path = os.path.join(POLY_VISUALIZATIONS_DIR, f"{page}_all_polygons.jpg")
        cv2.imwrite(viz_path, poly_viz_page_img)
        print(f"Polygon visualization saved to {viz_path}")

    print(f"Successfully generated {len(line_polygons_data)} line images and polygon data.")
    return line_polygons_data
---
