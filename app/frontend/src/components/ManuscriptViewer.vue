<template>
  <div class="manuscript-viewer">
    
    <!-- TOP RAIL: Navigation & Global Actions -->
<!-- TOP RAIL: Navigation & Global Actions -->
    <div class="top-bar fixed-ui-compensated" :style="fixedUiCompensationStyle">
      
      <!-- 1) TOP BAR LEFT -->
      <div class="top-bar-left top-bar-section">
        <div class="page-context" style="flex-direction: column; align-items: flex-start; gap: 4px;">
          <div class="page-meta">
            <span class="page-eyebrow">Manuscript</span>
            <span class="page-title">{{ manuscriptNameForDisplay }}</span>
          </div>
          <button class="nav-btn secondary" style="padding: 0; font-size: 0.85rem;" @click="$emit('back')">&larr; Back</button>
        </div>

        <div class="page-controls" style="flex-direction: column; align-items: flex-end; gap: 6px; margin-left: auto;">
          <label class="page-picker" style="padding: 4px 8px;">
            <span class="page-picker-label">Go to</span>
            <select class="page-select" :value="currentPageForDisplay" @change="handlePageSelect" style="min-width: 80px; padding: 2px 4px;">
               <option v-for="pg in localPageList" :key="pg" :value="pg">Page {{ pg }}</option>
            </select>
          </label>
          <div class="page-stepper" style="gap: 4px;">
            <span class="control-shell" :class="{ 'is-disabled': previousPageDisabled }" :title="previousPageButtonTitle">
              <button class="nav-btn" style="min-height: 24px; padding: 2px 6px; font-size: 0.8rem;" @click="previousPage" :disabled="previousPageDisabled">
                Previous Page
              </button>
            </span>
            <span class="control-shell" :class="{ 'is-disabled': nextPageDisabled }" :title="nextPageButtonTitle">
              <button class="nav-btn" style="min-height: 24px; padding: 2px 6px; font-size: 0.8rem;" @click="nextPage" :disabled="nextPageDisabled">
                Next Page
              </button>
            </span>
          </div>
        </div>
      </div>

      <!-- 2) TOP BAR CENTER -->
      <div class="top-bar-center workflow-panel" style="justify-content: center; align-items: center; padding: 8px;">
        <div class="workflow-controls" style="justify-content: center; width: 100%; gap: 12px;">
          
          <!-- Active Learning Toggle (Always shown) -->
          <div class="workflow-toggle-group">
            <label class="toggle-switch">
              <input type="checkbox" v-model="activeLearningEnabled">
              <span class="slider"></span>
            </label>
            <div class="workflow-toggle-copy">
              <span class="workflow-toggle-label">Improve Future Reading</span>
              <span class="workflow-toggle-subcopy">{{ activeLearningStatus }}</span>
            </div>
          </div>

          <!-- Recognition Mode Specific Controls -->
          <div
            class="workflow-recognition-controls"
            :class="{ 'is-inactive': !recognitionModeActive }"
            :aria-hidden="!recognitionModeActive"
          >
            <!-- Moved OCR Engine Dropdown -->
            <div class="recognition-engine-panel" style="padding: 4px 10px; margin: 0; background: rgba(0,0,0,0.2); border-radius: 8px; gap: 8px;">
              <span class="recognition-engine-label" style="font-size: 0.65rem;">Read Text With</span>
              <select
                v-model="recognitionEngine"
                class="workflow-select"
                :disabled="isProcessingSave || recognitionInFlight"
                :title="recognitionEngineSelectTitle"
                style="padding: 2px 6px; font-size: 0.75rem;"
              >
                <option value="local">Built-in Reader</option>
                <option value="gemini">Gemini</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <!-- 3) TOP BAR RIGHT -->
      <div class="top-bar-right top-bar-section" style="justify-content: center; gap: 6px; padding: 8px 12px;">
        <div class="action-summary" style="align-items: center; margin-bottom: 2px; width: 100%;">
          <span class="action-eyebrow" style="font-size: 0.75rem; font-weight: bold; color: #8cb8a7;">
            {{ layoutModeActive ? 'Page Layout' : 'Text Review' }}
          </span>
          <span class="action-title" style="font-size: 0.8rem; color: #ccc; text-align: center;">
            {{ layoutModeActive ? 'Mark the lines and structure of the page before reviewing the text.' : 'Review the text line by line and correct any reading mistakes.' }}
          </span>
        </div>

        <div class="action-group" style="flex-wrap: nowrap; justify-content: center; gap: 8px; width: 100%;">
          <span class="control-shell action-slot" :class="{ 'is-disabled': primaryTopBarActionDisabled, 'is-ghost': primaryTopBarActionHidden }" :title="primaryTopBarActionHidden ? '' : primaryTopBarActionTitle">
            <button
              class="action-btn"
              :class="{ recommended: topBarActionState.recommendedAction === 'recognize' && !recognitionModeRequiresLayoutReturn }"
              @click="handlePrimaryTopBarAction"
              :disabled="primaryTopBarActionDisabled"
              :tabindex="primaryTopBarActionHidden ? -1 : 0"
              :aria-hidden="primaryTopBarActionHidden"
              style="padding: 6px 12px; min-height: 32px; font-size: 0.85rem;"
            >
              {{ primaryTopBarActionLabel }}
            </button>
          </span>
          <span class="control-shell action-slot" :class="{ 'is-disabled': commitActionDisabled, 'is-ghost': recognitionModeRequiresLayoutReturn }" :title="recognitionModeRequiresLayoutReturn ? '' : commitButtonTitle">
            <button
              class="action-btn"
              :class="{ recommended: topBarActionState.recommendedAction === 'commit' }"
              @click="saveCurrentPage"
              :disabled="commitActionDisabled"
              style="padding: 6px 12px; min-height: 32px; font-size: 0.85rem;"
            >
              Save Page
            </button>
          </span>
          <span class="control-shell action-slot" :class="{ 'is-disabled': commitAndNextDisabled, 'is-ghost': recognitionModeRequiresLayoutReturn }" :title="recognitionModeRequiresLayoutReturn ? '' : commitAndNextButtonTitle">
            <button class="action-btn forward-action" @click="saveAndGoNext" :disabled="commitAndNextDisabled" style="padding: 6px 12px; min-height: 32px; font-size: 0.85rem;">
              Save & Next Page
            </button>
          </span>
          <!-- EXPORT IMAGE BUTTON COMMENTED OUT
          <span class="control-shell" :class="{ 'is-disabled': exportImageDisabled }" :title="exportImageButtonTitle">
            <button class="action-btn secondary-action" @click="saveOverlay" :disabled="exportImageDisabled" style="padding: 6px 12px; min-height: 32px; font-size: 0.85rem;">
              Export Image
            </button>
          </span>
          -->
          <span class="control-shell" :class="{ 'is-disabled': downloadResultsDisabled }" :title="downloadResultsButtonTitle">
            <button class="action-btn secondary-action" @click="downloadResults" :disabled="downloadResultsDisabled" style="padding: 6px 12px; min-height: 32px; font-size: 0.85rem;">
              Download Manuscript
            </button>
          </span>
        </div>
      </div>
    </div>

    <!-- MAIN CONTENT: Visualization Area -->
    <div class="visualization-container" ref="container">
      
      <!-- 1. Unified Overlay for Saving OR Mode Switching (Foreground) -->
      <div v-if="isProcessingSave || recognitionInFlight" class="processing-save-notice">
        {{ recognitionInFlight ? recognitionBusyLabel : 'Saving your changes. Please wait.' }}
      </div>

      <div v-if="error" class="error-message">
        {{ error }}
      </div>

      <!-- 2. Loading Indicator (Only for initial page load) -->
      <div v-if="loading" class="loading">Loading this page...</div>

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
              :style="{ cursor: pointer}"
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
                :stroke-width="getEdgeStrokeWidth(edge)"
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
                :stroke-width="tempEdgeStrokeWidth"
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

            <div
              v-if="recognitionModeActive && !effectivePageWorkflow.can_edit_text && !isProcessingSave && !recognitionInFlight"
              class="recognition-guard-card"
            >
              <span class="recognition-guard-badge">{{ effectivePageWorkflow.label }}</span>
              <h3>{{ effectivePageWorkflow.hint }}</h3>
              <p v-if="effectivePageWorkflow.prediction.source_label">
                Latest visible text came from {{ effectivePageWorkflow.prediction.source_label }}.
              </p>
              <p v-else>
                Page Layout sets up the lines on the page. Text Review needs that page structure before it can show the text.
              </p>
              <button
                v-if="recognitionModeRequiresLayoutReturn"
                class="action-btn primary"
                @click="goToLayoutMode"
              >
                Open Page Layout
              </button>
              <!-- <button
                class="action-btn primary"
                @click="runRecognitionAction"
                :disabled="loading || isProcessingSave || recognitionInFlight || !canRecognizePage"
              >
                {{ recognizeButtonLabel }}
              </button> -->
            </div>

            <!-- Recognition Input Overlay Layer -->
            <div
                v-if="recognitionModeActive && effectivePageWorkflow.can_edit_text && focusedLineId && pagePolygons[focusedLineId]"
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
    <div
      class="bottom-panel fixed-ui-compensated"
      :class="{ 'is-collapsed': isPanelCollapsed }"
      :style="fixedUiCompensationStyle"
    >
      
      <!-- Mode Tabs (Always Visible) -->
      <div class="mode-tabs">
          <!-- REMOVED: View Mode Button -->
          <button 
           class="mode-tab" 
           :class="{ active: layoutModeActive }"
           @click="setMode('layout')"
           :disabled="isProcessingSave || !graphIsLoaded">
           Page Layout (W)
         </button>
         <button 
           class="mode-tab" 
           :class="{ active: recognitionModeActive }"
           @click="requestSwitchToRecognition()" 
           :disabled="isProcessingSave">
           Text Review (T)
         </button>

         <div
           v-show="recognitionModeActive"
           class="mode-tools-shell"
           :aria-hidden="!recognitionModeActive"
         >
           <div class="mode-tools-section">
             <div class="mode-tools-label">Typing Tools</div>
             <div class="mode-tools-controls">
               <div class="workflow-toggle-group compact bottom-tools-toggle">
                 <label class="toggle-switch">
                   <input type="checkbox" v-model="devanagariModeEnabled">
                   <span class="slider"></span>
                 </label>
                 <div class="workflow-toggle-copy">
                   <span class="workflow-toggle-label">Keyboard</span>
                   <span class="workflow-toggle-subcopy">Devanagari</span>
                 </div>
               </div>

               <div
                 v-show="devanagariModeEnabled"
                 class="bottom-palette-slot"
                 :aria-hidden="!devanagariModeEnabled"
               >
                 <CharacterPalette />
               </div>
             </div>
           </div>
         </div>

         <div class="tab-spacer"></div>

         <button class="panel-toggle-btn" @click="isPanelCollapsed = !isPanelCollapsed" :title="isPanelCollapsed ? 'Show help' : 'Hide help'">
            <span v-if="isPanelCollapsed">Show Help</span>
            <span v-else>Hide Help</span>
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
                <h4>Points</h4>
                <p><span class="key-badge">L-Click</span> Add point</p>
                <p><span class="key-badge">R-Click</span> Delete point</p>
              </div>
            </div>

            <!-- Edges Card -->
            <div class="help-card horizontal-layout">
              <div class="media-container-square">
                <video :src="edgeWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
              </div>
              <div class="card-text">
                <h4>Links</h4>
                <p>Hold <span class="key-badge">A</span> and hover to connect points</p>
                <p>Hold <span class="key-badge">D</span> and hover to remove a link</p>
              </div>
            </div>

            <!-- Regions Card -->
            <div class="help-card horizontal-layout">
              <div class="media-container-square">
                <video :src="regionWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
              </div>
              <div class="card-text">
                <h4>Regions</h4>
                <p>Hold <span class="key-badge">E</span> and hover to mark a region</p>
                <p>Release and repeat to start a new region</p>
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
           <!-- <div class="media-container">
             <div class="webm-placeholder" style="flex-direction:column; gap:10px;">
              <span>Text Review</span>
              <span v-if="devanagariModeEnabled" style="color:#4CAF50; font-size:0.8rem;">Devanagari keyboard on</span>
            </div>
           </div> -->
           <div class="instructions-container">
             <h3>Text Review</h3>
             <p>{{ effectivePageWorkflow.hint }}</p>
             <ul>
               <!-- <li><strong>Read Text:</strong> Press <code>R</code> to read the page or read it again.</li> -->
               <!-- <li><strong>Save:</strong> Press <code>S</code> to save, or <code>Shift+S</code>/<code>Ctrl+Enter</code> to save and open the next page.</li> -->
               <li><strong>Navigate:</strong> Press <code>Tab</code> for the next line, <code>Shift+Tab</code> for the previous line.</li>
               <li v-if="devanagariModeEnabled"><strong>Keys:</strong> Type phonetically (for example, <code>k</code> gives <code>क</code>). Use <code>q</code> for halant.</li>
             </ul>
             <div class="recognition-status-grid">
               <div class="recognition-status-card">
                 <span class="recognition-status-label">Page Status</span>
                 <strong>{{ effectivePageWorkflow.label }}</strong>
               </div>
               <div class="recognition-status-card">
                 <span class="recognition-status-label">Current Text Source</span>
                 <strong>{{ effectivePageWorkflow.prediction.source_label || 'Not read yet' }}</strong>
               </div>
               <!-- <div class="recognition-status-card">
                 <span class="recognition-status-label">Read Again Using</span>
                 <strong>{{ nextRecognitionSourceLabel }}</strong>
               </div> -->
             </div>

           </div>
        </div>
        
        <!-- Logs -->
        <div v-if="modifications.length > 0" class="log-sidebar">
            <div class="log-header">
              <span>Layout Changes: {{ modifications.length }}</span>
              <button class="text-btn" @click="resetModifications" :disabled="loading">Clear All</button>
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
const PAGE_ENTRY_LAYOUT = 'layout'
const PAGE_ENTRY_RECOGNITION_IF_COMMITTED_TEXT = 'recognition_if_committed_text'

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
const autoRecogEnabled = ref(localStorage.getItem('auto_prepare_next_page') === 'true')
const activeLearningEnabled = ref(localStorage.getItem('active_learning_enabled') !== 'false')
const activeLearningStatus = ref('Not updating right now')
const recognitionEngine = ref(localStorage.getItem('recognition_engine') || 'local') // NEW
const devanagariModeEnabled = ref(true) 
const recognitionInFlight = ref(false)
const recognitionDraftDirty = ref(false)
const suppressTextDirtyTracking = ref(false)
const pendingPageEntryPreference = ref(null)
const activeLearningMeta = reactive({
  label: 'Not updating right now',
  active_checkpoint_id: 'base',
  active_checkpoint_path: null,
  pending_jobs: [],
  needs_rebase: false,
})
const pageWorkflow = reactive({
  state: 'missing_page_xml',
  label: 'Set up the page first',
  hint: 'Open Page Layout first, check the lines on the page, and then move to Text Review.',
  needs_recognition: true,
  can_edit_text: false,
  can_resume_recognition: false,
  has_text: false,
  latest_revision_save_intent: null,
  correction_summary: { changed_line_count: 0, total_edit_distance: 0, normalized_edit_distance: 0 },
  prediction: {
    available: false,
    engine: null,
    checkpoint_id: null,
    checkpoint_path: null,
    recorded_at: null,
    source_label: null,
    layout_fingerprint: null,
    matches_current_layout: null,
    layout_match_known: false,
  },
})

// NEW: Persist keys/settings to local storage
watch(autoRecogEnabled, (val) => localStorage.setItem('auto_prepare_next_page', String(val)))
watch(recognitionEngine, (val) => localStorage.setItem('recognition_engine', val))
watch(activeLearningEnabled, (val) => localStorage.setItem('active_learning_enabled', String(val)))
watch(geminiKey, (val) => localStorage.setItem('gemini_key', val))
const localTextConfidence = reactive({}) 
const autoSaveInterval = ref(null) // NEW

const scaleFactor = 0.7
const DEFAULT_MEDIAN_NEIGHBOR_DISTANCE_RAW = 20
const MIN_NODE_RADIUS_PX = 2.2
const MAX_NODE_RADIUS_PX = 7
const MIN_EDGE_STROKE_PX = 1.1
const MAX_EDGE_STROKE_PX = 4

const pageMedianNeighborDistanceRaw = ref(DEFAULT_MEDIAN_NEIGHBOR_DISTANCE_RAW)
const baseNodeRadiusPx = ref(7)
const baseEdgeStrokePx = ref(4)

const getOuterToInnerRatio = () => {
  const innerWidth = window.innerWidth || 0
  const outerWidth = window.outerWidth || 0
  if (innerWidth <= 0 || outerWidth <= 0) return 1
  return outerWidth / innerWidth
}

const initialDevicePixelRatio = window.devicePixelRatio || 1
const initialOuterToInnerRatio = getOuterToInnerRatio()
const initialViewportWidth = window.visualViewport?.width || window.innerWidth || 1
const browserZoomLevel = ref(1)
let zoomUpdateRafId = null
let zoomPollIntervalId = null
let zoomShortcutTimeoutId = null

const manuscriptNameForDisplay = computed(() => localManuscriptName.value)
const currentPageForDisplay = computed(() => localCurrentPage.value)
const isFirstPage = computed(() => localPageList.value.indexOf(localCurrentPage.value) === 0)
const isLastPage = computed(() => localPageList.value.indexOf(localCurrentPage.value) === localPageList.value.length - 1)

const scaledWidth = computed(() => Math.floor(dimensions.value[0] * scaleFactor))
const scaledHeight = computed(() => Math.floor(dimensions.value[1] * scaleFactor))
const scaleX = (x) => x * scaleFactor
const scaleY = (y) => y * scaleFactor
const graphIsLoaded = computed(() => workingGraph.nodes && workingGraph.nodes.length > 0)
const hasUnsavedLayoutChanges = computed(() => modifications.value.length > 0)

const describeLocalCheckpoint = (checkpointId) => {
  if (!checkpointId || checkpointId === 'base') {
    return {
      modelLabel: 'Built-in Reader',
      fineTunedPagesLabel: 'not yet trained on this manuscript',
      fullLabel: 'the built-in reader',
      detailLabel: 'Uses the built-in reader for this manuscript.',
    }
  }

  return {
    modelLabel: 'Saved Reader For This Manuscript',
    fineTunedPagesLabel: null,
    fullLabel: 'the saved reader for this manuscript',
    detailLabel: 'Uses the saved reader learned from this manuscript.',
  }
}

const localCheckpointDescriptor = computed(() => describeLocalCheckpoint(activeLearningMeta.active_checkpoint_id))

const nextRecognitionSourceLabel = computed(() => {
  if (recognitionEngine.value === 'gemini') return 'Gemini'
  return localCheckpointDescriptor.value.modelLabel
})

const recognitionEngineLabel = computed(() => nextRecognitionSourceLabel.value)
const canRecognizePage = computed(() => recognitionEngine.value !== 'gemini' || Boolean(geminiKey.value))
const recognitionBusyLabel = computed(() => {
  if (recognitionEngine.value === 'gemini') return 'Reading the page with Gemini...'
  return `Reading the page with ${localCheckpointDescriptor.value.fullLabel}...`
})
const pageWorkflowRequiresLayoutMode = (workflow) =>
  Boolean(workflow?.state === 'missing_page_xml' && !workflow?.can_edit_text)
const effectivePageWorkflow = computed(() => {
  const prediction = { ...pageWorkflow.prediction }
  const correctionSummary = { ...pageWorkflow.correction_summary }
  if (recognitionInFlight.value) {
    return {
      ...pageWorkflow,
      prediction,
      correction_summary: correctionSummary,
      state: 'refreshing_ocr',
      label: 'Reading text',
      hint: `Reading the page with ${recognitionEngineLabel.value}.`,
      needs_recognition: false,
      can_edit_text: false,
    }
  }
  if (hasUnsavedLayoutChanges.value) {
    return {
      ...pageWorkflow,
      prediction,
      correction_summary: correctionSummary,
      state: 'layout_dirty',
      label: 'Page layout changed',
      hint: 'Save the page layout, then read the text again before correcting it.',
      needs_recognition: true,
      can_edit_text: false,
    }
  }
  return {
    ...pageWorkflow,
    prediction,
    correction_summary: correctionSummary,
  }
})
const recognitionModeRequiresLayoutReturn = computed(() =>
  recognitionModeActive.value && pageWorkflowRequiresLayoutMode(effectivePageWorkflow.value)
)

const workflowStateClass = computed(() => `state-${effectivePageWorkflow.value.state}`)
const workflowPanelEyebrow = computed(() => {
  if (layoutModeActive.value) return 'Step 1'
  if (recognitionModeActive.value) return 'Step 2'
  return 'Page Progress'
})
const workflowPanelHint = computed(() => {
  if (layoutModeActive.value) {
    if (hasUnsavedLayoutChanges.value) {
      return 'Save your page layout changes here before opening Text Review.'
    }
    return 'Check the lines and page structure here first. Then open Text Review to work on the text.'
  }
  if (recognitionModeRequiresLayoutReturn.value) {
    return effectivePageWorkflow.value.hint
  }
  if (recognitionModeActive.value && effectivePageWorkflow.value.needs_recognition) {
    return 'Choose a text-reading method here, then read the page.'
  }
  return effectivePageWorkflow.value.hint
})
const recognizeButtonLabel = computed(() => {
  if (hasUnsavedLayoutChanges.value) return 'Save Layout & Read Text'
  if (layoutModeActive.value) {
    if (effectivePageWorkflow.value.needs_recognition) return 'Read Text'
    return 'Open Text Review'
  }
  if (effectivePageWorkflow.value.needs_recognition) return 'Read Text'
  return 'Replace With New Reading'
})
const recognitionEngineDescription = computed(() => {
  if (recognitionEngine.value === 'gemini') {
    return canRecognizePage.value
      ? 'Uses Gemini to read the text on this page.'
      : 'Gemini needs an API key before it can read this page.'
  }
  return localCheckpointDescriptor.value.detailLabel
})
const topBarActionState = computed(() => {
  if (recognitionInFlight.value) {
    return {
      eyebrow: 'Working',
      title: 'Reading text on this page',
      hint: recognitionBusyLabel.value,
      recommendedAction: 'recognize',
    }
  }
  if (isProcessingSave.value) {
    return {
      eyebrow: 'Working',
      title: 'Saving this page',
      hint: 'Please wait until the save is finished before moving on.',
      recommendedAction: 'commit',
    }
  }
  if (layoutModeActive.value) {
    if (hasUnsavedLayoutChanges.value) {
      return {
        eyebrow: 'Page Layout',
        title: 'Save page layout changes',
        hint: 'Save your line and region changes before opening Text Review.',
        recommendedAction: 'commit',
      }
    }
    if (recognitionDraftDirty.value) {
      return {
        eyebrow: 'Page Layout',
        title: 'Save text changes',
        hint: 'Save your text changes before making more page layout changes or reading the page again.',
        recommendedAction: 'commit',
      }
    }
    return {
      eyebrow: 'Page Layout',
      title: 'Check the page structure first',
      hint: 'Adjust the lines and regions here. When the page structure looks right, open Text Review.',
      recommendedAction: null,
    }
  }
  if (recognitionModeActive.value) {
    if (recognitionModeRequiresLayoutReturn.value) {
      return {
        eyebrow: 'Text Review',
        title: 'Return to Page Layout',
        hint: effectivePageWorkflow.value.hint,
        recommendedAction: null,
      }
    }
    if (effectivePageWorkflow.value.needs_recognition) {
      return {
        eyebrow: 'Text Review',
        title: 'Read the text for this page',
        hint: 'Choose a text-reading method here, then read the page before correcting the text.',
        recommendedAction: 'recognize',
      }
    }
    if (recognitionDraftDirty.value) {
      return {
        eyebrow: 'Text Review',
        title: 'Save your current corrections',
        hint: 'Save this page so your text corrections stay attached to the current page.',
        recommendedAction: 'commit',
      }
    }
    if (effectivePageWorkflow.value.can_edit_text) {
      return {
        eyebrow: 'Text Review',
        title: 'Review and correct the text',
        hint: 'Edit the text line by line here, then save when you are done with this page.',
        recommendedAction: null,
      }
    }
  }
  if (effectivePageWorkflow.value.can_edit_text) {
    return {
      eyebrow: 'Ready',
      title: 'Page is ready to review',
      hint: 'You can correct the text now and save when you finish this page.',
      recommendedAction: null,
    }
  }
  return {
    eyebrow: 'Ready',
    title: 'Page is ready',
    hint: 'Review this page, download the results, or move to another page.',
    recommendedAction: null,
  }
})

const getBusyDisabledReason = (label) => {
  if (loading.value) return `${label} is not available while the page is loading.`
  if (recognitionInFlight.value) return `${label} is not available while the page is being read.`
  if (isProcessingSave.value) return `${label} is not available while changes are being saved.`
  return ''
}

const previousPageDisabled = computed(() => loading.value || isProcessingSave.value || recognitionInFlight.value || isFirstPage.value)
const nextPageDisabled = computed(() => loading.value || isProcessingSave.value || recognitionInFlight.value || isLastPage.value)
const recognizeActionDisabled = computed(() => loading.value || isProcessingSave.value || recognitionInFlight.value || !canRecognizePage.value || recognitionModeRequiresLayoutReturn.value)
const commitActionDisabled = computed(() => loading.value || isProcessingSave.value || recognitionInFlight.value || recognitionModeRequiresLayoutReturn.value)
const commitAndNextDisabled = computed(() => loading.value || isProcessingSave.value || recognitionInFlight.value || recognitionModeRequiresLayoutReturn.value)
const exportImageDisabled = computed(() => loading.value || isProcessingSave.value || recognitionInFlight.value || recognitionModeActive.value)
const downloadResultsDisabled = computed(() => loading.value || isProcessingSave.value || recognitionInFlight.value)

const previousPageButtonTitle = computed(() => {
  const busyReason = getBusyDisabledReason('Previous page')
  if (busyReason) return busyReason
  if (isFirstPage.value) return 'You are already on the first page.'
  return 'Open the previous page. Shortcut: [ '
})

const nextPageButtonTitle = computed(() => {
  const busyReason = getBusyDisabledReason('Next page')
  if (busyReason) return busyReason
  if (isLastPage.value) return 'You are already on the last page.'
  return 'Open the next page. Shortcut: ] '
})

const recognizeButtonTitle = computed(() => {
  const busyReason = getBusyDisabledReason(recognizeButtonLabel.value)
  if (busyReason) return busyReason
  if (!canRecognizePage.value) {
    return recognitionEngine.value === 'gemini'
      ? 'Add a Gemini API key or switch to the built-in reader before reading the page.'
      : 'Text reading is not available right now.'
  }
  if (layoutModeActive.value && hasUnsavedLayoutChanges.value) {
    return 'Save the updated page layout, open Text Review, and read the page text (R).'
  }
  if (layoutModeActive.value) {
    if (effectivePageWorkflow.value.needs_recognition) {
      return 'Open Text Review and read the page text (R).'
    }
    return 'Open Text Review and reopen the saved text for this page (R). Use Replace With New Reading there if you want to overwrite it with a fresh reading.'
  }
  if (effectivePageWorkflow.value.needs_recognition) return 'Read the page text now (R).'
  return 'Replace the current text on this page with a fresh reading using the current method and layout (R). Existing corrections will be overwritten.'
})

const commitButtonTitle = computed(() => {
  const busyReason = getBusyDisabledReason('Save Page')
  if (busyReason) return busyReason
  if (hasUnsavedLayoutChanges.value) return 'Save the current page layout changes (S).'
  if (recognitionDraftDirty.value) return 'Save the current text corrections on this page (S).'
  return 'Save the current page (S).'
})

const commitAndNextButtonTitle = computed(() => {
  const busyReason = getBusyDisabledReason('Save & Next Page')
  if (busyReason) return busyReason
  if (isLastPage.value) return 'Save the current page. This manuscript is already on its last page.'
  return 'Save the current page and open the next one (Shift+S).'
})

const exportImageButtonTitle = computed(() => {
  const busyReason = getBusyDisabledReason('Export Image')
  if (busyReason) return busyReason
  if (recognitionModeActive.value) return 'Export Image is only available in Page Layout.'
  return 'Save an image of the current page with the layout overlay.'
})

const downloadResultsButtonTitle = computed(() => {
  const busyReason = getBusyDisabledReason('Download Manuscript')
  if (busyReason) return busyReason
  return 'Download the digitized manuscript in PAGE-XML format.'
})
const goToLayoutModeButtonTitle = computed(() => {
  const busyReason = getBusyDisabledReason('Open Page Layout')
  if (busyReason) return busyReason
  return 'Return to Page Layout for this page.'
})
const primaryTopBarActionHidden = computed(() =>
  (
    layoutModeActive.value &&
    !recognitionModeRequiresLayoutReturn.value &&
    !hasUnsavedLayoutChanges.value &&
    !effectivePageWorkflow.value.needs_recognition
  ) ||
  rereadWillOverwriteExistingText.value
)
const primaryTopBarActionLabel = computed(() =>
  recognitionModeRequiresLayoutReturn.value ? 'Open Page Layout' : recognizeButtonLabel.value
)
const primaryTopBarActionTitle = computed(() =>
  recognitionModeRequiresLayoutReturn.value ? goToLayoutModeButtonTitle.value : recognizeButtonTitle.value
)
const primaryTopBarActionDisabled = computed(() =>
  primaryTopBarActionHidden.value
    ? true
    : recognitionModeRequiresLayoutReturn.value
    ? loading.value || isProcessingSave.value || recognitionInFlight.value
    : recognizeActionDisabled.value
)
const rereadWillOverwriteExistingText = computed(() =>
  recognitionModeActive.value &&
  !effectivePageWorkflow.value.needs_recognition &&
  effectivePageWorkflow.value.can_edit_text &&
  effectivePageWorkflow.value.has_text
)
const recognitionEngineSelectTitle = computed(() => {
  if (recognitionInFlight.value) return 'The text-reading method cannot be changed while the page is being read.'
  if (isProcessingSave.value) return 'The text-reading method cannot be changed while changes are being saved.'
  return 'Choose the method used when you read the text on this page.'
})

const confirmReplaceWithNewReading = () => {
  if (!rereadWillOverwriteExistingText.value) return true
  const warning = recognitionDraftDirty.value
    ? 'This will replace the current text on this page with a new reading and overwrite your existing corrections, including unsaved changes. Continue?'
    : 'This will replace the current text on this page with a new reading and overwrite the existing corrections on this page. Continue?'
  return window.confirm(warning)
}

const replaceLocalRecognitionData = (textPayload = {}, confidencePayload = {}) => {
  suppressTextDirtyTracking.value = true
  Object.keys(localTextContent).forEach((key) => delete localTextContent[key])
  Object.keys(localTextConfidence).forEach((key) => delete localTextConfidence[key])
  Object.assign(localTextContent, textPayload || {})
  Object.assign(localTextConfidence, confidencePayload || {})
  recognitionDraftDirty.value = false
  nextTick(() => {
    suppressTextDirtyTracking.value = false
  })
}

const applyActiveLearningState = (payload = {}) => {
  activeLearningStatus.value = payload.label || 'Not updating right now'
  activeLearningMeta.label = activeLearningStatus.value
  activeLearningMeta.active_checkpoint_id = payload.active_checkpoint_id || 'base'
  activeLearningMeta.active_checkpoint_path = payload.active_checkpoint_path || null
  activeLearningMeta.pending_jobs = Array.isArray(payload.pending_jobs) ? payload.pending_jobs : []
  activeLearningMeta.needs_rebase = Boolean(payload.needs_rebase)
}

const applyPageWorkflow = (payload = {}) => {
  pageWorkflow.state = payload.state || 'missing_page_xml'
  pageWorkflow.label = payload.label || 'Set up the page first'
  pageWorkflow.hint = payload.hint || 'Open Page Layout first, check the lines on the page, and then move to Text Review.'
  pageWorkflow.needs_recognition = Boolean(payload.needs_recognition)
  pageWorkflow.can_edit_text = Boolean(payload.can_edit_text)
  pageWorkflow.can_resume_recognition = Boolean(payload.can_resume_recognition)
  pageWorkflow.has_text = Boolean(payload.has_text)
  pageWorkflow.latest_revision_save_intent = payload?.latest_revision_save_intent || null
  pageWorkflow.correction_summary = {
    changed_line_count: Number(payload?.correction_summary?.changed_line_count || 0),
    total_edit_distance: Number(payload?.correction_summary?.total_edit_distance || 0),
    normalized_edit_distance: Number(payload?.correction_summary?.normalized_edit_distance || 0),
  }
  pageWorkflow.prediction = {
    available: Boolean(payload?.prediction?.available),
    engine: payload?.prediction?.engine || null,
    checkpoint_id: payload?.prediction?.checkpoint_id || null,
    checkpoint_path: payload?.prediction?.checkpoint_path || null,
    recorded_at: payload?.prediction?.recorded_at || null,
    source_label: payload?.prediction?.source_label || null,
    layout_fingerprint: payload?.prediction?.layout_fingerprint || null,
    matches_current_layout: payload?.prediction?.matches_current_layout ?? null,
    layout_match_known: Boolean(payload?.prediction?.layout_match_known),
  }
}

const goToLayoutMode = () => {
  if (isProcessingSave.value || recognitionInFlight.value) return
  setMode('layout')
}

const handlePrimaryTopBarAction = () => {
  if (recognitionModeRequiresLayoutReturn.value) {
    goToLayoutMode()
    return
  }
  runRecognitionAction()
}

const shouldResumeRecognitionForWorkflow = (workflow = {}) =>
  Boolean(
    workflow?.can_resume_recognition
  )

watch(
  localTextContent,
  () => {
    if (!suppressTextDirtyTracking.value) {
      recognitionDraftDirty.value = true
    }
  },
  { deep: true }
)

const clamp = (value, min, max) => Math.min(max, Math.max(min, value))

const readBrowserZoomLevel = () => {
  const zoomCandidates = []

  const viewportScale = window.visualViewport?.scale
  if (typeof viewportScale === 'number' && Number.isFinite(viewportScale) && viewportScale > 0) {
    zoomCandidates.push(viewportScale)
  }

  const currentDpr = window.devicePixelRatio || initialDevicePixelRatio
  if (initialDevicePixelRatio > 0) {
    zoomCandidates.push(currentDpr / initialDevicePixelRatio)
  }

  const currentOuterToInnerRatio = getOuterToInnerRatio()
  if (initialOuterToInnerRatio > 0 && currentOuterToInnerRatio > 0) {
    zoomCandidates.push(currentOuterToInnerRatio / initialOuterToInnerRatio)
  }

  if (window.visualViewport?.width && initialViewportWidth > 0) {
    zoomCandidates.push(initialViewportWidth / window.visualViewport.width)
  }

  const validCandidates = zoomCandidates
    .filter((value) => Number.isFinite(value) && value > 0)
    .map((value) => clamp(value, 0.25, 4))

  if (validCandidates.length === 0) return 1

  // Median is stable against noisy signals from resize/frame metrics.
  validCandidates.sort((a, b) => a - b)
  return validCandidates[Math.floor(validCandidates.length / 2)]
}

const updateBrowserZoomLevel = () => {
  const measuredZoom = readBrowserZoomLevel()
  if (Math.abs(measuredZoom - browserZoomLevel.value) > 0.002) {
    browserZoomLevel.value = measuredZoom
  }
}

const scheduleBrowserZoomLevelUpdate = () => {
  if (zoomUpdateRafId !== null) return
  zoomUpdateRafId = window.requestAnimationFrame(() => {
    zoomUpdateRafId = null
    updateBrowserZoomLevel()
  })
}

const schedulePostZoomShortcutUpdate = () => {
  if (zoomShortcutTimeoutId !== null) {
    window.clearTimeout(zoomShortcutTimeoutId)
  }
  // Let browser apply zoom first, then measure.
  zoomShortcutTimeoutId = window.setTimeout(() => {
    zoomShortcutTimeoutId = null
    scheduleBrowserZoomLevelUpdate()
  }, 40)
}

const handleCtrlWheelZoom = (event) => {
  if (!event.ctrlKey) return
  schedulePostZoomShortcutUpdate()
}

const fixedUiCompensationStyle = computed(() => {
  const inverseZoom = 1 / browserZoomLevel.value
  const normalizedScale = clamp(inverseZoom, 0.25, 4)
  return {
    '--fixed-ui-zoom': normalizedScale.toFixed(4),
    '--fixed-ui-transform-scale': normalizedScale.toFixed(4),
  }
})

const tempEdgeStrokeWidth = computed(() =>
  clamp(baseEdgeStrokePx.value * 0.95, MIN_EDGE_STROKE_PX, MAX_EDGE_STROKE_PX)
)
const nodeHoverRadiusPx = computed(() => Math.max(baseNodeRadiusPx.value * 1.6, baseNodeRadiusPx.value + 2.5))
const edgeHoverThresholdPx = computed(() => Math.max(baseEdgeStrokePx.value * 1.8, 4))

const getMedian = (values) => {
  if (!Array.isArray(values) || values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  if (sorted.length % 2 === 1) return sorted[mid]
  return (sorted[mid - 1] + sorted[mid]) / 2
}

const computeMedianNeighborDistanceFromGraph = (nodes, edges) => {
  if (!Array.isArray(nodes) || nodes.length < 2) return null

  const minDistances = new Array(nodes.length).fill(Number.POSITIVE_INFINITY)
  const uniqueEdges = new Set()
  for (const edge of edges || []) {
    const source = Number(edge?.source)
    const target = Number(edge?.target)
    if (
      !Number.isInteger(source) || !Number.isInteger(target) ||
      source < 0 || target < 0 ||
      source >= nodes.length || target >= nodes.length ||
      source === target
    ) {
      continue
    }
    const key = source < target ? `${source}-${target}` : `${target}-${source}`
    if (uniqueEdges.has(key)) continue
    uniqueEdges.add(key)

    const n1 = nodes[source]
    const n2 = nodes[target]
    const distance = Math.hypot((n1?.x || 0) - (n2?.x || 0), (n1?.y || 0) - (n2?.y || 0))
    if (!Number.isFinite(distance) || distance <= 0) continue
    if (distance < minDistances[source]) minDistances[source] = distance
    if (distance < minDistances[target]) minDistances[target] = distance
  }

  const validDistances = minDistances.filter((d) => Number.isFinite(d) && d > 0)
  if (validDistances.length < Math.min(8, Math.max(3, Math.floor(nodes.length * 0.2)))) return null
  return getMedian(validDistances)
}

const computeMedianNearestDistanceFallback = (nodes) => {
  if (!Array.isArray(nodes) || nodes.length < 2) return null

  const SAMPLE_LIMIT = 1200
  const step = Math.max(1, Math.ceil(nodes.length / SAMPLE_LIMIT))
  const sampledIndices = []
  for (let i = 0; i < nodes.length; i += step) {
    sampledIndices.push(i)
  }

  const nearestDistances = []
  for (const sourceIndex of sampledIndices) {
    const sourceNode = nodes[sourceIndex]
    if (!sourceNode) continue
    let nearest = Number.POSITIVE_INFINITY
    for (let j = 0; j < nodes.length; j++) {
      if (j === sourceIndex) continue
      const targetNode = nodes[j]
      if (!targetNode) continue
      const distance = Math.hypot((sourceNode.x || 0) - (targetNode.x || 0), (sourceNode.y || 0) - (targetNode.y || 0))
      if (distance > 0 && distance < nearest) nearest = distance
    }
    if (Number.isFinite(nearest) && nearest > 0) nearestDistances.push(nearest)
  }

  return getMedian(nearestDistances)
}

const updatePageDynamicSizing = (nodes, edges) => {
  const safeNodes = Array.isArray(nodes) ? nodes : []
  const safeEdges = Array.isArray(edges) ? edges : []

  let medianDistance = computeMedianNeighborDistanceFromGraph(safeNodes, safeEdges)
  if (!medianDistance) medianDistance = computeMedianNearestDistanceFallback(safeNodes)
  if (!medianDistance || !Number.isFinite(medianDistance) || medianDistance <= 0) {
    medianDistance = DEFAULT_MEDIAN_NEIGHBOR_DISTANCE_RAW
  }

  pageMedianNeighborDistanceRaw.value = medianDistance
  const medianDistanceScaled = medianDistance * scaleFactor
  baseNodeRadiusPx.value = clamp(medianDistanceScaled * 0.28, MIN_NODE_RADIUS_PX, MAX_NODE_RADIUS_PX)
  baseEdgeStrokePx.value = clamp(baseNodeRadiusPx.value * 0.58, MIN_EDGE_STROKE_PX, MAX_EDGE_STROKE_PX)
}


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
    if (!effectivePageWorkflow.value.can_edit_text) return;
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
    if (!effectivePageWorkflow.value.can_edit_text) return;
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
    if (!workingGraph.nodes || workingGraph.nodes.length === 0) return pageMedianNeighborDistanceRaw.value;
    const sum = workingGraph.nodes.reduce((acc, n) => acc + (n.s || pageMedianNeighborDistanceRaw.value), 0);
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

const fetchPageData = async (manuscript, page, isRefresh = false, autoPrepareRecognition = false) => {
  if (!manuscript || !page) return;
  
  if (!isRefresh) {
      loading.value = true;
      imageData.value = ''; 
  }

  error.value = null
  modifications.value = []
  
  Object.keys(textlineLabels).forEach(k => delete textlineLabels[k])
  replaceLocalRecognitionData({}, {})
  pagePolygons.value = {}
  sortedLineIds.value = []
  let shouldAutoPrepareCurrentPage = false
  let pageData = null

  try {
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscript}/${page}`
    )
    if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch page data')
    const data = await response.json()
    pageData = data

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
    replaceLocalRecognitionData(data.textContent || {}, data.textConfidences || {})
    if (data.activeLearning) {
      applyActiveLearningState(data.activeLearning)
    }
    if (data.pageWorkflow) {
      applyPageWorkflow(data.pageWorkflow)
    }
    shouldAutoPrepareCurrentPage = Boolean(
      autoPrepareRecognition &&
      recognitionModeActive.value &&
      data?.pageWorkflow?.needs_recognition &&
      canRecognizePage.value
    )

    updatePageDynamicSizing(graph.value?.nodes || [], graph.value?.edges || [])
    resetWorkingGraph()
    sortLinesTopToBottom()
  } catch (err) {
    console.error(err)
    error.value = err.message
  } finally {
    loading.value = false
  }
  if (!error.value && shouldAutoPrepareCurrentPage) {
    await recognizeCurrentPage({ focusAfter: true, suppressErrors: true })
  }
  return pageData
}

const getConfidenceColor = (score) => {
    if (score === undefined || score === null) return '#fff'; 
    if (score >= 0.8) return '#4CAF50'; 
    if (score >= 0.5) return '#FFC107'; 
    return '#FF5252';                   
}

const recognizeCurrentPage = async ({ focusAfter = false, suppressErrors = false } = {}) => {
  if (!localManuscriptName.value || !localCurrentPage.value || recognitionInFlight.value || isProcessingSave.value) {
    return false
  }
  if (!canRecognizePage.value) {
    const message = recognitionEngine.value === 'gemini'
      ? 'Gemini needs an API key before it can read this page.'
      : 'Text reading is not available right now.'
    error.value = message
    if (!suppressErrors) alert(message)
    return false
  }

  recognitionInFlight.value = true
  error.value = null
  try {
    const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/recognize-text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        manuscript: localManuscriptName.value,
        page: localCurrentPage.value,
        apiKey: geminiKey.value,
        recognitionEngine: recognitionEngine.value,
      }),
    })
    if (!response.ok) {
      const payload = await response.json()
      throw new Error(payload.error || 'Could not read the page')
    }

    const data = await response.json()
    replaceLocalRecognitionData(data.text || {}, data.confidences || {})
    if (data.activeLearning) applyActiveLearningState(data.activeLearning)
    if (data.pageWorkflow) applyPageWorkflow(data.pageWorkflow)
    sortLinesTopToBottom()
    if (focusAfter && sortedLineIds.value.length > 0) {
      activateInput(sortedLineIds.value[0])
    }
    return true
  } catch (err) {
    error.value = err.message
    if (!suppressErrors) alert(`Could not read the page: ${err.message}`)
    return false
  } finally {
    recognitionInFlight.value = false
  }
}

const runRecognitionAction = async () => {
  if (recognitionModeRequiresLayoutReturn.value) {
    return
  }
  if (recognitionModeActive.value && !hasUnsavedLayoutChanges.value) {
    if (!confirmReplaceWithNewReading()) {
      return
    }
    await recognizeCurrentPage({ focusAfter: true })
    return
  }
  const shouldRefreshOnEnter = hasUnsavedLayoutChanges.value || effectivePageWorkflow.value.needs_recognition
  await requestSwitchToRecognition(shouldRefreshOnEnter)
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
  if (edgeCount === 2) return '4CAF50'
  return '#2196F3'
}

const getNodeRadius = (nodeIndex) => {
  const baseRadius = baseNodeRadiusPx.value
  if (layoutModeActive.value && isEKeyPressed.value) {
    return (hoveredTextlineId.value === nodeToTextlineMap.value[nodeIndex])
      ? clamp(baseRadius * 1.2, MIN_NODE_RADIUS_PX, MAX_NODE_RADIUS_PX + 1)
      : baseRadius
  }
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return clamp(baseRadius * 1.2, MIN_NODE_RADIUS_PX, MAX_NODE_RADIUS_PX + 1)
  if (isNodeSelected(nodeIndex)) return clamp(baseRadius * 1.25, MIN_NODE_RADIUS_PX, MAX_NODE_RADIUS_PX + 1.2)
  return nodeEdgeCounts.value[nodeIndex] < 2 ? clamp(baseRadius * 0.95, MIN_NODE_RADIUS_PX, MAX_NODE_RADIUS_PX) : baseRadius
}
const getEdgeColor = (edge) => (edge.modified ? '#f44336' : '#ffffff')
const getEdgeStrokeWidth = (edge) => {
  const baseWidth = baseEdgeStrokePx.value
  if (isEdgeSelected(edge)) return clamp(baseWidth * 1.35, MIN_EDGE_STROKE_PX, MAX_EDGE_STROKE_PX + 1)
  return edge.modified
    ? clamp(baseWidth * 1.1, MIN_EDGE_STROKE_PX, MAX_EDGE_STROKE_PX + 0.5)
    : baseWidth
}
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

const saveOverlay = async () => {
    // Add visual loading feedback to cursor
    const originalCursor = document.body.style.cursor;
    document.body.style.cursor = 'wait';
    
    try {
        const payload = { graph: workingGraph };
        const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/save-overlay/${localManuscriptName.value}/${localCurrentPage.value}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) throw new Error((await res.json()).error || "Failed to save overlay to backend");
        
        alert(`✅ Image saved successfully to backend for page ${localCurrentPage.value}!`);
    } catch (err) {
        console.error("Error saving overlay:", err);
        alert(`❌ Error saving overlay: ${err.message}`);
    } finally {
        document.body.style.cursor = originalCursor;
    }
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
      if (Math.hypot(mouseX - scaleX(node.x), mouseY - scaleY(node.y)) < nodeHoverRadiusPx.value) {
        newHoveredTextlineId = nodeToTextlineMap.value[i]
        break 
      }
    }
    if (newHoveredTextlineId === null) {
        for(const edge of workingGraph.edges) {
             const n1 = workingGraph.nodes[edge.source], n2 = workingGraph.nodes[edge.target];
             if(n1 && n2 && distanceToLineSegment(mouseX, mouseY, scaleX(n1.x), scaleY(n1.y), scaleX(n2.x), scaleY(n2.y)) < edgeHoverThresholdPx.value) {
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
  if (recognitionInFlight.value) return

  const key = e.key.toLowerCase()
  const isZoomShortcut = (e.ctrlKey || e.metaKey) &&
    (key === '+' || key === '-' || key === '=' || key === '0' || e.code === 'NumpadAdd' || e.code === 'NumpadSubtract')
  if (isZoomShortcut) {
    schedulePostZoomShortcutUpdate()
  }

  if (
    recognitionModeRequiresLayoutReturn.value &&
    !e.repeat &&
    !isInput &&
    (
      key === 'r' ||
      key === 's' ||
      ((e.ctrlKey || e.metaKey) && key === 'enter')
    )
  ) {
    e.preventDefault()
    return
  }

  if ((key === 's' && e.shiftKey && !e.repeat && !isInput) || ((e.ctrlKey || e.metaKey) && key === 'enter' && !isInput)) {
    e.preventDefault()
    saveAndGoNext()
    return
  }

  if (key === 's' && !e.repeat && !isInput) {
    e.preventDefault()
    saveCurrentPage()
    return
  }

  if (key === 'r' && !e.repeat && !isInput) {
    e.preventDefault()
    runRecognitionAction()
    return
  }

  if (key === '[' && !e.repeat && !isInput) {
    e.preventDefault()
    previousPage()
    return
  }

  if (key === ']' && !e.repeat && !isInput) {
    e.preventDefault()
    nextPage()
    return
  }
  
  if (key === 'w' && !e.repeat && !isInput) { e.preventDefault(); setMode('layout'); return }
  if (key === 't' && !e.repeat && !isInput) { e.preventDefault(); requestSwitchToRecognition(); return }
  if (key === 'escape' && recognitionModeActive.value && isInput) { e.preventDefault(); focusedLineId.value = null; return }
  
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
    if (n1 && n2 && distanceToLineSegment(mouseX, mouseY, scaleX(n1.x), scaleY(n1.y), scaleX(n2.x), scaleY(n2.y)) < edgeHoverThresholdPx.value) {
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
    if (Math.hypot(mouseX - scaleX(node.x), mouseY - scaleY(node.y)) < nodeHoverRadiusPx.value)
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
  const textContentForSave = hasUnsavedLayoutChanges.value ? {} : { ...localTextContent }
  const requestBody = {
    graph: workingGraph, 
    modifications: modifications.value,
    textlineLabels: dummyTextlineLabels, 
    textboxLabels: labelsToSend,
    textContent: textContentForSave,
    runRecognition: false,
    apiKey: geminiKey.value,
    recognitionEngine: recognitionEngine.value, // <--- NEW PARAMETER
    activeLearningEnabled: activeLearningEnabled.value,
    saveIntent: background ? 'draft' : 'commit',
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
    if (data.activeLearning) applyActiveLearningState(data.activeLearning)
    if (data.pageWorkflow) applyPageWorkflow(data.pageWorkflow)

    modifications.value = []
    recognitionDraftDirty.value = false
    error.value = null
  } catch (err) {
    error.value = err.message
    throw err
  }
}


const requestSwitchToRecognition = async (forceRecognition = false) => {
    const shouldForceRecognition = forceRecognition === true
    if (recognitionInFlight.value) return;
    if (recognitionModeActive.value && !shouldForceRecognition && !hasUnsavedLayoutChanges.value) return;

    isProcessingSave.value = true;
    try {
        if (hasUnsavedLayoutChanges.value) {
            await saveModifications(); 
            await fetchPageData(localManuscriptName.value, localCurrentPage.value, true, false);
        }
        setMode('recognition');
    } catch (e) {
        alert("Error switching mode: " + e.message);
    } finally {
        isProcessingSave.value = false;
    }
    if (error.value) return;
    if (!hasUnsavedLayoutChanges.value && effectivePageWorkflow.value.can_resume_recognition && !shouldForceRecognition) {
      if (sortedLineIds.value.length > 0) {
        activateInput(sortedLineIds.value[0]);
      }
      return;
    }
    if (shouldForceRecognition || effectivePageWorkflow.value.needs_recognition) {
      await recognizeCurrentPage({ focusAfter: true });
    } else if (effectivePageWorkflow.value.can_edit_text && sortedLineIds.value.length > 0) {
      activateInput(sortedLineIds.value[0]);
    }
}


const confirmAndNavigate = async (navAction) => {
  if (isProcessingSave.value || recognitionInFlight.value) return
  if (hasUnsavedLayoutChanges.value || recognitionDraftDirty.value) {
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
      recognitionDraftDirty.value = false
      navAction()
    }
  } else {
    navAction()
  }
}

const navigateToPage = (page) => {
  pendingPageEntryPreference.value = recognitionModeActive.value
    ? PAGE_ENTRY_RECOGNITION_IF_COMMITTED_TEXT
    : PAGE_ENTRY_LAYOUT
  emit('page-changed', page)
}
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
  if (loading.value || isProcessingSave.value || recognitionInFlight.value || recognitionModeRequiresLayoutReturn.value) return
  isProcessingSave.value = true
  try {
    await saveModifications()
    // Optional: Flash a small 'Saved' toast
  } catch (err) { alert(`Save failed: ${err.message}`) } 
  finally { isProcessingSave.value = false }
}

const saveAndGoNext = async () => {
  if (loading.value || isProcessingSave.value || recognitionInFlight.value || recognitionModeRequiresLayoutReturn.value) return
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
            if (recognitionInFlight.value || isProcessingSave.value || !recognitionDraftDirty.value) return;
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
  updateBrowserZoomLevel()
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

    await fetchPageData(props.manuscriptName, localCurrentPage.value, false, false)
  }
  window.addEventListener('resize', scheduleBrowserZoomLevelUpdate, { passive: true })
  window.addEventListener('wheel', handleCtrlWheelZoom, { passive: true })
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', scheduleBrowserZoomLevelUpdate, { passive: true })
  }
  zoomPollIntervalId = window.setInterval(updateBrowserZoomLevel, 500)
  window.addEventListener('keydown', handleGlobalKeyDown)
  window.addEventListener('keyup', handleGlobalKeyUp)
})

onBeforeUnmount(() => {
  if (zoomUpdateRafId !== null) {
    window.cancelAnimationFrame(zoomUpdateRafId)
    zoomUpdateRafId = null
  }
  if (zoomShortcutTimeoutId !== null) {
    window.clearTimeout(zoomShortcutTimeoutId)
    zoomShortcutTimeoutId = null
  }
  if (zoomPollIntervalId !== null) {
    window.clearInterval(zoomPollIntervalId)
    zoomPollIntervalId = null
  }
  window.removeEventListener('resize', scheduleBrowserZoomLevelUpdate)
  window.removeEventListener('wheel', handleCtrlWheelZoom)
  if (window.visualViewport) {
    window.visualViewport.removeEventListener('resize', scheduleBrowserZoomLevelUpdate)
  }
  window.removeEventListener('keydown', handleGlobalKeyDown)
  window.removeEventListener('keyup', handleGlobalKeyUp)
  if(autoSaveInterval.value) clearInterval(autoSaveInterval.value);
})

watch(() => props.pageName, async (newPageName) => {
    if (newPageName && newPageName !== localCurrentPage.value) {
      const entryPreference = pendingPageEntryPreference.value
      pendingPageEntryPreference.value = null
      const shouldEvaluateRecognitionResume = entryPreference === PAGE_ENTRY_RECOGNITION_IF_COMMITTED_TEXT
      const shouldResetToLayoutMode = entryPreference === PAGE_ENTRY_LAYOUT || shouldEvaluateRecognitionResume
      localCurrentPage.value = newPageName
      if (shouldResetToLayoutMode) {
        setMode('layout')
      }
      const shouldAutoPrepareRecognition =
        !shouldResetToLayoutMode && recognitionModeActive.value && autoRecogEnabled.value
      const pageData = await fetchPageData(
        localManuscriptName.value,
        newPageName,
        false,
        shouldAutoPrepareRecognition,
      )
      if (shouldEvaluateRecognitionResume && shouldResumeRecognitionForWorkflow(pageData?.pageWorkflow)) {
        setMode('recognition')
      }
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
/* Replace existing .top-bar */
.top-bar {
  display: grid;
  grid-template-columns: minmax(240px, 1.2fr) minmax(320px, 1.05fr) minmax(520px, 2.15fr);
  align-items: stretch;
  gap: 10px;
  padding: 8px 10px;
  min-height: 72px; /* Reduced height */
  background-color: #2c2c2c;
  border-bottom: 1px solid #3d3d3d;
  flex-shrink: 0;
  z-index: 10;
}

.fixed-ui-compensated {
  zoom: var(--fixed-ui-zoom, 1);
}

@supports not (zoom: 1) {
  .fixed-ui-compensated {
    transform: scale(var(--fixed-ui-transform-scale, 1));
    transform-origin: top left;
    width: calc(100% / var(--fixed-ui-transform-scale, 1));
  }
}
.top-bar-left, .top-bar-right, .action-group { display: flex; align-items: center; gap: 16px; }
.top-bar-left, .top-bar-right {
  min-width: 0;
  flex-shrink: 0;
}
.top-bar-center { flex: 1; min-width: 0; }
.top-bar-section {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: center;
  gap: 10px;
  padding: 10px 12px;
  border: 1px solid #3b3b3b;
  border-radius: 12px;
  background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015));
  min-width: 0;
}

.top-bar-left {
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.page-context {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.page-meta {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}

.page-eyebrow {
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #8cb8a7;
}

.page-title-row {
  display: flex;
  align-items: baseline;
  gap: 8px;
  min-width: 0;
}

.page-title {
  font-size: 1rem;
  color: #fff;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.page-divider {
  color: #5f5f5f;
}

.page-current {
  font-size: 0.86rem;
  color: #cfd9ff;
  white-space: nowrap;
}

.page-controls {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 10px;
  flex-wrap: wrap;
  margin-left: auto;
}

.page-picker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 9px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.05);
  background: rgba(255,255,255,0.03);
  min-width: 0;
  transition: transform 0.18s ease, border-color 0.18s ease, background-color 0.18s ease;
}

.page-picker-label {
  font-size: 0.64rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #9a9a9a;
  white-space: nowrap;
}

.page-stepper {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.page-stepper .nav-btn {
  min-height: 34px;
  padding: 7px 12px;
  border: 1px solid #4a4a4a;
  background: rgba(255,255,255,0.035);
  color: #e7e7e7;
}

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
    padding: 5px 10px;
    border-radius: 4px;
    outline: none;
    font-size: 0.84rem;
    cursor: pointer;
    min-width: 102px;
}
.page-select:hover { border-color: #666; }

.workflow-panel {
  display: flex;
  align-items: stretch;
  gap: 18px;
  padding: 12px 16px;
  background: linear-gradient(135deg, rgba(23, 23, 23, 0.95), rgba(41, 41, 41, 0.92));
  border: 1px solid #3b3b3b;
  border-radius: 12px;
  min-width: 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}

.workflow-summary {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  flex: 1;
}

.workflow-eyebrow {
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #8cb8a7;
}

.workflow-pill-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
}

.workflow-pill {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  padding: 0 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  border: 1px solid #4d4d4d;
  background: rgba(255,255,255,0.06);
  color: #f4f4f4;
}

.workflow-pill.subtle {
  color: #d3d3d3;
  background: rgba(255,255,255,0.04);
}

.workflow-pill.state-ready,
.workflow-pill.state-manual_only {
  background: rgba(64, 145, 108, 0.2);
  border-color: rgba(97, 201, 149, 0.45);
  color: #bff0d7;
}

.workflow-pill.state-layout_dirty,
.workflow-pill.state-stale_layout {
  background: rgba(191, 111, 59, 0.18);
  border-color: rgba(240, 152, 94, 0.4);
  color: #ffd2b6;
}

.workflow-pill.state-missing_page_xml,
.workflow-pill.state-refreshing_ocr {
  background: rgba(48, 116, 170, 0.18);
  border-color: rgba(108, 181, 240, 0.4);
  color: #c8e8ff;
}

.workflow-hint {
  color: #bababa;
  font-size: 0.78rem;
  line-height: 1.25;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.workflow-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: flex-end;
  align-content: center;
}

.workflow-recognition-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: nowrap;
  justify-content: flex-end;
  align-content: center;
  min-height: 44px;
  min-width: 0;
}

.workflow-recognition-controls.is-inactive,
.workflow-palette-slot.is-inactive {
  visibility: hidden;
  pointer-events: none;
}

.workflow-palette-slot {
  display: inline-flex;
  align-items: center;
  min-height: 40px;
}

.workflow-toggle-group {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 10px;
  border: 1px solid transparent;
  background: rgba(255,255,255,0.03);
  transition: transform 0.18s ease, border-color 0.18s ease, background-color 0.18s ease;
}

.workflow-toggle-group.compact {
  padding-right: 2px;
}

.workflow-toggle-copy {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.workflow-toggle-label {
  font-size: 0.78rem;
  color: #efefef;
  line-height: 1;
}

.workflow-toggle-subcopy {
  font-size: 0.68rem;
  color: #9fd4ff;
  line-height: 1;
}

.workflow-select {
  background: #333;
  color: #fff;
  border: 1px solid #555;
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 0.72rem;
  outline: none;
  cursor: pointer;
}

.top-bar-right {
  justify-content: center;
}

.action-summary {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}

.action-eyebrow {
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #8cb8a7;
}

.action-title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.action-title {
  color: #fff;
  font-size: 0.98rem;
  font-weight: 600;
}

.action-badge {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  padding: 0 9px;
  border-radius: 999px;
  background: rgba(97, 201, 149, 0.14);
  border: 1px solid rgba(97, 201, 149, 0.32);
  color: #c6f5da;
  font-size: 0.72rem;
}

.action-hint {
  color: #bababa;
  font-size: 0.78rem;
  line-height: 1.35;
}

.recognition-engine-panel {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(108, 181, 240, 0.16);
  background: linear-gradient(180deg, rgba(48, 116, 170, 0.14), rgba(48, 116, 170, 0.06));
}

.recognition-engine-label {
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #9fd4ff;
  white-space: nowrap;
}

.recognition-engine-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.recognition-engine-select {
  min-width: 132px;
}

.recognition-engine-hint {
  color: #c6d7e6;
  font-size: 0.76rem;
  line-height: 1.3;
}

.top-bar-right .action-group {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  align-items: stretch;
}

.control-shell {
  display: inline-flex;
  max-width: 100%;
}

.action-slot {
  width: 100%;
  min-width: 0;
}

.action-slot.is-ghost {
  visibility: hidden;
  pointer-events: none;
}

.control-shell.is-disabled {
  cursor: not-allowed;
}

.control-shell > button:disabled {
  pointer-events: none;
}

.primary-actions {
  justify-content: flex-end;
}

.secondary-actions {
  justify-content: flex-end;
  gap: 10px;
}

.top-bar-right .action-btn {
  min-height: 40px;
  width: 100%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.top-bar-right .action-btn.recommended {
  background: linear-gradient(180deg, #59ab6c, #458a57);
  border-color: #66c37d;
  color: #f5fff7;
  box-shadow: 0 0 0 1px rgba(102, 195, 125, 0.18), 0 12px 24px rgba(46, 92, 58, 0.22);
}

.top-bar-right .action-btn.forward-action {
  background: rgba(99, 123, 173, 0.12);
  border-color: rgba(132, 161, 223, 0.32);
  color: #dde7ff;
}

.secondary-action {
  background: transparent;
  color: #d0d0d0;
  border-color: #4c4c4c;
}

.top-bar .nav-btn:hover:not(:disabled),
.top-bar .action-btn:hover:not(:disabled),
.workflow-toggle-group:hover,
.page-picker:hover {
  transform: translateY(-1px);
}

.workflow-toggle-group:hover,
.page-picker:hover {
  border-color: rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.05);
}

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

.recognition-guard-card {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 200;
  width: min(440px, calc(100% - 40px));
  padding: 22px 24px;
  border-radius: 16px;
  background: rgba(13, 13, 13, 0.92);
  border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.45);
  text-align: left;
  backdrop-filter: blur(10px);
}

.recognition-guard-card h3 {
  margin: 10px 0 8px;
  color: #fff;
  font-size: 1.1rem;
}

.recognition-guard-card p {
  margin: 0 0 16px;
  color: #c8c8c8;
  line-height: 1.45;
}

.recognition-guard-badge {
  display: inline-flex;
  align-items: center;
  padding: 0 10px;
  min-height: 26px;
  border-radius: 999px;
  border: 1px solid rgba(108, 181, 240, 0.35);
  background: rgba(48, 116, 170, 0.18);
  color: #c8e8ff;
  font-size: 0.78rem;
}

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
.mode-tools-shell {
  display: flex;
  align-items: stretch;
  min-width: 0;
  padding: 4px 10px;
  background: #212121;
  border-left: 1px solid #323232;
}
.mode-tools-shell.is-inactive {
  visibility: hidden;
  pointer-events: none;
}
.mode-tools-section {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
  padding: 0 12px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.06);
  background: rgba(255,255,255,0.035);
}
.mode-tools-label {
  font-size: 0.64rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #9a9a9a;
  white-space: nowrap;
}
.mode-tools-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.bottom-tools-toggle {
  background: rgba(255,255,255,0.02);
  border-color: rgba(255,255,255,0.04);
}
.bottom-palette-slot {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
}
.bottom-palette-slot.is-inactive {
  visibility: hidden;
  pointer-events: none;
}
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

.recognition-status-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 14px;
}

.recognition-status-card {
  background: #252525;
  border: 1px solid #3d3d3d;
  border-radius: 8px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.recognition-status-label {
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #8cb8a7;
}

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

@media (max-width: 1380px) {
  .top-bar {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .top-bar-left {
    flex-direction: column;
    align-items: stretch;
    justify-content: center;
  }

  .page-controls {
    justify-content: flex-start;
    margin-left: 0;
  }

  .top-bar-center.workflow-panel {
    grid-column: 1 / -1;
  }

  .workflow-panel {
    flex-direction: column;
    align-items: stretch;
  }

  .workflow-controls {
    justify-content: flex-start;
  }

  .top-bar-right .action-group {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .primary-actions {
    justify-content: flex-start;
  }

  .secondary-actions {
    justify-content: flex-start;
  }

  .recognition-engine-panel {
    align-items: flex-start;
  }

  .recognition-status-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 980px) {
  .top-bar {
    grid-template-columns: minmax(0, 1fr);
  }

  .top-bar-center.workflow-panel {
    grid-column: auto;
  }

  .page-stepper {
    width: 100%;
  }

  .workflow-recognition-controls {
    width: 100%;
    justify-content: flex-start;
  }

  .top-bar-right .action-group {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
