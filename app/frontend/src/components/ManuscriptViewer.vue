<template>
  <div class="manuscript-viewer">
    
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
              <span class="workflow-toggle-meta">{{ activeReaderStatusLabel }}</span>
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
                <option value="gemini" :disabled="!isRecognitionEngineAvailable('gemini')">Gemini</option>
              </select>
              <span v-if="readerSwitchNotice" class="recognition-engine-note">
                {{ readerSwitchNotice }}
              </span>
              <span v-else-if="recognitionEngine === 'gemini' && !isRecognitionEngineAvailable('gemini')" class="recognition-engine-note">
                {{ recognitionEngineUnavailableMessage('gemini') }}
              </span>
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

      <div v-if="error && !recognitionRecoveryPrompt" class="error-message">
        {{ error }}
      </div>

      <div
        v-if="recognitionRecoveryPrompt"
        class="recognition-recovery-backdrop"
        role="dialog"
        aria-modal="true"
        aria-labelledby="recognition-recovery-title"
      >
        <div class="recognition-recovery-card">
          <span class="recognition-recovery-badge">Text Reading Failed</span>
          <h3 id="recognition-recovery-title">Gemini could not read this page</h3>
          <p>{{ recognitionRecoveryPrompt.message }}</p>
          <div class="recognition-recovery-actions">
            <button
              class="action-btn secondary-action"
              @click="retryRecognitionAfterFailure('gemini')"
              :disabled="recognitionInFlight || isProcessingSave || !isRecognitionEngineAvailable('gemini')"
            >
              Try Gemini Again
            </button>
            <button
              class="action-btn"
              @click="retryRecognitionAfterFailure('local')"
              :disabled="recognitionInFlight || isProcessingSave"
            >
              Use Built-in Reader
            </button>
            <button class="action-btn secondary-action" @click="dismissRecognitionRecovery">
              Dismiss
            </button>
          </div>
        </div>
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
              :style="{ cursor: isOKeyPressed ? 'crosshair' : 'pointer'}"
              @click="onBackgroundClick($event)"
              @contextmenu.prevent 
              @mousedown="handleSvgMouseDown"
              @mousemove="handleSvgMouseMove"
              @mouseup="handleSvgMouseUp"
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
                  !isEKeyPressed &&
                  !isOKeyPressed
                "
                :x1="scaleX(workingGraph.nodes[selectedNodes[0]].x)"
                :y1="scaleY(workingGraph.nodes[selectedNodes[0]].y)"
                :x2="tempEndPoint.x"
                :y2="tempEndPoint.y"
                stroke="#ff9500"
                :stroke-width="tempEdgeStrokeWidth"
                stroke-dasharray="5,5"
              />

              <g
                v-if="showReadingDirectionOverlay"
                class="reading-direction-overlay"
                aria-hidden="true"
              >
                <g
                  v-for="arrow in readingDirectionArrowOverlays"
                  :key="`reading-arrow-${arrow.id}`"
                  class="reading-direction-arrow"
                  :transform="arrow.transform"
                >
                  <path d="M -9 -5 L 1.5 -5 L 1.5 -8.5 L 12 0 L 1.5 8.5 L 1.5 5 L -9 5 Z" />
                </g>
              </g>
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
              v-if="showRecognitionGuardCard"
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
                :class="{ 'has-line-preview': activeLineImagePreview }"
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
                    v-if="activeLineImagePreview"
                    class="line-image-preview"
                >
                    <img
                        :src="backendAssetUrl(activeLineImagePreview.imageUrl)"
                        class="line-image-preview-img"
                        alt=""
                        draggable="false"
                    />
                </div>
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
                <p>Hold <span class="key-badge">a</span> and hover to connect points</p>
                <p>Hold <span class="key-badge">d</span> and hover to remove a link</p>
              </div>
            </div>

            <!-- Regions Card -->
            <div class="help-card horizontal-layout">
              <div class="media-container-square">
                <video :src="regionWebm" autoplay loop muted playsinline preload="auto" class="tutorial-video"></video>
              </div>
              <div class="card-text">
                <h4>Regions</h4>
                <p>Hold <span class="key-badge">e</span> and hover to mark a region</p>
                <p>Release and repeat to start a new region</p>
              </div>
            </div>

            <!-- Orientation Card -->
            <div class="help-card horizontal-layout">
              <div class="media-container-square orientation-help-visual" aria-hidden="true">
                <span class="orientation-help-key">q</span>
                <span class="orientation-help-line"></span>
              </div>
              <div class="card-text">
                <h4>Direction</h4>
                <p>Hold <span class="key-badge">q</span> and hover to mark orientation</p>
                <p>Hover over a text-line from it's bottom to it's top</p>
              </div>
            </div>

          </div>

          <!-- Hotkey Footer -->
          <div class="hotkey-footer">
            <span class="key-hint"><span class="key-badge">v</span> Hold to Hide Graph</span>
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
const RECOGNITION_READER_LABELS = {
  local: 'Built-in Reader',
  gemini: 'Gemini',
}
const normalizeRecognitionEngine = (value) => (value === 'gemini' ? 'gemini' : 'local')
const activeLearningPollDelayMs = {
  immediate: 0,
  active: 1000,
  idle: 60000,
}

// UI State
const isPanelCollapsed = ref(true)
const activeInput = ref(null) 

const setMode = (mode) => {
  layoutModeActive.value = false
  recognitionModeActive.value = false
  
  isAKeyPressed.value = false
  isDKeyPressed.value = false
  isEKeyPressed.value = false
  isOKeyPressed.value = false
  resetReadingDirectionHoverState()
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
const isOKeyPressed = ref(false)

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
const savedTextboxLabelsSnapshot = ref('[]')
const readingDirectionAnnotations = ref({})
const readingDirectionDraft = ref(null)
const savedReadingDirectionAnnotationsSnapshot = ref('[]')
let suppressNextBackgroundClick = false
let readingDirectionHoverStartPoint = null
let pendingReadingDirectionHoverPoint = null
let readingDirectionHoverRafId = null
const readingDirectionHoverAnnotatedLineIds = new Set()

function cancelPendingReadingDirectionHover() {
  if (readingDirectionHoverRafId !== null) {
    window.cancelAnimationFrame(readingDirectionHoverRafId)
    readingDirectionHoverRafId = null
  }
  pendingReadingDirectionHoverPoint = null
}

function resetReadingDirectionHoverState() {
  cancelPendingReadingDirectionHover()
  readingDirectionHoverStartPoint = null
  readingDirectionHoverAnnotatedLineIds.clear()
  readingDirectionDraft.value = null
}

// Recognition Data
const localTextContent = reactive({}) 
const pagePolygons = ref({}) 
const lineImagePreviews = ref({})
const focusedLineId = ref(null)
const sortedLineIds = ref([])
const autoRecogEnabled = ref(localStorage.getItem('auto_prepare_next_page') === 'true')
const activeLearningEnabled = ref(localStorage.getItem('active_learning_enabled') !== 'false')
const activeLearningStatus = ref('Not updating right now')
const recognitionEngine = ref(normalizeRecognitionEngine(localStorage.getItem('recognition_engine') || 'local'))
const devanagariModeEnabled = ref(true) 
const recognitionInFlight = ref(false)
const recognitionDraftDirty = ref(false)
const suppressTextDirtyTracking = ref(false)
const pendingPageEntryPreference = ref(null)
const readerSwitchNotice = ref('')
const recognitionRecoveryPrompt = ref(null)
const readerCapabilities = reactive({
  local: {
    available: true,
    label: RECOGNITION_READER_LABELS.local,
    unavailableReason: null,
  },
  gemini: {
    available: false,
    label: RECOGNITION_READER_LABELS.gemini,
    unavailableReason: 'Gemini is not configured on this server.',
  },
})
const activeLearningMeta = reactive({
  code: 'idle',
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
  latest_supervised_commit_revision_number: null,
  review_status: 'layout_ready_no_text',
  has_ground_truth: false,
  ground_truth_revision_number: null,
  current_revision_is_ground_truth: false,
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
watch(activeLearningEnabled, (val) => localStorage.setItem('active_learning_enabled', String(val)))
const localTextConfidence = reactive({}) 
const autoSaveInterval = ref(null) // NEW
let activeLearningPollTimeoutId = null
let activeLearningPollInFlight = false
let suppressRecognitionEngineWatcher = false
let readerSwitchNoticeTimeoutId = null

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

const buildTextboxLabelsPayload = (numNodes = 0) => {
  const safeNodeCount = Math.max(0, Number(numNodes) || 0)
  const labels = new Array(safeNodeCount).fill(0)
  Object.keys(textlineLabels).forEach((nodeIndex) => {
    const parsedIndex = Number(nodeIndex)
    if (!Number.isInteger(parsedIndex) || parsedIndex < 0 || parsedIndex >= safeNodeCount) return
    labels[parsedIndex] = Number(textlineLabels[nodeIndex] ?? 0)
  })
  return labels
}

const normalizeReadingDirectionAnnotation = (annotation) => {
  if (!annotation?.reading_direction || !annotation?.cut_start || !annotation?.cut_end) return null
  const cutStart = annotation.cut_start.map(Number)
  const cutEnd = annotation.cut_end.map(Number)
  const cutMidpoint = annotation.cut_midpoint ? annotation.cut_midpoint.map(Number) : [
    (cutStart[0] + cutEnd[0]) / 2,
    (cutStart[1] + cutEnd[1]) / 2,
  ]
  const readingDirection = annotation.reading_direction.map(Number)
  if (
    cutStart.length < 2 ||
    cutEnd.length < 2 ||
    readingDirection.length < 2 ||
    !cutStart.every(Number.isFinite) ||
    !cutEnd.every(Number.isFinite) ||
    !cutMidpoint.every(Number.isFinite) ||
    !readingDirection.every(Number.isFinite)
  ) return null
  const componentNodeIndices = Array.isArray(annotation.component_node_indices)
    ? [...new Set(annotation.component_node_indices.map(Number).filter(Number.isInteger))].sort((a, b) => a - b)
    : []
  return {
    annotation_id: String(annotation.annotation_id || annotation.frontend_line_id || ''),
    frontend_line_id: String(annotation.frontend_line_id || annotation.annotation_id || ''),
    component_node_indices: componentNodeIndices,
    cut_start: cutStart,
    cut_end: cutEnd,
    cut_midpoint: cutMidpoint,
    reading_direction: readingDirection,
    source: annotation.source || 'user_cross_cut',
    updated_at: annotation.updated_at || '',
  }
}

const findFrontendTextlineForComponent = (componentNodeIndices, fallbackLineId = null) => {
  const componentSet = new Set(
    Array.isArray(componentNodeIndices)
      ? componentNodeIndices.map(Number).filter(Number.isInteger)
      : []
  )
  if (componentSet.size === 0 && fallbackLineId !== null && textlines.value[fallbackLineId]) {
    return String(fallbackLineId)
  }
  let best = { lineId: null, overlap: 0, ratio: 0 }
  Object.entries(textlines.value).forEach(([lineId, nodeIndices]) => {
    const overlap = nodeIndices.filter((nodeIndex) => componentSet.has(Number(nodeIndex))).length
    const ratio = componentSet.size > 0 ? overlap / componentSet.size : 0
    if (overlap > best.overlap || (overlap === best.overlap && ratio > best.ratio)) {
      best = { lineId, overlap, ratio }
    }
  })
  if (best.lineId !== null && best.overlap > 0 && best.ratio >= 0.5) return String(best.lineId)
  if (fallbackLineId !== null && textlines.value[fallbackLineId]) return String(fallbackLineId)
  return null
}

const loadReadingDirectionAnnotationsFromPageData = (metadata) => {
  const loaded = {}
  const lineAnnotations = Array.isArray(metadata?.lineAnnotations) ? metadata.lineAnnotations : []
  lineAnnotations.forEach((annotation) => {
    const fallbackLineId = annotation.frontend_line_id ?? annotation.line_id ?? annotation.resolved_line_numeric_id ?? null
    const textlineId = findFrontendTextlineForComponent(annotation.component_node_indices, fallbackLineId)
    if (textlineId === null) return
    const normalized = normalizeReadingDirectionAnnotation({
      ...annotation,
      annotation_id: textlineId,
      frontend_line_id: textlineId,
    })
    if (normalized) loaded[textlineId] = normalized
  })
  readingDirectionAnnotations.value = loaded
  syncSavedReadingDirectionAnnotationsSnapshot()
}

const buildReadingDirectionAnnotationsPayload = () =>
  Object.values(readingDirectionAnnotations.value)
    .map(normalizeReadingDirectionAnnotation)
    .filter(Boolean)
    .sort((a, b) => a.annotation_id.localeCompare(b.annotation_id))

const readingDirectionAnnotationList = computed(() => buildReadingDirectionAnnotationsPayload())
const READING_DIRECTION_OVERLAY_LOG_LIMIT = 40
const readingDirectionOverlayLogKeys = new Set()

const showReadingDirectionOverlay = computed(() =>
  graphIsLoaded.value &&
  layoutModeActive.value &&
  isOKeyPressed.value &&
  !recognitionModeActive.value
)

const logReadingDirectionOverlayIssue = (reason, payload = {}) => {
  const annotationId = payload.annotationId || payload.lineId || 'unknown'
  const key = `${reason}:${annotationId}`
  if (readingDirectionOverlayLogKeys.has(key)) return
  if (readingDirectionOverlayLogKeys.size >= READING_DIRECTION_OVERLAY_LOG_LIMIT) return
  readingDirectionOverlayLogKeys.add(key)
  console.debug('[reading-direction-overlay]', reason, payload)
}

const svgNumber = (value) => Number.isFinite(value) ? value.toFixed(2) : '0'

const toFinitePoint = (point) => {
  if (!Array.isArray(point) || point.length < 2) return null
  const x = Number(point[0])
  const y = Number(point[1])
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null
  return [x, y]
}

const normalizeVector2d = (x, y) => {
  const dx = Number(x)
  const dy = Number(y)
  const length = Math.hypot(dx, dy)
  if (!Number.isFinite(length) || length <= 1e-9) return null
  return [dx / length, dy / length]
}

const dot2d = (a, b) => (a[0] * b[0]) + (a[1] * b[1])

const nodePointRaw = (nodeIndex) => {
  const node = workingGraph.nodes[Number(nodeIndex)]
  if (!node) return null
  const x = Number(node.x)
  const y = Number(node.y)
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null
  return [x, y]
}

const annotationReadingDirectionVector = (annotation) => {
  const vector = Array.isArray(annotation?.reading_direction) ? annotation.reading_direction : null
  return vector ? normalizeVector2d(vector[0], vector[1]) : null
}

const annotationCutPoint = (annotation) => {
  const midpoint = toFinitePoint(annotation?.cut_midpoint)
  if (midpoint) return midpoint
  const start = toFinitePoint(annotation?.cut_start)
  const end = toFinitePoint(annotation?.cut_end)
  if (!start || !end) return null
  return [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]
}

const lineNodeIndicesForAnnotation = (annotation) => {
  const lineId = String(annotation?.frontend_line_id || annotation?.annotation_id || '')
  const liveLineNodes = lineId !== '' ? textlines.value[lineId] : null
  const rawIndices = Array.isArray(liveLineNodes) && liveLineNodes.length > 0
    ? liveLineNodes
    : annotation?.component_node_indices
  if (!Array.isArray(rawIndices)) return []
  return [...new Set(
    rawIndices
      .map(Number)
      .filter((nodeIndex) => Number.isInteger(nodeIndex) && nodePointRaw(nodeIndex))
  )]
}

const createEmptyBoundingBox = () => ({
  minX: Infinity,
  minY: Infinity,
  maxX: -Infinity,
  maxY: -Infinity,
})

const expandBoundingBoxWithPoint = (bbox, point) => {
  if (!point || !Number.isFinite(point[0]) || !Number.isFinite(point[1])) return
  bbox.minX = Math.min(bbox.minX, point[0])
  bbox.minY = Math.min(bbox.minY, point[1])
  bbox.maxX = Math.max(bbox.maxX, point[0])
  bbox.maxY = Math.max(bbox.maxY, point[1])
}

const finalizeBoundingBox = (bbox) => (
  Number.isFinite(bbox.minX) && Number.isFinite(bbox.minY) &&
  Number.isFinite(bbox.maxX) && Number.isFinite(bbox.maxY)
    ? bbox
    : null
)

const expandBoundingBox = (bbox, padding) => ({
  minX: bbox.minX - padding,
  minY: bbox.minY - padding,
  maxX: bbox.maxX + padding,
  maxY: bbox.maxY + padding,
})

const boundingBoxesOverlap = (left, right) => (
  left.minX <= right.maxX &&
  left.maxX >= right.minX &&
  left.minY <= right.maxY &&
  left.maxY >= right.minY
)

const textlineGeometryIndex = computed(() => {
  const edgeBuckets = new Map()

  workingGraph.edges.forEach((edge) => {
    const source = Number(edge.source)
    const target = Number(edge.target)
    if (!Number.isInteger(source) || !Number.isInteger(target) || source === target) return

    const sourceLineId = nodeToTextlineMap.value[source]
    const targetLineId = nodeToTextlineMap.value[target]
    if (sourceLineId === undefined || sourceLineId === null) return
    if (targetLineId === undefined || targetLineId === null) return
    if (String(sourceLineId) !== String(targetLineId)) return

    const sourcePoint = nodePointRaw(source)
    const targetPoint = nodePointRaw(target)
    if (!sourcePoint || !targetPoint) return

    const lineId = String(sourceLineId)
    if (!edgeBuckets.has(lineId)) edgeBuckets.set(lineId, [])
    edgeBuckets.get(lineId).push({
      sourceIndex: source,
      targetIndex: target,
      start: sourcePoint,
      end: targetPoint,
    })
  })

  const entries = []
  const byLineId = new Map()

  Object.entries(textlines.value).forEach(([rawLineId, rawNodeIndices]) => {
    const lineId = String(rawLineId)
    const bbox = createEmptyBoundingBox()
    const nodePoints = []
    const nodeIndices = Array.isArray(rawNodeIndices)
      ? rawNodeIndices.map(Number).filter(Number.isInteger)
      : []

    nodeIndices.forEach((nodeIndex) => {
      const point = nodePointRaw(nodeIndex)
      if (!point) return
      nodePoints.push({ nodeIndex, point })
      expandBoundingBoxWithPoint(bbox, point)
    })

    const edgeSegments = edgeBuckets.get(lineId) || []
    const degreeByNode = new Map()
    edgeSegments.forEach((segment) => {
      degreeByNode.set(segment.sourceIndex, (degreeByNode.get(segment.sourceIndex) || 0) + 1)
      degreeByNode.set(segment.targetIndex, (degreeByNode.get(segment.targetIndex) || 0) + 1)
      expandBoundingBoxWithPoint(bbox, segment.start)
      expandBoundingBoxWithPoint(bbox, segment.end)
    })

    const finalizedBox = finalizeBoundingBox(bbox)
    if (!finalizedBox) return

    const endpointIndices = nodeIndices.filter((nodeIndex) => (degreeByNode.get(nodeIndex) || 0) <= 1)
    const geometry = {
      lineId,
      nodeIndices,
      nodePoints,
      edgeSegments,
      endpointIndices,
      bbox: finalizedBox,
    }
    entries.push(geometry)
    byLineId.set(lineId, geometry)
  })

  return { entries, byLineId }
})

const pointWithMaxProjection = (points, readingDirection) =>
  points.reduce((best, point) => {
    if (!point) return best
    const projection = dot2d(point, readingDirection)
    if (!best || projection > best.projection) return { point, projection }
    return best
  }, null)?.point ?? null

const readingDirectionArrowAnchorForAnnotation = (annotation, readingDirection) => {
  const lineId = String(annotation?.frontend_line_id || annotation?.annotation_id || '')
  const lineGeometry = lineId !== '' ? textlineGeometryIndex.value.byLineId.get(lineId) : null
  const cutPoint = annotationCutPoint(annotation)

  if (lineGeometry?.edgeSegments?.length > 1 && lineGeometry.endpointIndices.length === 0 && cutPoint) {
    return cutPoint
  }

  if (lineGeometry?.nodePoints?.length) {
    const endpointSet = new Set(lineGeometry.endpointIndices)
    const endpointPoints = lineGeometry.nodePoints
      .filter(({ nodeIndex }) => endpointSet.has(nodeIndex))
      .map(({ point }) => point)
    const candidatePoints = endpointPoints.length > 0
      ? endpointPoints
      : lineGeometry.nodePoints.map(({ point }) => point)
    const endpoint = pointWithMaxProjection(candidatePoints, readingDirection)
    if (endpoint) return endpoint
  }

  const annotationPoints = lineNodeIndicesForAnnotation(annotation).map(nodePointRaw).filter(Boolean)
  return pointWithMaxProjection(annotationPoints, readingDirection) || cutPoint
}

const buildReadingDirectionTerminalArrow = (annotation, fallbackIndex) => {
  const annotationId = annotation.annotation_id || annotation.frontend_line_id || String(fallbackIndex)
  const readingDirection = annotationReadingDirectionVector(annotation)
  if (!readingDirection) {
    logReadingDirectionOverlayIssue('missing-reading-direction', { annotationId })
    return null
  }

  const anchor = readingDirectionArrowAnchorForAnnotation(annotation, readingDirection)
  if (!anchor) {
    logReadingDirectionOverlayIssue('missing-arrow-anchor', { annotationId })
    return null
  }

  const angle = Math.atan2(readingDirection[1], readingDirection[0]) * 180 / Math.PI
  return {
    id: annotationId,
    transform: `translate(${svgNumber(scaleX(anchor[0]))} ${svgNumber(scaleY(anchor[1]))}) rotate(${svgNumber(angle)})`,
  }
}

const readingDirectionOverlay = computed(() => {
  if (!showReadingDirectionOverlay.value) return { arrows: [] }
  return {
    arrows: readingDirectionAnnotationList.value
      .map(buildReadingDirectionTerminalArrow)
      .filter(Boolean),
  }
})

const readingDirectionArrowOverlays = computed(() => readingDirectionOverlay.value.arrows)

const syncSavedTextboxLabelsSnapshot = (numNodes = workingGraph.nodes?.length || graph.value?.nodes?.length || 0) => {
  savedTextboxLabelsSnapshot.value = JSON.stringify(buildTextboxLabelsPayload(numNodes))
}

const syncSavedReadingDirectionAnnotationsSnapshot = () => {
  savedReadingDirectionAnnotationsSnapshot.value = JSON.stringify(buildReadingDirectionAnnotationsPayload())
}

const hasUnsavedTextboxLabelChanges = computed(() => {
  const nodeCount = workingGraph.nodes?.length || graph.value?.nodes?.length || 0
  return JSON.stringify(buildTextboxLabelsPayload(nodeCount)) !== savedTextboxLabelsSnapshot.value
})
const hasUnsavedReadingDirectionChanges = computed(() =>
  JSON.stringify(buildReadingDirectionAnnotationsPayload()) !== savedReadingDirectionAnnotationsSnapshot.value
)

const hasUnsavedGraphChanges = computed(() => modifications.value.length > 0)
const hasUnsavedLayoutChanges = computed(() =>
  hasUnsavedGraphChanges.value || hasUnsavedTextboxLabelChanges.value || hasUnsavedReadingDirectionChanges.value
)
const currentPageHasTextContent = computed(() =>
  Object.values(localTextContent).some((value) => String(value || '').trim().length > 0)
)
const reviewStatusRequiresGroundTruthCommit = (status) =>
  [
    'ocr_prediction_unreviewed',
    'draft_saved',
    'ground_truth_with_draft_changes',
    'legacy_text_needs_review',
  ].includes(String(status || ''))
const groundTruthCommitPending = computed(() =>
  currentPageHasTextContent.value &&
  !pageWorkflow.current_revision_is_ground_truth &&
  (
    recognitionDraftDirty.value ||
    reviewStatusRequiresGroundTruthCommit(pageWorkflow.review_status)
  )
)
const determineSaveScope = ({ background = false, forceLayoutSave = false } = {}) => {
  if (forceLayoutSave) return 'layout'
  if (hasUnsavedLayoutChanges.value) return 'layout'
  if (
    recognitionModeActive.value &&
    pageWorkflow.can_edit_text &&
    currentPageHasTextContent.value &&
    !effectivePageWorkflow.value.needs_recognition
  ) {
    return 'text_only'
  }
  return 'layout'
}

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
const activeReaderStatusLabel = computed(() => `Trainable Reader: ${localCheckpointDescriptor.value.modelLabel}`)

const recognitionEngineOptionLabel = (engine) =>
  readerCapabilities[engine]?.label || RECOGNITION_READER_LABELS[engine] || String(engine || 'Reader')

const isRecognitionEngineAvailable = (engine) => {
  const normalizedEngine = normalizeRecognitionEngine(engine)
  return Boolean(readerCapabilities[normalizedEngine]?.available)
}

const recognitionEngineUnavailableMessage = (engine = recognitionEngine.value) => {
  const normalizedEngine = normalizeRecognitionEngine(engine)
  return (
    readerCapabilities[normalizedEngine]?.unavailableReason ||
    `${recognitionEngineOptionLabel(normalizedEngine)} is not available right now.`
  )
}

const nextRecognitionSourceLabel = computed(() => {
  if (recognitionEngine.value === 'gemini') return recognitionEngineOptionLabel('gemini')
  return localCheckpointDescriptor.value.modelLabel
})

const recognitionEngineLabel = computed(() => nextRecognitionSourceLabel.value)
const canRecognizePage = computed(() => isRecognitionEngineAvailable(recognitionEngine.value))
const recognitionBusyLabel = computed(() => {
  if (recognitionEngine.value === 'gemini') return `Reading the page with ${recognitionEngineOptionLabel('gemini')}...`
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
      hint: pageWorkflow.has_ground_truth
        ? 'This page has saved ground truth. Save the layout changes only if you are ready to re-review the text.'
        : 'Save the page layout, then read the text again before correcting it.',
      review_status: pageWorkflow.has_ground_truth ? 'ground_truth_stale_layout' : pageWorkflow.review_status,
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
const showRecognitionGuardCard = computed(() =>
  recognitionModeActive.value &&
  !recognitionRecoveryPrompt.value &&
  !effectivePageWorkflow.value.can_edit_text &&
  !isProcessingSave.value &&
  !recognitionInFlight.value
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
      : recognitionEngineUnavailableMessage('gemini')
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
    if (recognitionDraftDirty.value || groundTruthCommitPending.value) {
      return {
        eyebrow: 'Text Review',
        title: 'Save ground truth',
        hint: 'Save this reviewed text as ground truth before moving on.',
        recommendedAction: 'commit',
      }
    }
    if (effectivePageWorkflow.value.can_edit_text) {
      if (effectivePageWorkflow.value.current_revision_is_ground_truth) {
        return {
          eyebrow: 'Text Review',
          title: 'Ground truth saved',
          hint: 'This page is saved as reviewed ground truth.',
          recommendedAction: null,
        }
      }
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
    return recognitionEngineUnavailableMessage(recognitionEngine.value)
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
  if (recognitionModeActive.value && (recognitionDraftDirty.value || groundTruthCommitPending.value)) {
    return 'Save the current text as ground truth for this page (S).'
  }
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
  return 'Download the digitized manuscript in PAGE-XML format, and as [image, text] pairs for fine-tuning OCR models.'
})
const goToLayoutModeButtonTitle = computed(() => {
  const busyReason = getBusyDisabledReason('Open Page Layout')
  if (busyReason) return busyReason
  return 'Return to Page Layout for this page.'
})
const primaryTopBarActionHidden = computed(() =>
  layoutModeActive.value &&
  !recognitionModeRequiresLayoutReturn.value &&
  !hasUnsavedLayoutChanges.value &&
  !effectivePageWorkflow.value.needs_recognition
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
  return 'Choose the method used the next time this page is read. Changing this does not alter the current text.'
})

const confirmReplaceWithNewReading = () => {
  if (!rereadWillOverwriteExistingText.value) return true
  const warning = pageWorkflow.has_ground_truth
    ? 'This page is saved as ground truth. Re-reading will overwrite the current text for this page. Continue?'
    : recognitionDraftDirty.value
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

const applyReaderCapabilities = (payload = {}) => {
  const readers = payload.readers || {}
  ;['local', 'gemini'].forEach((engine) => {
    const reader = readers[engine] || {}
    readerCapabilities[engine].available = engine === 'local' ? true : Boolean(reader.available)
    readerCapabilities[engine].label = reader.label || RECOGNITION_READER_LABELS[engine]
    readerCapabilities[engine].unavailableReason = reader.unavailableReason || null
  })
  if (!readerCapabilities.gemini.available && !readerCapabilities.gemini.unavailableReason) {
    readerCapabilities.gemini.unavailableReason = 'Gemini is not configured on this server.'
  }
  if (!isRecognitionEngineAvailable(recognitionEngine.value)) {
    setRecognitionEngineSilently(normalizeRecognitionEngine(payload.defaultEngine))
  }
}

const refreshReaderCapabilities = async () => {
  try {
    const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/recognition/readers`)
    if (!response.ok) return
    const data = await response.json()
    applyReaderCapabilities(data)
  } catch (err) {
    console.warn('Reader capability refresh failed', err)
  }
}

const showReaderSwitchNotice = (message) => {
  readerSwitchNotice.value = message || ''
  if (readerSwitchNoticeTimeoutId !== null) {
    window.clearTimeout(readerSwitchNoticeTimeoutId)
    readerSwitchNoticeTimeoutId = null
  }
  if (readerSwitchNotice.value) {
    readerSwitchNoticeTimeoutId = window.setTimeout(() => {
      readerSwitchNotice.value = ''
      readerSwitchNoticeTimeoutId = null
    }, 7000)
  }
}

const shouldOfferRecognitionRecovery = (payload = {}, attemptedEngine = recognitionEngine.value) => {
  const failedEngine = normalizeRecognitionEngine(payload.failedEngine || payload.recognitionEngine || attemptedEngine)
  return failedEngine === 'gemini' && payload.retryable !== false
}

const showRecognitionRecovery = (payload = {}, attemptedEngine = recognitionEngine.value) => {
  const message = payload.error || 'Gemini could not return usable text for this page.'
  recognitionRecoveryPrompt.value = {
    failedEngine: normalizeRecognitionEngine(payload.failedEngine || attemptedEngine),
    errorCode: payload.errorCode || null,
    fallbackEngines: Array.isArray(payload.fallbackEngines) ? payload.fallbackEngines : ['local'],
    message,
  }
}

const dismissRecognitionRecovery = () => {
  recognitionRecoveryPrompt.value = null
}

const retryRecognitionAfterFailure = async (engine) => {
  const nextEngine = normalizeRecognitionEngine(engine)
  recognitionRecoveryPrompt.value = null
  if (!isRecognitionEngineAvailable(nextEngine)) {
    const message = recognitionEngineUnavailableMessage(nextEngine)
    error.value = message
    alert(message)
    return false
  }
  setRecognitionEngineSilently(nextEngine)
  showReaderSwitchNotice(readerSelectionNotice(nextEngine))
  return recognizeCurrentPage({ focusAfter: true })
}

const logRecognitionReaderSelection = (eventName, details = {}) => {
  console.info('[recognition] reader selection', {
    event: eventName,
    manuscript: localManuscriptName.value,
    page: localCurrentPage.value,
    reader: recognitionEngine.value,
    recognitionModeActive: recognitionModeActive.value,
    hasUnsavedLayoutChanges: hasUnsavedLayoutChanges.value,
    recognitionDraftDirty: recognitionDraftDirty.value,
    workflowState: effectivePageWorkflow.value.state,
    ...details,
  })
}

const readerSelectionNotice = (engine) => {
  const label = recognitionEngineOptionLabel(engine)
  if (recognitionModeActive.value && rereadWillOverwriteExistingText.value) {
    return `${label} selected. Current text is unchanged. Click the Replace With New Reading button to overwrite this page.`
  }
  if (recognitionModeActive.value && effectivePageWorkflow.value.needs_recognition) {
    return `${label} selected. Click Read Text to read this page.`
  }
  return `${label} will be used the next time you read a page.`
}

const setRecognitionEngineSilently = (engine) => {
  suppressRecognitionEngineWatcher = true
  recognitionEngine.value = normalizeRecognitionEngine(engine)
  localStorage.setItem('recognition_engine', recognitionEngine.value)
  nextTick(() => {
    suppressRecognitionEngineWatcher = false
  })
}

const applyActiveLearningState = (payload = {}) => {
  activeLearningMeta.code = payload.code || 'idle'
  activeLearningStatus.value = payload.label || 'Not updating right now'
  activeLearningMeta.label = activeLearningStatus.value
  activeLearningMeta.active_checkpoint_id = payload.active_checkpoint_id || 'base'
  activeLearningMeta.active_checkpoint_path = payload.active_checkpoint_path || null
  activeLearningMeta.pending_jobs = Array.isArray(payload.pending_jobs) ? payload.pending_jobs : []
  activeLearningMeta.needs_rebase = Boolean(payload.needs_rebase)
}

const activeLearningNeedsFastPolling = () =>
  activeLearningMeta.pending_jobs.length > 0 ||
  ['queued', 'running', 'paused_for_ocr'].includes(activeLearningMeta.code)

const currentActiveLearningPollDelay = () =>
  activeLearningNeedsFastPolling() ? activeLearningPollDelayMs.active : activeLearningPollDelayMs.idle

const scheduleNextActiveLearningPoll = (delayMs = currentActiveLearningPollDelay()) => {
  if (activeLearningPollTimeoutId !== null) {
    window.clearTimeout(activeLearningPollTimeoutId)
  }
  activeLearningPollTimeoutId = window.setTimeout(async () => {
    activeLearningPollTimeoutId = null
    await refreshActiveLearningState()
    scheduleNextActiveLearningPoll()
  }, delayMs)
}

const rescheduleActiveLearningPolling = (delayMs = currentActiveLearningPollDelay()) => {
  if (activeLearningPollTimeoutId === null) return
  scheduleNextActiveLearningPoll(delayMs)
}

const refreshActiveLearningState = async () => {
  if (!localManuscriptName.value || activeLearningPollInFlight) return
  activeLearningPollInFlight = true
  try {
    const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/manuscript/${localManuscriptName.value}/active-learning`)
    if (!response.ok) return
    const data = await response.json()
    applyActiveLearningState(data)
  } catch (err) {
    console.warn('Active learning state refresh failed', err)
  } finally {
    activeLearningPollInFlight = false
  }
}

const startActiveLearningPolling = () => {
  if (activeLearningPollTimeoutId !== null) return
  scheduleNextActiveLearningPoll(currentActiveLearningPollDelay())
}

const stopActiveLearningPolling = () => {
  if (activeLearningPollTimeoutId === null) return
  window.clearTimeout(activeLearningPollTimeoutId)
  activeLearningPollTimeoutId = null
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
  pageWorkflow.latest_supervised_commit_revision_number = payload?.latest_supervised_commit_revision_number ?? null
  pageWorkflow.review_status = payload?.review_status || 'layout_ready_no_text'
  pageWorkflow.has_ground_truth = Boolean(payload?.has_ground_truth)
  pageWorkflow.ground_truth_revision_number = payload?.ground_truth_revision_number ?? null
  pageWorkflow.current_revision_is_ground_truth = Boolean(payload?.current_revision_is_ground_truth)
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

const handleRecognitionEngineChange = async (newEngine, previousEngine) => {
  const normalizedNewEngine = normalizeRecognitionEngine(newEngine)
  const normalizedPreviousEngine = normalizeRecognitionEngine(previousEngine)
  if (normalizedNewEngine !== newEngine) {
    logRecognitionReaderSelection('normalize_reader_value', {
      requestedReader: newEngine,
      normalizedReader: normalizedNewEngine,
    })
    setRecognitionEngineSilently(normalizedNewEngine)
    return
  }

  localStorage.setItem('recognition_engine', normalizedNewEngine)
  if (suppressRecognitionEngineWatcher) return
  logRecognitionReaderSelection('reader_selected', {
    previousReader: normalizedPreviousEngine,
    selectedReader: normalizedNewEngine,
  })

  if (!isRecognitionEngineAvailable(normalizedNewEngine)) {
    const message = recognitionEngineUnavailableMessage(normalizedNewEngine)
    error.value = message
    showReaderSwitchNotice(message)
    logRecognitionReaderSelection('reader_unavailable_reverted', {
      requestedReader: normalizedNewEngine,
      revertedReader: normalizedPreviousEngine,
      reason: message,
    })
    setRecognitionEngineSilently(normalizedPreviousEngine)
    return
  }

  showReaderSwitchNotice(readerSelectionNotice(normalizedNewEngine))
}

watch(recognitionEngine, (newEngine, previousEngine) => {
  if (suppressRecognitionEngineWatcher) return
  handleRecognitionEngineChange(newEngine, previousEngine)
})

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
const readingDirectionHoverMinStrokeRaw = computed(() =>
  clamp(pageMedianNeighborDistanceRaw.value * 0.35, 8, 30)
)

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

const backendAssetUrl = (path) => {
    if (!path) return "";
    if (/^(https?:|data:)/i.test(path)) return path;
    const baseUrl = (import.meta.env.VITE_BACKEND_URL || "").replace(/\/$/, "");
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;
    return `${baseUrl}${normalizedPath}`;
}

const activeLineImagePreview = computed(() => {
    if (!focusedLineId.value) return null;
    const preview = lineImagePreviews.value[String(focusedLineId.value)];
    return preview?.imageUrl ? preview : null;
})

const getLinePreviewDisplayWidthPx = (preview) => {
    if (!preview) return null;
    const naturalWidth = Number(preview.imageWidth);
    const maxWidth = Math.max(240, Math.min(760, scaledWidth.value - 8));
    if (!Number.isFinite(naturalWidth) || naturalWidth <= 0) return Math.min(420, maxWidth);
    return clamp(naturalWidth * scaleFactor, 260, maxWidth);
}

const clampFloaterLeft = (left, width) => {
    const maxLeft = Math.max(4, scaledWidth.value - width - 4);
    return clamp(left, 4, maxLeft);
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
    const previewWidth = getLinePreviewDisplayWidthPx(activeLineImagePreview.value);

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
        const targetWidth = previewWidth || INPUT_WIDTH_PX;
        style.width = `${targetWidth}px`;

        if (polyCenterX > pageCenterX) {
            style.left = `${clampFloaterLeft(scaleX(minX) - targetWidth - 10, targetWidth)}px`;
        } else {
            style.left = `${clampFloaterLeft(scaleX(maxX) + 10, targetWidth)}px`;
        }
    } else {
        const targetWidth = previewWidth || scaleX(rawWidth);
        style.top = `${scaleY(maxY) + 5}px`;
        style.left = `${clampFloaterLeft(scaleX(minX), targetWidth)}px`;
        style.width = `${targetWidth}px`;
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
    const width = getLinePreviewDisplayWidthPx(activeLineImagePreview.value) || ((Math.max(...xs) - Math.min(...xs)) * scaleFactor);
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
  recognitionRecoveryPrompt.value = null
  modifications.value = []
  readingDirectionAnnotations.value = {}
  readingDirectionDraft.value = null
  readingDirectionOverlayLogKeys.clear()
  syncSavedReadingDirectionAnnotationsSnapshot()
  
  Object.keys(textlineLabels).forEach(k => delete textlineLabels[k])
  replaceLocalRecognitionData({}, {})
  pagePolygons.value = {}
  lineImagePreviews.value = {}
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
    lineImagePreviews.value = data.lineImagePreviews || {}
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
    loadReadingDirectionAnnotationsFromPageData(data.readingDirectionAnnotations)
    syncSavedTextboxLabelsSnapshot(graph.value?.nodes?.length || 0)
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
    const message = recognitionEngineUnavailableMessage(recognitionEngine.value)
    error.value = message
    if (!suppressErrors) alert(message)
    return false
  }

  recognitionInFlight.value = true
  error.value = null
  recognitionRecoveryPrompt.value = null
  const attemptedEngine = recognitionEngine.value
  try {
    const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/recognize-text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        manuscript: localManuscriptName.value,
        page: localCurrentPage.value,
        recognitionEngine: attemptedEngine,
      }),
    })
    if (!response.ok) {
      let payload = {}
      try {
        payload = await response.json()
      } catch (parseError) {
        payload = {}
      }
      const requestError = new Error(payload.error || 'Could not read the page')
      requestError.payload = payload
      requestError.failedEngine = payload.failedEngine || payload.recognitionEngine || attemptedEngine
      throw requestError
    }

    const data = await response.json()
    replaceLocalRecognitionData(data.text || {}, data.confidences || {})
    if (data.activeLearning) applyActiveLearningState(data.activeLearning)
    rescheduleActiveLearningPolling(activeLearningPollDelayMs.immediate)
    if (data.pageWorkflow) applyPageWorkflow(data.pageWorkflow)
    sortLinesTopToBottom()
    if (focusAfter && sortedLineIds.value.length > 0) {
      activateInput(sortedLineIds.value[0])
    }
    return true
  } catch (err) {
    error.value = err.message
    if (!suppressErrors) {
      const payload = err.payload || { error: err.message, failedEngine: err.failedEngine || attemptedEngine }
      if (shouldOfferRecognitionRecovery(payload, attemptedEngine)) {
        showRecognitionRecovery(payload, attemptedEngine)
      } else {
        alert(`Could not read the page: ${err.message}`)
      }
    }
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
const getEdgeColor = (edge) => (edge.modified ? '#ffffff' : '#ffffff')
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

const imagePointFromMouseEvent = (event) => {
  if (!svgOverlayRef.value) return null
  const rect = svgOverlayRef.value.getBoundingClientRect()
  return [
    (event.clientX - rect.left) / scaleFactor,
    (event.clientY - rect.top) / scaleFactor,
  ]
}

const distancePointToSegmentRaw = (px, py, x1, y1, x2, y2) => {
  const denom = Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1
  const ratio = Math.max(0, Math.min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / denom))
  const nx = x1 + ratio * (x2 - x1)
  const ny = y1 + ratio * (y2 - y1)
  return Math.hypot(px - nx, py - ny)
}

const closestPointOnSegmentRaw = (px, py, x1, y1, x2, y2) => {
  const denom = Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1
  const ratio = clamp(((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / denom, 0, 1)
  const x = x1 + ratio * (x2 - x1)
  const y = y1 + ratio * (y2 - y1)
  return {
    point: [x, y],
    ratio,
    distance: Math.hypot(px - x, py - y),
  }
}

const cross2d = (ax, ay, bx, by) => (ax * by) - (ay * bx)

const segmentIntersectionRaw = (startA, endA, startB, endB) => {
  const ax = startA[0]
  const ay = startA[1]
  const rx = endA[0] - ax
  const ry = endA[1] - ay
  const bx = startB[0]
  const by = startB[1]
  const sx = endB[0] - bx
  const sy = endB[1] - by
  const denominator = cross2d(rx, ry, sx, sy)
  if (Math.abs(denominator) < 1e-9) return null

  const qpx = bx - ax
  const qpy = by - ay
  const cutRatio = cross2d(qpx, qpy, sx, sy) / denominator
  const textlineRatio = cross2d(qpx, qpy, rx, ry) / denominator
  if (cutRatio < 0 || cutRatio > 1 || textlineRatio < 0 || textlineRatio > 1) return null

  const point = [ax + (cutRatio * rx), ay + (cutRatio * ry)]
  return {
    cutPoint: point,
    textlinePoint: point,
    cutRatio,
    textlineRatio,
    distance: 0,
  }
}

const closestSegmentPairRaw = (cutStart, cutEnd, lineStart, lineEnd) => {
  const intersection = segmentIntersectionRaw(cutStart, cutEnd, lineStart, lineEnd)
  if (intersection) return intersection

  const candidates = []
  const addCandidate = (candidate) => {
    if (candidate && Number.isFinite(candidate.distance)) candidates.push(candidate)
  }

  const cutToLineStart = closestPointOnSegmentRaw(lineStart[0], lineStart[1], cutStart[0], cutStart[1], cutEnd[0], cutEnd[1])
  addCandidate({
    cutPoint: cutToLineStart.point,
    textlinePoint: lineStart,
    cutRatio: cutToLineStart.ratio,
    distance: cutToLineStart.distance,
  })

  const cutToLineEnd = closestPointOnSegmentRaw(lineEnd[0], lineEnd[1], cutStart[0], cutStart[1], cutEnd[0], cutEnd[1])
  addCandidate({
    cutPoint: cutToLineEnd.point,
    textlinePoint: lineEnd,
    cutRatio: cutToLineEnd.ratio,
    distance: cutToLineEnd.distance,
  })

  const lineToCutStart = closestPointOnSegmentRaw(cutStart[0], cutStart[1], lineStart[0], lineStart[1], lineEnd[0], lineEnd[1])
  addCandidate({
    cutPoint: cutStart,
    textlinePoint: lineToCutStart.point,
    cutRatio: 0,
    textlineRatio: lineToCutStart.ratio,
    distance: lineToCutStart.distance,
  })

  const lineToCutEnd = closestPointOnSegmentRaw(cutEnd[0], cutEnd[1], lineStart[0], lineStart[1], lineEnd[0], lineEnd[1])
  addCandidate({
    cutPoint: cutEnd,
    textlinePoint: lineToCutEnd.point,
    cutRatio: 1,
    textlineRatio: lineToCutEnd.ratio,
    distance: lineToCutEnd.distance,
  })

  return candidates.reduce((best, candidate) => (
    !best || candidate.distance < best.distance ? candidate : best
  ), null)
}

const findTextlineCutMatches = (cutStart, cutEnd) => {
  if (!graphIsLoaded.value) return []
  const threshold = Math.max(pageMedianNeighborDistanceRaw.value * 0.75, 16)
  const strokeBbox = expandBoundingBox({
    minX: Math.min(cutStart[0], cutEnd[0]),
    minY: Math.min(cutStart[1], cutEnd[1]),
    maxX: Math.max(cutStart[0], cutEnd[0]),
    maxY: Math.max(cutStart[1], cutEnd[1]),
  }, threshold)
  const matches = []

  textlineGeometryIndex.value.entries.forEach((lineGeometry) => {
    if (!boundingBoxesOverlap(lineGeometry.bbox, strokeBbox)) return

    let bestMatch = null
    const considerMatch = (candidate) => {
      if (!candidate || !Number.isFinite(candidate.distance)) return
      if (!bestMatch || candidate.distance < bestMatch.distance) bestMatch = candidate
    }

    lineGeometry.nodePoints.forEach(({ point }) => {
      const closest = closestPointOnSegmentRaw(point[0], point[1], cutStart[0], cutStart[1], cutEnd[0], cutEnd[1])
      considerMatch({
        cutPoint: closest.point,
        textlinePoint: point,
        cutRatio: closest.ratio,
        distance: closest.distance,
      })
    })

    lineGeometry.edgeSegments.forEach((segment) => {
      considerMatch(closestSegmentPairRaw(
        cutStart,
        cutEnd,
        segment.start,
        segment.end,
      ))
    })

    if (bestMatch && bestMatch.distance <= threshold) {
      matches.push({
        lineId: lineGeometry.lineId,
        cutPoint: bestMatch.cutPoint,
        cutRatio: bestMatch.cutRatio,
        distance: bestMatch.distance,
        midpoint: bestMatch.textlinePoint || bestMatch.cutPoint,
      })
    }
  })

  return matches.sort((a, b) => (a.cutRatio - b.cutRatio) || (a.distance - b.distance))
}

const localCutSegmentForTextlineMatch = (match, cutStart, cutEnd, strokeLength, useOriginalStroke) => {
  if (useOriginalStroke) {
    return {
      cutStart,
      cutEnd,
      cutMidpoint: [
        (cutStart[0] + cutEnd[0]) / 2,
        (cutStart[1] + cutEnd[1]) / 2,
      ],
    }
  }

  const directionX = (cutEnd[0] - cutStart[0]) / strokeLength
  const directionY = (cutEnd[1] - cutStart[1]) / strokeLength
  const halfLength = Math.min(
    strokeLength / 2,
    Math.max(pageMedianNeighborDistanceRaw.value * 0.8, 12),
  )
  const midpoint = match.midpoint || match.cutPoint
  return {
    cutStart: [
      midpoint[0] - (directionX * halfLength),
      midpoint[1] - (directionY * halfLength),
    ],
    cutEnd: [
      midpoint[0] + (directionX * halfLength),
      midpoint[1] + (directionY * halfLength),
    ],
    cutMidpoint: midpoint,
  }
}

const nearestTextlineIdForImagePoint = (point) => {
  if (!point || !graphIsLoaded.value) return null
  const [x, y] = point
  let best = { id: null, distance: Infinity }
  const maxDistance = Math.max(pageMedianNeighborDistanceRaw.value * 2.5, 35)
  const searchBbox = expandBoundingBox({ minX: x, minY: y, maxX: x, maxY: y }, maxDistance)

  textlineGeometryIndex.value.entries.forEach((lineGeometry) => {
    if (!boundingBoxesOverlap(lineGeometry.bbox, searchBbox)) return

    lineGeometry.nodePoints.forEach(({ point: nodePoint }) => {
      const distance = Math.hypot(x - nodePoint[0], y - nodePoint[1])
      if (distance < best.distance) best = { id: lineGeometry.lineId, distance }
    })

    lineGeometry.edgeSegments.forEach((segment) => {
      const distance = distancePointToSegmentRaw(x, y, segment.start[0], segment.start[1], segment.end[0], segment.end[1])
      if (distance < best.distance) best = { id: lineGeometry.lineId, distance }
    })
  })

  return best.distance <= maxDistance ? best.id : null
}

const commitReadingDirectionStroke = (start, endPoint, options = {}) => {
  const dx = endPoint[0] - start[0]
  const dy = endPoint[1] - start[1]
  const strokeLength = Math.hypot(dx, dy)
  if (!Number.isFinite(strokeLength) || strokeLength < 4) return []

  const excludedLineIds = options.excludedLineIds || null
  const midpoint = [(start[0] + endPoint[0]) / 2, (start[1] + endPoint[1]) / 2]
  const readingDirection = [-dy / strokeLength, dx / strokeLength]
  let textlineMatches = findTextlineCutMatches(start, endPoint)
  if (textlineMatches.length === 0) {
    const fallbackTextlineId = nearestTextlineIdForImagePoint(midpoint)
    if (fallbackTextlineId !== null && textlines.value[fallbackTextlineId]) {
      textlineMatches = [{
        lineId: String(fallbackTextlineId),
        cutPoint: midpoint,
        cutRatio: 0.5,
        distance: 0,
        midpoint,
      }]
    }
  }
  if (excludedLineIds?.size) {
    textlineMatches = textlineMatches.filter((match) => !excludedLineIds.has(String(match.lineId)))
  }
  if (textlineMatches.length === 0) return []

  const committedLineIds = []
  const nextAnnotations = { ...readingDirectionAnnotations.value }
  const useOriginalStroke = textlineMatches.length === 1
  const updatedAt = new Date().toISOString()
  textlineMatches.forEach((match) => {
    const textlineId = String(match.lineId)
    if (!textlines.value[textlineId]) return
    const localCut = localCutSegmentForTextlineMatch(match, start, endPoint, strokeLength, useOriginalStroke)
    const annotation = {
      annotation_id: String(textlineId),
      frontend_line_id: String(textlineId),
      component_node_indices: [...textlines.value[textlineId]].sort((a, b) => a - b),
      cut_start: localCut.cutStart,
      cut_end: localCut.cutEnd,
      cut_midpoint: localCut.cutMidpoint,
      reading_direction: readingDirection,
      source: 'user_cross_cut',
      updated_at: updatedAt,
    }
    const previousAnnotation = readingDirectionAnnotations.value[String(textlineId)] || null
    nextAnnotations[String(textlineId)] = annotation
    modifications.value.push({
      type: 'reading_direction',
      lineId: String(textlineId),
      previousAnnotation,
    })
    committedLineIds.push(textlineId)
  })
  if (committedLineIds.length > 0) {
    readingDirectionAnnotations.value = nextAnnotations
  }
  return committedLineIds
}

const handleReadingDirectionHover = (point) => {
  if (!point) return
  if (!readingDirectionHoverStartPoint) {
    readingDirectionHoverStartPoint = point
    readingDirectionDraft.value = {
      cut_start: point,
      cut_end: point,
    }
    return
  }

  const start = readingDirectionHoverStartPoint
  const strokeLength = Math.hypot(point[0] - start[0], point[1] - start[1])
  readingDirectionDraft.value = {
    cut_start: start,
    cut_end: point,
  }
  if (!Number.isFinite(strokeLength) || strokeLength < readingDirectionHoverMinStrokeRaw.value) return

  const committedLineIds = commitReadingDirectionStroke(start, point, {
    excludedLineIds: readingDirectionHoverAnnotatedLineIds,
  })
  committedLineIds.forEach((lineId) => readingDirectionHoverAnnotatedLineIds.add(String(lineId)))
  readingDirectionHoverStartPoint = point
}

const flushPendingReadingDirectionHover = () => {
  if (readingDirectionHoverRafId !== null) {
    window.cancelAnimationFrame(readingDirectionHoverRafId)
    readingDirectionHoverRafId = null
  }
  const point = pendingReadingDirectionHoverPoint
  pendingReadingDirectionHoverPoint = null
  if (!point || !layoutModeActive.value || !isOKeyPressed.value || recognitionModeActive.value) return
  handleReadingDirectionHover(point)
}

const scheduleReadingDirectionHover = (point) => {
  pendingReadingDirectionHoverPoint = point
  if (readingDirectionHoverRafId !== null) return

  readingDirectionHoverRafId = window.requestAnimationFrame(() => {
    readingDirectionHoverRafId = null
    const queuedPoint = pendingReadingDirectionHoverPoint
    pendingReadingDirectionHoverPoint = null
    if (!queuedPoint || !layoutModeActive.value || !isOKeyPressed.value || recognitionModeActive.value) return
    handleReadingDirectionHover(queuedPoint)
  })
}

const handleSvgMouseDown = (event) => {
  if (!layoutModeActive.value || !isOKeyPressed.value || recognitionModeActive.value) return
  event.preventDefault()
  event.stopPropagation()
  suppressNextBackgroundClick = true
  resetSelection()
}

const handleSvgMouseUp = (event) => {
  if (!layoutModeActive.value || !isOKeyPressed.value || recognitionModeActive.value) return
  event.preventDefault()
  event.stopPropagation()
  suppressNextBackgroundClick = true
}

const onEdgeClick = (edge, event) => {
  if (isAKeyPressed.value || isDKeyPressed.value || isEKeyPressed.value || isOKeyPressed.value || recognitionModeActive.value) return
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
    if (suppressNextBackgroundClick) {
        suppressNextBackgroundClick = false
        return
    }
    if (isOKeyPressed.value) return
    
    if (layoutModeActive.value && !isAKeyPressed.value && !isDKeyPressed.value && !isEKeyPressed.value && !isOKeyPressed.value) {
        addNode(event.clientX, event.clientY);
        return;
    }
    
    resetSelection();
}

const onNodeClick = (nodeIndex, event) => {
    event.stopPropagation(); 
    if (!layoutModeActive.value || recognitionModeActive.value) return;
    if (isAKeyPressed.value || isDKeyPressed.value || isEKeyPressed.value || isOKeyPressed.value) return;
    
    const existingIndex = selectedNodes.value.indexOf(nodeIndex);
    if (existingIndex !== -1) selectedNodes.value.splice(existingIndex, 1);
    else selectedNodes.value.length < 2 ? selectedNodes.value.push(nodeIndex) : (selectedNodes.value = [nodeIndex]);
}

const onNodeRightClick = (nodeIndex, event) => {
    if (layoutModeActive.value && !isAKeyPressed.value && !isDKeyPressed.value && !isEKeyPressed.value && !isOKeyPressed.value) {
        event.preventDefault(); 
        deleteNode(nodeIndex);
    }
}

const handleSvgMouseMove = (event) => {
  if (!svgOverlayRef.value || !layoutModeActive.value) return
  const { left, top } = svgOverlayRef.value.getBoundingClientRect()
  const mouseX = event.clientX - left
  const mouseY = event.clientY - top

  if (isOKeyPressed.value) {
    event.preventDefault()
    scheduleReadingDirectionHover([mouseX / scaleFactor, mouseY / scaleFactor])
    tempEndPoint.value = null
    return
  }

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
  if (isOKeyPressed.value) flushPendingReadingDirectionHover()
  resetReadingDirectionHoverState()
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
  if (key === 'escape' && !e.repeat && !isInput && layoutModeActive.value && isOKeyPressed.value) {
    e.preventDefault()
    isOKeyPressed.value = false
    resetReadingDirectionHoverState()
    return
  }
  if (key === 'escape' && recognitionModeActive.value && isInput) { e.preventDefault(); focusedLineId.value = null; return }
  
  // NEW: Visibility Hotkey 'v'
  if (key === 'v' && !isInput) {
      isVKeyPressed.value = true
      return
  }

  if (layoutModeActive.value && !isInput) {
      if (key === 'q') {
        e.preventDefault()
        if (!isOKeyPressed.value) {
          isOKeyPressed.value = true
          resetReadingDirectionHoverState()
          hoveredNodesForMST.clear()
          resetSelection()
        }
        return
      }
      if (isOKeyPressed.value) {
        e.preventDefault()
        return
      }
      if (e.repeat) return
      if (key === 'e') { e.preventDefault(); isEKeyPressed.value = true; return }
      if (key === 'd') { e.preventDefault(); isDKeyPressed.value = true; resetSelection(); return }
      if (key === 'a') { e.preventDefault(); isAKeyPressed.value = true; hoveredNodesForMST.clear(); resetSelection(); return }
  }
}

const handleGlobalKeyUp = (e) => {
  const key = e.key.toLowerCase()
  if (key === 'v') { isVKeyPressed.value = false }

  if (layoutModeActive.value) {
      if (key === 'q') {
        flushPendingReadingDirectionHover()
        isOKeyPressed.value = false
        resetReadingDirectionHoverState()
        return
      }
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

const handleWindowBlur = () => {
  if (!isOKeyPressed.value) return
  flushPendingReadingDirectionHover()
  isOKeyPressed.value = false
  resetReadingDirectionHoverState()
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
  } else if (mod.type === 'reading_direction') {
    if (mod.previousAnnotation) {
      readingDirectionAnnotations.value = {
        ...readingDirectionAnnotations.value,
        [String(mod.lineId)]: mod.previousAnnotation,
      }
    } else {
      const nextAnnotations = { ...readingDirectionAnnotations.value }
      delete nextAnnotations[String(mod.lineId)]
      readingDirectionAnnotations.value = nextAnnotations
    }
  }
}


const resetModifications = () => {
  resetWorkingGraph()
  try {
    const restored = {}
    JSON.parse(savedReadingDirectionAnnotationsSnapshot.value || '[]').forEach((annotation) => {
      const normalized = normalizeReadingDirectionAnnotation(annotation)
      if (normalized) restored[String(normalized.annotation_id)] = normalized
    })
    readingDirectionAnnotations.value = restored
  } catch (err) {
    readingDirectionAnnotations.value = {}
  }
  readingDirectionDraft.value = null
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

const saveModifications = async (background = false, options = {}) => {
  const forceLayoutSave = options?.forceLayoutSave === true
  const saveScope = options?.saveScope || determineSaveScope({ background, forceLayoutSave })
  const numNodes = workingGraph.nodes.length
  const labelsToSend = buildTextboxLabelsPayload(numNodes)
  const readingDirectionAnnotationsToSend = buildReadingDirectionAnnotationsPayload()
  const dummyTextlineLabels = new Array(numNodes).fill(-1);
  const textContentForSave = saveScope === 'text_only'
    ? { ...localTextContent }
    : hasUnsavedLayoutChanges.value
    ? {}
    : {}
  const requestBody = {
    graph: workingGraph, 
    modifications: modifications.value,
    textlineLabels: dummyTextlineLabels, 
    textboxLabels: labelsToSend,
    readingDirectionAnnotations: readingDirectionAnnotationsToSend,
    textContent: textContentForSave,
    runRecognition: false,
    recognitionEngine: recognitionEngine.value, // <--- NEW PARAMETER
    activeLearningEnabled: activeLearningEnabled.value,
    saveIntent: background ? 'draft' : 'commit',
    saveScope,
  }
  console.info('[page-save] submitting', {
    manuscript: localManuscriptName.value,
    page: localCurrentPage.value,
    background,
    saveScope,
    recognitionModeActive: recognitionModeActive.value,
    hasUnsavedLayoutChanges: hasUnsavedLayoutChanges.value,
    recognitionDraftDirty: recognitionDraftDirty.value,
    groundTruthCommitPending: groundTruthCommitPending.value,
    reviewStatus: pageWorkflow.review_status,
  })
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
    rescheduleActiveLearningPolling(activeLearningPollDelayMs.immediate)
    if (data.pageWorkflow) applyPageWorkflow(data.pageWorkflow)

    if (saveScope === 'layout') {
      modifications.value = []
      syncSavedTextboxLabelsSnapshot(numNodes)
      syncSavedReadingDirectionAnnotationsSnapshot()
    }
    if (saveScope === 'text_only') {
      recognitionDraftDirty.value = false
    }
    error.value = null
    console.info('[page-save] completed', {
      manuscript: localManuscriptName.value,
      page: localCurrentPage.value,
      saveScope,
      saveIntent: background ? 'draft' : 'commit',
      reviewStatus: data?.pageWorkflow?.review_status,
      queuedJobIds: data?.activeLearningQueuedJobIds || [],
    })
  } catch (err) {
    error.value = err.message
    console.warn('[page-save] failed', {
      manuscript: localManuscriptName.value,
      page: localCurrentPage.value,
      saveScope,
      error: err.message,
    })
    throw err
  }
}

const saveCurrentPageForCurrentMode = async ({ background = false, forceLayoutSave = false } = {}) => {
  const saveScope = determineSaveScope({ background, forceLayoutSave })
  if (saveScope === 'layout' && !hasUnsavedLayoutChanges.value && !forceLayoutSave) {
    console.info('[page-save] skipped unchanged layout save', {
      manuscript: localManuscriptName.value,
      page: localCurrentPage.value,
      background,
      saveScope,
    })
    return true
  }
  if (saveScope === 'text_only' && (!currentPageHasTextContent.value || effectivePageWorkflow.value.needs_recognition)) {
    console.warn('[page-save] skipped invalid text review save', {
      manuscript: localManuscriptName.value,
      page: localCurrentPage.value,
      background,
      saveScope,
      hasText: currentPageHasTextContent.value,
      needsRecognition: effectivePageWorkflow.value.needs_recognition,
    })
    return false
  }
  await saveModifications(background, { forceLayoutSave, saveScope })
  return true
}


const requestSwitchToRecognition = async (forceRecognition = false) => {
    const shouldForceRecognition = forceRecognition === true
    const requiresSavedLayoutForRecognition = pageWorkflowRequiresLayoutMode(effectivePageWorkflow.value)
    if (recognitionInFlight.value) return;
    if (recognitionModeActive.value && !shouldForceRecognition && !hasUnsavedLayoutChanges.value) return;

    isProcessingSave.value = true;
    try {
        if (hasUnsavedLayoutChanges.value || requiresSavedLayoutForRecognition) {
            await saveCurrentPageForCurrentMode({ forceLayoutSave: true }); 
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


const navigateToPage = (page) => {
  pendingPageEntryPreference.value = recognitionModeActive.value
    ? PAGE_ENTRY_RECOGNITION_IF_COMMITTED_TEXT
    : PAGE_ENTRY_LAYOUT
  emit('page-changed', page)
}

const discardCurrentPageLocalChanges = () => {
  if (hasUnsavedLayoutChanges.value) {
    resetModifications()
  }
  if (recognitionDraftDirty.value) {
    recognitionDraftDirty.value = false
  }
}

const navigationSavePrompt = () => {
  if (hasUnsavedLayoutChanges.value) {
    if (pageWorkflow.has_ground_truth) {
      return 'Save layout changes before leaving? This page has saved ground truth, and layout changes may require re-reviewing the text.'
    }
    return 'Save layout changes before leaving?'
  }
  if (recognitionModeActive.value && recognitionDraftDirty.value) {
    return 'Save this page as ground truth before leaving?'
  }
  if (recognitionModeActive.value && groundTruthCommitPending.value) {
    if (pageWorkflow.review_status === 'draft_saved' || pageWorkflow.review_status === 'ground_truth_with_draft_changes') {
      return 'This page has draft text that is not saved as ground truth. Save as ground truth before leaving?'
    }
    return 'Save this page as ground truth before leaving?'
  }
  return ''
}

const navigateToPageWithPolicy = async (targetPage, { exitAction = 'prompt' } = {}) => {
  if (!targetPage || targetPage === localCurrentPage.value || isProcessingSave.value || recognitionInFlight.value) return

  if (exitAction === 'save') {
    isProcessingSave.value = true
    try {
      const saved = await saveCurrentPageForCurrentMode()
      if (saved) navigateToPage(targetPage)
    } catch (err) {
      alert(`Save failed, navigation cancelled: ${err.message}`)
    } finally {
      isProcessingSave.value = false
    }
    return
  }

  const promptMessage = navigationSavePrompt()
  if (promptMessage) {
    if (confirm(promptMessage)) {
      isProcessingSave.value = true
      try {
        const saved = await saveCurrentPageForCurrentMode()
        if (saved) navigateToPage(targetPage)
      } catch (err) {
        alert(`Save failed, navigation cancelled: ${err.message}`)
      } finally {
        isProcessingSave.value = false
      }
    } else {
      discardCurrentPageLocalChanges()
      navigateToPage(targetPage)
    }
    return
  }

  if (!recognitionModeActive.value && groundTruthCommitPending.value) {
    if (confirm('This page has text that is not saved as ground truth. Leave without saving it as ground truth?')) {
      navigateToPage(targetPage)
    }
    return
  }

  navigateToPage(targetPage)
}

const previousPage = () => {
    const idx = localPageList.value.indexOf(localCurrentPage.value)
    if (idx > 0) navigateToPageWithPolicy(localPageList.value[idx - 1], { exitAction: 'prompt' })
}
const nextPage = () => {
    const idx = localPageList.value.indexOf(localCurrentPage.value)
    if (idx < localPageList.value.length - 1) navigateToPageWithPolicy(localPageList.value[idx + 1], { exitAction: 'prompt' })
}

const handlePageSelect = (event) => {
    const selectedPage = event.target.value;
    if (selectedPage === localCurrentPage.value) return;
    
    navigateToPageWithPolicy(selectedPage, { exitAction: 'prompt' });
}

// NEW: Save current page logic (no nav)
const saveCurrentPage = async () => {
  if (loading.value || isProcessingSave.value || recognitionInFlight.value || recognitionModeRequiresLayoutReturn.value) return
  isProcessingSave.value = true
  try {
    await saveCurrentPageForCurrentMode()
    // Optional: Flash a small 'Saved' toast
  } catch (err) { alert(`Save failed: ${err.message}`) } 
  finally { isProcessingSave.value = false }
}

const saveAndGoNext = async () => {
  if (loading.value || isProcessingSave.value || recognitionInFlight.value || recognitionModeRequiresLayoutReturn.value) return
  const idx = localPageList.value.indexOf(localCurrentPage.value)
  if (idx < localPageList.value.length - 1) {
    await navigateToPageWithPolicy(localPageList.value[idx + 1], { exitAction: 'save' })
  } else {
    isProcessingSave.value = true
    try {
      await saveCurrentPageForCurrentMode()
      alert('Last page saved!')
    } catch (err) { alert(`Save failed: ${err.message}`) }
    finally { isProcessingSave.value = false }
  }
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
                await saveCurrentPageForCurrentMode({ background: true });
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

    await refreshReaderCapabilities()
    await fetchPageData(props.manuscriptName, localCurrentPage.value, false, false)
    await refreshActiveLearningState()
    startActiveLearningPolling()
  }
  window.addEventListener('resize', scheduleBrowserZoomLevelUpdate, { passive: true })
  window.addEventListener('wheel', handleCtrlWheelZoom, { passive: true })
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', scheduleBrowserZoomLevelUpdate, { passive: true })
  }
  zoomPollIntervalId = window.setInterval(updateBrowserZoomLevel, 500)
  window.addEventListener('keydown', handleGlobalKeyDown)
  window.addEventListener('keyup', handleGlobalKeyUp)
  window.addEventListener('blur', handleWindowBlur)
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
  window.removeEventListener('blur', handleWindowBlur)
  if(autoSaveInterval.value) clearInterval(autoSaveInterval.value);
  if (readerSwitchNoticeTimeoutId !== null) {
    window.clearTimeout(readerSwitchNoticeTimeoutId)
    readerSwitchNoticeTimeoutId = null
  }
  stopActiveLearningPolling()
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

.workflow-toggle-meta {
  font-size: 0.66rem;
  color: #c9d0d7;
  line-height: 1.1;
  white-space: nowrap;
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

.recognition-engine-note {
  color: #ffd89f;
  font-size: 0.68rem;
  line-height: 1.25;
  max-width: 260px;
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
.reading-direction-overlay {
  pointer-events: none;
}
.reading-direction-arrow {
  pointer-events: none;
}
.reading-direction-arrow path {
  fill: #fff2a8;
  stroke: rgba(52, 38, 0, 0.62);
  stroke-width: 0.8;
  paint-order: stroke fill;
  vector-effect: non-scaling-stroke;
}

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

.input-floater.has-line-preview .line-input {
    box-sizing: border-box;
}

.line-image-preview {
    box-sizing: border-box;
    width: 100%;
    margin-top: 4px;
    padding: 4px;
    background: rgba(0, 0, 0, 0.78);
    border: 1px solid rgba(255, 213, 79, 0.65);
    border-radius: 4px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.45);
}

.line-image-preview-img {
    display: block;
    width: 100%;
    height: auto;
    max-height: 140px;
    object-fit: contain;
    user-select: none;
    pointer-events: none;
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

.recognition-recovery-backdrop {
  position: absolute;
  inset: 0;
  z-index: 10001;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: rgba(0, 0, 0, 0.45);
}

.recognition-recovery-card {
  width: min(460px, 100%);
  padding: 22px 24px;
  border-radius: 8px;
  background: rgba(18, 18, 18, 0.96);
  border: 1px solid rgba(255, 255, 255, 0.14);
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.5);
  color: #fff;
  text-align: left;
}

.recognition-recovery-card h3 {
  margin: 10px 0 8px;
  font-size: 1.08rem;
}

.recognition-recovery-card p {
  margin: 0 0 18px;
  color: #d6d6d6;
  line-height: 1.45;
}

.recognition-recovery-badge {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 0 9px;
  border-radius: 999px;
  border: 1px solid rgba(255, 216, 159, 0.38);
  background: rgba(255, 216, 159, 0.12);
  color: #ffd89f;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.recognition-recovery-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

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
  flex: 1 1 230px;
  min-width: 0;
  width: auto;
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

.orientation-help-visual {
  position: relative;
  overflow: hidden;
}

.orientation-help-key {
  color: #ffb74d;
  font-family: monospace;
  font-size: 1.4rem;
  font-weight: bold;
  padding: 6px 10px;
  border: 1px solid #555;
  border-radius: 4px;
  background: #242424;
  z-index: 1;
}

.orientation-help-line {
  position: absolute;
  width: 72%;
  height: 3px;
  background: #ffd54f;
  border-radius: 999px;
  transform: rotate(-24deg);
  box-shadow: 0 0 8px rgba(255, 213, 79, 0.45);
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
