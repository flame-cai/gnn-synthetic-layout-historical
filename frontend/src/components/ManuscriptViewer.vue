<template>
  <div class="manuscript-viewer">
    
    <!-- TOP RAIL: Navigation & Global Actions -->
    <div class="top-bar">
      <div class="top-bar-left">
        <button class="nav-btn secondary" @click="$emit('back')" aria-label="Back">
          <i class="fa fa-arrow-left" aria-hidden="true"></i>
        </button>
        <span class="page-title">{{ manuscriptNameForDisplay }} <span class="divider">/</span></span>
        
        <!-- NEW: Page Dropdown -->
        <select class="page-select" :value="currentPageForDisplay" @change="handlePageSelect">
           <option v-for="pg in localPageList" :key="pg" :value="pg">{{ pg }}</option>
        </select>

        <div class="separator"></div>

        <div class="action-group">
           <button class="nav-btn" @click="previousPage" :disabled="loading || isProcessingSave || isFirstPage">
            Previous
          </button>
          <button class="nav-btn" @click="nextPage" :disabled="loading || isProcessingSave || isLastPage">
            Next
          </button>
        </div>

        <div class="separator"></div>

        <div class="action-group zoom-controls">
          <button class="action-btn zoom-btn" @click="zoomOut" :disabled="loading || !imageData" title="Zoom out">-</button>
          <span class="zoom-readout" title="Current zoom">{{ zoomPercent }}%</span>
          <button class="action-btn zoom-btn" @click="zoomIn" :disabled="loading || !imageData" title="Zoom in">+</button>
          <button class="action-btn zoom-btn" @click="resetZoom" :disabled="loading || !imageData" title="Fit to screen">Fit</button>
        </div>
      </div>

      <div class="top-bar-right">
        <template v-if="recognitionModeActive">
          <div class="separator"></div>

          <div class="action-group recognition-controls-group">
            <div class="center-control-group auto-recog-group">
              <label class="toggle-switch">
                <input type="checkbox" v-model="autoRecogEnabled">
                <span class="slider"></span>
              </label>
              <span class="center-control-label auto-recog-icon" title="Auto-Recognize" aria-label="Auto-Recognize">
                <i class="fa-solid fa-glasses" aria-hidden="true"></i>
              </span>
              <div v-if="autoRecogEnabled" class="auto-recog-options">
                <select v-model="recognitionEngine" class="auto-recog-select" title="Auto-Recognition Engine">
                  <option value="local">Local OCR</option>
                  <option value="gemini">Gemini</option>
                </select>
                <input v-if="recognitionEngine === 'gemini'" type="password" v-model="geminiKey" placeholder="API Key" class="auto-recog-key" title="Enter Gemini API Key" />
              </div>
            </div>

            <div class="recognition-toolbar">
              <label class="toolbar-field">
                <span class="toolbar-label">
                  <i class="fas fa-language toolbar-label-icon" aria-hidden="true"></i>
                  <span>Script</span>
                </span>
                <select v-model="recognitionScriptMode" class="toolbar-select">
                  <option value="plain">Roman</option>
                  <option value="devanagari">Devanagari</option>
                  <option value="grantha">Grantha</option>
                </select>
              </label>
              <button
                class="toolbar-icon-btn"
                :class="{ active: wordCutModeActive, delete: wordCutDeleteMode }"
                @click="toggleWordCutMode"
                title="Word cut mode"
                aria-label="Word cut mode"
              >
                ✂
              </button>
              <button
                v-if="wordCutModeActive"
                class="toolbar-icon-btn preview-btn"
                :class="{ active: wordPreviewVisible }"
                @click="handlePreviewToggle"
                title="Preview words"
                aria-label="Preview words"
              >
                <i class="fas fa-eye" aria-hidden="true"></i>
              </button>
            </div>
          </div>
        </template>

        <div class="separator"></div>

        <div class="action-group">
          <button
            v-if="!recognitionModeActive"
            class="action-btn"
            @click="saveOverlay"
            :disabled="loading || isProcessingSave"
            title="Save image with graph to backend"
          >
            Export Image
          </button>
          <button
            v-if="!recognitionModeActive"
            class="action-btn"
            @click="downloadResults"
            :disabled="loading || isProcessingSave"
          >
            Download PAGE-XMLs
          </button>
          <button
            v-if="!recognitionModeActive"
            class="action-btn"
            @click="runHeuristic"
            :disabled="loading"
          >
            Auto-Link
          </button>

          <button class="action-btn" @click="saveCurrentPage" :disabled="loading || isProcessingSave">
            Save (S)
          </button>
          <button class="action-btn primary" @click="saveAndGoNext" :disabled="loading || isProcessingSave">
            Save & Next
          </button>
        </div>
      </div>
    </div>

    <!-- MAIN CONTENT: Visualization Area -->
    <div class="visualization-container" ref="container" :style="visualizationContainerStyle" @wheel="handleViewerWheel">
      
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
        class="visualization-stage"
        :style="visualizationStageStyle"
      >
        <div
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
            ref="pageImageRef"
          />
          <div
            v-else
            class="placeholder-image"
            :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }"
          >
            No image available
          </div>

        <!-- NEW: Wrapper to hide everything when 'v' is pressed -->
        <div :style="{ opacity: recognitionModeActive ? 1 : (isVKeyPressed ? 0 : 1), transition: 'opacity 0.1s' }">
            <img
              v-if="recognitionModeActive && focusedBinaryOverlayUrl && focusedLineBounds"
              :src="focusedBinaryOverlayUrl"
              class="focused-binary-overlay"
              :style="getFocusedBinaryOverlayStyle()"
              alt="Focused binary line overlay"
            />
            
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
                stroke-width="4"
              />

              <circle
                v-for="(node, nodeIndex) in workingGraph.nodes"
                :key="`node-${nodeIndex}`"
                :cx="scaleX(node.x)"
                :cy="scaleY(node.y)"
                :r="getNodeRadius(nodeIndex)"
                :fill="getNodeColor(nodeIndex)"
                @contextmenu.stop.prevent="onNodeRightClick(nodeIndex, $event)"
              />
            </svg>

            <!-- SVG Polygon Layer (Visible in Recognition Mode) -->
            <svg
              v-if="recognitionModeActive"
              class="graph-overlay is-visible"
              :width="scaledWidth"
              :height="scaledHeight"
              :viewBox="`0 0 ${scaledWidth} ${scaledHeight}`"
              :style="{ cursor: recognitionOverlayCursor }"
              @click.stop="handleRecognitionOverlayClick"
              @mousemove="handleRecognitionOverlayMove"
              @mouseleave="handleRecognitionOverlayLeave"
              ref="recognitionOverlayRef"
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
                fill="transparent"
                :stroke="wordCutModeActive ? 'rgba(196, 196, 196, 0.9)' : '#00e5ff'"
                :stroke-width="wordCutModeActive ? 1.5 : 0"
                class="polygon-active"
              />

              <template v-if="wordCutModeActive && focusedLineBounds">
                <line
                  v-for="(splitX, splitIndex) in getWordCutsForLine(focusedLineId)"
                  :key="`split-${focusedLineId}-${splitIndex}`"
                  :x1="scaleX(splitX)"
                  :y1="scaleY(focusedLineBounds.minY)"
                  :x2="scaleX(splitX)"
                  :y2="scaleY(focusedLineBounds.maxY)"
                  class="word-cut-line"
                  :class="{ deletable: hoveredWordCutIndex === splitIndex && wordCutDeleteMode }"
                />
                <line
                  v-if="wordCutHoverX !== null"
                  :x1="scaleX(wordCutHoverX)"
                  :y1="scaleY(focusedLineBounds.minY)"
                  :x2="scaleX(wordCutHoverX)"
                  :y2="scaleY(focusedLineBounds.maxY)"
                  class="word-cut-line hover"
                  :class="{ delete: wordCutDeleteMode }"
                />
              </template>
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
                    @keydown.ctrl.p.prevent.stop="handlePreviewToggle"
                    @keydown.meta.p.prevent.stop="handlePreviewToggle"
                    @keydown.up.prevent="focusAdjacentLine(-1)"
                    @keydown.down.prevent="focusAdjacentLine(1)"
                    placeholder="Type text here..."
                    :style="{ 
                        fontSize: `${getRecognitionInputFontSizePx(focusedLineId)}px`,
                        fontFamily:
                          recognitionScriptMode === 'grantha'
                            ? `'Noto Sans Grantha', 'Anek Tamil', serif`
                            : recognitionScriptMode === 'devanagari'
                              ? `'Noto Sans Devanagari', 'Nirmala UI', 'Mangal', serif`
                              : 'monospace',
                        height: `${getRecognitionInputHeightPx(focusedLineId)}px`,
                        maxHeight: `${getRecognitionInputHeightPx(focusedLineId)}px`,
                        lineHeight: `${Math.max(getRecognitionInputHeightPx(focusedLineId) - (getRecognitionInputPaddingPx(focusedLineId) * 2), getRecognitionInputFontSizePx(focusedLineId) + 2)}px`,
                        padding: `${getRecognitionInputPaddingPx(focusedLineId)}px ${Math.max(10, getRecognitionInputPaddingPx(focusedLineId))}px`,
                        marginBottom: '0px' 
                    }"
                />
                <div
                    v-if="(recognitionScriptMode === 'grantha' || recognitionScriptMode === 'devanagari') && focusedLineId"
                    class="transliteration-preview-strip"
                    :style="{
                      fontSize: `${getRecognitionPreviewFontSizePx(focusedLineId)}px`,
                      padding: `1px ${Math.max(8, getRecognitionInputPaddingPx(focusedLineId) - 1)}px 3px`
                    }"
                >
                    {{ getRecognitionTranscription(localTextContent[focusedLineId] || '') }}
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
                            fontSize: `${getRecognitionInputFontSizePx(focusedLineId)}px`
                        }"
                    >{{ char }}</span>
                </div>
                <div
                    v-if="wordCutModeActive && wordPreviewVisible && focusedWordPreviewItems.length > 0"
                    class="word-preview-strip"
                >
                    <div
                      v-for="(item, index) in focusedWordPreviewItems"
                      :key="`word-preview-${index}`"
                      class="word-preview-card"
                    >
                      <img
                        :src="item.image"
                        class="word-preview-image"
                        :style="{
                          height: `${getActivePreviewHeight()}px`,
                          maxHeight: `${getActivePreviewHeight()}px`
                        }"
                      />
                      <span class="word-preview-label">{{ item.label }}</span>
                    </div>
                </div>
            </div>
          </div> <!-- End of Visibility Wrapper -->
        </div>
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
              <span
                v-if="recognitionScriptMode === 'grantha' || recognitionScriptMode === 'devanagari'"
                style="color:var(--success); font-size:0.8rem;"
              >
                {{ recognitionScriptMode === 'grantha' ? 'Grantha Transliteration ON' : 'Devanagari Transliteration ON' }}
              </span>
            </div>
           </div>
           <div class="instructions-container">
             <h3>Recognition Mode</h3>
             <p>Transcribe line-by-line on the focused binary overlay. Auto-save is active every 20 seconds.</p>
             <ul>
               <li><strong>Navigate:</strong> Use <code>↑</code> or <code>↓</code> to move between lines.</li>
               <li><strong>Save:</strong> Press <code>Ctrl+S</code> or use the top-bar save button.</li>
               <li v-if="recognitionScriptMode === 'devanagari'"><strong>Devanagari:</strong> Type with Devanagari selected and use the preview strip to confirm the roman reading.</li>
               <li><strong>OCR:</strong> Turn on the bolt toggle to use either <code>Local OCR</code> or <code>Gemini</code> during save.</li>
               <li><strong>Word Cuts:</strong> Click the scissor to enter cut mode, click inside the active line to place cuts, and hold <code>Delete</code> to remove a cut with the hand cursor.</li>
               <li><strong>Preview:</strong> In word-cut mode, use the eye button or <code>Ctrl+P</code> to preview the split word images.</li>
             </ul>
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
import Sanscript from '@indic-transliteration/sanscript'
import { generateLayoutGraph } from '../layout-analysis-utils/LayoutGraphGenerator.js'
// Assuming these imports exist in your project structure
import edgeWebm from '../tutorial/_edge.webm'
import regionWebm from '../tutorial/_textbox.webm'
import nodeWebm from '../tutorial/_node.webm'
import { handleInput as handleDevanagariInput } from '../typing-utils/devanagariInputUtils.js'
import { handleGranthaInput, reverseTransliterateGrantha } from '../typing-utils/granthaInputUtils.js'

const props = defineProps({
  manuscriptName: { type: String, default: null },
  pageName: { type: String, default: null },
})

const emit = defineEmits(['page-changed', 'back'])

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
    nextTick(() => ensureFocusedRecognitionLine())
  }
  isPanelCollapsed.value = false
}
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

// Key states
const isDKeyPressed = ref(false)
const isAKeyPressed = ref(false)
const isEKeyPressed = ref(false) 
const isVKeyPressed = ref(false) // NEW for Visibility

const hoveredNodesForMST = reactive(new Set())
const container = ref(null)
const svgOverlayRef = ref(null)
const recognitionOverlayRef = ref(null)
const pageImageRef = ref(null)
const focusedBinaryOverlayUrl = ref('')
const viewerResizeObserver = ref(null)

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
const normalTextContent = reactive({})
const wordModeTextContent = reactive({})
const pagePolygons = ref({}) 
const focusedLineId = ref(null)
const sortedLineIds = ref([])
const autoRecogEnabled = ref(false)
const recognitionEngine = ref(localStorage.getItem('recognition_engine') || 'local') // NEW
const recognitionScriptMode = ref(localStorage.getItem('recognition_script_mode') || 'plain')
const wordCuts = reactive({})
const wordCutModeActive = ref(false)
const wordCutDeleteMode = ref(false)
const wordCutHoverX = ref(null)
const hoveredWordCutIndex = ref(null)
const wordPreviewVisible = ref(false)
const focusedWordPreviewItems = ref([])
const lastSavedRecognitionSignature = ref('')
const isSyncingTextBuffers = ref(false)

// NEW: Persist keys/settings to local storage
watch(recognitionEngine, (val) => localStorage.setItem('recognition_engine', val))
watch(geminiKey, (val) => localStorage.setItem('gemini_key', val))
watch(recognitionScriptMode, (val) => localStorage.setItem('recognition_script_mode', val))
const localTextConfidence = reactive({}) 
const autoSaveInterval = ref(null) // NEW

const fitScale = ref(1)
const zoomLevel = ref(1)
const containerViewportWidth = ref(1)
const containerViewportHeight = ref(1)
const NODE_HOVER_RADIUS = 7
const EDGE_HOVER_THRESHOLD = 5
const ZOOM_STEP = 0.1
const MIN_ZOOM_LEVEL = 0.5
const MAX_ZOOM_LEVEL = 4

const manuscriptNameForDisplay = computed(() => localManuscriptName.value)
const currentPageForDisplay = computed(() => localCurrentPage.value)
const isFirstPage = computed(() => localPageList.value.indexOf(localCurrentPage.value) === 0)
const isLastPage = computed(() => localPageList.value.indexOf(localCurrentPage.value) === localPageList.value.length - 1)

const scaleFactor = computed(() => fitScale.value * zoomLevel.value)
const zoomPercent = computed(() => Math.round(zoomLevel.value * 100))
const scaledWidth = computed(() => Math.floor(dimensions.value[0] * scaleFactor.value))
const scaledHeight = computed(() => Math.floor(dimensions.value[1] * scaleFactor.value))
const visualizationContainerStyle = computed(() => ({
  justifyContent: scaledWidth.value > containerViewportWidth.value ? 'flex-start' : 'center',
  alignItems: scaledHeight.value > containerViewportHeight.value ? 'flex-start' : 'center',
}))
const visualizationStageStyle = computed(() => ({
  width: `${Math.max(scaledWidth.value, containerViewportWidth.value)}px`,
  height: `${Math.max(scaledHeight.value, containerViewportHeight.value)}px`,
  justifyContent: scaledWidth.value > containerViewportWidth.value ? 'flex-start' : 'center',
  alignItems: scaledHeight.value > containerViewportHeight.value ? 'flex-start' : 'center',
}))
const scaleX = (x) => x * scaleFactor.value
const scaleY = (y) => y * scaleFactor.value
const graphIsLoaded = computed(() => workingGraph.nodes && workingGraph.nodes.length > 0)
const focusedLineBounds = computed(() => {
  if (!focusedLineId.value || !pagePolygons.value[focusedLineId.value]) return null

  const points = pagePolygons.value[focusedLineId.value]
  const xs = points.map(([x]) => x)
  const ys = points.map(([, y]) => y)

  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  }
})
const currentFocusedLineIndex = computed(() => sortedLineIds.value.indexOf(focusedLineId.value))
const hasPreviousLine = computed(() => currentFocusedLineIndex.value > 0)
const hasNextLine = computed(() => currentFocusedLineIndex.value > -1 && currentFocusedLineIndex.value < sortedLineIds.value.length - 1)
const recognitionOverlayCursor = computed(() => {
  if (!recognitionModeActive.value || !wordCutModeActive.value || wordCutHoverX.value === null) return 'default'
  return wordCutDeleteMode.value ? 'pointer' : 'cell'
})

const updateViewerScale = () => {
  if (!container.value || !dimensions.value[0] || !dimensions.value[1]) {
    fitScale.value = 1
    return
  }

  const containerStyles = window.getComputedStyle(container.value)
  const paddingX = (parseFloat(containerStyles.paddingLeft) || 0) + (parseFloat(containerStyles.paddingRight) || 0)
  const paddingY = (parseFloat(containerStyles.paddingTop) || 0) + (parseFloat(containerStyles.paddingBottom) || 0)
  const availableWidth = Math.max(container.value.clientWidth - paddingX, 1)
  const availableHeight = Math.max(container.value.clientHeight - paddingY, 1)
  containerViewportWidth.value = availableWidth
  containerViewportHeight.value = availableHeight

  const fitScaleValue = Math.min(
    availableWidth / dimensions.value[0],
    availableHeight / dimensions.value[1],
  )

  fitScale.value = Math.max(0.05, fitScaleValue)
}

const clampZoomLevel = (value) => Math.min(MAX_ZOOM_LEVEL, Math.max(MIN_ZOOM_LEVEL, value))

const setZoomLevel = (nextZoomLevel) => {
  zoomLevel.value = clampZoomLevel(nextZoomLevel)
}

const zoomIn = () => setZoomLevel(zoomLevel.value + ZOOM_STEP)
const zoomOut = () => setZoomLevel(zoomLevel.value - ZOOM_STEP)
const resetZoom = () => {
  zoomLevel.value = 1
}

const handleViewerWheel = (event) => {
  if (!event.ctrlKey && !event.metaKey) return
  event.preventDefault()
  if (event.deltaY < 0) zoomIn()
  else zoomOut()
}

const getRecognitionTranscription = (value) => {
    const text = value || ''
    if (!text) return ''

    if (recognitionScriptMode.value === 'grantha') {
        return reverseTransliterateGrantha(text)
    }

    if (recognitionScriptMode.value === 'devanagari') {
        return Sanscript.t(text, 'devanagari', 'hk')
    }

    return ''
}


// --- RECOGNITION MODE LOGIC ---

const handleRecognitionInput = (event) => {
    if (event.ctrlKey || event.metaKey) {
        const lowerKey = event.key.toLowerCase()
        if (wordCutModeActive.value && lowerKey === 'p') {
            event.preventDefault()
            toggleWordPreview()
        }
        if (wordCutModeActive.value && lowerKey === 'r') {
            event.preventDefault()
            clearFocusedWordCuts()
        }
        return
    }
    if (event.altKey || !focusedLineId.value) return;

    const textRef = {
        get value() {
            return localTextContent[focusedLineId.value] || ''
        },
        set value(val) {
            localTextContent[focusedLineId.value] = val
        }
    }

    if (recognitionScriptMode.value === 'devanagari') {
        handleDevanagariInput(event, textRef)
        return
    }

    if (recognitionScriptMode.value === 'grantha') {
        handleGranthaInput(event, textRef)
    }
}

const pointsToSvgString = (pts) => {
    if(!pts) return "";
    return pts.map(p => `${scaleX(p[0])},${scaleY(p[1])}`).join(" ");
}

const getPolygonHeight = (lineId) => {
    if (!lineId || !pagePolygons.value[lineId]) return 0
    const ys = pagePolygons.value[lineId].map((point) => point[1])
    return Math.max(...ys) - Math.min(...ys)
}

const getRecognitionInputHeightPx = (lineId) => {
    if (!lineId) return 28
    return Math.max(scaleY(getPolygonHeight(lineId)), 28)
}

const getRecognitionInputFontSizePx = (lineId) => {
    const inputHeight = getRecognitionInputHeightPx(lineId)
    return Math.max(11, Math.min(24, Math.round(inputHeight * 0.4)))
}

const getRecognitionPreviewFontSizePx = (lineId) => {
    return Math.max(9, getRecognitionInputFontSizePx(lineId) - 2)
}

const getRecognitionInputPaddingPx = (lineId) => {
    const inputHeight = getRecognitionInputHeightPx(lineId)
    return Math.max(6, Math.min(14, Math.round(inputHeight * 0.18)))
}

const getActivePreviewHeight = () => {
    if (!focusedLineId.value) return 28
    return getRecognitionInputHeightPx(focusedLineId.value)
}

const getFocusedBinaryOverlayStyle = () => {
    if (!focusedLineBounds.value) return { display: 'none' }

    const width = Math.max(1, Math.ceil(scaleX(focusedLineBounds.value.maxX - focusedLineBounds.value.minX)))
    const height = Math.max(1, Math.ceil(scaleY(focusedLineBounds.value.maxY - focusedLineBounds.value.minY)))

    return {
        position: 'absolute',
        left: `${scaleX(focusedLineBounds.value.minX)}px`,
        top: `${scaleY(focusedLineBounds.value.minY)}px`,
        width: `${width}px`,
        height: `${height}px`,
        pointerEvents: 'none',
        zIndex: 2,
    }
}

const buildFocusedBinaryCanvas = () => {
    if (
      !recognitionModeActive.value ||
      !focusedLineId.value ||
      !focusedLineBounds.value ||
      !pagePolygons.value[focusedLineId.value] ||
      !pageImageRef.value ||
      !imageLoaded.value
    ) {
      return null
    }

    const bounds = focusedLineBounds.value
    const polygon = pagePolygons.value[focusedLineId.value]
    const width = Math.max(1, Math.round(bounds.maxX - bounds.minX))
    const height = Math.max(1, Math.round(bounds.maxY - bounds.minY))
    if (width <= 0 || height <= 0) {
      return null
    }

    const sourceCanvas = document.createElement('canvas')
    sourceCanvas.width = width
    sourceCanvas.height = height
    const sourceCtx = sourceCanvas.getContext('2d', { willReadFrequently: true })
    if (!sourceCtx) {
      return null
    }

    sourceCtx.drawImage(
      pageImageRef.value,
      bounds.minX,
      bounds.minY,
      width,
      height,
      0,
      0,
      width,
      height,
    )

    const sourceData = sourceCtx.getImageData(0, 0, width, height)
    const sourcePixels = sourceData.data

    const maskCanvas = document.createElement('canvas')
    maskCanvas.width = width
    maskCanvas.height = height
    const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true })
    if (!maskCtx) {
      return null
    }

    maskCtx.beginPath()
    polygon.forEach(([x, y], index) => {
      const shiftedX = x - bounds.minX
      const shiftedY = y - bounds.minY
      if (index === 0) maskCtx.moveTo(shiftedX, shiftedY)
      else maskCtx.lineTo(shiftedX, shiftedY)
    })
    maskCtx.closePath()
    maskCtx.fillStyle = '#ffffff'
    maskCtx.fill()

    const maskData = maskCtx.getImageData(0, 0, width, height).data
    const grayscale = new Uint8ClampedArray(width * height)
    const polygonPixels = []

    for (let index = 0; index < grayscale.length; index += 1) {
      const pixelOffset = index * 4
      const grayValue = Math.round(
        (sourcePixels[pixelOffset] * 0.299) +
        (sourcePixels[pixelOffset + 1] * 0.587) +
        (sourcePixels[pixelOffset + 2] * 0.114)
      )
      grayscale[index] = grayValue
      if (maskData[pixelOffset + 3] > 0) {
        polygonPixels.push(grayValue)
      }
    }

    if (polygonPixels.length === 0) {
      return null
    }

    const sortedPolygonPixels = polygonPixels.slice().sort((a, b) => a - b)
    const medianValue = sortedPolygonPixels[Math.floor(sortedPolygonPixels.length / 2)]
    const filledGray = new Uint8ClampedArray(width * height)

    for (let index = 0; index < filledGray.length; index += 1) {
      const pixelOffset = index * 4
      filledGray[index] = maskData[pixelOffset + 3] > 0 ? grayscale[index] : medianValue
    }

    const sortedFilled = Array.from(filledGray).sort((a, b) => a - b)
    const darkCutoff = sortedFilled[Math.max(0, Math.floor(sortedFilled.length * 0.15) - 1)]
    const darkPixels = Array.from(filledGray).filter((value) => value <= darkCutoff)
    const sortedDarkPixels = darkPixels.sort((a, b) => a - b)
    const binaryThreshold = sortedDarkPixels.length > 0
      ? sortedDarkPixels[Math.min(sortedDarkPixels.length - 1, Math.floor(sortedDarkPixels.length * 0.72))]
      : darkCutoff

    const overlayCanvas = document.createElement('canvas')
    overlayCanvas.width = width
    overlayCanvas.height = height
    const overlayCtx = overlayCanvas.getContext('2d')
    if (!overlayCtx) {
      return null
    }

    const overlayImage = overlayCtx.createImageData(width, height)
    const overlayPixels = overlayImage.data

    for (let index = 0; index < filledGray.length; index += 1) {
      const pixelOffset = index * 4
      if (maskData[pixelOffset + 3] === 0) {
        overlayPixels[pixelOffset + 3] = 0
        continue
      }

      const binaryValue = filledGray[index] <= binaryThreshold ? 0 : 255
      overlayPixels[pixelOffset] = binaryValue
      overlayPixels[pixelOffset + 1] = binaryValue
      overlayPixels[pixelOffset + 2] = binaryValue
      overlayPixels[pixelOffset + 3] = 255
    }

    overlayCtx.putImageData(overlayImage, 0, 0)
    return overlayCanvas
}

const buildFocusedBinaryOverlay = () => {
    const overlayCanvas = buildFocusedBinaryCanvas()
    if (!overlayCanvas) {
      focusedBinaryOverlayUrl.value = ''
      return
    }
    focusedBinaryOverlayUrl.value = overlayCanvas.toDataURL('image/png')
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

const ensureFocusedRecognitionLine = () => {
    sortLinesTopToBottom()
    if (focusedLineId.value && pagePolygons.value[focusedLineId.value]) return
    if (sortedLineIds.value.length > 0) {
        activateInput(sortedLineIds.value[0])
    }
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
    const scaledGap = Math.max(5, Math.round(8 * scaleFactor.value))

    const style = {
        position: 'absolute',
        minHeight: `${Math.max(scaleY(rawHeight), 28)}px`,
        overflow: 'visible',
        zIndex: 100
    };

    if (isVertical) {
        const pageCenterX = dimensions.value[0] / 2;
        const polyCenterX = minX + (rawWidth / 2);
        
        const INPUT_WIDTH_PX = Math.max(250, Math.round(250 * scaleFactor.value)); 
        
        style.top = `${scaleY(minY)}px`; 
        style.width = `${INPUT_WIDTH_PX}px`;

        if (polyCenterX > pageCenterX) {
            style.left = `${scaleX(minX) - INPUT_WIDTH_PX - scaledGap}px`;
        } else {
            style.left = `${scaleX(maxX) + scaledGap}px`;
        }
    } else {
        style.top = `${scaleY(maxY) + scaledGap}px`;
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
    const width = (Math.max(...xs) - Math.min(...xs)) * scaleFactor.value;
    let calcSize = (width / charCount) * 1.8;
    calcSize = Math.max(14, Math.min(calcSize, 40));
    return `${calcSize}px`;
}

const activateInput = (lineId) => {
    if (!lineId || !pagePolygons.value[lineId]) return
    focusedLineId.value = lineId;
    wordPreviewVisible.value = false
    focusedWordPreviewItems.value = []
    wordCutDeleteMode.value = false
    nextTick(() => {
        if(activeInput.value) {
            activeInput.value.focus();
        }
    });
}

const handleInputBlur = () => {
    setTimeout(() => {
       if (recognitionModeActive.value) {
         ensureFocusedRecognitionLine()
         return
       }
       if (wordCutModeActive.value) return;
       if (document.activeElement && document.activeElement.tagName === 'INPUT') return;
       if (document.activeElement && document.activeElement.closest('.input-floater')) return;
       focusedLineId.value = null; 
       wordPreviewVisible.value = false
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

const focusAdjacentLine = (direction) => {
    if (!focusedLineId.value || sortedLineIds.value.length === 0) return
    const currentIndex = sortedLineIds.value.indexOf(focusedLineId.value)
    if (currentIndex === -1) return

    const nextIndex = currentIndex + direction
    if (nextIndex < 0 || nextIndex >= sortedLineIds.value.length) return
    activateInput(sortedLineIds.value[nextIndex])
}

const getWordCutsForLine = (lineId) => {
    if (!lineId) return []
    return wordCuts[lineId] || []
}

const replaceReactiveTextMap = (target, source) => {
    Object.keys(target).forEach((key) => delete target[key])
    Object.assign(target, source || {})
}

const hasSavedWordModeStateForLine = (lineId) => {
    const hasWordText = Object.prototype.hasOwnProperty.call(wordModeTextContent, lineId)
    const hasWordCuts = getWordCutsForLine(lineId).length > 0
    return hasWordText || hasWordCuts
}

const getWordModeDisplayText = (lineId) => {
    if (hasSavedWordModeStateForLine(lineId)) {
      return wordModeTextContent[lineId] || ''
    }
    return normalTextContent[lineId] || ''
}

const loadLocalTextContentForCurrentMode = () => {
    isSyncingTextBuffers.value = true
    try {
      if (wordCutModeActive.value) {
        const lineIds = new Set([
          ...Object.keys(normalTextContent),
          ...Object.keys(wordModeTextContent),
          ...Object.keys(wordCuts),
        ])
        const nextWordModeBuffer = {}
        lineIds.forEach((lineId) => {
          nextWordModeBuffer[lineId] = getWordModeDisplayText(lineId)
        })
        replaceReactiveTextMap(localTextContent, nextWordModeBuffer)
      } else {
        replaceReactiveTextMap(localTextContent, normalTextContent)
      }
    } finally {
      isSyncingTextBuffers.value = false
    }
}

const syncLocalTextContentToActiveStore = () => {
    if (isSyncingTextBuffers.value) return
    if (wordCutModeActive.value) {
      replaceReactiveTextMap(wordModeTextContent, localTextContent)
    } else {
      replaceReactiveTextMap(normalTextContent, localTextContent)
    }
}

const setWordCutsForLine = (lineId, cuts) => {
    if (!lineId) return
    const uniqueCuts = Array.from(new Set((cuts || []).map((cut) => Math.round(cut)))).sort((a, b) => a - b)
    wordCuts[lineId] = uniqueCuts
}

const syncWordLabelSegmentsForLine = (lineId) => {
    if (!lineId) return

    const desiredSegments = Math.max(getWordCutsForLine(lineId).length + 1, 1)
    const currentValue = localTextContent[lineId] || ''
    let segments = currentValue.split('_')

    if (segments.length < desiredSegments) {
      while (segments.length < desiredSegments) {
        segments.push('')
      }
    } else if (segments.length > desiredSegments) {
      const preservedSegments = segments.slice(0, desiredSegments - 1)
      const mergedTail = segments.slice(desiredSegments - 1).join('')
      segments = [...preservedSegments, mergedTail]
    }

    const nextValue = segments.join('_')
    if (nextValue !== currentValue) {
      localTextContent[lineId] = nextValue
    }
}

const syncWordLabelSegmentsForAllLines = () => {
    const lineIds = new Set([
      ...Object.keys(localTextContent),
      ...Object.keys(wordCuts),
    ])

    lineIds.forEach((lineId) => syncWordLabelSegmentsForLine(lineId))
}

const toggleWordCutMode = () => {
    syncLocalTextContentToActiveStore()
    wordCutModeActive.value = !wordCutModeActive.value
    if (wordCutModeActive.value) {
      ensureFocusedRecognitionLine()
    }
    loadLocalTextContentForCurrentMode()
    if (wordCutModeActive.value) {
      syncWordLabelSegmentsForAllLines()
    }
    wordCutDeleteMode.value = false
    wordPreviewVisible.value = false
    focusedWordPreviewItems.value = []
    wordCutHoverX.value = null
    hoveredWordCutIndex.value = null
}

const clearFocusedWordCuts = () => {
    if (!focusedLineId.value) return
    wordCuts[focusedLineId.value] = []
    syncWordLabelSegmentsForLine(focusedLineId.value)
    wordPreviewVisible.value = false
    focusedWordPreviewItems.value = []
    hoveredWordCutIndex.value = null
}

const syncHoveredWordCutIndex = (pageX) => {
    const cuts = getWordCutsForLine(focusedLineId.value)
    if (cuts.length === 0) {
        hoveredWordCutIndex.value = null
        return
    }

    const pageThreshold = 10 / scaleFactor.value
    const nearestIndex = cuts.findIndex((cut) => Math.abs(cut - pageX) <= pageThreshold)
    hoveredWordCutIndex.value = nearestIndex === -1 ? null : nearestIndex
}

const handleRecognitionOverlayMove = (event) => {
    if (!wordCutModeActive.value || !focusedLineBounds.value || !recognitionOverlayRef.value) return

    const rect = recognitionOverlayRef.value.getBoundingClientRect()
    const pageX = (event.clientX - rect.left) / scaleFactor.value
    const pageY = (event.clientY - rect.top) / scaleFactor.value

    if (
      pageX < focusedLineBounds.value.minX ||
      pageX > focusedLineBounds.value.maxX ||
      pageY < focusedLineBounds.value.minY ||
      pageY > focusedLineBounds.value.maxY
    ) {
      wordCutHoverX.value = null
      hoveredWordCutIndex.value = null
      return
    }

    wordCutHoverX.value = pageX
    syncHoveredWordCutIndex(pageX)
}

const handleRecognitionOverlayLeave = () => {
    wordCutHoverX.value = null
    hoveredWordCutIndex.value = null
}

const handleRecognitionOverlayClick = (event) => {
    if (!wordCutModeActive.value || !focusedLineId.value || !focusedLineBounds.value || !recognitionOverlayRef.value) return

    const rect = recognitionOverlayRef.value.getBoundingClientRect()
    const pageX = (event.clientX - rect.left) / scaleFactor.value
    const pageY = (event.clientY - rect.top) / scaleFactor.value

    if (
      pageX < focusedLineBounds.value.minX ||
      pageX > focusedLineBounds.value.maxX ||
      pageY < focusedLineBounds.value.minY ||
      pageY > focusedLineBounds.value.maxY
    ) {
      return
    }

    const cuts = [...getWordCutsForLine(focusedLineId.value)]
    const pageThreshold = 10 / scaleFactor.value

    if (wordCutDeleteMode.value) {
      const cutIndex = cuts.findIndex((cut) => Math.abs(cut - pageX) <= pageThreshold)
      if (cutIndex !== -1) {
        cuts.splice(cutIndex, 1)
        setWordCutsForLine(focusedLineId.value, cuts)
        syncWordLabelSegmentsForLine(focusedLineId.value)
      }
    } else {
      cuts.push(pageX)
      setWordCutsForLine(focusedLineId.value, cuts)
      syncWordLabelSegmentsForLine(focusedLineId.value)
    }

    syncHoveredWordCutIndex(pageX)

    if (wordPreviewVisible.value) {
      focusedWordPreviewItems.value = buildWordPreviewItems(focusedLineId.value)
    }
  }

const buildWordPreviewItems = (lineId) => {
    if (!lineId || !focusedLineBounds.value) return []

    const width = Math.max(1, Math.round(focusedLineBounds.value.maxX - focusedLineBounds.value.minX))
    const height = Math.max(1, Math.round(focusedLineBounds.value.maxY - focusedLineBounds.value.minY))
    const cuts = getWordCutsForLine(lineId)
      .filter((cut) => cut > focusedLineBounds.value.minX && cut < focusedLineBounds.value.maxX)
      .map((cut) => cut - focusedLineBounds.value.minX)
    const boundaries = [0, ...cuts, width]
    const labels = (localTextContent[lineId] || '')
      .split('_')
      .map((label) => label.trim())

    const binaryCanvas = buildFocusedBinaryCanvas()
    if (!binaryCanvas) return []

    const previewItems = []
    for (let index = 0; index < boundaries.length - 1; index += 1) {
      const sliceWidth = Math.round(boundaries[index + 1] - boundaries[index])
      if (sliceWidth <= 0) continue

      const sliceCanvas = document.createElement('canvas')
      sliceCanvas.width = sliceWidth
      sliceCanvas.height = height
      const sliceContext = sliceCanvas.getContext('2d')
      if (!sliceContext) continue

      sliceContext.drawImage(
        binaryCanvas,
        boundaries[index],
        0,
        sliceWidth,
        height,
        0,
        0,
        sliceWidth,
        height,
      )

      previewItems.push({
        image: sliceCanvas.toDataURL('image/png'),
        label: labels[index] || '?',
      })
    }

    return previewItems
}

const toggleWordPreview = () => {
    if (!focusedLineId.value) return
    if (wordPreviewVisible.value) {
      wordPreviewVisible.value = false
      focusedWordPreviewItems.value = []
      return
    }

    focusedWordPreviewItems.value = buildWordPreviewItems(focusedLineId.value)
    wordPreviewVisible.value = focusedWordPreviewItems.value.length > 0
}

const handlePreviewToggle = () => {
    if (!wordCutModeActive.value) return
    ensureFocusedRecognitionLine()
    toggleWordPreview()
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
    const x = (clientX - rect.left) / scaleFactor.value;
    const y = (clientY - rect.top) / scaleFactor.value;
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
      imageLoaded.value = false;
  }

  error.value = null
  modifications.value = []
  
  Object.keys(textlineLabels).forEach(k => delete textlineLabels[k])
  Object.keys(localTextContent).forEach(k => delete localTextContent[k])
  Object.keys(localTextConfidence).forEach(k => delete localTextConfidence[k]) 
  Object.keys(wordCuts).forEach(k => delete wordCuts[k])
  pagePolygons.value = {}
  sortedLineIds.value = []
  focusedLineId.value = null
  wordCutModeActive.value = false
  wordCutDeleteMode.value = false
  wordCutHoverX.value = null
  hoveredWordCutIndex.value = null
  wordPreviewVisible.value = false
  focusedWordPreviewItems.value = []
  Object.keys(normalTextContent).forEach(k => delete normalTextContent[k])
  Object.keys(wordModeTextContent).forEach(k => delete wordModeTextContent[k])

  try {
    const response = await fetch(
      `${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscript}/${page}`
    )
    if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch page data')
    const data = await response.json()

    dimensions.value = data.dimensions
    
    if (data.image) imageData.value = data.image;
    else imageLoaded.value = false;
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
        Object.assign(normalTextContent, data.textContent);
    }
    if (data.wordTextContent) {
        Object.assign(wordModeTextContent, data.wordTextContent);
    }
    if (data.textConfidences) {
        Object.assign(localTextConfidence, data.textConfidences);
    }
    if (data.wordCuts) {
        Object.entries(data.wordCuts).forEach(([lineId, cuts]) => {
            setWordCutsForLine(lineId, cuts)
        })
    }

    loadLocalTextContentForCurrentMode()
    resetWorkingGraph()
    sortLinesTopToBottom()
    lastSavedRecognitionSignature.value = buildRecognitionSaveSignature()
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
  const edgeCount = nodeEdgeCounts.value[nodeIndex]
  if (edgeCount < 2) return '#f44336'
  if (edgeCount === 2) return '#4CAF50'
  return '#2196F3'
}

const getNodeRadius = (nodeIndex) => {
  if (layoutModeActive.value && isEKeyPressed.value) {
    return (hoveredTextlineId.value === nodeToTextlineMap.value[nodeIndex]) ? 7 : 7
  }
  if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return 7
  return nodeEdgeCounts.value[nodeIndex] < 2 ? 7 : 7
}
const getEdgeColor = (edge) => (edge.modified ? '#f44336' : 'var(--viewer-text-primary)')

const resetSelection = () => {}

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
}

const handleSvgMouseLeave = () => {
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
  if ((e.ctrlKey || e.metaKey) && !e.repeat && (key === '=' || key === '+' || key === '-' || key === '_' || key === '0')) {
    e.preventDefault()
    e.stopPropagation()
    if (key === '0') resetZoom()
    else if (key === '-' || key === '_') zoomOut()
    else zoomIn()
    return
  }
  if ((e.ctrlKey || e.metaKey) && key === 's' && !e.repeat) {
    e.preventDefault()
    e.stopPropagation()
    saveCurrentPage()
    return
  }
  if (wordCutModeActive.value && (e.ctrlKey || e.metaKey) && key === 'p') {
    e.preventDefault()
    e.stopPropagation()
    handlePreviewToggle()
    return
  }
  if (wordCutModeActive.value && (e.ctrlKey || e.metaKey) && key === 'r') {
    e.preventDefault()
    clearFocusedWordCuts()
    return
  }

  if (key === 's' && !e.repeat && !isInput) {
    e.preventDefault()
    saveCurrentPage()
    return
  }
  
  if (key === 'w' && !e.repeat && !isInput) { e.preventDefault(); setMode('layout'); return }
  if (key === 't' && !e.repeat && !isInput) { e.preventDefault(); requestSwitchToRecognition(); return }
  
  // NEW: Visibility Hotkey 'v'
  if (key === 'v' && !isInput && !recognitionModeActive.value) {
      isVKeyPressed.value = true
      return
  }

  if (wordCutModeActive.value && key === 'delete') {
      e.preventDefault()
      e.stopPropagation()
      wordCutDeleteMode.value = true
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
  if (key === 'delete') { wordCutDeleteMode.value = false }

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

const buildTextContentForSave = () => {
  return Object.fromEntries(
    Object.entries(localTextContent).map(([lineId, value]) => {
      const textValue = value || ''
      return [lineId, wordCutModeActive.value ? textValue.replaceAll('_', '') : textValue]
    })
  )
}

const buildWordTextContentForSave = () => (
  wordCutModeActive.value
    ? Object.fromEntries(Object.entries(localTextContent).map(([lineId, value]) => [lineId, value || '']))
    : null
)

const buildWordCutsForSave = () => (
  wordCutModeActive.value
    ? Object.fromEntries(Object.entries(wordCuts).map(([lineId, cuts]) => [lineId, cuts || []]))
    : null
)

const buildRecognitionSaveSignature = () => JSON.stringify({
  textContent: buildTextContentForSave(),
  wordTextContent: buildWordTextContentForSave(),
  wordCuts: buildWordCutsForSave(),
  wordModeActive: wordCutModeActive.value,
  modifications: modifications.value,
})

const saveModifications = async (background = false) => {
  const numNodes = workingGraph.nodes.length
  const labelsToSend = new Array(numNodes).fill(0) 
  for (const nodeIndex in textlineLabels) {
    if (nodeIndex < numNodes) labelsToSend[nodeIndex] = textlineLabels[nodeIndex]
  }
  const dummyTextlineLabels = new Array(numNodes).fill(-1);
  const textContent = buildTextContentForSave()
  const wordTextContent = buildWordTextContentForSave()
  const wordCutsForSave = buildWordCutsForSave()
  const saveSignature = JSON.stringify({
    textContent,
    wordTextContent,
    wordCuts: wordCutsForSave,
    wordModeActive: wordCutModeActive.value,
    modifications: modifications.value,
  })

  if (background && saveSignature === lastSavedRecognitionSignature.value) {
    return
  }

  const requestBody = {
    graph: workingGraph, 
    modifications: modifications.value,
    recognitionModeActive: recognitionModeActive.value,
    textlineLabels: dummyTextlineLabels, 
    textboxLabels: labelsToSend,
    textContent,
    wordTextContent,
    wordCuts: wordCutsForSave,
    wordModeActive: wordCutModeActive.value,
    runRecognition: autoRecogEnabled.value && !background, // Don't run GNN/AI on auto-save
    apiKey: geminiKey.value,
    recognitionEngine: recognitionEngine.value // <--- NEW PARAMETER
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
        syncLocalTextContentToActiveStore()
    }

    replaceReactiveTextMap(normalTextContent, textContent)
    if (wordCutModeActive.value) {
      replaceReactiveTextMap(wordModeTextContent, wordTextContent || {})
    }

    modifications.value = []
    lastSavedRecognitionSignature.value = buildRecognitionSaveSignature()
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
        if (Object.keys(pagePolygons.value).length === 0) {
            await fetchPageData(localManuscriptName.value, localCurrentPage.value, true);
        }
        setMode('recognition');
        nextTick(() => ensureFocusedRecognitionLine())
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
  if (typeof ResizeObserver !== 'undefined') {
    viewerResizeObserver.value = new ResizeObserver(() => {
      updateViewerScale()
    })
    if (container.value) {
      viewerResizeObserver.value.observe(container.value)
    }
  }
  window.addEventListener('resize', updateViewerScale)
  window.addEventListener('keydown', handleGlobalKeyDown, true)
  window.addEventListener('keyup', handleGlobalKeyUp)
})

onBeforeUnmount(() => {
  viewerResizeObserver.value?.disconnect()
  window.removeEventListener('resize', updateViewerScale)
  window.removeEventListener('keydown', handleGlobalKeyDown, true)
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
        nextTick(() => ensureFocusedRecognitionLine())
    } else {
        focusedBinaryOverlayUrl.value = ''
        wordCutModeActive.value = false
        wordCutDeleteMode.value = false
        wordPreviewVisible.value = false
        focusedWordPreviewItems.value = []
    }
})

watch(localTextContent, () => {
    syncLocalTextContentToActiveStore()
    if (wordPreviewVisible.value && focusedLineId.value) {
      focusedWordPreviewItems.value = buildWordPreviewItems(focusedLineId.value)
    }
}, { deep: true })

watch(wordCuts, () => {
    if (wordPreviewVisible.value && focusedLineId.value) {
      focusedWordPreviewItems.value = buildWordPreviewItems(focusedLineId.value)
    }
}, { deep: true })

watch(
  [focusedLineId, recognitionModeActive, imageLoaded, () => imageData.value],
  () => {
    buildFocusedBinaryOverlay()
  }
)

watch(pagePolygons, () => {
  if (recognitionModeActive.value) {
    nextTick(() => ensureFocusedRecognitionLine())
  }
  buildFocusedBinaryOverlay()
}, { deep: true })

watch(
  [dimensions, imageLoaded, isPanelCollapsed],
  () => {
    nextTick(() => updateViewerScale())
  },
  { deep: true }
)
</script>

<style scoped>
/* Basic Layout */
.manuscript-viewer {
  display: flex; flex-direction: column; height: 100vh; width: 100%;
  background-color: var(--viewer-panel-bg); color: var(--viewer-text-primary); font-family: 'Roboto', sans-serif; overflow: hidden;
}

/* Top Bar */
.top-bar {
  display: flex; justify-content: flex-start; align-items: center; padding: 0 16px;
  height: 60px; background-color: var(--viewer-topbar-bg); border-bottom: 1px solid var(--viewer-border); flex-shrink: 0; z-index: 10;
}
.top-bar-left, .top-bar-right, .action-group { display: flex; align-items: center; gap: 16px; }
.top-bar-left {
  flex: 1 1 auto;
  min-width: 0;
}
.top-bar-right {
  flex: 0 0 auto;
  margin-left: auto;
  padding-left: 28px;
  min-width: max-content;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.page-title { font-size: 1.1rem; color: var(--viewer-text-primary); white-space: nowrap; }
.separator { width: 1px; height: 24px; background-color: var(--viewer-border); margin: 0 4px; }
button { border: none; cursor: pointer; border-radius: 4px; font-size: 0.9rem; transition: all 0.2s; }
.nav-btn { background: transparent; color: var(--viewer-text-secondary); padding: 8px 12px; display: flex; align-items: center; }
.nav-btn:hover:not(:disabled) { background: var(--viewer-highlight-soft); color: var(--viewer-text-primary); }
.action-btn { background: var(--viewer-panel-alt-bg); color: var(--viewer-text-primary); padding: 8px 16px; border: 1px solid var(--viewer-border); font-size: 0.82rem; }
.action-btn.primary { background-color: var(--success); border-color: var(--success); color: #fff; }
.action-btn:hover:not(:disabled) { background-color: var(--viewer-panel-hover); }
.action-btn.primary:hover:not(:disabled) { background-color: #5cb860; }
button:disabled { opacity: 0.5; cursor: not-allowed; }

/* Page Select Dropdown */
.page-select {
    background: var(--input-bg);
    color: var(--text-primary);
    border: 1px solid var(--viewer-border);
    padding: 6px 12px;
    border-radius: 4px;
    outline: none;
    font-size: 0.9rem;
    cursor: pointer;
}
.page-select:hover { border-color: var(--surface-strong); }
.recognition-controls-group {
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
  flex-shrink: 0;
}
.center-control-group {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 34px;
  padding: 0 10px;
  border: 1px solid var(--viewer-border);
  border-radius: 4px;
  background: var(--viewer-panel-alt-bg);
}
.center-control-label {
  font-size: 0.82rem;
  color: var(--viewer-text-secondary);
  white-space: nowrap;
}
.auto-recog-group {
  flex-wrap: nowrap;
  gap: 6px;
  padding: 0 8px;
  flex-shrink: 0;
}
.auto-recog-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 16px;
}
.auto-recog-options {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}
.auto-recog-select,
.auto-recog-key {
  height: 30px;
  box-sizing: border-box;
  align-self: center;
  font-size: 0.82rem;
  padding: 5px 7px;
  background: var(--viewer-input-bg);
  color: var(--viewer-text-primary);
  border: 1px solid var(--viewer-border);
  border-radius: 6px;
  outline: none;
}
.auto-recog-select {
  cursor: pointer;
  min-width: 74px;
}
.auto-recog-key {
  width: 92px;
}
.recognition-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: nowrap;
  flex-shrink: 0;
}
.toolbar-field {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--viewer-text-secondary);
  font-size: 0.82rem;
  white-space: nowrap;
  flex-shrink: 0;
}
.toolbar-label {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.toolbar-label-icon {
  width: 16px;
  height: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--viewer-border);
  border-radius: 3px;
  background: var(--viewer-panel-alt-bg);
  color: var(--viewer-text-secondary);
  font-size: 0.63rem;
  font-weight: 700;
  line-height: 1;
}
.toolbar-select {
  background: var(--viewer-input-bg);
  color: var(--viewer-text-primary);
  border: 1px solid var(--viewer-border);
  border-radius: 6px;
  padding: 5px 7px;
  font-size: 0.82rem;
  outline: none;
}
.toolbar-icon-btn {
  width: 30px;
  height: 30px;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: 1px solid var(--viewer-border);
  color: var(--viewer-text-secondary);
  border-radius: 4px;
  font-size: 0.9rem;
}
.toolbar-icon-btn:hover:not(:disabled) {
  background: var(--viewer-panel-hover);
  color: var(--viewer-text-primary);
}
.toolbar-icon-btn.active {
  background: var(--viewer-toggle-active-bg);
  border-color: var(--viewer-toggle-active-border);
  color: var(--viewer-toggle-active-text);
}
.toolbar-icon-btn.delete {
  border-color: rgba(255, 120, 120, 0.6);
  color: #ff8d8d;
}
.preview-btn.active {
  background: var(--viewer-toggle-active-bg);
  border-color: var(--viewer-toggle-active-border);
  color: var(--viewer-toggle-active-text);
}
.zoom-controls {
  gap: 8px;
}
.zoom-btn {
  min-width: 38px;
  padding: 8px 10px;
}
.zoom-readout {
  min-width: 54px;
  text-align: center;
  color: var(--viewer-text-secondary);
  font-size: 0.88rem;
  font-variant-numeric: tabular-nums;
}

/* Main Visualization */
.visualization-container {
  position: relative; overflow: auto; flex-grow: 1; display: flex;
  justify-content: center; align-items: center; padding: 1rem; background-color: var(--viewer-canvas-bg);
}
.visualization-stage {
  display: flex;
  flex: 0 0 auto;
}
.image-container { position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.6); }
.manuscript-image { display: block; user-select: none; opacity: var(--viewer-image-opacity); }
.focused-binary-overlay { position: absolute; object-fit: fill; pointer-events: none; z-index: 2; }
.graph-overlay { position: absolute; top: 0; left: 0; opacity: 0; pointer-events: none; transition: opacity 0.2s; z-index: 3; }
.graph-overlay.is-visible { opacity: 1; pointer-events: auto; }

/* Input Floater */
.input-floater {
    z-index: 100;
    display: flex;
    flex-direction: column;
    gap: 0;
    overflow: visible;
}
.line-input {
    width: 100%;
    background: var(--viewer-input-bg);
    color: var(--viewer-text-primary);
    border: 1px solid var(--viewer-border);
    box-sizing: border-box;
    padding: 8px 12px;
    box-shadow: none;
    border-radius: 4px;
    font-family: monospace;
    outline: none;
    transition: font-size 0.2s;
}
.line-input:focus {
    border-color: var(--viewer-border);
    box-shadow: none;
}
.transliteration-preview-strip {
    background: var(--viewer-transcription-bg);
    color: var(--viewer-transcription-text);
    border: 1px solid var(--viewer-transcription-border);
    border-radius: 0;
    padding: 1px 10px 3px;
    line-height: 1.15;
    opacity: 0.92;
    font-style: italic;
}
.word-preview-strip {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-top: 6px;
    min-height: 0;
    padding: 6px 8px;
    border: 1px solid var(--viewer-border);
    border-radius: 8px;
    background: var(--viewer-overlay-bg);
    overflow-x: auto;
}
.word-preview-card {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
    align-items: center;
    padding-right: 8px;
    border-right: 1px dashed var(--viewer-border);
}
.word-preview-card:last-child {
    padding-right: 0;
    border-right: none;
}
.word-preview-image {
    display: block;
    width: auto;
    border: 1px solid var(--viewer-border);
    background: #fff;
    object-fit: contain;
    image-rendering: crisp-edges;
}
.word-preview-label {
    font-size: 0.68rem;
    color: var(--viewer-text-secondary);
    text-align: center;
}
.word-cut-line {
    stroke: #ff5f5f;
    stroke-width: 2.5;
    stroke-dasharray: none;
    pointer-events: none;
}
.word-cut-line.hover {
    stroke: rgba(180, 180, 180, 0.95);
}
.word-cut-line.hover.delete,
.word-cut-line.deletable {
    stroke: #ff5f5f;
    stroke-width: 3;
    stroke-dasharray: none;
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
.processing-save-notice { background: var(--viewer-overlay-bg); border: 1px solid var(--viewer-border); color: var(--viewer-text-primary); }
.error-message { background: #c62828; color: white; }
.loading { font-size: 1.2rem; color: var(--viewer-text-muted); background: var(--viewer-overlay-bg); }

/* Bottom Rail */
.bottom-panel {
  background-color: var(--viewer-panel-bg); border-top: 1px solid var(--viewer-border); flex-shrink: 0; display: flex; flex-direction: column;
  height: 280px; transition: height 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.bottom-panel.is-collapsed { height: 45px; }
.mode-tabs { display: flex; background: var(--viewer-panel-alt-bg); height: 45px; flex-shrink: 0; }
.mode-tab { flex: 1; border-bottom: 3px solid transparent; color: var(--viewer-text-secondary); text-transform: uppercase; display: flex; align-items: center; justify-content: center; background: transparent; }
.mode-tab:hover:not(:disabled) { background: var(--viewer-panel-hover); color: var(--viewer-text-primary); }
.mode-tab.active { background: var(--viewer-panel-bg); color: var(--viewer-highlight); border-bottom-color: var(--viewer-highlight); font-weight: 500; }
.tab-spacer { flex-grow: 1; background: var(--viewer-panel-alt-bg); }
.panel-toggle-btn { background: var(--viewer-panel-alt-bg); color: var(--viewer-text-secondary); border-left: 1px solid var(--viewer-border); padding: 0 16px; min-width: 100px; }

/* Help Area */
.help-content-area { padding: 16px 24px; display: flex; gap: 24px; flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; }
.help-section { display: flex; gap: 24px; flex-grow: 1; min-height: 0; }
.media-container { width: 200px; height: 200px; background: var(--viewer-media-bg); border: 1px solid var(--viewer-border); flex-shrink: 0; position: relative; }
.tutorial-video { width: 100%; height: 100%; object-fit: contain; }
.instructions-container { flex-grow: 1; min-width: 0; max-width: 700px; overflow: visible; color: var(--viewer-text-muted); }
.instructions-container h3 { color: var(--viewer-text-primary); margin-top: 0; }
.instructions-container h4 { color: var(--viewer-text-secondary); margin-bottom: 5px; margin-top: 0; }
code { background: var(--viewer-code-bg); color: var(--viewer-code-text); padding: 2px 4px; border-radius: 3px; font-family: monospace; }
.webm-placeholder { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: var(--viewer-text-muted); background: var(--viewer-panel-hover); }

/* Sidebar Log */
.log-sidebar { width: 200px; background: var(--viewer-sidebar-bg); border: 1px solid var(--viewer-border); display: flex; flex-direction: column; min-height: 0; overflow-y: auto; }
.log-header { padding: 8px 10px; background: var(--viewer-panel-alt-bg); border-bottom: 1px solid var(--viewer-border); display: flex; justify-content: space-between; }
.log-list { list-style: none; padding: 0; margin: 0; overflow: visible; max-height: none; }
.log-list li { padding: 6px 10px; border-bottom: 1px solid var(--viewer-border); display: flex; justify-content: space-between; color: var(--viewer-text-secondary); }
.undo-icon { background: none; color: var(--viewer-text-muted); font-size: 1.1rem; }
.undo-icon:hover { color: var(--viewer-text-primary); }

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
input:checked + .slider { background-color: var(--viewer-switch-on-bg); }
input:checked + .slider:before { transform: translateX(14px); }

.confidence-strip {
    background: var(--viewer-overlay-bg);
    padding: 4px 12px;
    border-radius: 4px;
    white-space: pre; 
    pointer-events: none; 
    display: flex;
    flex-wrap: wrap;
    margin-top: 4px; 
    border: 1px solid var(--viewer-border);
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
  background: var(--viewer-sidebar-bg);
  border: 1px solid var(--viewer-border);
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition: transform 0.2s;
}

.help-card:hover {
  border-color: var(--surface-strong);
  background: var(--viewer-panel-hover);
}

.media-container-small {
  width: 100%;
  height: 110px; 
  background: var(--viewer-media-bg);
  border-bottom: 1px solid var(--viewer-border);
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
  color: var(--viewer-text-primary);
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.card-text p {
  margin: 4px 0;
  font-size: 0.85rem;
  color: var(--viewer-text-muted);
}

.key-badge {
  background: var(--viewer-code-bg);
  color: var(--viewer-code-text);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: monospace;
  font-weight: bold;
  border: 1px solid var(--viewer-border);
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
  background: var(--viewer-media-bg);
  border-right: 1px solid var(--viewer-border);
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
  border-top: 1px solid var(--viewer-border);
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background: var(--viewer-highlight-soft);
  border-radius: 4px;
  margin-top: 8px;
}

.key-hint {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.9rem;
  color: var(--viewer-text-muted);
}

@media (max-width: 1200px) {
  .top-bar {
    height: auto;
    min-height: 60px;
    padding-top: 10px;
    padding-bottom: 10px;
    flex-wrap: wrap;
    gap: 10px;
  }

  .top-bar-left,
  .top-bar-right {
    width: 100%;
    justify-content: flex-start;
    margin-left: 0;
  }

  .recognition-controls-group {
    flex-wrap: wrap;
    justify-content: flex-start;
  }
}
</style>
