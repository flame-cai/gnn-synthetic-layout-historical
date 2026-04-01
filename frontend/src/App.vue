<template>
  <div class="app-container">
    <div class="theme-toggle">
      <button
        class="theme-toggle-btn icon-only"
        @click="toggleTheme"
        :title="themeMode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'"
        aria-label="Toggle theme"
      >
        <i :class="themeMode === 'light' ? 'fa-regular fa-moon' : 'fa-regular fa-lightbulb'" aria-hidden="true"></i>
      </button>
    </div>

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
          <label class="label-with-help">
            <span>Resize Dimension (Longest Side):</span>
            <span class="help-tooltip">
              <span class="help-icon">?</span>
              <span class="help-tooltip-text">
                Sets the maximum page size before processing. Increase if small characters are missed. Reduce if processing is slow or memory-heavy.
              </span>
            </span>
          </label>
          <input v-model.number="formLongestSide" type="number" />
        </div>
        
        <div class="form-group">
          <label class="label-with-help">
            <span>Min Distance (Peak Detection):</span>
            <span class="help-tooltip">
              <span class="help-icon">?</span>
              <span class="help-tooltip-text">
                Controls how close detected character centers can be. Increase if one character gets multiple points. Reduce if nearby characters are being merged.
              </span>
            </span>
          </label>
          <input v-model.number="formMinDistance" type="number" title="Distance between char centers" />
        </div>

        <div class="form-group">
          <label class="label-with-help">
            <span>Wide Character Spacing?</span>
            <span class="help-tooltip">
              <span class="help-icon">?</span>
              <span class="help-tooltip-text">
                Use this when characters are spread wider than the gap between lines. Turn it on if lines get confused with neighboring lines. Leave it off if spacing already looks normal.
              </span>
            </span>
          </label>
          <div class="radio-group">
            <label class="radio-option">
              <input v-model="formCharSpacingMoreThanLineSpacing" :value="true" type="radio" />
              <span>Yes</span>
            </label>
            <label class="radio-option">
              <input v-model="formCharSpacingMoreThanLineSpacing" :value="false" type="radio" />
              <span>No</span>
            </label>
          </div>
        </div>

        <div v-if="formCharSpacingMoreThanLineSpacing" class="scale-panel">
          <div class="scale-panel-header">Segmentation Scaling</div>
          <div class="scale-grid">
            <div class="form-group compact">
              <label>X Scale</label>
              <input v-model.number="formXScale" type="number" step="0.01" min="0.01" />
            </div>
            <div class="form-group compact">
              <label>Y Scale</label>
              <input v-model.number="formYScale" type="number" step="0.01" min="0.01" />
            </div>
          </div>
        </div>
        
        <div class="form-group">
          <label>Images:</label>
          <input type="file" multiple @change="handleFileChange" accept="image/*" />
        </div>
        <button @click="upload" :disabled="uploading">
          {{ uploading ? 'Processing (Step 1-3)...' : 'Start Processing' }}
        </button>
        <div v-if="uploading" class="progress-panel">
          <div class="progress-meta">
            <span>{{ uploadStageLabel }}</span>
            <span>{{ uploadPercent }}%</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" :style="{ width: `${uploadPercent}%` }"></div>
          </div>
          <div class="progress-detail">
            {{ uploadStatus }}
            <span v-if="uploadTotal !== null && uploadCompleted !== null">
              ({{ uploadCompleted }}/{{ uploadTotal }})
            </span>
          </div>
        </div>
        <div v-if="uploadStatus && !uploading" class="status">{{ uploadStatus }}</div>
      </div>
    </div>

    <div v-else>
      <!-- Main Workstation -->
      <ManuscriptViewer 
        :manuscriptName="currentManuscript" 
        :pageName="currentPage"
        @page-changed="handlePageChange"
        @back="resetSelection" 
      />
    </div>
  </div>
</template>

<script setup>

import { ref, onMounted, onBeforeUnmount, watch, computed } from 'vue'
import ManuscriptViewer from './components/ManuscriptViewer.vue'

// Basic State
const currentManuscript = ref(null)
const currentPage = ref(null)
const pageList = ref([])

// Upload Form State
const formName = ref('my_manuscript')
const formLongestSide = ref(2500)
const formMinDistance = ref(20)
const formCharSpacingMoreThanLineSpacing = ref(false)
const formXScale = ref(0.25)
const formYScale = ref(0.5)
const selectedFiles = ref([])
const uploading = ref(false)
const uploadStatus = ref('')
const uploadPercent = ref(0)
const uploadStage = ref('idle')
const uploadCompleted = ref(null)
const uploadTotal = ref(null)
const activeUploadRequest = ref(null)
const existingManuscripts = ref([])
const selectedExisting = ref('')
const themeMode = ref(
  typeof window !== 'undefined'
    ? localStorage.getItem('manuscript-theme-mode') || 'light'
    : 'light'
)

const applyTheme = (theme) => {
  if (typeof document === 'undefined') return
  document.documentElement.setAttribute('data-theme', theme)
  if (typeof window !== 'undefined') {
    localStorage.setItem('manuscript-theme-mode', theme)
  }
}

const uploadStageLabel = computed(() => {
  const labels = {
    idle: 'Idle',
    uploading_files: 'Uploading Files',
    queued: 'Queued',
    starting: 'Starting',
    preparing_images: 'Preparing Images',
    segmenting_pages: 'Segmenting Pages',
    finalizing: 'Finalizing',
    completed: 'Completed',
    failed: 'Failed',
  }
  return labels[uploadStage.value] || 'Processing'
})

onMounted(async () => {
    applyTheme(themeMode.value)
    try {
        const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/existing-manuscripts`)
        if(res.ok) {
            existingManuscripts.value = await res.json()
        }
    } catch(e) {
        console.error("Failed to load existing manuscripts", e)
    }
})

watch(themeMode, applyTheme)

onBeforeUnmount(() => {
  if (activeUploadRequest.value) {
    activeUploadRequest.value.abort()
  }
})

const handleFileChange = (e) => {
  selectedFiles.value = Array.from(e.target.files)
}

const toggleTheme = () => {
  themeMode.value = themeMode.value === 'light' ? 'dark' : 'light'
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const pollUploadJob = async (jobId) => {
  while (true) {
    const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/upload-progress/${jobId}`)
    if (!res.ok) {
      throw new Error('Could not read upload progress')
    }

    const data = await res.json()
    uploadStage.value = data.stage || 'processing'
    uploadPercent.value = data.progressPercent ?? uploadPercent.value
    uploadStatus.value = data.message || 'Processing...'
    uploadCompleted.value = data.completed ?? null
    uploadTotal.value = data.total ?? null

    if (data.status === 'completed') {
      uploadStage.value = 'completed'
      uploadPercent.value = 100
      return data
    }

    if (data.status === 'failed') {
      uploadStage.value = 'failed'
      throw new Error(data.error || data.message || 'Processing failed')
    }

    await sleep(1000)
  }
}

const submitUploadRequest = (formData) => {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    activeUploadRequest.value = xhr

    xhr.open('POST', `${import.meta.env.VITE_BACKEND_URL}/upload`)

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return
      uploadStage.value = 'uploading_files'
      uploadPercent.value = Math.round((event.loaded / event.total) * 100)
      uploadStatus.value = `Uploading ${selectedFiles.value.length} file(s)...`
      uploadCompleted.value = null
      uploadTotal.value = null
    }

    xhr.onload = () => {
      activeUploadRequest.value = null
      let data = {}
      try {
        data = JSON.parse(xhr.responseText || '{}')
      } catch (_err) {
        data = {}
      }

      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(data.error || 'Upload failed'))
        return
      }

      uploadStage.value = data.stage || 'queued'
      uploadPercent.value = 0
      uploadStatus.value = data.message || 'Upload complete. Waiting for processing...'
      resolve(data)
    }

    xhr.onerror = () => {
      activeUploadRequest.value = null
      reject(new Error('Upload failed'))
    }

    xhr.onabort = () => {
      activeUploadRequest.value = null
      reject(new Error('Upload cancelled'))
    }

    xhr.send(formData)
  })
}

const upload = async () => {
  if (selectedFiles.value.length === 0) return alert('Select files')
  uploading.value = true
  uploadStage.value = 'uploading_files'
  uploadPercent.value = 0
  uploadCompleted.value = null
  uploadTotal.value = null
  uploadStatus.value = 'Starting upload...'

  const formData = new FormData()
  formData.append('manuscriptName', formName.value)
  formData.append('longestSide', formLongestSide.value)
  formData.append('minDistance', formMinDistance.value)
  formData.append('charSpacingMoreThanLineSpacing', String(formCharSpacingMoreThanLineSpacing.value))
  formData.append('xScale', formCharSpacingMoreThanLineSpacing.value ? String(formXScale.value || 0.25) : '1')
  formData.append('yScale', formCharSpacingMoreThanLineSpacing.value ? String(formYScale.value || 0.5) : '1')
  selectedFiles.value.forEach(file => formData.append('images', file))

  try {
    const uploadResponse = await submitUploadRequest(formData)
    const data = await pollUploadJob(uploadResponse.jobId)

    pageList.value = data.pages || []
    if (pageList.value.length > 0) {
      currentManuscript.value = formName.value
      currentPage.value = pageList.value[0]
    } else {
      uploadStatus.value = 'No pages processed.'
    }
  } catch (e) {
    uploadStage.value = 'failed'
    uploadStatus.value = 'Error: ' + e.message
  } finally {
    uploading.value = false
    activeUploadRequest.value = null
  }
}

const handlePageChange = (newPage) => {
  currentPage.value = newPage
}

// FIXED: Handle new backend response format { pages: [], last_edited: ... }
const loadExisting = async () => {
    if(!selectedExisting.value) return
    
    try {
        const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/manuscript/${selectedExisting.value}/pages`)
        if(!res.ok) throw new Error("Could not load pages")
        
        const data = await res.json()
        // Extract pages array
        const pages = data.pages || []
        const lastEdited = data.last_edited
        
        if(pages.length > 0) {
            pageList.value = pages
            currentManuscript.value = selectedExisting.value
            // Jump to last edited if available, otherwise first page
            currentPage.value = lastEdited && pages.includes(lastEdited) ? lastEdited : pages[0]
        } else {
            alert("This manuscript has no processed pages.")
        }
    } catch(e) {
        alert(e.message)
    }
}

const resetSelection = () => {
    currentManuscript.value = null
    // Reset internal state
    currentPage.value = null
    pageList.value = []
    // Refresh existing list
    onMounted() 
}
</script>

<style>
:root {
  --bg-app: #f5f5f5;
  --bg-elevated: rgba(255, 255, 255, 0.96);
  --bg-muted: #efefef;
  --surface-1: #ffffff;
  --surface-2: #f7f7f7;
  --surface-3: #ececec;
  --surface-strong: #d0d0d0;
  --text-primary: #111111;
  --text-secondary: #2d2d2d;
  --text-muted: #666666;
  --border-soft: rgba(0, 0, 0, 0.08);
  --border-strong: rgba(0, 0, 0, 0.18);
  --accent: #111111;
  --accent-strong: #000000;
  --accent-contrast: #ffffff;
  --success: #111111;
  --danger: #c62828;
  --shadow-strong: 0 20px 40px rgba(0, 0, 0, 0.08);
  --shadow-soft: 0 8px 20px rgba(0, 0, 0, 0.06);
  --input-bg: #ffffff;
  --panel-overlay: rgba(255, 255, 255, 0.96);
  --code-bg: #f0f0f0;
  --code-text: #111111;
  --viewer-image-opacity: 0.94;
  --viewer-topbar-bg: #ffffff;
  --viewer-panel-bg: #ffffff;
  --viewer-panel-alt-bg: #f6f6f6;
  --viewer-panel-hover: #ededed;
  --viewer-sidebar-bg: #fafafa;
  --viewer-canvas-bg: #f3f3f3;
  --viewer-media-bg: #ffffff;
  --viewer-overlay-bg: rgba(255, 255, 255, 0.96);
  --viewer-input-bg: rgba(255, 255, 255, 0.98);
  --viewer-border: rgba(0, 0, 0, 0.14);
  --viewer-text-primary: #111111;
  --viewer-text-secondary: #2d2d2d;
  --viewer-text-muted: #666666;
  --viewer-highlight: #111111;
  --viewer-highlight-soft: rgba(0, 0, 0, 0.06);
  --viewer-code-bg: #f0f0f0;
  --viewer-code-text: #111111;
  --viewer-toggle-active-bg: #000000;
  --viewer-toggle-active-border: #000000;
  --viewer-toggle-active-text: #ffffff;
  --viewer-switch-on-bg: #000000;
  --viewer-transcription-bg: rgba(255, 255, 255, 0.84);
  --viewer-transcription-border: rgba(0, 0, 0, 0.14);
  --viewer-transcription-text: #3a3a3a;
  --load-btn-bg: #000000;
  --load-btn-hover-bg: #111111;
}

:root[data-theme='dark'] {
  --bg-app: #0f1217;
  --bg-elevated: rgba(22, 27, 34, 0.94);
  --bg-muted: #171c24;
  --surface-1: #171c24;
  --surface-2: #1d2430;
  --surface-3: #283241;
  --surface-strong: #3b495d;
  --text-primary: #f3f6fb;
  --text-secondary: #d5dde8;
  --text-muted: #93a0b4;
  --border-soft: rgba(255, 255, 255, 0.08);
  --border-strong: rgba(255, 255, 255, 0.14);
  --accent: #dce3ec;
  --accent-strong: #ffffff;
  --accent-contrast: #0b0f14;
  --success: #7cc48a;
  --danger: #e17474;
  --shadow-strong: 0 24px 48px rgba(0, 0, 0, 0.4);
  --shadow-soft: 0 10px 24px rgba(0, 0, 0, 0.28);
  --input-bg: #171d26;
  --panel-overlay: rgba(13, 17, 24, 0.92);
  --code-bg: #1c2530;
  --code-text: #dce7f9;
  --viewer-image-opacity: 0.76;
  --viewer-topbar-bg: #141922;
  --viewer-panel-bg: #121720;
  --viewer-panel-alt-bg: #1a202b;
  --viewer-panel-hover: #202938;
  --viewer-sidebar-bg: #161c26;
  --viewer-canvas-bg: #0a0e14;
  --viewer-media-bg: #0d1218;
  --viewer-overlay-bg: rgba(17, 23, 31, 0.96);
  --viewer-input-bg: rgba(16, 21, 29, 0.95);
  --viewer-border: rgba(255, 255, 255, 0.12);
  --viewer-text-primary: #f3f6fb;
  --viewer-text-secondary: #d7deea;
  --viewer-text-muted: #99a7ba;
  --viewer-highlight: #8fb8ff;
  --viewer-highlight-soft: rgba(143, 184, 255, 0.14);
  --viewer-code-bg: #1b2330;
  --viewer-code-text: #d7e5ff;
  --viewer-toggle-active-bg: #5f6978;
  --viewer-toggle-active-border: #5f6978;
  --viewer-toggle-active-text: #f3f6fb;
  --viewer-switch-on-bg: #5f6978;
  --viewer-transcription-bg: rgba(255, 255, 255, 0.86);
  --viewer-transcription-border: rgba(255, 255, 255, 0.18);
  --viewer-transcription-text: #38404d;
  --load-btn-bg: #5f6978;
  --load-btn-hover-bg: #6b7584;
}

body {
  margin: 0;
  font-family: "Segoe UI", "Helvetica Neue", sans-serif;
  background: var(--bg-app);
  color: var(--text-primary);
}

.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: var(--bg-app);
}

.theme-toggle {
  position: fixed;
  bottom: 18px;
  left: 18px;
  z-index: 1000;
}

.theme-toggle-btn {
  padding: 9px 13px;
  border-radius: 999px;
  border: 1px solid var(--border-strong);
  background: var(--bg-elevated);
  color: var(--text-primary);
  box-shadow: var(--shadow-soft);
  font-size: 0.9rem;
}

.theme-toggle-btn.icon-only {
  width: 42px;
  height: 42px;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 1.05rem;
}

.upload-card {
  width: min(640px, calc(100vw - 48px));
  margin: 36px auto 24px;
  padding: 20px 22px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-soft);
  border-radius: 20px;
  box-shadow: var(--shadow-strong);
  backdrop-filter: blur(8px);
}

.upload-card h1 {
  margin-top: 0;
  margin-bottom: 14px;
  font-family: "Baskerville", "Palatino Linotype", serif;
  font-size: 1.6rem;
  color: var(--text-primary);
}

.upload-card h3 {
  margin: 0 0 10px;
  font-size: 1rem;
}

.form-group {
  margin-bottom: 10px;
  display: flex;
  flex-direction: column;
  color: var(--text-secondary);
}

.form-group label {
  font-weight: 600;
  font-size: 0.9rem;
}

.label-with-help {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  width: fit-content;
}

.help-tooltip {
  position: relative;
  display: inline-flex;
  align-items: center;
}

.help-icon {
  width: 16px;
  height: 16px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.68rem;
  font-weight: 700;
  background: var(--surface-3);
  color: var(--text-primary);
  border: 1px solid var(--border-strong);
  cursor: help;
  user-select: none;
}

.help-tooltip-text {
  position: absolute;
  left: calc(100% + 10px);
  top: 50%;
  transform: translateY(-50%);
  width: 240px;
  padding: 10px 12px;
  border-radius: 12px;
  background: var(--bg-elevated);
  color: var(--text-secondary);
  border: 1px solid var(--border-soft);
  box-shadow: var(--shadow-soft);
  font-size: 0.82rem;
  line-height: 1.35;
  font-weight: 500;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.16s ease;
  z-index: 20;
}

.help-tooltip:hover .help-tooltip-text {
  opacity: 1;
}

.compact {
  margin-bottom: 0;
}

input,
select,
button {
  font: inherit;
}

input,
.dropdown {
  padding: 8px 10px;
  background: var(--input-bg);
  border: 1px solid var(--border-strong);
  color: var(--text-primary);
  margin-top: 4px;
  border-radius: 12px;
  font-size: 0.92rem;
}

button {
  padding: 9px 12px;
  background: var(--accent);
  color: var(--accent-contrast);
  border: none;
  cursor: pointer;
  border-radius: 12px;
  transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
  font-size: 0.92rem;
}

button:hover:not(:disabled) {
  transform: translateY(-1px);
  background: var(--accent-strong);
  box-shadow: var(--shadow-soft);
}

button:disabled {
  background: var(--surface-strong);
  color: var(--text-muted);
  cursor: not-allowed;
}

.progress-panel {
  margin-top: 14px;
  padding: 14px;
  border: 1px solid var(--border-soft);
  border-radius: 14px;
  background: var(--surface-2);
}

.progress-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  color: var(--text-secondary);
  font-weight: 600;
}

.progress-track {
  width: 100%;
  height: 12px;
  border-radius: 999px;
  background: var(--surface-3);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), var(--accent-strong));
  transition: width 0.25s ease;
}

.progress-detail {
  margin-top: 8px;
  color: var(--text-muted);
  font-size: 0.92rem;
}

.section-divider {
  margin-bottom: 14px;
}

.separator {
  margin: 16px 0;
  border: 0;
  border-top: 1px solid var(--border-soft);
}

.load-group {
  display: flex;
  gap: 10px;
}

.dropdown {
  flex-grow: 1;
  margin-top: 0;
}

.load-btn {
  background: var(--load-btn-bg);
  color: #fff;
}

.load-btn:hover:not(:disabled) {
  background: var(--load-btn-hover-bg);
}

.status {
  margin-top: 10px;
  color: var(--text-muted);
  font-style: italic;
  font-size: 0.9rem;
}

.radio-group {
  display: flex;
  gap: 14px;
  margin-top: 6px;
}

.radio-option {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--text-primary);
  font-size: 0.92rem;
}

.radio-option input {
  margin: 0;
}

.scale-panel {
  margin-bottom: 12px;
  padding: 12px;
  border-radius: 16px;
  border: 1px solid var(--border-soft);
  background: var(--surface-2);
}

.scale-panel-header {
  margin-bottom: 8px;
  color: var(--text-primary);
  font-weight: 700;
  font-size: 0.92rem;
}

.scale-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

@media (max-width: 720px) {
  .upload-card {
    margin: 72px auto 32px;
    padding: 22px;
  }

  .help-tooltip-text {
    left: 0;
    top: calc(100% + 8px);
    transform: none;
    width: min(260px, calc(100vw - 96px));
  }

  .scale-grid,
  .load-group {
    grid-template-columns: 1fr;
    flex-direction: column;
  }

  .theme-toggle {
    top: 10px;
    right: 10px;
  }
}
</style>
