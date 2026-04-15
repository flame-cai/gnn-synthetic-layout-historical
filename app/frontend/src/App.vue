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
        
        <div class="form-group">
          <label>Min Distance (Peak Detection):</label>
          <input v-model.number="formMinDistance" type="number" title="Distance between char centers" />
        </div>
        
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

import { ref, onMounted } from 'vue'
import ManuscriptViewer from './components/ManuscriptViewer.vue'

// Basic State
const currentManuscript = ref(null)
const currentPage = ref(null)
const pageList = ref([])

// Upload Form State
const formName = ref('my_manuscript')
const formLongestSide = ref(2500)
const formMinDistance = ref(20)
const selectedFiles = ref([])
const uploading = ref(false)
const uploadStatus = ref('')
const existingManuscripts = ref([])
const selectedExisting = ref('')

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
  formData.append('minDistance', formMinDistance.value)
  selectedFiles.value.forEach(file => formData.append('images', file))

  try {
    const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/upload`, {
      method: 'POST',
      body: formData
    })
    if(!res.ok) throw new Error('Upload failed')
    const data = await res.json()
    
    // The /upload endpoint returns { message: "...", pages: [...] }
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
.status { margin-top: 15px; color: #aaa; font-style: italic; }
</style>