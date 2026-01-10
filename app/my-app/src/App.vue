<template>
  <div class="app-container">
    <div v-if="!currentManuscript">
      <!-- Upload Screen -->
      <div class="upload-card">
        <h1>Historical Manuscript Segmentation</h1>
        <div class="form-group">
          <label>Manuscript Name:</label>
          <input v-model="formName" type="text" placeholder="e.g. manuscript_1" />
        </div>
        <div class="form-group">
          <label>Resize Dimension (Longest Side):</label>
          <input v-model.number="formLongestSide" type="number" />
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
      <button class="back-btn" @click="currentManuscript = null">← Back to Upload</button>
      <ManuscriptViewer 
        :manuscriptName="currentManuscript" 
        :pageName="currentPage"
        @page-changed="handlePageChange"
      />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ManuscriptViewer from './components/ManuscriptViewer.vue'

// Basic State
const currentManuscript = ref(null)
const currentPage = ref(null)
const pageList = ref([])

// Upload Form State
const formName = ref('my_manuscript')
const formLongestSide = ref(2500)
const selectedFiles = ref([])
const uploading = ref(false)
const uploadStatus = ref('')

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
</script>

<style>
body { margin: 0; font-family: sans-serif; background: #222; color: white; }
.app-container { display: flex; flex-direction: column; height: 100vh; }
.upload-card { max-width: 500px; margin: 100px auto; padding: 20px; background: #333; border-radius: 8px; }
.form-group { margin-bottom: 15px; display: flex; flex-direction: column; }
input { padding: 8px; background: #444; border: 1px solid #555; color: white; margin-top: 5px; }
button { padding: 10px; background: #4CAF50; color: white; border: none; cursor: pointer; }
button:disabled { background: #555; }
.back-btn { position: absolute; top: 10px; left: 10px; z-index: 1000; background: #444; }
</style>