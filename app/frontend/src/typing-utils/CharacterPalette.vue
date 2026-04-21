<template>
  <div class="character-palette-container" ref="paletteContainer">
    <!-- Trigger Button -->
    <button 
      class="palette-toggle-btn" 
      :class="{ 'is-active': isOpen }"
      @click="togglePalette"
      title="Insert Rare Characters"
    >
      <span class="icon">अ+</span>
      <span class="label">Rare</span>
    </button>

    <!-- Popover Panel -->
    <transition name="fade-slide">
      <div v-if="isOpen" class="palette-popover">
        <div class="palette-header">
          <span class="palette-eyebrow">Devanagari</span>
          <span class="palette-title" :style="{ color: copyMessage ? '#00e5ff' : '#fff' }">
            {{ copyMessage || 'Special Characters' }}
          </span>
        </div>
        
        <div class="character-grid">
          <!-- NOTE: 'character-button' class is required so App.vue doesn't blur the input! -->
          <button
            v-for="(char, index) in characters"
            :key="index"
            class="char-btn character-button"
            @mousedown.prevent="copyCharacter(char)"
            :title="`Copy ${char}`"
          >
            {{ char }}
          </button>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';

// Array of rare/special Devanagari characters. Feel free to add more here.
const characters = [
  'ऀ',
  'ऄ',
  'ऎ',
  'ऒ',
  'ऴ',
  'ऺ',
  'ऻ',
  'ऽ',
  'ॆ',
  'ॊ',
  'ॎ',
  'ॏ',
  '॑',
  '॒',
  'ॕ',
  'ॖ',
  'ॗ',
  'ऌ',
  'ॡ',
  'ॢ',
  'ॢ',
  '॥',
  '॰',
  'ॲ',
  'ॳ',
  'ॴ',
  'ॵ',
  'ॶ',
  'ॷ',
  'ॸ',
  'ॹ',
  'ॺ',
  'ॻ',
  'ॼ',
  'ॽ',
  'ॾ',
  'ॿ',
  '꣠',
  '꣡',
  '꣢',
  '꣣',
  '꣤',
  '꣥',
  '꣦',
  '꣧',
  '꣨',
  '꣩',
  '꣪',
  '꣫',
  '꣬',
  '꣭',
  '꣮',
  '꣯',
  '꣰',
  '꣱',
  'ꣲ',
  'ꣳ',
  'ꣴ',
  'ꣵ',
  'ꣶ',
  'ꣷ',
  '꣸',
  '꣹',
  '꣺',
  'ꣻ',
  '꣼',
  'ꣽ',
  'ꣾ',
  'ꣿ',
];

const isOpen = ref(false);
const paletteContainer = ref(null);
const copyMessage = ref('');
let copyTimeout = null;

const togglePalette = () => {
  isOpen.value = !isOpen.value;
  if (isOpen.value) copyMessage.value = ''; // Reset message when opening
};

// Copy character to clipboard
const copyCharacter = async (char) => {
  try {
    await navigator.clipboard.writeText(char);
    copyMessage.value = `Copied '${char}'!`; // Show success text
    
    if (copyTimeout) clearTimeout(copyTimeout);
    copyTimeout = setTimeout(() => { copyMessage.value = ''; }, 1500);
  } catch (err) {
    console.error('Failed to copy character: ', err);
  }
};

// Close popover when clicking outside of it
const handleClickOutside = (event) => {
  if (isOpen.value && paletteContainer.value && !paletteContainer.value.contains(event.target)) {
    isOpen.value = false;
  }
};

onMounted(() => {
  document.addEventListener('mousedown', handleClickOutside);
});

onBeforeUnmount(() => {
  document.removeEventListener('mousedown', handleClickOutside);
});
</script>

<style scoped>
.character-palette-container {
  position: relative;
  display: inline-flex;
  align-items: center;
}

/* Toggle Button - Designed to match the Top Bar toggles */
.palette-toggle-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #efefef;
  padding: 4px 10px;
  border-radius: 8px;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
  min-height: 28px;
}

.palette-toggle-btn:hover {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.15);
}

.palette-toggle-btn.is-active {
  background: rgba(0, 229, 255, 0.1);
  border-color: rgba(0, 229, 255, 0.4);
  color: #00e5ff;
}

.icon {
  font-weight: 600;
  font-size: 0.8rem;
}

.label {
  letter-spacing: 0.02em;
}
/* Popover Panel */
.palette-popover {
  position: absolute;
  top: calc(100% + 12px);
  right: 0;
  width: 340px; /* Wider to comfortably fit characters */
  max-height: 400px; /* Prevents long lists from going off-screen */
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  background: rgba(20, 20, 20, 0.95);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid #3d3d3d;
  border-radius: 12px;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(0, 229, 255, 0.1);
  padding: 12px;
  z-index: 9999;
}

/* Little pointer triangle */
.palette-popover::before {
  content: '';
  position: absolute;
  top: -6px;
  right: 20px;
  transform: rotate(45deg);
  width: 10px;
  height: 10px;
  background: rgba(20, 20, 20, 0.95);
  border-top: 1px solid #3d3d3d;
  border-left: 1px solid #3d3d3d;
}

.palette-header {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  text-align: center;
}

.palette-eyebrow {
  font-size: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #8cb8a7;
}

.palette-title {
  font-size: 0.85rem;
  color: #fff;
  font-weight: 500;
}

/* Grid of Characters */
.character-grid {
  display: grid;
  /* Auto-fill automatically fits as many 36px columns as possible without overflowing */
  grid-template-columns: repeat(auto-fill, minmax(36px, 1fr));
  gap: 6px;
  overflow-y: auto; /* Makes it scrollable for large lists */
  padding-right: 4px; /* Space for the scrollbar */
  flex-grow: 1;
  min-height: 0;
}

/* Custom Scrollbar for the grid */
.character-grid::-webkit-scrollbar {
  width: 6px;
}
.character-grid::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
}
.character-grid::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
}
.character-grid::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 229, 255, 0.5);
}

.char-btn {
  box-sizing: border-box; /* Crucial: stops borders from expanding the width */
  padding: 0; /* Crucial: removes default button padding causing the cutoff */
  background: #2a2a2a;
  border: 1px solid #444;
  color: #e0e0e0;
  border-radius: 6px;
  font-size: 1.1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  aspect-ratio: 1;
  cursor: pointer;
  transition: all 0.15s ease;
  font-family: Arial, sans-serif;
}

.char-btn:hover {
  background: rgba(0, 229, 255, 0.15);
  border-color: #00e5ff;
  color: #00e5ff;
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
}

.char-btn:active {
  transform: translateY(0);
}

/* Animations */
.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.fade-slide-enter-from,
.fade-slide-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>