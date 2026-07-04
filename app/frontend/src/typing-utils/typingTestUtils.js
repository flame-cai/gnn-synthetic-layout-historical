import { handleInput, resetInputState } from './devanagariInputUtils.js'

class SimulatedInput {
  constructor(value = '') {
    this.value = value
    this.selectionStart = value.length
    this.selectionEnd = value.length
  }

  setSelectionRange(start, end) {
    this.selectionStart = start
    this.selectionEnd = end
  }
}

class SimulatedKeyboardEvent {
  constructor(key, input, options = {}) {
    this.key = key
    this.target = input
    this.ctrlKey = Boolean(options.ctrlKey)
    this.metaKey = Boolean(options.metaKey)
    this.altKey = Boolean(options.altKey)
    this.shiftKey = options.shiftKey ?? /^[A-Z]$/.test(key)
    this.defaultPrevented = false
  }

  preventDefault() {
    this.defaultPrevented = true
  }
}

const normalizeKeySpec = (keySpec) =>
  typeof keySpec === 'string' ? { key: keySpec } : keySpec

const applyBrowserDefault = (event, input) => {
  if (event.defaultPrevented) return

  const start = input.selectionStart
  const end = input.selectionEnd

  if (event.key === 'Backspace') {
    if (start !== end) {
      input.value = input.value.slice(0, start) + input.value.slice(end)
      input.setSelectionRange(start, start)
    } else if (start > 0) {
      input.value = input.value.slice(0, start - 1) + input.value.slice(start)
      input.setSelectionRange(start - 1, start - 1)
    }
    return
  }

  if (event.key.length === 1) {
    input.value = input.value.slice(0, start) + event.key + input.value.slice(end)
    const nextPosition = start + event.key.length
    input.setSelectionRange(nextPosition, nextPosition)
  }
}

export const simulateDevanagariKeySequence = (keySequence, options = {}) => {
  const input = new SimulatedInput(options.initialValue || '')
  const devanagariRef = {
    get value() {
      return input.value
    },
    set value(value) {
      input.value = value
    },
  }

  if (options.resetState !== false) resetInputState()

  for (const keySpec of keySequence.map(normalizeKeySpec)) {
    const event = new SimulatedKeyboardEvent(keySpec.key, input, keySpec)
    handleInput(event, devanagariRef)
    applyBrowserDefault(event, input)
    devanagariRef.value = input.value
  }

  return input.value
}

export const unicodeCodePoints = (value) =>
  Array.from(value)
    .map((char) => `U+${char.codePointAt(0).toString(16).toUpperCase().padStart(4, '0')}`)
    .join(' ')
