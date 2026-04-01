const VIRAMA = '𑍍'

const consonantBases = {
  k: '𑌕', kh: '𑌖', g: '𑌗', gh: '𑌘', G: '𑌙',
  c: '𑌚', ch: '𑌛', j: '𑌜', jh: '𑌝', J: '𑌞',
  T: '𑌟', Th: '𑌠', D: '𑌡', Dh: '𑌢', N: '𑌣',
  t: '𑌤', th: '𑌥', d: '𑌦', dh: '𑌧', n: '𑌨',
  p: '𑌪', ph: '𑌫', b: '𑌬', bh: '𑌭', m: '𑌮',
  y: '𑌯', r: '𑌰', l: '𑌲', L: '𑌳',
  v: '𑌵', z: '𑌶', 'ç': '𑌶', sh: '𑌶',
  S: '𑌷', s: '𑌸', h: '𑌹',
}

const vowelMarks = {
  aa: '𑌾', A: '𑌾', 'ā': '𑌾',
  i: '𑌿', ii: '𑍀', I: '𑍀', 'ī': '𑍀',
  u: '𑍁', uu: '𑍂', U: '𑍂', 'ū': '𑍂',
  e: '𑍇', ai: '𑍈', o: '𑍋', au: '𑍌', auu: '𑍗',
  Ri: '𑍃', 'rī': '𑍄', '-Ri': '𑍄', li: '𑍢', 'lī': '𑍣', '-li': '𑍣',
}

const independentVowels = {
  a: '𑌅', aa: '𑌆', A: '𑌆', 'ā': '𑌆',
  i: '𑌇', ii: '𑌈', I: '𑌈', 'ī': '𑌈',
  u: '𑌉', uu: '𑌊', U: '𑌊', 'ū': '𑌊',
  e: '𑌏', ai: '𑌐', o: '𑌓', au: '𑌔', auu: '𑍗',
  Ri: '𑌋', Rii: '𑍠', '-Ri': '𑍠', Li: '𑌌', Lii: '𑍡', '-Li': '𑍡',
}

const specialChars = {
  M: '𑌂', MM: '𑌁', H: '𑌃', '|': '।', '.': '।', "'": '𑌽', '&': VIRAMA,
  1: '௧', 2: '௨', 3: '௩', 4: '௪', 5: '௫',
  6: '௬', 7: '௭', 8: '௮', 9: '௯', 0: '௦',
}

const vowelMarkValues = new Set(Object.values(vowelMarks))

const isGranthaConsonant = (char) => char >= '𑌕' && char <= '𑌹'

const setValueAtCursor = (input, textRef, value, cursorPosition) => {
  textRef.value = value
  input.value = value
  input.setSelectionRange(cursorPosition, cursorPosition)
}

const reconcileTail = (value) => {
  const chars = Array.from(value)
  if (chars.length < 2) return value

  const a = chars[chars.length - 1]
  const b = chars[chars.length - 2]
  const c = chars[chars.length - 3]
  const d = chars[chars.length - 4]
  const e = chars[chars.length - 5]

  if (a === VIRAMA && b === '𑌹' && c === VIRAMA && d) {
    const aspiratedMap = {
      '𑌕': '𑌖', '𑌗': '𑌘', '𑌚': '𑌛', '𑌜': '𑌝', '𑌟': '𑌠',
      '𑌡': '𑌢', '𑌤': '𑌥', '𑌦': '𑌧', '𑌪': '𑌫', '𑌬': '𑌭', '𑌸': '𑌶',
    }
    if (aspiratedMap[d]) {
      chars.splice(chars.length - 4, 4, aspiratedMap[d], VIRAMA)
      return chars.join('')
    }
  }

  if (a === VIRAMA && b && c === VIRAMA && d && e === VIRAMA) return chars.join('')

  if (a === VIRAMA && b === '𑌞' && c === VIRAMA && d === '𑌜') {
    chars.splice(chars.length - 4, 4, '𑌜', VIRAMA, '𑌞', VIRAMA)
    return chars.join('')
  }

  if (a === VIRAMA && b === '𑌶' && c === VIRAMA && d === '𑌕') {
    chars.splice(chars.length - 4, 4, '𑌕', VIRAMA, '𑌷', VIRAMA)
    return chars.join('')
  }

  if (a === '𑌅' && b === VIRAMA && isGranthaConsonant(c)) {
    chars.splice(chars.length - 2, 2)
    return chars.join('')
  }

  if (isGranthaConsonant(b)) {
    if (a === '𑌇') {
      chars.splice(chars.length - 1, 1, '𑍈')
      return chars.join('')
    }
    if (a === '𑌉') {
      chars.splice(chars.length - 1, 1, '𑍌')
      return chars.join('')
    }
  }

  if (a === '𑌉') {
    if (b === '𑌅') {
      chars.splice(chars.length - 2, 2, '𑌔')
      return chars.join('')
    }
    if (b === '𑌔') {
      chars.splice(chars.length - 2, 2, '𑍗')
      return chars.join('')
    }
    if (isGranthaConsonant(b)) {
      chars.splice(chars.length - 1, 1, '𑍌')
      return chars.join('')
    }
    if (b === '𑍌') {
      chars.splice(chars.length - 2, 2, '𑍗')
      return chars.join('')
    }
  }

  if (b === VIRAMA && isGranthaConsonant(c)) {
    for (const [roman, glyph] of Object.entries(independentVowels)) {
      if (a === glyph && vowelMarks[roman]) {
        chars.splice(chars.length - 2, 2, vowelMarks[roman])
        return chars.join('')
      }
    }
  }

  return chars.join('')
}

const applyGranthaKey = (valueWithKey) => {
  if (valueWithKey.endsWith('&')) {
    return valueWithKey.slice(0, -1) + VIRAMA
  }

  for (let len = 2; len >= 1; len -= 1) {
    const chunk = valueWithKey.slice(-len)
    if (consonantBases[chunk]) {
      return valueWithKey.slice(0, -len) + consonantBases[chunk] + VIRAMA
    }
    if (independentVowels[chunk]) {
      return valueWithKey.slice(0, -len) + independentVowels[chunk]
    }
    if (specialChars[chunk]) {
      return valueWithKey.slice(0, -len) + specialChars[chunk]
    }
  }

  return valueWithKey
}

export const reverseTransliterateGrantha = (value) => {
  if (!value) return ''

  const reverseConsonants = Object.fromEntries(Object.entries(consonantBases).map(([key, glyph]) => [glyph, key]))
  const reverseVowels = Object.fromEntries(Object.entries(vowelMarks).map(([key, glyph]) => [glyph, key]))
  const reverseIndependent = Object.fromEntries(Object.entries(independentVowels).map(([key, glyph]) => [glyph, key]))

  reverseConsonants['𑌙'] = 'ṅ'
  reverseConsonants['𑌞'] = 'ñ'
  reverseConsonants['𑌶'] = 'ś'
  reverseConsonants['𑌷'] = 'ṣ'

  const chars = Array.from(value)
  let result = ''

  for (let index = 0; index < chars.length; index += 1) {
    const current = chars[index]
    const next = chars[index + 1]

    if (reverseIndependent[current]) {
      result += reverseIndependent[current]
    } else if (reverseConsonants[current]) {
      if (next === VIRAMA) {
        result += reverseConsonants[current]
        index += 1
      } else if (next && reverseVowels[next]) {
        result += reverseConsonants[current] + reverseVowels[next]
        index += 1
      } else {
        result += `${reverseConsonants[current]}a`
      }
    } else if (current === '𑌂') {
      result += 'M'
    } else if (current === '𑌃') {
      result += 'H'
    } else if (current === '।') {
      result += '/'
    } else {
      result += current
    }
  }

  return result.replace(/aa/g, 'ā')
}

export const handleGranthaInput = (event, textRef) => {
  const { key, ctrlKey, metaKey, altKey, target } = event
  if (ctrlKey || metaKey || altKey || !target) return
  if (key === 'Tab' || key === 'Enter') return

  const selectionStart = target.selectionStart ?? 0
  const selectionEnd = target.selectionEnd ?? selectionStart
  const before = target.value.slice(0, selectionStart)
  const after = target.value.slice(selectionEnd)

  if (key === 'Backspace') {
    if (selectionStart !== selectionEnd || selectionStart === 0) return

    const chars = Array.from(before)
    const lastChar = chars[chars.length - 1]
    const previousChar = chars[chars.length - 2]

    if (vowelMarkValues.has(lastChar) && isGranthaConsonant(previousChar)) {
      event.preventDefault()
      chars.splice(chars.length - 1, 1, VIRAMA)
      const updatedBefore = chars.join('')
      setValueAtCursor(target, textRef, updatedBefore + after, updatedBefore.length)
    }
    return
  }

  if (key.length !== 1) return

  const transformedBefore = reconcileTail(applyGranthaKey(before + key))
  if (transformedBefore === before + key) return

  event.preventDefault()
  setValueAtCursor(target, textRef, transformedBefore + after, transformedBefore.length)
}
