import assert from 'node:assert/strict'
import { simulateDevanagariKeySequence, unicodeCodePoints } from './typingTestUtils.js'

const cases = [
  {
    name: 'R+u applies vocalic R matra after a single consonant',
    keys: ['h', 'R', 'u'],
    expected: 'हृ',
  },
  {
    name: 'R+u applies vocalic R matra after a conjunct tail',
    keys: ['r', 'n', 'R', 'u'],
    expected: 'र्नृ',
  },
  {
    name: 'R+u applies vocalic R matra after a multi-consonant cluster',
    keys: ['n', 'm', 'R', 'u'],
    expected: 'न्मृ',
  },
  {
    name: 'R+u at independent start keeps previous literal-R behavior',
    keys: ['R', 'u'],
    expected: 'Rउ',
  },
  {
    name: 'existing schwa deletion behavior remains unchanged',
    keys: ['k', 'a'],
    expected: 'क',
  },
  {
    name: 'existing aa matra behavior remains unchanged',
    keys: ['k', 'a', 'a'],
    expected: 'का',
  },
  {
    name: 'existing aspirate behavior remains unchanged',
    keys: ['k', 'h', 'a'],
    expected: 'ख',
  },
  {
    name: 'existing double danda behavior remains unchanged',
    keys: ['f', 'f'],
    expected: '॥',
  },
  {
    name: 'middle backspace protects a halant before the right-side consonant',
    initialValue: '\u092E\u094D\u0915\u094D\u200C\u092F',
    initialCursor: 5,
    keys: ['Backspace'],
    expected: '\u092E\u094D\u200C\u092F',
  },
  {
    name: 'middle replacement can change open म्क्‌य to म्नय without joining म to य',
    initialValue: '\u092E\u094D\u0915\u094D\u200C\u092F',
    initialCursor: 5,
    keys: ['Backspace', 'n', 'a'],
    expected: '\u092E\u094D\u0928\u092F',
  },
  {
    name: 'middle replacement can change closed म्कय to म्नय without joining म to य',
    initialValue: '\u092E\u094D\u0915\u092F',
    initialCursor: 3,
    keys: ['Backspace', 'n', 'a'],
    expected: '\u092E\u094D\u0928\u092F',
  },
  {
    name: 'existing middle edit त्यम to त्वम remains unchanged',
    initialValue: '\u0924\u094D\u092F\u092E',
    initialCursor: 3,
    keys: ['Backspace', 'v', 'a'],
    expected: '\u0924\u094D\u0935\u092E',
  },
  {
    name: 'existing middle edit श्चैमके to श्वमके remains unchanged',
    initialValue: '\u0936\u094D\u091A\u0948\u092E\u0915\u0947',
    initialCursor: 4,
    keys: ['Backspace', 'Backspace', 'v', 'a'],
    expected: '\u0936\u094D\u0935\u092E\u0915\u0947',
  },
]

let failures = 0

for (const testCase of cases) {
  const originalLog = console.log
  console.log = () => {}
  let actual
  try {
    actual = simulateDevanagariKeySequence(testCase.keys, testCase)
    await Promise.resolve()
  } finally {
    console.log = originalLog
  }

  try {
    assert.equal(actual, testCase.expected)
    console.log(`PASS ${testCase.name}`)
  } catch (error) {
    failures += 1
    console.error(`FAIL ${testCase.name}`)
    console.error(`  keys:     ${testCase.keys.join(' ')}`)
    console.error(`  expected: ${testCase.expected} (${unicodeCodePoints(testCase.expected)})`)
    console.error(`  actual:   ${actual} (${unicodeCodePoints(actual)})`)
    console.error(`  ${error.message}`)
  }
}

if (failures > 0) {
  process.exitCode = 1
} else {
  console.log(`All ${cases.length} typing tests passed.`)
}
