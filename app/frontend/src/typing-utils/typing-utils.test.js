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
]

let failures = 0

for (const testCase of cases) {
  const originalLog = console.log
  console.log = () => {}
  let actual
  try {
    actual = simulateDevanagariKeySequence(testCase.keys)
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
