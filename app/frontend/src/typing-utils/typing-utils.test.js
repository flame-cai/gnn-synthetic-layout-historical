import assert from 'node:assert/strict'
import { simulateDevanagariKeySequence, unicodeCodePoints } from './typingTestUtils.js'

const cases = [
  {
    name: 'R+R+i inserts independent vocalic R',
    keys: ['R', 'R', 'i'],
    expected: '\u090B',
  },
  {
    name: 'R+R+I inserts independent long vocalic R',
    keys: ['R', 'R', 'I'],
    expected: '\u0960',
  },
  {
    name: 'independent vocalic R can be lengthened by the existing replacement rule',
    keys: ['R', 'R', 'i', 'I'],
    expected: '\u0960',
  },
  {
    name: 'incomplete R+R independent-vowel prefix remains literal',
    keys: ['R', 'R'],
    expected: 'RR',
  },
  {
    name: 'divergent R+R+u sequence keeps its existing literal fallback',
    keys: ['R', 'R', 'u'],
    expected: 'RR\u0909',
  },
  {
    name: 'existing literal RR text is not mistaken for a typed vowel prefix',
    initialValue: 'RR',
    keys: ['i'],
    expected: 'RR\u0907',
  },
  {
    name: 'R+R+i inserts independent vocalic R at a middle cursor',
    initialValue: '\u092E\u092F',
    initialCursor: 1,
    keys: ['R', 'R', 'i'],
    expected: '\u092E\u090B\u092F',
  },
  {
    name: 'L+l+i applies vocalic L matra before anusvara',
    keys: ['h', 'L', 'l', 'i', 'M'],
    expected: '\u0939\u0962\u0902',
  },
  {
    name: 'L+l+I applies long vocalic L matra before anusvara',
    keys: ['h', 'L', 'l', 'I', 'M'],
    expected: '\u0939\u0963\u0902',
  },
  {
    name: 'L+l+i applies vocalic L matra after a conjunct tail',
    keys: ['n', 'm', 'L', 'l', 'i'],
    expected: '\u0928\u094D\u092E\u0962',
  },
  {
    name: 'vocalic L matra can be lengthened by the existing replacement rule',
    keys: ['h', 'L', 'l', 'i', 'i'],
    expected: '\u0939\u0963',
  },
  {
    name: 'uppercase L keeps its existing consonant behavior',
    keys: ['h', 'L', 'a'],
    expected: '\u0939\u094D\u0933',
  },
  {
    name: 'incomplete L+l vowel prefix keeps its existing conjunct behavior',
    keys: ['h', 'L', 'l', 'a'],
    expected: '\u0939\u094D\u0933\u094D\u0932',
  },
  {
    name: 'L+l+i at independent start keeps its existing consonant behavior',
    keys: ['L', 'l', 'i'],
    expected: '\u0933\u094D\u0932\u093F',
  },
  {
    name: 'an existing L+l consonant suffix is not mistaken for a vowel prefix',
    initialValue: '\u0939\u094D\u0933\u094D\u0932\u094D\u200C',
    keys: ['i'],
    expected: '\u0939\u094D\u0933\u094D\u0932\u093F',
  },
  {
    name: 'L+l+i inserts vocalic L matra at a middle cursor without changing right-side text',
    initialValue: '\u092E\u092F',
    initialCursor: 1,
    keys: ['h', 'L', 'l', 'i', 'M'],
    expected: '\u092E\u0939\u0962\u0902\u092F',
  },
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
    name: 'backspace over i matra reopens the consonant for vowel replacement',
    initialValue: '\u0928\u093F',
    initialCursor: 2,
    keys: ['Backspace'],
    expected: '\u0928\u094D\u200C',
  },
  {
    name: 'single backspace can change ni to no',
    initialValue: '\u0928\u093F',
    initialCursor: 2,
    keys: ['Backspace', 'o'],
    expected: '\u0928\u094B',
  },
  {
    name: 'single backspace can change ni to nau',
    initialValue: '\u0928\u093F',
    initialCursor: 2,
    keys: ['Backspace', 'o', 'u'],
    expected: '\u0928\u094C',
  },
  {
    name: 'single backspace can change cai to ce',
    initialValue: '\u091A\u0948',
    initialCursor: 2,
    keys: ['Backspace', 'e'],
    expected: '\u091A\u0947',
  },
  {
    name: 'single backspace can change lam to lem',
    initialValue: '\u0932\u0902',
    initialCursor: 2,
    keys: ['Backspace', 'e', 'M'],
    expected: '\u0932\u0947\u0902',
  },
  {
    name: 'backspace over modifier after matra only removes the modifier',
    initialValue: '\u0932\u0947\u0902',
    initialCursor: 3,
    keys: ['Backspace'],
    expected: '\u0932\u0947',
  },
  {
    name: 'middle edit can replace a matra without joining the next consonant',
    initialValue: '\u0928\u093F\u092E',
    initialCursor: 2,
    keys: ['Backspace', 'e'],
    expected: '\u0928\u0947\u092E',
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
