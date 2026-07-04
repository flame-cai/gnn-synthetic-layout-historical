# Devanagari Typing Tests

The Read Mode Devanagari keyboard can be tested without opening the GUI.
The test runner simulates browser `keydown` events against the same
`handleInput` function used by `ManuscriptViewer.vue`.

## Run The Tests

From the repository root:

```powershell
npm --prefix app/frontend run test:typing
```

From `app/frontend`:

```powershell
npm run test:typing
```

## Files

- `app/frontend/src/typing-utils/devanagariInputUtils.js`
  - Production keydown handler.
  - Exports `resetInputState()` so tests can start each simulated sequence
    without inheriting the previous sequence key state.
- `app/frontend/src/typing-utils/InputClusterCode.js`
  - Character constants, consonant maps, vowel maps, special character maps,
    and helper functions used by the input handler.
- `app/frontend/src/typing-utils/typingTestUtils.js`
  - Minimal non-GUI input and keyboard-event simulator.
  - Calls the production handler and then applies browser-like default text
    insertion when the handler does not call `preventDefault()`.
- `app/frontend/src/typing-utils/typing-utils.test.js`
  - Node-based assertions for expected key sequence output.

## Add A New Typing Case

Add an object to the `cases` array in `typing-utils.test.js`:

```js
{
  name: 'short behavior description',
  keys: ['r', 'n', 'R', 'u'],
  expected: 'र्नृ',
}
```

Use the same key values that `KeyboardEvent.key` would provide. For uppercase
letters, pass the uppercase key, for example `R` or `T`.

For a middle-cursor edit, provide the initial text and cursor index:

```js
{
  name: 'middle replacement keeps the right-side consonant separate',
  initialValue: '\u092E\u094D\u0915\u094D\u200C\u092F',
  initialCursor: 5,
  keys: ['Backspace', 'n', 'a'],
  expected: '\u092E\u094D\u0928\u092F',
}
```

`initialCursor` is a JavaScript string index, not a visual glyph index. For
Devanagari tests, prefer Unicode escapes when invisible characters such as
halant (`\u094D`) and ZWNJ (`\u200C`) matter.

## What The Simulator Checks

The simulator verifies the final text value after a sequence of key presses.
It intentionally exercises the production handler instead of duplicating the
mapping logic. This catches bugs where a map entry exists but the keydown flow
does not reach it.

Useful typing invariants to preserve:

- A consonant that is waiting for a vowel is represented as
  `consonant + halant + ZWNJ`.
- Backspace over a dependent vowel should reopen the preceding consonant into
  that same waiting state, so the next vowel key replaces the vowel without
  retyping the consonant.
- Backspace over `ं`, `ः`, or `ँ` reopens the consonant only when the modifier
  is directly after a bare consonant. If the modifier follows a matra, only the
  modifier is removed.

When a test fails, the runner prints both the expected and actual Unicode code
points. This is useful for invisible characters such as halant (`U+094D`),
ZWNJ (`U+200C`), and ZWJ (`U+200D`).
