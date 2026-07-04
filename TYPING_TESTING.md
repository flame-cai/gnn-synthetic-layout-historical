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

## What The Simulator Checks

The simulator verifies the final text value after a sequence of key presses.
It intentionally exercises the production handler instead of duplicating the
mapping logic. This catches bugs where a map entry exists but the keydown flow
does not reach it.

When a test fails, the runner prints both the expected and actual Unicode code
points. This is useful for invisible characters such as halant (`U+094D`),
ZWNJ (`U+200C`), and ZWJ (`U+200D`).
