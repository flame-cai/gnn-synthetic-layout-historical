# App User Flows

This guide explains the main ways a reader, editor, or research assistant can use the app.

The app has two main work areas:

- `Page Layout`: tell the app where the lines and regions are on the page
- `Text Review`: check the text the app read, correct it, and save it

The simplest way to think about the workflow is:

1. Open a page.
2. Check the page structure in `Page Layout`.
3. Open `Text Review`.
4. Correct the text.
5. Save the page and move on.

## 1. What Happens When A Manuscript Opens

When a manuscript opens, the app loads:

- the list of pages
- the current page image
- the current line and region structure
- any saved text for that page
- any saved corrections from earlier work

If the manuscript has been edited before, the app may reopen the most recently used page.

## 2. The Two Main Work Areas

### `Page Layout`

Use this area to describe the page structure.

Typical work here:

- add missing points
- remove wrong points
- connect points that belong to the same line
- mark regions that group lines together

This area answers the question:

- "Where are the lines on the page?"

### `Text Review`

Use this area when the page structure is ready and you want to work on the words.

Typical work here:

- ask the app to read the page text
- review the suggested text line by line
- correct spelling, missed letters, or missed words
- save the corrected text

This area answers the question:

- "What does each line say?"

## 3. Main User Flows

### Flow A: Work On A New Page

This is the standard flow for a page that has not been finished yet.

1. Open the page in `Page Layout`.
2. Fix the line and region structure if needed.
3. Save the layout.
4. Open `Text Review`.
5. Read the text if needed.
6. Correct the text.
7. Save the page.

### Flow B: Open A Page That Already Has Saved Text

If a page already has saved editable text from an earlier session:

- the app can open that page directly in `Text Review`
- the saved text is reused
- the app does not read the page again automatically

This is important because it protects earlier human corrections.

If the user wants a fresh reading on purpose, they can click:

- `Read Text Again`

### Flow C: Correct Text, Then Move To The Next Page

From `Text Review`, the user can move on in several ways:

- `Next Page`
- `Save & Next Page`
- the page menu

What happens on the destination page:

- if the next page already has saved editable text, it may open directly in `Text Review`
- otherwise, it opens in `Page Layout`

This keeps the workflow safe while still saving time on pages that were already prepared earlier.

### Flow D: Go Back To Layout After Correcting Text

Sometimes a user notices that the line structure still needs work after they have already corrected text.

In that case:

1. Open `Page Layout`.
2. Fix the structure.
3. Save the layout.
4. Return to `Text Review`.

What happens next depends on whether the layout really changed:

- if the layout did not change in a way that affects the text, the app reuses the saved corrected text
- if the layout did change in a way that affects the lines, the app asks the user to read the page again

### Flow E: Continue After A Pause Or Interruption

The app can keep partial progress while a user is still typing in `Text Review`.

That means:

- text changes may be saved in the background as draft work
- this helps recover work after a refresh, interruption, or accidental navigation
- the app still avoids reading the page again automatically over saved human corrections

Draft saving helps with recovery. Explicit saving still marks the page as deliberately saved by the user.

### Flow F: Read The Page Again On Purpose

Sometimes the user wants to try a new reading method or ask the app to read the page again with the latest saved manuscript reader.

In that case:

1. Open the page in `Text Review`.
2. Click `Read Text Again`.
3. Compare the new reading with the current text.
4. Keep correcting and save when finished.

The app does not do this automatically just because a newer reader exists. The user must ask for it.

## 4. Saving Work

There are two main save actions:

- `Save Page`
- `Save & Next Page`

What they mean:

- in `Page Layout`, saving stores the current line and region structure
- in `Text Review`, saving stores the corrected text for that page

There is also background draft saving while the user types in `Text Review`.

Draft saving:

- helps recover in-progress work
- does not force the app to read the page again
- does not overwrite saved corrections by itself

## 5. Ways To Move Around

The app lets the user move through a manuscript in several ways:

- `Previous Page`
- `Next Page`
- `Save & Next Page`
- the page picker

The user can also switch between:

- `Page Layout`
- `Text Review`

The app tries to respect the user's earlier work while moving between pages and modes.

## 6. Safety And Guard Flows

Sometimes `Text Review` is not safe yet.

Examples:

- the page layout is not ready
- the page structure changed and the old text no longer matches it

In these cases, the app shows a clear message and offers:

- `Open Page Layout`

This helps the user return to the correct step instead of editing text that may no longer match the page structure.

## 7. Quick Examples

### Example: New page from scratch

1. Open page.
2. Fix layout.
3. Open `Text Review`.
4. Read text.
5. Correct text.
6. Save page.

### Example: Continue work from an earlier session

1. Open manuscript.
2. Open a page with saved text.
3. Resume in `Text Review`.
4. Continue correcting.
5. Save page.

### Example: Fix layout after text correction

1. Work in `Text Review`.
2. Notice a structural problem.
3. Return to `Page Layout`.
4. Fix the structure.
5. Save the layout.
6. Return to `Text Review`.

### Example: Ask for a new reading

1. Open a page that already has saved text.
2. Stay in `Text Review`.
3. Click `Read Text Again`.
4. Compare the new reading with the saved text.
5. Keep correcting and save.

## 8. Mental Model For Users

The app is easiest to understand in two steps:

- `Page Layout` tells the app where the lines are
- `Text Review` tells the app what the lines say

The app tries to protect two kinds of human work:

- page structure corrections
- text corrections

That is why it avoids reading the page again automatically when doing so could overwrite saved human corrections.
