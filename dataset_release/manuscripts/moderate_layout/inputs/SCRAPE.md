These files cannot be included because of copyright restrictions of distribution. The copyright owners have allowed use of the manuscript images for research purposes. The images can be found at the link:
https://dav.splrarebooks.com/collection/view/yajnavalakyasmritih-acharadhyayah

Out of all pages of the manuscript we are using the following 15 pages in this research project:
233_0002.jpg
233_0003.jpg
233_0004.jpg
233_0005.jpg
233_0006.jpg
233_0007.jpg
233_0008.jpg
233_0009.jpg
233_0010.jpg
233_0011.jpg
233_0012.jpg
233_0013.jpg
233_0014.jpg
233_0015.jpg
233_0016.jpg



# Website Scraping Specification: DAV SPL Rare Books Collections

## 1. Objective

Build a Python scraper that downloads every page image from one or more book or manuscript collections hosted on:

`https://dav.splrarebooks.com`

The scraper must visit each configured collection URL, open the website's full-book viewer, determine the total number of pages, and download each page image into a separate local folder.

This document describes the required behavior without prescribing a single exact implementation. A future coding agent should use it as the functional specification for recreating or improving the scraper.

---

## 2. Example Collection URLs

The scraper should accept a configurable list of collection URLs.

Initial examples:

- `https://dav.splrarebooks.com/collection/view/yajnavalakyasmritih-acharadhyayah`
- `https://dav.splrarebooks.com/collection/view/amaranathamahatmyam`
- `https://dav.splrarebooks.com/collection/view/bhagavatamahatmyam-padmapuranagatam1`

The URL list must be easy to extend without changing the scraping logic.

---

## 3. Recommended Technology

Use the following Python tools:

- **Playwright** for browser automation and interaction with the JavaScript-based book viewer.
- **Requests** for downloading image files directly from their image URLs.
- Standard Python modules for:
  - file and directory management,
  - delays,
  - random delay generation,
  - logging,
  - exception handling.

A Chromium browser should be launched through Playwright.

---

## 4. Configurable Settings

The implementation should expose the following settings near the top of the program or in a configuration file.

### Collection URL list

A list containing all collection pages that should be scraped.

### Request delay range

Wait a random amount of time between page changes.

Recommended default:

- Minimum: 2 seconds
- Maximum: 5 seconds

The delay helps reduce load on the website and makes requests less aggressive.

### Headless mode

Allow the browser to run either:

- headlessly, without a visible browser window, or
- visibly, for debugging.

Recommended default: headless mode enabled.

### User-Agent

Use a normal desktop browser User-Agent for both:

- the Playwright browser context, and
- direct image-download requests.

Example:

`Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36`

### Timeouts

Recommended defaults:

- Main page navigation: 60 seconds
- Initial image-container lookup: 30 seconds
- Full-book viewer opening: 10 seconds
- Active image lookup: 20 seconds
- Image HTTP download: 30 seconds

These values should be configurable.

---

## 5. Logging Requirements

Create one log for the entire scraping session.

Recommended log filename:

`master_scraper.log`

Log messages should be written to both:

- the terminal, and
- the log file.

Include timestamps and severity levels.

Use appropriate levels such as:

- `INFO` for normal progress,
- `WARNING` for unexpected but recoverable states,
- `ERROR` for failures affecting one page or collection,
- `CRITICAL` for severe errors while processing a collection.

Important progress messages should include:

- browser startup,
- collection URL currently being processed,
- viewer-opening status,
- total page count,
- current page navigation,
- successful image downloads,
- failed downloads,
- skipped collections,
- browser shutdown.

---

## 6. Output Structure

Create one directory per collection.

The directory name should be derived from the final path segment of the collection URL.

Example:

Collection URL:

`https://dav.splrarebooks.com/collection/view/amaranathamahatmyam`

Output directory:

`amaranathamahatmyam/`

Each collection directory should contain:

```text
amaranathamahatmyam/
├── link.txt
├── <original-image-filename-1>
├── <original-image-filename-2>
├── <original-image-filename-3>
└── ...
```

### `link.txt`

Store the original collection URL in a file named:

`link.txt`

This allows the downloaded collection to be traced back to its source.

### Image filenames

By default, preserve the original filename provided by the website or image URL.

For example, if the active image URL is:

`https://example.com/path/manuscript_00017.jpg`

save it as:

`manuscript_00017.jpg`

Do not rename images using the collection slug or a generated page-number pattern unless a fallback is required.

When deriving the filename:

1. Read the final path component of the image URL.
2. Remove URL query parameters and fragments.
3. Decode URL-encoded characters when safe.
4. Sanitize characters that are invalid for the local operating system.
5. Preserve the original file extension instead of forcing `.jpg`.

If the image URL does not contain a usable filename, use this fallback format:

`<collection-slug>_page_<three-digit-page-number>.<extension>`

If two different image URLs resolve to the same local filename, avoid overwriting an existing file. Add a deterministic suffix such as `_2`, `_3`, or a short URL hash.

---

## 7. Browser Session Behavior

Use one browser session for the full scraping run.

Recommended sequence:

1. Start Playwright.
2. Launch Chromium.
3. Create a browser context using the configured User-Agent.
4. Open one page or tab.
5. Reuse that page while processing each collection URL.
6. Close the browser after all URLs have been attempted.

A future implementation may recreate the page or browser context after a serious failure, but this is optional.

---

## 8. Per-Collection Scraping Workflow

For each collection URL, perform the following steps.

### Step 1: Identify the collection slug

Take the final path segment of the URL.

Example:

```text
https://dav.splrarebooks.com/collection/view/amaranathamahatmyam
```

becomes:

```text
amaranathamahatmyam
```

Use this value for:

- the output directory name,
- image filename prefixes,
- log messages.

### Step 2: Create the output directory

Create the collection directory if it does not already exist.

Do not fail if the directory already exists.

### Step 3: Save the source URL

Write the full collection URL into:

`<collection-directory>/link.txt`

If this operation fails, log the error and continue attempting to scrape the collection.

### Step 4: Navigate to the collection page

Open the collection URL with Playwright.

Wait until network activity becomes idle, subject to the configured navigation timeout.

### Step 5: Open the full-book viewer

The collection page initially displays a preview image.

Wait for and click this element:

```css
div.plates .preview .img-container
```

After clicking it, wait for the active full-book viewer:

```css
div.full-book.section.active
```

If the preview element is not found, the click fails, or the viewer does not become active within the timeout:

1. Log an error.
2. Stop processing this collection.
3. Continue with the next collection URL.

### Step 6: Read the total page count

Read the text content of:

```css
span.total
```

Convert the value to an integer.

This value determines how many viewer pages should be processed.

If the value is missing or cannot be converted to an integer, log the error and stop processing the current collection.

### Step 7: Process each page

Loop from page 1 through the reported total page count.

For each page:

1. Wait for the currently active main image:
   ```css
   .owl-item.active .main-img
   ```
2. Read its `src` attribute.
3. Confirm that the image URL is not empty.
4. Check whether the same image URL has already been processed.
5. If it is new, download the image.
6. If this is not the final page, click the next-page control.
7. Wait a random delay before reading the next page.

Use this selector for the next-page control:

```css
li.next
```

Before clicking, inspect the element's CSS class.

If its class contains `disabled` before the expected final page:

1. Log a warning.
2. Stop processing the current collection.
3. Preserve all images already downloaded.

---

## 9. Duplicate Image Protection

Maintain an in-memory set of image URLs already downloaded for the current collection.

Before downloading an image, check whether its URL is in the set.

This prevents duplicate downloads if:

- the viewer briefly remains on the same page,
- a click does not advance,
- the carousel repeats an image,
- the total page count and carousel state do not perfectly match.

The duplicate-tracking set should be reset for each collection.

---

## 10. Image Download Procedure

Use an HTTP client such as `requests` to download each image directly.

For every image:

1. Send a `GET` request to the image URL.
2. Include the configured User-Agent header.
3. Enable streamed downloading.
4. Use a reasonable timeout, such as 30 seconds.
5. Raise an exception for unsuccessful HTTP status codes.
6. Ensure the filename ends in `.jpg`.
7. Write the response body to disk in chunks.

Recommended chunk size:

`8192` bytes

The download function should return a success indicator.

### Successful download

Log the completed filename.

### Failed download

Catch network-related exceptions, log the failed URL and error, and return failure without crashing the entire scraper.

Do not assume every source image is JPEG. Preserve the extension from the source filename when available, and use the HTTP `Content-Type` header to determine a suitable extension when a fallback filename is required.

---

## 11. Delays and Polite Request Behavior

After clicking the next-page button, pause for a random interval within the configured range.

Recommended range:

2 to 5 seconds

Do not remove the delay unless the website operator explicitly permits faster automated access.

A future implementation should also consider:

- respecting the website's terms of service,
- checking applicable `robots.txt` guidance,
- limiting concurrency,
- avoiding parallel downloads unless permitted,
- stopping when repeated server errors occur.

---

## 12. Error-Handling Rules

The scraper must isolate failures so one broken collection does not stop the entire run.

### Failure while downloading one image

- Log the error.
- Return a failed status.
- Continue the page loop when practical.

### Failure while opening the viewer

- Log the error.
- Stop the current collection.
- Move to the next URL.

### Unexpected failure during a page loop

- Log the page number, collection URL, and exception.
- Stop the remainder of the current collection.
- Move to the next URL.

### Critical failure while processing one collection

- Catch the error in the outer URL loop.
- Log it as critical.
- Continue with the next URL.

### Failure to write `link.txt`

- Log the error.
- Continue scraping the collection.

### Cleanup

The browser should be closed after the URL loop finishes, including when errors occur. Prefer a context manager or `try/finally` structure to guarantee cleanup.

---

## 13. Functional Pseudocode

```text
configure URL list, delays, browser mode, user-agent, timeouts, and logging

define download_image(image_url, output_folder, image_filename):
    verify output folder exists
    request image using user-agent and timeout
    validate HTTP response
    ensure filename has .jpg extension
    stream response into local file
    log success
    return true
    on network error:
        log failure
        return false

define scrape_collection(browser_page, collection_url):
    log collection start

    derive collection slug from URL
    create output directory
    write collection URL to link.txt

    navigate to collection page

    wait for preview image container
    click preview image container
    wait for active full-book viewer
    if viewer cannot be opened:
        log error
        return

    read total page count from span.total
    convert total page count to integer

    create empty set of downloaded image URLs

    for page_number from 1 to total page count:
        wait for active main image
        read active image src

        if src is missing:
            treat as an error for the current collection

        if src has not been downloaded:
            add src to duplicate-tracking set
            derive and sanitize the original filename from the image URL
            if no usable filename exists, build a zero-padded fallback filename
            ensure the chosen filename will not overwrite a different image
            download image

        if page_number is not the final page:
            inspect next button
            if next button is disabled:
                log warning
                stop collection loop

            click next button
            sleep for random configured delay

        if an unexpected error occurs:
            log page and collection details
            stop collection loop

    log collection completion

define main():
    start Playwright
    launch Chromium with configured headless setting
    create browser context with configured user-agent
    open a page

    for each collection URL:
        try:
            scrape_collection(page, URL)
        on critical collection error:
            log error
            continue

    close browser

run main only when the script is executed directly
```

---

## 14. Website-Specific Selectors

The following selectors are essential to the current scraping workflow:

| Purpose | CSS selector |
|---|---|
| Preview image that opens the viewer | `div.plates .preview .img-container` |
| Active full-book viewer | `div.full-book.section.active` |
| Total page count | `span.total` |
| Active page image | `.owl-item.active .main-img` |
| Next-page button | `li.next` |

These selectors are implementation details of the current website and may change.

A future agent should verify them in the browser's developer tools before relying on them.

If the website structure has changed, locate equivalent elements by inspecting:

- the preview image,
- the visible full-screen or full-book viewer,
- the page-count indicator,
- the active carousel image,
- the next-navigation control.

Prefer stable attributes such as IDs, semantic labels, or data attributes when available.

---

## 15. Suggested Improvements for a Future Agent

The recreated scraper may improve reliability while preserving the behavior described above.

### Resume support

Before downloading an image, derive its preserved source filename and check whether that file already exists and has a nonzero size.

This allows interrupted collections to resume without downloading every page again.

### Wait for image change

After clicking the next button, wait until the active image's `src` differs from the previous page instead of relying only on a fixed sleep.

The random delay can still be retained for polite pacing.

### Retry logic

Retry failed image downloads a small number of times with exponential backoff.

Recommended:

- 2 or 3 attempts,
- increasing delay after each failure,
- no infinite retry loop.

### Image validation

After download, verify that:

- the file exists,
- its size is greater than zero,
- the response has an image content type,
- the file can optionally be opened by an image library.

### URL normalization

If the image `src` is relative, resolve it against the collection page's base URL before downloading.

### Safer directory names

Sanitize the collection slug before using it as a local directory name.

### Per-collection summary

At the end of each collection, log:

- reported total pages,
- successful downloads,
- failed downloads,
- duplicates skipped,
- last processed page.

### Final session summary

After all URLs are processed, log:

- collections attempted,
- collections completed,
- collections failed or partially completed,
- total images downloaded,
- total image failures.

### Browser recovery

If a collection leaves the page in a broken state, close that page and create a fresh page before processing the next URL.

### Configuration file or command-line input

Allow URLs and settings to be supplied through:

- a text file,
- JSON or YAML configuration,
- command-line arguments.

### Screenshot on failure

When browser interaction fails, save a screenshot and optionally the page HTML for debugging.

---

## 16. Validation Checklist

A future agent should confirm all of the following before considering the implementation complete:

- [ ] The browser launches successfully.
- [ ] Every configured collection URL is attempted.
- [ ] A separate directory is created for each collection.
- [ ] Each directory contains `link.txt`.
- [ ] Clicking the preview opens the full-book viewer.
- [ ] The total page count is read correctly.
- [ ] The active image URL is extracted from each page.
- [ ] Images retain the original filenames supplied by the website or image URLs by default.
- [ ] Fallback filenames are used only when an image URL has no usable filename.
- [ ] Filename collisions do not overwrite previously downloaded images.
- [ ] Duplicate image URLs are not downloaded twice.
- [ ] The next button is clicked only when another page is expected.
- [ ] A disabled next button ends the collection safely.
- [ ] Random delays occur between page transitions.
- [ ] One collection failure does not stop later collections.
- [ ] Network failures are logged.
- [ ] The browser closes cleanly.
- [ ] The scraper's use complies with the website's applicable rules and permissions.

---

## 17. Expected High-Level Result

After running the scraper, the working directory should contain one folder for each collection URL, with all successfully downloaded page images and a record of the original source URL.

The scraper should favor reliability, traceability, polite request pacing, and graceful failure over maximum speed.