# rAI-project

Camera software for a two-bin item sorter, configured for recycling and trash by default. `camera_sorter.py` watches an overhead camera, detects a substantial change from an empty platform, waits for the scene to settle, and sends the cropped photo to the OpenAI API for classification. A localhost dashboard shows the live feed, accepts optional information for the next item, and keeps item names and totals for each bin. The script also prints structured decisions for a future motor controller; it does not drive a pusher.

## Setup

Python 3.9 or newer is required. The OpenAI SDK is pinned to a version compatible with the Python 3.9 installed here. A local `.venv` has been created for this checkout. To recreate it:

```sh
cd rAI-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. `OPENAI_MODEL` defaults to `gpt-6-luna` and can also be selected with `--model`. Keep the API key out of source control; `.env` is ignored.

## Configure the bins

Edit `bins.json` to name the bins and define their sorting rules. The sorter has exactly two physical bin IDs, `A` and `B`; their names and accepted materials are configurable. Select another file with `--bins-config PATH`. The configuration is loaded and validated once at startup, before opening the camera or using the API. Restart the application after editing it.

The default configuration names bin A **Recycling** and bin B **Trash**, with B as the fallback. Only properly recyclable items go to A: visibly clean, dry, empty items made from ordinary uncoated paper/cardboard, metal food/drink cans, glass bottles/jars, or rigid plastic beverage bottles/food containers visibly marked PET #1 or HDPE #2. A recycling symbol alone does not establish acceptance. Contaminated items, uncertain materials, wrappers, films, foam, lined cups, and mixed materials go to B. Adjust these rules for your collection facility's accepted materials.

For example, this configuration sorts metal cans into A and other ordinary items into B:

```json
{
  "bins": {
    "A": {
      "name": "Metal",
      "rules": ["Accept clean, dry, empty metal food or drink cans only."]
    },
    "B": {
      "name": "Other",
      "rules": ["Accept all other nonhazardous items."]
    }
  },
  "fallback_bin": "B",
  "hold_rules": ["Hold hazardous items, including batteries and sharp objects."]
}
```

`fallback_bin` must be `A` or `B`. Ambiguous or unmatched items go to that bin, and a classification for the other bin is redirected there if its confidence is below `--min-confidence`. `hold_rules` defines cases that should remain on the platform. Hands, multiple items, and an unusable view also prevent sorting. `hold` and `empty` are reserved outcomes, not additional physical bins.

## Run

Start with the platform empty and the camera fixed in place:

```sh
source .venv/bin/activate
python camera_sorter.py --dry-run
```

Dry run exercises the camera, change detection, and local captures without an API key or API calls. Remove `--dry-run` to classify items through the API:

```sh
python camera_sorter.py
```

Open **http://localhost:8000** for the live camera feed and controls. Choose a different port with `--port 8080`. The server listens on this computer only. The feed uses the same platform crop as the detector and continues streaming during classification and cooldown.

Allow the two-second camera warmup to finish with the platform empty, then place one item on it. The detector waits for one second of stability before capturing. Keep hands clear of the platform while it settles. An unchanged item is submitted once; removing it to reveal a stable empty platform rearms detection. A substantial change to the item can also trigger a new capture after the cooldown.

After each capture, no new photos are taken for **5 seconds** by default. Set the interval with `--cooldown-seconds`, including `0` to disable the cooldown:

```sh
python camera_sorter.py --cooldown-seconds 0
```

Change detection and stability checks still apply with no cooldown. Clearing or recalibrating the platform does not shorten an active cooldown.

Stop with Ctrl+C in the terminal. The dashboard has a recalibration button; use it with an empty platform. To also show the original OpenCV desktop window, add `--desktop-preview`; that window supports **q**/**Esc** to quit and **r** to recalibrate. `--no-preview` disables the desktop window while keeping the browser feed. `--no-web` disables the dashboard. Your operating system may ask for camera permission for the terminal or IDE running Python.

Select a different camera or limit the watched and transmitted area to the platform:

```sh
python camera_sorter.py --camera 1 --roi 120 80 400 300
```

The four ROI values are `X Y WIDTH HEIGHT` in camera pixels. A tight crop helps exclude movement outside the platform. To classify an existing photo without opening a camera:

```sh
python camera_sorter.py --image item.jpg
```

Add `--dry-run` to test the one-photo path without an API call.

## Next-item information and history

Before placing an item, type any extra information into the dashboard's note box and save it. The queued note is optional, can be replaced or cleared, and is consumed when the next item is captured. The API receives it alongside the image as context for that item; it cannot override the configured bin rules. If classification retries, the note stays with that same item. Information entered after an item has been captured is reserved for the following item.

Clear the platform between items and allow it to appear empty for the settling interval (one second by default). This separates item identities: multiple captures of the same item update one history entry rather than increasing the count. Replacing an item without a stable empty interval is treated as another view of the current item.

The dashboard displays **Bin A / its configured name**, **Bin B / its configured name**, their rule lists, and combined totals, plus item names supplied by the model, materials, timestamps, and explanations. Only completed `A` or `B` classifications enter the history; `hold`, `empty`, and dry-run results do not increase totals. These are classification destinations until a physical pusher or bin sensor confirms delivery.

History and totals persist across restarts in `data/sorter.sqlite3`; change the location with `--history-file PATH`. Counts follow the fixed physical IDs A/B across configuration changes, while each history entry keeps the bin name used when that item was classified. Older `recycle`/`trash` entries are retained as A/B with their original Recycling/Trash names. Editing the configuration does not delete history; use Reset history if you want fresh counts after changing the rules. This database is ignored by Git. Pending notes are kept only for the current run. Existing files in `captures/` are not automatically imported.

The temporary **Reset history** button clears counts, stored history, and the latest result. An item already being processed when reset is clicked cannot repopulate the cleared history. The queued next-item note and saved capture files remain available; the live feed keeps running.

## Tuning

Run `python camera_sorter.py --help` for all options.

| Option | Default | Meaning |
| --- | --- | --- |
| `--change-fraction` | `0.04` | Fraction of changed pixels needed to recognize a substantial scene change. Lower for smaller items. |
| `--pixel-threshold` | `25` | Pixel brightness difference considered a change. Raise to ignore minor image noise. |
| `--motion-fraction` | `0.01` | Changed-pixel fraction tolerated while the scene settles. |
| `--reset-fraction` | `0.01` | Difference from the empty baseline allowed when rearming. |
| `--settle-seconds` | `1` | Required stable time before capture. |
| `--cooldown-seconds` | `5` | Seconds between captures; `0` disables the cooldown. |
| `--warmup-seconds` | `2` | Camera warmup before establishing the empty baseline. |
| `--min-confidence` | `0.75` | Minimum confidence for the nonfallback bin; lower-confidence decisions go to `fallback_bin`. |
| `--bins-config` | `bins.json` | Names, rules, fallback bin, and hold rules, loaded at startup. |
| `--output-dir` | `captures` | Local destination for captured JPEGs and result JSON files. |
| `--port` | `8000` | Localhost dashboard port. |
| `--history-file` | `data/sorter.sqlite3` | Persistent item names and sorting history. |

Use steady lighting. Camera movement, shadows, and exposure changes can look like new objects. If the camera or platform moves, clear the platform and recalibrate. Tiny items below the change threshold may require a smaller ROI or a lower `--change-fraction`.

## Results and integration

Each completed capture produces one JSON line on standard output and saves a JPEG plus a result JSON file under `captures/`. Records include `item_id` and `additional_info` so retries can be associated with the same item, plus `bin_name` for the destination name at classification time. Diagnostic logs go to standard error. `.venv/`, `.env`, `captures/`, and `data/` are ignored by Git.

Decisions are `A`, `B`, `hold`, or `empty`, with an item description, material, confidence, reason, and capture metadata. A future actuator should move an item only for `A` or `B`; `hold` means leave it in place and `empty` means no item to move. Ambiguous sorting decisions use the configured fallback bin. Operational problems such as hands or another obstruction, multiple items, an unusable view, uncertain item presence, or API failures produce `hold`, as do cases listed in `hold_rules`. The default configuration holds hazardous items. If the scene changes while a request is in flight, the earlier classification is held instead of being used to move the current item.

Only triggered photos are sent to the API; the camera video stays local. Captured photos remain on disk until you remove them.

The integration uses the [Responses API image input](https://developers.openai.com/api/docs/guides/images-vision) and [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests use synthetic frames, a simulated camera, Flask's test client, temporary databases, and mocked API responses. They require no camera or API key. A live camera and API call must be checked on the target hardware with your key and bin configuration.
