# rAI-project

Camera software for a two-bin item sorter, configured for recycling and trash by default. `camera_sorter.py` watches an overhead camera, detects a substantial change from an empty platform, waits for the scene to settle, and sends the cropped photo to the OpenAI API for classification. Each captured image represents one load, whether it contains one object or several. A localhost dashboard shows the live feed, accepts optional information for the next capture, and keeps load names and totals for each bin. The script also prints structured decisions for a future motor controller; it does not drive a pusher.

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

The default configuration names bin A **Recycling** and bin B **Trash**, with B as the fallback. A accepts recognizable paper/cardboard, including paper cups and cartons; cans and small metal packaging; glass bottles/jars; common rigid plastic bottles, cups, tubs, and containers; and all electronics. The model can infer materials from normal appearance and context without a visible resin code or recycling symbol. Mostly empty is sufficient: minor food traces, drops, light stains or dampness, small grease marks, light coatings, labels, caps, and tape do not automatically send an item to B.

B mainly accepts food/organic waste, tissues/napkins/paper towels, heavily soiled or food-filled packaging, films/bags/wrappers, foam, and ceramics. Materially unclear loads also use B. These are permissive app defaults; edit `bins.json` to match acceptance at your facility.

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
      "rules": ["Accept all items that do not satisfy bin A unless an immediate emergency is visible."]
    }
  },
  "fallback_bin": "B",
  "hold_rules": ["Hold only for visible fire, smoke, sparks, an actively burning or venting battery, or a serious active chemical leak."]
}
```

`fallback_bin` must be `A` or `B`. Ambiguous or unmatched items go to that bin, and a classification for the other bin is redirected there if its confidence is below `--min-confidence`. Confidence remains the model's honest estimate; choosing a destination does not raise it.

`hold_rules` defines immediate emergency conditions only. The default rules reserve `hold` for visible fire, smoke, sparks, an actively burning or venting battery, or a serious active chemical leak. Ordinary batteries, electronics, or closed containers alone are not emergencies. Custom hold rules remain subject to the requirement that an actual immediate emergency is visible. `hold` and `empty` are reserved outcomes, not additional physical bins.

The classifier ignores incidental hands, arms, and other body parts rather than treating them as items or hazards. It classifies everything else in the submitted photo together as one load, with one combined label and one A/B destination. A partially obscured or uncertain load uses the best decision or fallback; a mixed load with conflicting bin requirements uses the fallback. A view containing only body parts and no item returns `empty`. Change detection still compares pixels: movement of a hand can affect settling outside cooldown; there is no body-part masking in the detector.

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

Allow the two-second camera warmup to finish with the platform empty, then place a load on it. The detector waits for one second of stability before capturing. A still scene does not automatically repeat captures. A substantial new change after cooldown can trigger another capture after settling; the platform does not need to become empty between captures.

After each capture, the detector ignores all scene changes and motion for **5 seconds** by default, treating the scene as stable during that interval. Movement during cooldown does not queue a capture or retry and does not count toward settling. The live feed continues streaming.

When cooldown expires, the first frame establishes a fresh reference. A new change or motion after that frame, followed by the normal settling interval, is required for another capture. For example, an item placed entirely during cooldown and left still will not trigger a photo until it is moved after cooldown.

Set the interval with `--cooldown-seconds`, including `0` to disable this ignore period:

```sh
python camera_sorter.py --cooldown-seconds 0
```

With no cooldown, normal change detection and stability checks apply continuously. Clearing or recalibrating the platform does not shorten an active cooldown or queue a capture.

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

## Next-capture information and history

Before a capture, type any extra information into the dashboard's note box and save it. The queued note is optional, can be replaced or cleared, and is consumed by the next captured image. The API receives it alongside that image as context for the whole load; it cannot override the configured bin rules. A later capture gets its own queued note, even if it shows the same physical objects. Information entered after a photo has been captured is reserved for the following capture.

Each successful A/B capture adds one history entry and one count to its destination bin. Several objects in one image count as one load. A later distinct capture counts as another load even if the physical objects are the same or the photos look identical. There is no physical-object identity tracking and no stable-empty requirement between loads. Cooldown and fresh-motion/settling checks still determine when a new photo can be taken.

Each image has a unique `capture_id`, also used as its `item_id`. Delivering the same capture's result twice counts it once; a repeated delivery does not overwrite its original bin or label.

The dashboard displays **Bin A / its configured name**, **Bin B / its configured name**, their rule lists, and combined totals, plus item names supplied by the model, materials, timestamps, and explanations. Only completed `A` or `B` classifications enter the history; `hold`, `empty`, `error`, `discarded`, and dry-run `preview` results do not increase totals. These are classification destinations until a physical pusher or bin sensor confirms delivery.

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
| `--cooldown-seconds` | `5` | Seconds after capture during which scene changes are ignored; afterward, a fresh change and settling are required. `0` disables this period. |
| `--warmup-seconds` | `2` | Camera warmup before establishing the empty baseline. |
| `--min-confidence` | `0.75` | Minimum confidence for the nonfallback bin; lower-confidence decisions go to `fallback_bin`. |
| `--bins-config` | `bins.json` | Names, rules, fallback bin, and hold rules, loaded at startup. |
| `--output-dir` | `captures` | Local destination for captured JPEGs and result JSON files. |
| `--port` | `8000` | Localhost dashboard port. |
| `--history-file` | `data/sorter.sqlite3` | Persistent captured-load names and sorting history. |

Use steady lighting. Camera movement, shadows, and exposure changes can look like new objects. If the camera or platform moves, clear the platform and recalibrate. Tiny items below the change threshold may require a smaller ROI or a lower `--change-fraction`.

## Results and integration

Each completed capture produces one JSON line on standard output and saves a JPEG plus a result JSON file under `captures/`. Records include matching `capture_id` and `item_id` values, the capture's `additional_info`, and `bin_name` for the destination name at classification time. Diagnostic logs go to standard error. `.venv/`, `.env`, `captures/`, and `data/` are ignored by Git.

Classifier decisions are `A`, `B`, `hold`, or `empty`, with a required `emergency` boolean, combined load description, material, confidence, reason, and capture metadata. A future actuator should move a load only for `A` or `B`. `hold` is reserved for a visible immediate emergency: `emergency: true` always produces `hold`. A nonemergency `hold` response is redirected to the configured fallback bin. `empty` means no load to move. Ordinary uncertainty, partial obstruction, multiple objects, and low confidence use one best bin decision or fallback for the entire load while preserving the reported confidence.

Saved records and JSON output also use operational outcomes with `emergency: false`: `error` for an API failure and `preview` for dry run. Legacy records may also contain `discarded`. These outcomes are separate from an emergency hold and do not trigger sorting or increase history totals.

An API response always belongs to its captured photo. Later movement in the live view or recalibration does not discard that result. Reset history still prevents requests captured before the reset from adding counts afterward. Taking another photo requires the normal fresh change after cooldown followed by settling.

Only triggered photos are sent to the API; the camera video stays local. Captured photos remain on disk until you remove them.

The integration uses the [Responses API image input](https://developers.openai.com/api/docs/guides/images-vision) and [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests use synthetic frames, a simulated camera, Flask's test client, temporary databases, and mocked API responses. They require no camera or API key. A live camera and API call must be checked on the target hardware with your key and bin configuration.
