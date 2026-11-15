# rAI-project

Camera software for a trash/recycling sorter. `camera_sorter.py` watches an overhead camera, detects a substantial change from an empty platform, waits for the scene to settle, and sends the cropped photo to the OpenAI API for classification. It prints a structured decision for a future motor controller; it does not drive a pusher.

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

The default policy in `recycling_rules.txt` sends only properly recyclable items to recycling; everything else goes to trash. It accepts visibly clean, dry, empty items made from ordinary uncoated paper/cardboard, metal food/drink cans, glass bottles/jars, or rigid plastic beverage bottles/food containers visibly marked PET #1 or HDPE #2. A recycling symbol alone does not establish acceptance. Contaminated items, uncertain materials, wrappers, films, foam, lined cups, and mixed materials go to trash. Adjust the file for your collection facility's accepted materials, or select another policy with `--rules-file PATH`.

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

Allow the two-second camera warmup to finish with the platform empty, then place one item on it. The detector waits for one second of stability before capturing. Keep hands clear of the platform while it settles. An unchanged item is submitted once; removing it to reveal a stable empty platform rearms detection. A substantial change to the item can also trigger a new capture after the cooldown.

Press **q** or **Esc** to quit. Press **r** to recalibrate the baseline while the platform is empty. For a headless setup, use `--no-preview` and stop with Ctrl+C. Your operating system may ask for camera permission for the terminal or IDE running Python.

Select a different camera or limit the watched and transmitted area to the platform:

```sh
python camera_sorter.py --camera 1 --roi 120 80 400 300
```

The four ROI values are `X Y WIDTH HEIGHT` in camera pixels. A tight crop helps exclude movement outside the platform. To classify an existing photo without opening a camera:

```sh
python camera_sorter.py --image item.jpg
```

Add `--dry-run` to test the one-photo path without an API call.

## Tuning

Run `python camera_sorter.py --help` for all options.

| Option | Default | Meaning |
| --- | --- | --- |
| `--change-fraction` | `0.04` | Fraction of changed pixels needed to recognize a substantial scene change. Lower for smaller items. |
| `--pixel-threshold` | `25` | Pixel brightness difference considered a change. Raise to ignore minor image noise. |
| `--motion-fraction` | `0.01` | Changed-pixel fraction tolerated while the scene settles. |
| `--reset-fraction` | `0.01` | Difference from the empty baseline allowed when rearming. |
| `--settle-seconds` | `1` | Required stable time before capture. |
| `--cooldown-seconds` | `3` | Minimum time between submissions. |
| `--warmup-seconds` | `2` | Camera warmup before establishing the empty baseline. |
| `--min-confidence` | `0.75` | Minimum model confidence accepted for recycling; lower-confidence recycling decisions become trash. |
| `--output-dir` | `captures` | Local destination for captured JPEGs and result JSON files. |

Use steady lighting. Camera movement, shadows, and exposure changes can look like new objects. If the camera or platform moves, clear the platform and recalibrate. Tiny items below the change threshold may require a smaller ROI or a lower `--change-fraction`.

## Results and integration

Each completed capture produces one JSON line on standard output and saves a JPEG plus a result JSON file under `captures/`. Diagnostic logs go to standard error. `.venv/`, `.env`, and `captures/` are ignored by Git.

Decisions are `recycle`, `trash`, `hold`, or `empty`, with an item description, material, confidence, reason, and capture metadata. A future actuator should move an item only for `recycle` or `trash`; `hold` means leave it in place and `empty` means no item to move. Uncertain recyclability defaults to trash, including recycling classifications below the confidence threshold. Operational problems such as hands or another obstruction, multiple items, hazardous items, uncertain item presence, or API failures produce `hold`. If the scene changes while a request is in flight, the earlier classification is held instead of being used to move the current item.

Only triggered photos are sent to the API; the camera video stays local. Captured photos remain on disk until you remove them.

The integration uses the [Responses API image input](https://developers.openai.com/api/docs/guides/images-vision) and [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The detector tests use synthetic frames and require no camera or API key. A live camera and API call must be checked on the target hardware with your key and recycling policy.
