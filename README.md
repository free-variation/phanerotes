# Phanerotes

Audio-reactive video generation using convolutional autoencoders and a Forth-like scripting language.

## Overview

Phanerotes generates music videos by morphing between image tiles using a trained autoencoder. Source images are divided into tiles, encoded into a latent space, and transitions between tiles are driven by audio features. The system uses spherical linear interpolation (slerp) for smooth morphs and supports composable visual effects.

### Key Features

- **Tile-based generation**: Source images are divided into tiles and encoded via CNN autoencoder
- **Audio-reactive timing**: Transition speed modulated by audio energy (flux + RMS)
- **Theme system**: Tiles are associated with audio sections for visual coherence
- **Slerp interpolation**: Spherical interpolation in latent space for natural morphs
- **Composable effects**: Camera motion (wobble) and chromatic aberration scriptable per-frame
- **Forth scripting**: Stack-based DSL for flexible pipeline control

## Building

Requires:
- Fortran compiler (gfortran or ifort)
- BLAS library
- fpm (Fortran Package Manager)

```bash
fpm build --profile release --flag "-march=native -ffast-math -fopenmp"
```

## Running

Interactive REPL:
```bash
fpm run --profile release --flag "-march=native -ffast-math -fopenmp"
```

Execute a script:
```bash
fpm run --profile release --flag "-march=native -ffast-math -fopenmp" -- path/to/script.phan
```

## Training

Train an autoencoder on images in `images/training_data/`:

```bash
fpm run --example train_full_images --profile release --flag "-march=native -ffast-math -fopenmp" -- <tile_width> [options]
```

Options:
- `--height N` - tile height (default: same as width)
- `--epochs N` - number of training epochs (default: 10)
- `--batch N` - batch size (default: 8)
- `--add` - use addition for skip connections
- `--concat` - use concatenation for skip connections (default)

Examples:
```bash
# 512x512 tiles, 50 epochs, concatenation mode
fpm run --example train_full_images --profile release --flag "-march=native -ffast-math -fopenmp" -- 512 --epochs 50

# 576x384 tiles, addition mode
fpm run --example train_full_images --profile release --flag "-march=native -ffast-math -fopenmp" -- 576 --height 384 --add
```

Output: `ae-weights-{width}x{height}-{mode}.bin`

## Testing

```bash
fpm test --profile release --flag "-march=native -ffast-math -fopenmp"
```

---

## Video Generation Pipeline

1. **Load model**: Load pre-trained autoencoder weights
2. **Prepare tiles**: Extract tiles from source images, compute latents and cosine similarities
3. **Analyze audio**: Extract per-frame audio features, compute normalized energy
4. **Establish themes**: Select dispersed theme tiles, segment audio into sections
5. **Generate transitions**: Select tiles based on theme affinity and continuity, interpolate frames
6. **Apply effects**: Camera motion, chromatic aberration (scriptable per-frame)
7. **Finalize video**: Assemble frames with FFmpeg, apply fades, mux audio

---

## Forth Language Reference

Phanerotes uses a stack-based language inspired by Forth. There are three stacks:

- **Number stack**: Floating-point values
- **String stack**: Text strings
- **Image stack**: 3D arrays (channels, height, width)

### Syntax

- Numbers are pushed directly: `42`, `3.14`, `-0.5`
- Strings are quoted: `"hello world"`
- Comments start with `#`: `# this is a comment`
- Word definitions: `: word-name ... ;`

### Stack Notation

Stack effects are written as `( before -- after )` where the rightmost item is top of stack.

---

## Word Reference

### Output

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `.` | `( n -- )` | Print number and pop |
| `s.` | `( s -- )` | Print string and pop |
| `cr` | `( -- )` | Print newline |

### Number Stack

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `dup` | `( n -- n n )` | Duplicate top |
| `drop` | `( n -- )` | Discard top |
| `swap` | `( a b -- b a )` | Swap top two |
| `over` | `( a b -- a b a )` | Copy second to top |

### Arithmetic

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `+` | `( a b -- a+b )` | Add |
| `-` | `( a b -- a-b )` | Subtract |
| `*` | `( a b -- a*b )` | Multiply |
| `/` | `( a b -- a/b )` | Divide |

### String Stack

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `sdup` | `( s -- s s )` | Duplicate top string |

### Image Stack

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `idup` | `( img -- img img )` | Duplicate top image |
| `idrop` | `( img -- )` | Discard top image |
| `iswap` | `( a b -- b a )` | Swap top two images |
| `iover` | `( a b -- a b a )` | Copy second image to top |

### Control Flow

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `repeat` | `( n -- )` | Execute rest of line n times |
| `0?` | `( n -- n )` | Stop execution if top is zero (leaves value) |
| `>=?` | `( a b -- )` | Stop execution if a >= b (consumes both) |
| `empty-string-stack?` | `( -- )` | Stop execution if string stack empty |
| `empty-image-stack?` | `( -- )` | Stop execution if image stack empty |
| `:` | `( -- )` | Begin word definition |
| `;` | `( -- )` | End word definition |
| `quit` | `( -- )` | Exit interpreter |

### Image I/O

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `load` | `( s:path -- img )` | Load image from file |
| `save` | `( img s:path -- )` | Save image to file |
| `save-in` | `( img s:path s:dir -- )` | Save image to dir with same filename |
| `list-files` | `( s:dir -- s... n:count )` | Push sorted filenames and count |
| `load-frames` | `( s:dir -- [img s:path]... n:count )` | Load all frame_*.bmp files |

### Image Manipulation

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `transpose` | `( img -- img )` | Swap width and height |
| `fliph` | `( img -- img )` | Flip horizontally |
| `flipv` | `( img -- img )` | Flip vertically |
| `transform` | `( img n:a n:b n:c n:d n:tx n:ty -- img )` | Apply affine transform |
| `split` | `( img -- r g b )` | Split into color channels |
| `merge` | `( r g b -- img )` | Merge color channels |
| `interpolate` | `( img1 img2 n:alpha -- img )` | Linear blend (alpha=1 gives img1) |
| `interpolate-frames` | `( img1 img2 n:count -- img... )` | Generate interpolated sequence |

---

## Video Generation Words

### Setup

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `clear-video` | `( -- )` | Reset all video generation state |
| `load-model` | `( s:path -- )` | Load autoencoder weights |
| `set-seed` | `( n:seed -- )` | Set random seed for reproducibility |

### Initialization

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `prepare-tiles` | `( n:batch n:height n:width s:dir -- )` | Extract and encode tiles from images in directory |
| `analyze-audio` | `( n:bpm n:fps s:path -- )` | Analyze audio file, compute per-frame energy |
| `establish-themes` | `( n:num-themes -- )` | Select dispersed theme tiles, segment audio |

**prepare-tiles** reads all `*.jpg` files from the project directory, divides them into non-overlapping tiles, runs each through the encoder, and precomputes cosine similarities between all tile latents.

**analyze-audio** runs `scripts/audio_features.sh` to extract spectral features at the target fps. Energy is computed as smoothed, normalized flux + RMS, then re-normalized after smoothing to preserve full 0-1 dynamic range.

**establish-themes** uses greedy max-min selection to pick visually dispersed tiles as themes, then finds audio section boundaries based on timbral character (centroid + flatness).

### Generation

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `generate-transition` | `( n:theme-weight n:clock-div -- [img s:path]... n:frame n:count )` | Generate one transition |
| `get-total-frames` | `( -- n )` | Push total audio frames |
| `get-energy` | `( -- n )` | Push energy at current frame (0-1) |

**generate-transition** selects the next tile based on:
- **Theme affinity**: Cosine similarity to current theme tile (weighted by theme-weight)
- **Continuity**: Cosine similarity to previous tile (weighted by 1 - theme-weight)

Tiles are selected from a pool at minimum hit count, ensuring even distribution across passes.

The number of frames is computed as: `fps * 60 / bpm / clock-division`

Frames are decoded using slerp interpolation of latents and skip connections, then pushed to the image stack with their output filenames.

**Output logging** shows for each transition:
```
transition 1-7 ( 2.00x, 7 frames)
  tile 12 -> 25 (theme 1, pool 149)
  score= 0.732 [theme(0.50)= 0.987 + cont(0.50)= 0.477]
```

This shows:
- Frame range and clock division
- Tile transition, current theme section, pool size
- Score breakdown: total score with theme affinity and continuity components

### Effects

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `set-camera-motion` | `( n:pan-x n:pan-y n:rot n:zoom n:freq -- )` | Configure camera wobble |
| `wobble` | `( n:frame img -- img )` | Apply sinusoidal camera motion |
| `set-chroma` | `( n:prob n:max-offset n:period -- )` | Configure chromatic aberration |
| `roll-chroma` | `( -- )` | Roll dice for chroma activation |
| `aberrate` | `( n:frame img -- img )` | Apply chromatic aberration |

**Camera motion** applies sinusoidal pan, rotation, and zoom based on frame number:
- `pan-x`, `pan-y`: Maximum translation in pixels
- `rot`: Maximum rotation in degrees
- `zoom`: Maximum zoom factor (e.g., 0.02 = 2%)
- `freq`: Oscillation frequency (cycles per video)

Example: `3 2 0.5 0.005 0.3 set-camera-motion` sets subtle, slow motion.

**Chromatic aberration** shifts R and B channels in opposite directions:
- `prob`: Probability of activation when rolled (0-1)
- `max-offset`: Maximum pixel offset (scaled by energy)
- `period`: Seconds between automatic re-rolls (0 = manual only)

The `aberrate` word automatically calls `roll-chroma` at period boundaries.

Example: `0.5 8.0 2.0 set-chroma` gives 50% chance, up to 8px offset, re-rolling every 2 seconds.

### Finalization

| Word | Stack Effect | Description |
|------|--------------|-------------|
| `finalize-video` | `( n:scale n:fade-in n:fade-out s:color-in s:color-out s:frames s:output s:audio -- )` | Assemble final video |
| `preview-video` | `( n:scale n:fade-in n:fade-out s:color-in s:color-out s:frames s:output s:audio -- )` | Fast preview (no denoise) |

Both words assemble frames using FFmpeg with:
- Lanczos upscaling (if scale > 1)
- Fade in/out with specified colors
- Audio muxing
- H.264 encoding

**finalize-video** applies `nlmeans` denoising and `unsharp` for quality output.

**preview-video** skips denoising and uses `-preset ultrafast -threads 0` for fast iteration.

Example:
```forth
1 2.0 5.0 "black" "white"
    "projects/my-project/frames"
    "projects/my-project/output.mp4"
    "projects/my-project/audio.mp3"
    finalize-video
```

---

## Example Scripts

### Full Video Generation

```forth
clear-video

"models/ae-weights-576x384-add.bin" load-model
4 768 1152 "projects/my-project" prepare-tiles
120 24 "projects/my-project/audio.mp3" analyze-audio
4 establish-themes

# subtle effects
3 2 0.5 0.005 0.3 set-camera-motion
0.5 8.0 2.0 set-chroma

# helper: compute frame number from stack state
: effect-frame over over - ;

: flush-frames
    0?
    sdup s. cr
    effect-frame wobble
    effect-frame aberrate
    save
    1 - flush-frames ;

: do-transition
    0.5 get-energy 0.5 * 0.25 + generate-transition
    flush-frames
    drop ;

: main-loop
    get-total-frames >=?
    do-transition
    main-loop ;

1 main-loop

1 2.0 5.0 "black" "white"
    "projects/my-project/frames"
    "projects/my-project/output.mp4"
    "projects/my-project/audio.mp3"
    finalize-video
```

### Apply Effects to Existing Frames

```forth
# analyze audio for energy values
96 24 "projects/my-project/audio.mp3" analyze-audio

# set up effects
3 2 0.5 0.005 0.3 set-camera-motion
0.5 8.0 2.0 set-chroma

# load all frames
"projects/my-project/frames" list-files

: process-one
    0?
    sdup load
    dup wobble
    dup aberrate
    sdup s. cr
    "projects/my-project/frames-effects" save-in
    1 - process-one ;

process-one
drop

1 2.0 5.0 "black" "white"
    "projects/my-project/frames-effects"
    "projects/my-project/output-effects.mp4"
    "projects/my-project/audio.mp3"
    preview-video
```

### Clock Division Formula

The `generate-transition` word takes a clock division that determines transition length:

```
frames_per_transition = fps * 60 / bpm / clock_division
```

A common pattern modulates this by energy:

```forth
0.5 get-energy 0.5 * 0.25 + generate-transition
```

This computes: `clock_division = 0.25 + (energy * 0.5)`
- energy=0 (quiet): 0.25x = longer transitions
- energy=1 (loud): 0.75x = shorter transitions

At 96 bpm, 24 fps (15 frames/beat):
- 0.25x = 60 frames/transition
- 0.75x = 20 frames/transition

---

## Technical Details

### Tensor Layout

All tensors use Fortran column-major order: `(channels, height, width, batch)`

### Slerp Interpolation

Latent space interpolation uses spherical linear interpolation for smoother morphs:

```
slerp(a, b, t) = sin((1-t)*omega)/sin(omega) * a + sin(t*omega)/sin(omega) * b
```

where `omega = arccos(a . b / |a||b|)`. Falls back to linear interpolation when vectors are nearly parallel (dot product > 0.9995) or either vector has zero magnitude.

Skip connections are also slerp-interpolated for consistency.

### Audio Features

The `scripts/audio_features.sh` script extracts per-frame features using FFmpeg:

| Feature | Description |
|---------|-------------|
| centroid_l/r | Spectral centroid (brightness) |
| spread_l/r | Spectral spread |
| flux_l/r | Spectral flux (change) |
| flatness_l/r | Spectral flatness (noisiness) |
| rolloff_l/r | Spectral rolloff |
| rms_l/r | RMS level |
| peak_l/r | Peak level |

Energy is computed as:
1. Normalize flux and RMS to 0-1 range
2. Average: `(flux_norm + rms_norm) / 2`
3. Smooth with exponential moving average (alpha=0.25)
4. Re-normalize after smoothing to restore full 0-1 dynamic range

### Tile Selection Algorithm

1. Find tiles at minimum hit count (ensures even usage across passes)
2. For each candidate in pool, compute score:
   - `theme_weight * cosine(candidate, theme_tile)`
   - `+ (1 - theme_weight) * cosine(candidate, previous_tile)`
3. Select highest-scoring tile
4. When all tiles have equal hits, the pool resets (new pass begins)

### Theme Establishment

1. **Theme tiles**: Greedy max-min selection picks tiles maximally dispersed in latent space
2. **Audio boundaries**: Peaks in smoothed timbral character (centroid + flatness) define section changes
3. Current frame determines which theme section is active

---

## Project Structure

```
phanerotes/
├── src/fortran/
│   ├── phanerotes.f90      # Main entry point
│   ├── interpreter.f90     # Forth interpreter
│   ├── command.f90         # Stack operations, basic commands
│   ├── video.f90           # Video generation pipeline
│   ├── image.f90           # Image manipulation (affine transforms)
│   ├── cnn_autoencoder.f90 # Autoencoder forward/backward, slerp
│   ├── stb_bindings.f90    # C bindings for image I/O
│   └── utilities.f90       # File utilities
├── scripts/
│   └── audio_features.sh   # Audio feature extraction
├── models/                 # Trained autoencoder weights
├── projects/               # Project directories
│   └── <project>/
│       ├── *.jpg           # Source images
│       ├── *.mp3           # Audio file
│       ├── frames/         # Generated frames
│       └── *.phan          # Generation scripts
└── test/                   # Test suite
```
