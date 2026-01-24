# phanerotes

Convolutional autoencoder with U-Net style skip connections for image reconstruction and latent space interpolation.

## Building

Requires:
- Fortran compiler (gfortran or ifort)
- BLAS library
- fpm (Fortran Package Manager)

```bash
fpm build --profile release --flag "-march=native -ffast-math -fopenmp"
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

## Video Generation

Generate a video by interpolating between image tiles in latent space:

```bash
fpm run --example make_video --profile release --flag "-march=native -ffast-math -fopenmp" -- <model_file> <image_file> <tile_width> [options]
```

Options:
- `--height N` - tile height (default: same as width)
- `--frames N` - total frames (default: 7200)
- `--fps N` - frames per second (default: 24)
- `--scale N` - upscale ratio (default: 1, no upscaling)
- `--sharpen` - apply sharpening (default)
- `--no-sharpen` - no sharpening
- `--pixel` - interpolate pixels (default, fast)
- `--latent` - interpolate in latent space (slower, smoother)

Examples:
```bash
# 512x512 tiles, pixel interpolation
fpm run --example make_video --profile release --flag "-march=native -ffast-math -fopenmp" -- model.bin image.jpg 512

# 576x384 tiles, latent interpolation
fpm run --example make_video --profile release --flag "-march=native -ffast-math -fopenmp" -- model.bin image.jpg 576 --height 384 --latent

# 512x512 tiles, upscale 2x to 1024x1024
fpm run --example make_video --profile release --flag "-march=native -ffast-math -fopenmp" -- model.bin image.jpg 512 --scale 2
```

Output: `video_<width>x<height>.mp4` (requires ffmpeg)

## Testing

```bash
fpm test
```
