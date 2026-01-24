# phanerotes

Convolutional autoencoder with U-Net style skip connections for image reconstruction and latent space interpolation.

## Building

Requires:
- Fortran compiler (gfortran or ifort)
- BLAS library
- fpm (Fortran Package Manager)

```bash
fpm build --profile release
```

For OpenMP support (parallel tile processing):
```bash
fpm build --profile release --flag "-fopenmp"
```

## Training

Train an autoencoder on images in `images/training_data/`:

```bash
fpm run --example train_full_images --profile release -- <tile_width> [options]
```

Options:
- `--height N` - tile height (default: same as width)
- `--epochs N` - number of training epochs (default: 10)
- `--add` - use addition for skip connections
- `--concat` - use concatenation for skip connections (default)

Examples:
```bash
# 512x512 tiles, 50 epochs, concatenation mode
fpm run --example train_full_images --profile release -- 512 --epochs 50

# 576x384 tiles, addition mode
fpm run --example train_full_images --profile release -- 576 --height 384 --add
```

Output: `ae-weights-{width}x{height}-{mode}.bin`

## Video Generation

Generate a video by interpolating between image tiles in latent space:

```bash
fpm run --example make_video --profile release --flag "-fopenmp" -- <model_file> <image_file> [options]
```

Options:
- `--sharpen` - apply sharpening (default)
- `--no-sharpen` - no sharpening
- `--pixel` - interpolate pixels (default, fast)
- `--latent` - interpolate in latent space (slower, smoother)

Examples:
```bash
# Pixel interpolation (fast)
fpm run --example make_video --profile release --flag "-fopenmp" -- ae-weights-512x512-concat.bin image.jpg

# Latent space interpolation (smooth)
fpm run --example make_video --profile release --flag "-fopenmp" -- ae-weights-512x512-concat.bin image.jpg --latent
```

Output: `video_1024.mp4` (requires ffmpeg)

## Testing

```bash
fpm test
```
