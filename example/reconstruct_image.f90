program reconstruct_image
    use image
    use cnn_autoencoder
    use omp_lib
    implicit none

    type(autoencoder_config) :: config
    type(autoencoder) :: net
    real, allocatable :: img(:,:,:), reconstructed(:,:,:)
    real, allocatable :: tile(:,:,:,:), latent_mu(:,:,:,:), latent_log_var(:,:,:,:), output(:,:,:,:)
    integer :: width, height, channels
    integer :: tile_size, i, j, tiles_x, tiles_y
    integer :: x_start, x_end, y_start, y_end
    integer :: n, total_tiles
    logical :: success

    config%input_channels = 3
    config%num_layers = 3
    config%base_channels = 32
    config%max_channels = 128
    config%kernel_width = 3
    config%kernel_height = 3
    config%stride = 2
    config%padding = 1
    config%beta = 0.001

    tile_size = 512

    print *, "Initializing autoencoder..."
    net = autoencoder_init(config)

    print *, "Loading weights from models/ae-weights-512.bin..."
    call load_weights(net, "models/ae-weights-512.bin")
    call set_training(net, .false.)

    print *, "Loading image..."
    img = load_image("images/training_data/DSCF6922.jpg")
    channels = size(img, 1)
    width = size(img, 2)
    height = size(img, 3)
    print *, "Image size:", channels, "x", width, "x", height

    tiles_x = width / tile_size
    tiles_y = height / tile_size
    print *, "Tiles:", tiles_x, "x", tiles_y, "=", tiles_x * tiles_y

    allocate(reconstructed(channels, tiles_x * tile_size, tiles_y * tile_size))
    total_tiles = tiles_x * tiles_y

    !$omp parallel do private(n, i, j, x_start, x_end, y_start, y_end, tile, latent_mu, latent_log_var, output) schedule(dynamic)
    do n = 1, total_tiles
        i = mod(n - 1, tiles_x) + 1
        j = (n - 1) / tiles_x + 1

        x_start = (i-1) * tile_size + 1
        x_end = i * tile_size
        y_start = (j-1) * tile_size + 1
        y_end = j * tile_size

        allocate(tile(1, channels, tile_size, tile_size))
        tile(1, :, :, :) = img(:, x_start:x_end, y_start:y_end)
        call autoencoder_forward(net, tile, latent_mu, latent_log_var, output)
        reconstructed(:, x_start:x_end, y_start:y_end) = output(1, :, :, :)
        deallocate(tile)

        !$omp critical
        print '(A,I0,A,I0,A,I0,A,I0,A)', "  Tile ", n, "/", total_tiles, " (", i, ",", j, ") done"
        !$omp end critical
    end do
    !$omp end parallel do

    print *, "Saving reconstructed image..."
    call save_image("reconstructed.bmp", reconstructed, success)
    if (success) then
        print *, "Saved to reconstructed.bmp"
    else
        print *, "Failed to save"
    end if

end program
