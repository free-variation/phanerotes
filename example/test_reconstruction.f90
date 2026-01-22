program test_reconstruction
    use image
    use cnn_autoencoder
    implicit none

    type(autoencoder_config) :: config
    type(autoencoder) :: net
    real, allocatable :: img(:,:,:)
    real, allocatable :: tile(:,:,:,:), latent_mu(:,:,:,:), latent_log_var(:,:,:,:), output(:,:,:,:)
    real, allocatable :: tile_3d(:,:,:), output_3d(:,:,:)
    integer, allocatable :: tile_x(:), tile_y(:)
    integer :: width, height, channels
    integer :: tile_size, i, j, n, tiles_x, tiles_y
    logical :: success
    character(len=256) :: input_file, output_file, image_file

    ! Config must match what was used for training
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

    print *, "Loading weights from autoencoder_weights.bin..."
    call load_weights(net, "autoencoder_weights.bin")

    ! Load a specific image
    print *, "Loading test image..."
    image_file = "test1_resized.jpg"
    print *, "Using:", trim(image_file)
    img = load_image("images/" // trim(image_file))
    channels = size(img, 1)
    width = size(img, 2)
    height = size(img, 3)
    print *, "Image size:", channels, "x", width, "x", height
    print *, "Image brightness range:", minval(img), maxval(img)

    ! Get tiles from a 3x3 grid across the image
    tiles_x = width / tile_size
    tiles_y = height / tile_size

    allocate(tile_x(9))
    allocate(tile_y(9))

    ! Sample at 1/4, 1/2, 3/4 positions
    n = 0
    do j = 1, 3
        do i = 1, 3
            n = n + 1
            tile_x(n) = (tiles_x * i) / 4
            tile_y(n) = (tiles_y * j) / 4
        end do
    end do

    ! Process 9 tiles from grid
    allocate(tile(1, channels, tile_size, tile_size))
    allocate(tile_3d(channels, tile_size, tile_size))
    allocate(output_3d(channels, tile_size, tile_size))

    do n = 1, 9
        i = tile_x(n)
        j = tile_y(n)

        ! Extract tile
        tile(1, :, :, :) = img(:, (i-1)*tile_size+1:i*tile_size, (j-1)*tile_size+1:j*tile_size)

        ! Run through autoencoder
        call autoencoder_forward(net, tile, latent_mu, latent_log_var, output)

        print *, ""
        print *, "Tile", n, "at grid pos (", i, ",", j, ")"
        print *, "  Input range:", minval(tile), maxval(tile)
        print *, "  Output range:", minval(output), maxval(output)

        ! Save input tile
        tile_3d = tile(1, :, :, :)
        write(input_file, '(A,I0,A)') "tile_", n, "_input.bmp"
        call save_image(trim(input_file), tile_3d, success)
        print *, "  Saved:", trim(input_file)

        ! Save output reconstruction
        output_3d = output(1, :, :, :)
        write(output_file, '(A,I0,A)') "tile_", n, "_output.bmp"
        call save_image(trim(output_file), output_3d, success)
        print *, "  Saved:", trim(output_file)
    end do

    print *, ""
    print *, "Done. Check tile_*_input.bmp and tile_*_output.bmp files."

end program
