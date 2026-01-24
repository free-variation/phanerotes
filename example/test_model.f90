program test_model
    use image
    use cnn_autoencoder
    implicit none

    type(autoencoder) :: net
    type(autoencoder_config) :: config
    real, allocatable :: img(:,:,:)
    real, allocatable :: tile(:,:,:,:), latent(:,:,:,:), output(:,:,:,:)
    integer :: width, height, channels
    integer :: tile_width, tile_height
    logical :: success
    real :: mse

    tile_width = 576
    tile_height = 384

    print *, "Creating fresh autoencoder..."
    config%input_channels = 3
    config%num_layers = 3
    config%base_channels = 32
    config%max_channels = 128
    config%kernel_width = 3
    config%kernel_height = 3
    config%stride = 2
    config%padding = 1
    config%concatenate = .false.
    net = autoencoder_init(config)
    call set_training(net, .false.)

    print *, "Loading image..."
    img = load_image("images/training_data/DSCF5543.jpg")
    channels = size(img, 1)
    height = size(img, 2)
    width = size(img, 3)
    print *, "Image size:", channels, "x", height, "x", width

    ! Extract one tile from center of image
    allocate(tile(channels, tile_height, tile_width, 1))
    tile(:, :, :, 1) = img(:, height/2:height/2+tile_height-1, width/2:width/2+tile_width-1)

    print *, "Running forward pass..."
    call autoencoder_forward(net, tile, 0.0, latent, output)

    print *, "Latent range:", minval(latent), "-", maxval(latent)

    print *, "Input shape:", shape(tile)
    print *, "Latent shape:", shape(latent)
    print *, "Output shape:", shape(output)

    ! Check values
    print *, "Input R:", minval(tile(1,:,:,:)), "-", maxval(tile(1,:,:,:))
    print *, "Input G:", minval(tile(2,:,:,:)), "-", maxval(tile(2,:,:,:))
    print *, "Input B:", minval(tile(3,:,:,:)), "-", maxval(tile(3,:,:,:))
    print *, "Output R:", minval(output(1,:,:,:)), "-", maxval(output(1,:,:,:))
    print *, "Output G:", minval(output(2,:,:,:)), "-", maxval(output(2,:,:,:))
    print *, "Output B:", minval(output(3,:,:,:)), "-", maxval(output(3,:,:,:))

    mse = sum((output - tile)**2) / size(output)
    print *, "MSE:", mse

    ! Find brightest pixel in input
    print *, "Sample dark pixel (192,288):", tile(:, 192, 288, 1), "->", output(:, 192, 288, 1)

    ! Check pixel stats
    print *, "Input mean:", sum(tile) / size(tile)
    print *, "Output mean:", sum(output) / size(output)

    ! Save input and output for visual comparison
    call save_image("test_input.bmp", tile(:,:,:,1), success)
    print *, "Saved test_input.bmp:", success

    call save_image("test_output.bmp", output(:,:,:,1), success)
    print *, "Saved test_output.bmp:", success

end program
