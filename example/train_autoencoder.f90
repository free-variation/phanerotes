program train_autoencoder
    use image
    use cnn_autoencoder
    use train
    implicit none

    type(autoencoder_config) :: config
    type(autoencoder) :: net
    real, allocatable :: img(:,:,:)
    real, allocatable :: tiles(:,:,:,:)
    integer :: width, height, channels
    integer :: tile_size, tiles_x, tiles_y, num_tiles
    integer :: i, j, idx, batch_size

    batch_size = 32

    ! Load image
    print *, "Loading image..."
    img = load_image("images/test1.jpg")
    channels = size(img, 1)
    width = size(img, 2)
    height = size(img, 3)
    print *, "Image size:", channels, "x", width, "x", height

    ! Chop into tiles
    tile_size = 32
    tiles_x = width / tile_size
    tiles_y = height / tile_size
    num_tiles = tiles_x * tiles_y
    print *, "Creating", num_tiles, "tiles of size", tile_size

    allocate(tiles(num_tiles, channels, tile_size, tile_size))
    idx = 1
    do j = 1, tiles_y
        do i = 1, tiles_x
            tiles(idx, :, :, :) = img(:, (i-1)*tile_size+1:i*tile_size, (j-1)*tile_size+1:j*tile_size)
            idx = idx + 1
        end do
    end do

    ! Create autoencoder
    config%input_channels = channels
    config%num_layers = 3
    config%base_channels = 32
    config%max_channels = 128
    config%kernel_width = 3
    config%kernel_height = 3
    config%stride = 2
    config%padding = 1

    print *, "Initializing autoencoder..."
    net = autoencoder_init(config)

    ! Train and save weights
    call train_network(net, tiles, batch_size, 20, 0.01, 0.2, "autoencoder")
    print *, "Training complete."

end program
