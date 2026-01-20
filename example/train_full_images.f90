program train_full_images
    use image
    use cnn_autoencoder
    use train
    implicit none

    type(autoencoder_config) :: config
    type(autoencoder) :: net
    real, allocatable :: tiles(:,:,:,:)
    real, allocatable :: img(:,:,:)
    character(len=512) :: filename
    character(len=64) :: training_dir
    integer :: i, j, k, idx, num_images, channels, width, height, unit_num, ios
    integer :: tile_size, tiles_x, tiles_y, tiles_per_image, total_tiles, batch_size

    training_dir = "images/training_data/"
    tile_size = 512
    batch_size = 4

    ! Get file count by listing directory to temp file
    call execute_command_line("ls " // trim(training_dir) // " > /tmp/image_list.txt")

    ! Count images
    num_images = 0
    open(newunit=unit_num, file="/tmp/image_list.txt", status="old")
    do
        read(unit_num, '(A)', iostat=ios) filename
        if (ios /= 0) exit
        num_images = num_images + 1
    end do
    close(unit_num)

    print *, "Found", num_images, "images"

    ! Load first image to get dimensions
    open(newunit=unit_num, file="/tmp/image_list.txt", status="old")
    read(unit_num, '(A)') filename
    close(unit_num)

    img = load_image(trim(training_dir) // trim(filename))
    channels = size(img, 1)
    width = size(img, 2)
    height = size(img, 3)
    print *, "Image dimensions:", channels, "x", width, "x", height

    ! Calculate tiles
    tiles_x = width / tile_size
    tiles_y = height / tile_size
    tiles_per_image = tiles_x * tiles_y
    total_tiles = num_images * tiles_per_image
    print *, "Tiles per image:", tiles_per_image, "(", tiles_x, "x", tiles_y, ")"
    print *, "Total tiles:", total_tiles

    ! Allocate tiles array
    allocate(tiles(total_tiles, channels, tile_size, tile_size))

    ! Load each image and extract tiles
    idx = 1
    open(newunit=unit_num, file="/tmp/image_list.txt", status="old")
    do k = 1, num_images
        read(unit_num, '(A)') filename
        print *, "Loading:", trim(filename)
        img = load_image(trim(training_dir) // trim(filename))

        do j = 1, tiles_y
            do i = 1, tiles_x
                tiles(idx, :, :, :) = img(:, (i-1)*tile_size+1:i*tile_size, (j-1)*tile_size+1:j*tile_size)
                idx = idx + 1
            end do
        end do
    end do
    close(unit_num)

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

    ! Train
    print *, "Training..."
    call train_network(net, tiles, batch_size, 10, 0.01)

    call save_weights(net, "autoencoder_weights.bin")
    print *, "Saved weights to autoencoder_weights.bin"

end program
