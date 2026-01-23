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
    character(len=64) :: training_dir, arg
    integer :: i, j, k, idx, num_images, channels, width, height, unit_num, ios
    integer :: tile_width, tile_height, tiles_x, tiles_y, tiles_per_image, total_tiles, batch_size

    ! Parse arguments
    if (command_argument_count() < 1) then
        print *, "Usage: train_full_images <tile_width> [tile_height]"
        print *, "  tile_width  - width of tiles (e.g., 512)"
        print *, "  tile_height - height of tiles (default: same as width)"
        print *, "Examples:"
        print *, "  train_full_images 512        # 512x512 square tiles"
        print *, "  train_full_images 576 384    # 576x384 (3:2) tiles"
        stop 1
    end if

    call get_command_argument(1, arg)
    read(arg, *) tile_width

    if (command_argument_count() >= 2) then
        call get_command_argument(2, arg)
        read(arg, *) tile_height
    else
        tile_height = tile_width
    end if

    training_dir = "images/training_data/"
    batch_size = 8

    print *, "Tile size:", tile_width, "x", tile_height

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
    tiles_x = width / tile_width
    tiles_y = height / tile_height
    tiles_per_image = tiles_x * tiles_y
    total_tiles = num_images * tiles_per_image
    print *, "Tiles per image:", tiles_per_image, "(", tiles_x, "x", tiles_y, ")"
    print *, "Total tiles:", total_tiles

    ! Allocate tiles array
    allocate(tiles(total_tiles, channels, tile_width, tile_height))

    ! Load each image and extract tiles
    idx = 1
    open(newunit=unit_num, file="/tmp/image_list.txt", status="old")
    do k = 1, num_images
        read(unit_num, '(A)') filename
        print *, "Loading:", trim(filename)
        img = load_image(trim(training_dir) // trim(filename))

        do j = 1, tiles_y
            do i = 1, tiles_x
                tiles(idx, :, :, :) = img(:, (i-1)*tile_width+1:i*tile_width, (j-1)*tile_height+1:j*tile_height)
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

    write(filename, '(A,I0,A,I0,A)') "ae-weights-", tile_width, "x", tile_height, ".bin"
    call save_weights(net, trim(filename))
    print *, "Saved weights to ", trim(filename)

end program
