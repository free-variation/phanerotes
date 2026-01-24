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
    character(len=512), allocatable :: filenames(:)
    character(len=64) :: training_dir, arg
    integer :: i, j, k, idx, num_images, channels, width, height, unit_num, ios
    integer :: base_idx
    integer :: tile_width, tile_height, tiles_x, tiles_y, tiles_per_image, total_tiles, batch_size
    integer :: num_epochs
    logical :: use_concatenate
    integer :: concat_mode  ! -1 = not set, 0 = add, 1 = concat
    character(len=16) :: mode_str

    ! Parse arguments
    if (command_argument_count() < 1) then
        print *, "Usage: train_full_images <tile_width> [options]"
        print *, "  tile_width  - width of tiles (e.g., 512)"
        print *, "Options:"
        print *, "  --height N  - tile height (default: same as width)"
        print *, "  --epochs N  - number of training epochs (default: 10)"
        print *, "  --batch N   - batch size (default: 8)"
        print *, "  --add       - use addition for skip connections"
        print *, "  --concat    - use concatenation for skip connections (default)"
        print *, "Examples:"
        print *, "  train_full_images 512                      # 512x512, 10 epochs"
        print *, "  train_full_images 512 --epochs 50          # 512x512, 50 epochs"
        print *, "  train_full_images 576 --height 384 --add   # 576x384, addition mode"
        stop 1
    end if

    call get_command_argument(1, arg)
    read(arg, *) tile_width

    tile_height = tile_width
    concat_mode = -1  ! not set
    num_epochs = 10
    batch_size = 8

    i = 2
    do while (i <= command_argument_count())
        call get_command_argument(i, arg)
        if (trim(arg) == "--add") then
            if (concat_mode == 1) then
                print *, "Error: --add and --concat are mutually exclusive"
                stop 1
            end if
            concat_mode = 0
        else if (trim(arg) == "--concat") then
            if (concat_mode == 0) then
                print *, "Error: --add and --concat are mutually exclusive"
                stop 1
            end if
            concat_mode = 1
        else if (trim(arg) == "--height") then
            i = i + 1
            call get_command_argument(i, arg)
            read(arg, *) tile_height
        else if (trim(arg) == "--epochs") then
            i = i + 1
            call get_command_argument(i, arg)
            read(arg, *) num_epochs
        else if (trim(arg) == "--batch") then
            i = i + 1
            call get_command_argument(i, arg)
            read(arg, *) batch_size
        end if
        i = i + 1
    end do

    ! Default to concatenate if not specified
    use_concatenate = (concat_mode /= 0)

    training_dir = "images/training_data/"

    print *, "Tile size:", tile_width, "x", tile_height
    print *, "Epochs:", num_epochs
    print *, "Batch size:", batch_size

    ! Get file count by listing directory to temp file
    call execute_command_line("ls " // trim(training_dir) // " > /tmp/image_list.txt")

    ! Count images and store filenames
    num_images = 0
    open(newunit=unit_num, file="/tmp/image_list.txt", status="old")
    do
        read(unit_num, '(A)', iostat=ios) filename
        if (ios /= 0) exit
        num_images = num_images + 1
    end do
    close(unit_num)

    print *, "Found", num_images, "images"

    ! Store all filenames
    allocate(filenames(num_images))
    open(newunit=unit_num, file="/tmp/image_list.txt", status="old")
    do k = 1, num_images
        read(unit_num, '(A)') filenames(k)
    end do
    close(unit_num)

    ! Load first image to get dimensions
    img = load_image(trim(training_dir) // trim(filenames(1)))
    ! Image layout: (channels, height, width)
    channels = size(img, 1)
    height = size(img, 2)
    width = size(img, 3)
    print *, "Image dimensions:", channels, "x", height, "x", width

    ! Calculate tiles
    tiles_x = width / tile_width
    tiles_y = height / tile_height
    tiles_per_image = tiles_x * tiles_y
    total_tiles = num_images * tiles_per_image
    print *, "Tiles per image:", tiles_per_image, "(", tiles_x, "x", tiles_y, ")"
    print *, "Total tiles:", total_tiles

    ! Allocate tiles array: (channels, tile_height, tile_width, total_tiles)
    allocate(tiles(channels, tile_height, tile_width, total_tiles))

    ! Load each image and extract tiles (parallel)
    print *, "Loading images..."
    !$omp parallel do private(k, img, i, j, idx, base_idx) schedule(dynamic)
    do k = 1, num_images
        img = load_image(trim(training_dir) // trim(filenames(k)))
        base_idx = (k - 1) * tiles_per_image

        do j = 1, tiles_y
            do i = 1, tiles_x
                idx = base_idx + (j - 1) * tiles_x + i
                ! img is (channels, height, width), tile is (channels, tile_height, tile_width)
                tiles(:, :, :, idx) = img(:, (j-1)*tile_height+1:j*tile_height, (i-1)*tile_width+1:i*tile_width)
            end do
        end do

        !$omp critical
        print *, "Loaded:", trim(filenames(k))
        !$omp end critical
    end do
    !$omp end parallel do

    ! Create autoencoder
    config%input_channels = channels
    config%num_layers = 3
    config%base_channels = 32
    config%max_channels = 128
    config%kernel_width = 3
    config%kernel_height = 3
    config%stride = 2
    config%padding = 1
    config%concatenate = use_concatenate

    if (use_concatenate) then
        mode_str = "concat"
    else
        mode_str = "add"
    end if
    print *, "Skip connection mode:", trim(mode_str)
    print *, "Initializing autoencoder..."
    net = autoencoder_init(config)

    ! Train
    print *, "Training..."
    call train_network(net, tiles, batch_size, num_epochs, 0.01, 0.2)

    write(filename, '(A,I0,A,I0,A,A,A)') "ae-weights-", tile_width, "x", tile_height, "-", trim(mode_str), ".bin"
    call save_autoencoder(net, trim(filename))
    print *, "Saved weights to ", trim(filename)

end program
