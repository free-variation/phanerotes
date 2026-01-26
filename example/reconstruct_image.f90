program reconstruct_image
    use image
    use cnn_autoencoder
    implicit none

    type(autoencoder) :: net
    real, allocatable :: img(:,:,:), input(:,:,:,:), latent(:,:,:,:), output(:,:,:,:)
    integer :: width, height, channels, i, ios, unit_num
    logical :: success
    character(len=512) :: model_file, input_dir, output_dir, image_file, out_file
    character(len=512), allocatable :: image_files(:)
    integer :: num_images, img_idx

    if (command_argument_count() < 3) then
        print *, "Usage: reconstruct_image <model_file> <input_dir> <output_dir>"
        stop 1
    end if

    call get_command_argument(1, model_file)
    call get_command_argument(2, input_dir)
    call get_command_argument(3, output_dir)

    print *, "Loading autoencoder from ", trim(model_file)
    net = load_autoencoder(trim(model_file))
    call set_training(net, .false.)

    ! List images in directory
    call execute_command_line("mkdir -p " // trim(output_dir))
    call execute_command_line("ls " // trim(input_dir) // "/*.jpg " // trim(input_dir) // "/*.JPG " // &
        trim(input_dir) // "/*.bmp " // trim(input_dir) // "/*.png 2>/dev/null > /tmp/image_list.txt")

    ! Count images
    num_images = 0
    open(newunit=unit_num, file="/tmp/image_list.txt", status="old")
    do
        read(unit_num, '(A)', iostat=ios) image_file
        if (ios /= 0) exit
        num_images = num_images + 1
    end do
    close(unit_num)
    print *, "Found", num_images, "images"

    ! Read image filenames
    allocate(image_files(num_images))
    open(newunit=unit_num, file="/tmp/image_list.txt", status="old")
    do i = 1, num_images
        read(unit_num, '(A)') image_files(i)
    end do
    close(unit_num)

    ! Process each image
    do img_idx = 1, num_images
        print *, "Processing ", trim(image_files(img_idx))
        img = load_image(trim(image_files(img_idx)))

        channels = size(img, 1)
        height = size(img, 2)
        width = size(img, 3)
        print *, "  Size:", channels, "x", height, "x", width

        ! Wrap in batch dimension: (channels, height, width, 1)
        allocate(input(channels, height, width, 1))
        input(:, :, :, 1) = img

        call autoencoder_forward(net, input, 0.0, latent, output)

        ! Extract output filename
        i = index(image_files(img_idx), "/", back=.true.)
        out_file = trim(output_dir) // "/" // image_files(img_idx)(i+1:)
        i = index(out_file, ".", back=.true.)
        out_file = out_file(1:i) // "bmp"

        call save_image(trim(out_file), output(:,:,:,1), success)
        if (success) then
            print *, "  Saved:", trim(out_file)
        else
            print *, "  Failed to save"
        end if

        deallocate(input)
    end do

end program
