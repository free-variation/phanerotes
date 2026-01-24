program test_reconstruction
    use image
    use cnn_autoencoder
    use train
    implicit none

    type(autoencoder_config) :: config
    type(autoencoder) :: net
    real, allocatable :: images(:,:,:,:), img(:,:,:)
    real, allocatable :: latent(:,:,:,:), output(:,:,:,:)
    character(len=256) :: resized_file, out_file, cmd
    character(len=64) :: src_files(4)
    integer :: i, img_size, num_images, num_epochs, mode
    logical :: use_concatenate, success
    real :: dropout_rate
    character(len=16) :: mode_name
    character(len=8) :: size_str

    src_files(1) = "images/training_data/DSCF6833.JPG"
    src_files(2) = "images/training_data/DSCF6949.jpg"
    src_files(3) = "images/training_data/DSCF6974.jpg"
    src_files(4) = "images/training_data/P2050076.jpg"

    img_size = 128
    num_images = 4
    num_epochs = 200

    write(size_str, '(I0)') img_size

    ! Resize images using sips
    print *, "Resizing images to", img_size, "x", img_size
    do i = 1, num_images
        write(resized_file, '(A,I0,A)') "images/test_input_", i, ".png"
        cmd = "sips -z " // trim(size_str) // " " // trim(size_str) // &
              " " // trim(src_files(i)) // " --out " // trim(resized_file) // " >/dev/null 2>&1"
        call execute_command_line(trim(cmd))
        print *, "  ", trim(resized_file)
    end do

    ! Load resized images
    print *, "Loading images..."
    allocate(images(num_images, 3, img_size, img_size))
    do i = 1, num_images
        write(resized_file, '(A,I0,A)') "images/test_input_", i, ".png"
        img = load_image(trim(resized_file))
        images(i, :, :, :) = img
    end do

    ! Test three modes: concat, add, add with no skip connections
    do mode = 1, 3
        if (mode == 1) then
            use_concatenate = .true.
            dropout_rate = 0.1
            mode_name = "concat"
            print *, "=== Testing CONCAT mode ==="
        else if (mode == 2) then
            use_concatenate = .false.
            dropout_rate = 0.1
            mode_name = "add"
            print *, "=== Testing ADD mode ==="
        else
            use_concatenate = .false.
            dropout_rate = 1.0
            mode_name = "noskip"
            print *, "=== Testing ADD mode (no skip, dropout=1.0) ==="
        end if

        config%input_channels = 3
        config%num_layers = 3
        config%base_channels = 32
        config%max_channels = 128
        config%kernel_width = 3
        config%kernel_height = 3
        config%stride = 2
        config%padding = 1
        config%concatenate = use_concatenate

        net = autoencoder_init(config)

        print *, "Training..."
        call train_network(net, images, num_images, num_epochs, 0.1, dropout_rate)

        print *, "Generating reconstructions..."
        call set_training(net, .false.)

        do i = 1, num_images
            call autoencoder_forward(net, images(i:i,:,:,:), 0.0, latent, output)
            write(out_file, '(A,I0,A,A,A)') "images/test_output_", i, "_", trim(mode_name), ".bmp"
            call save_image(trim(out_file), output(1,:,:,:), success)
            print *, "  Saved:", trim(out_file)
        end do
    end do

    print *, "Done. Compare test_input_*.png with test_output_*_{concat,add,noskip}.bmp"

end program
