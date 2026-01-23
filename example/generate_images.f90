program generate_images
    use image
    use cnn_autoencoder
    implicit none

    type(autoencoder_config) :: config
    type(autoencoder) :: net
    real, allocatable :: latent(:,:,:,:), output(:,:,:,:)
    real, allocatable :: u1(:,:,:,:), u2(:,:,:,:)
    character(len=256) :: weights_file, output_file, arg
    integer :: i, tile_width, tile_height, latent_w, latent_h, latent_channels
    logical :: success

    if (command_argument_count() < 1) then
        print *, "Usage: generate_images <weights_file> [tile_width] [tile_height]"
        print *, "  weights_file - path to trained weights (e.g., ae-weights-576x384.bin)"
        print *, "  tile_width   - width of output images (default: 576)"
        print *, "  tile_height  - height of output images (default: 384)"
        stop 1
    end if

    call get_command_argument(1, weights_file)

    tile_width = 576
    tile_height = 384
    if (command_argument_count() >= 2) then
        call get_command_argument(2, arg)
        read(arg, *) tile_width
    end if
    if (command_argument_count() >= 3) then
        call get_command_argument(3, arg)
        read(arg, *) tile_height
    end if

    config%input_channels = 3
    config%num_layers = 3
    config%base_channels = 32
    config%max_channels = 128
    config%kernel_width = 3
    config%kernel_height = 3
    config%stride = 2
    config%padding = 1
    config%beta = 0.0

    net = autoencoder_init(config)
    call set_training(net, .false.)
    call load_weights(net, trim(weights_file))
    print *, "Loaded weights from ", trim(weights_file)

    latent_channels = min(config%base_channels * 2**config%num_layers, config%max_channels)
    latent_w = tile_width / (config%stride ** config%num_layers)
    latent_h = tile_height / (config%stride ** config%num_layers)
    print *, "Latent shape: ", latent_channels, latent_w, latent_h

    allocate(latent(1, latent_channels, latent_w, latent_h))
    allocate(u1(1, latent_channels, latent_w, latent_h))
    allocate(u2(1, latent_channels, latent_w, latent_h))

    do i = 1, 30
        call random_number(u1)
        call random_number(u2)
        u1 = max(u1, 1.0e-10)
        latent = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)

        call decode_latent(net, latent, output)

        write(output_file, '(A,I0,A)') "generated_", i, ".bmp"
        call save_image(trim(output_file), output(1,:,:,:), success)
        if (success) then
            print *, "Saved ", trim(output_file)
        else
            print *, "Failed to save ", trim(output_file)
        end if
    end do

end program
