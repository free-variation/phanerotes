program make_video
    use image
    use cnn_autoencoder
    use omp_lib
    implicit none

    type(autoencoder_config) :: config
    type(autoencoder) :: net
    real, allocatable :: img(:,:,:)
    real, allocatable :: tile(:,:,:,:), latent(:,:,:,:), output(:,:,:,:)
    real, allocatable :: all_latents(:,:,:,:,:)  ! (num_tiles, batch, channels, w, h)
    real, allocatable :: all_outputs(:,:,:,:,:)  ! (num_tiles, batch, channels, w, h)
    real, allocatable :: latent_flat(:,:)        ! (num_tiles, flattened_size)
    real, allocatable :: norms(:)
    integer, allocatable :: sequence(:)
    logical, allocatable :: used(:)

    integer :: width, height, channels
    integer :: tile_size, i, j, n, tiles_x, tiles_y, num_tiles
    integer :: x_start, x_end, y_start, y_end
    integer :: latent_size, current, best_next
    real :: best_sim, sim

    ! Video parameters
    integer :: fps, duration, total_frames, frames_per_transition, num_transitions
    integer :: frame_num, trans, f
    real :: t
    real, allocatable :: frame(:,:,:)
    character(len=256) :: frame_file, cmd, input_image, output_video, sharpen_arg
    logical :: success
    integer :: sharpen_mode  ! 0=none, 1=sharpen

    ! Setup
    config%input_channels = 3
    config%num_layers = 3
    config%base_channels = 32
    config%max_channels = 128
    config%kernel_width = 3
    config%kernel_height = 3
    config%stride = 2
    config%padding = 1

    tile_size = 512
    fps = 24
    duration = 300
    total_frames = fps * duration  ! 7200 frames

    ! Parse arguments
    if (command_argument_count() < 1) then
        print *, "Usage: make_video <image_file> [none|sharpen]"
        print *, "  none   - no sharpening"
        print *, "  sharpen - apply sharpening (default)"
        stop 1
    end if

    call get_command_argument(1, input_image)
    output_video = "video_1024.mp4"

    sharpen_mode = 1  ! default: sharpen
    if (command_argument_count() >= 2) then
        call get_command_argument(2, sharpen_arg)
        if (trim(sharpen_arg) == "none") then
            sharpen_mode = 0
        end if
    end if

    print *, "Input:", trim(input_image)
    print *, "Output:", trim(output_video)
    if (sharpen_mode == 0) then
        print *, "Sharpening: none"
    else
        print *, "Sharpening: enabled"
    end if

    print *, "Initializing autoencoder..."
    net = autoencoder_init(config)
    call load_weights(net, "ae-weights-576x384.bin")
    call set_training(net, .false.)

    print *, "Loading image..."
    img = load_image(trim(input_image))
    channels = size(img, 1)
    width = size(img, 2)
    height = size(img, 3)
    print *, "Image size:", channels, "x", width, "x", height

    tiles_x = width / tile_size
    tiles_y = height / tile_size
    num_tiles = tiles_x * tiles_y
    print *, "Tiles:", tiles_x, "x", tiles_y, "=", num_tiles

    ! Number of transitions is num_tiles - 1 (for pairs: 1+2, 2+3, ..., (n-1)+n)
    num_transitions = num_tiles - 1
    frames_per_transition = total_frames / num_transitions
    print *, "Transitions:", num_transitions
    print *, "Frames per transition:", frames_per_transition

    ! Extract tiles and compute latents
    print *, "Processing tiles..."

    ! First pass to get latent dimensions
    allocate(tile(1, channels, tile_size, tile_size))
    tile(1, :, :, :) = img(:, 1:tile_size, 1:tile_size)
    call autoencoder_forward(net, tile, 0.0, latent, output)
    latent_size = size(latent, 2) * size(latent, 3) * size(latent, 4)

    allocate(all_latents(num_tiles, size(latent,1), size(latent,2), size(latent,3), size(latent,4)))
    allocate(all_outputs(num_tiles, size(output,1), size(output,2), size(output,3), size(output,4)))
    allocate(latent_flat(num_tiles, latent_size))
    allocate(norms(num_tiles))
    deallocate(tile)

    !$omp parallel do private(n, i, j, x_start, x_end, y_start, y_end, tile, latent, output) schedule(dynamic)
    do n = 1, num_tiles
        i = mod(n - 1, tiles_x) + 1
        j = (n - 1) / tiles_x + 1

        x_start = (i-1) * tile_size + 1
        x_end = i * tile_size
        y_start = (j-1) * tile_size + 1
        y_end = j * tile_size

        allocate(tile(1, channels, tile_size, tile_size))
        tile(1, :, :, :) = img(:, x_start:x_end, y_start:y_end)
        call autoencoder_forward(net, tile, 0.0, latent, output)

        all_latents(n, :, :, :, :) = latent
        all_outputs(n, :, :, :, :) = output
        latent_flat(n, :) = reshape(latent, [latent_size])
        norms(n) = sqrt(sum(latent_flat(n, :)**2))
        deallocate(tile)

        !$omp critical
        print '(A,I0,A,I0)', "  Tile ", n, "/", num_tiles
        !$omp end critical
    end do
    !$omp end parallel do

    ! Build sequence using cosine similarity (greedy nearest neighbor)
    print *, "Building sequence..."
    allocate(sequence(num_tiles))
    allocate(used(num_tiles))
    used = .false.

    ! Random starting tile
    call random_number(sim)
    sequence(1) = int(sim * num_tiles) + 1
    if (sequence(1) > num_tiles) sequence(1) = num_tiles
    used(sequence(1)) = .true.
    print '(A,I0)', "  Starting tile: ", sequence(1)

    do n = 2, num_tiles
        current = sequence(n-1)
        best_sim = -2.0
        best_next = -1

        do i = 1, num_tiles
            if (.not. used(i)) then
                sim = sum(latent_flat(current, :) * latent_flat(i, :)) / (norms(current) * norms(i))
                if (sim > best_sim) then
                    best_sim = sim
                    best_next = i
                end if
            end if
        end do

        sequence(n) = best_next
        used(best_next) = .true.
        print '(A,I0,A,I0,A,F6.3)', "  Step ", n, ": tile ", best_next, " (sim=", best_sim, ")"
    end do

    ! Create frames directory
    call execute_command_line("mkdir -p /tmp/video_frames")

    ! Generate frames
    ! Each frame shows pair of tiles: sequence(trans) + sequence(trans+1)
    ! Interpolate from pair (trans, trans+1) to pair (trans+1, trans+2)
    print *, "Generating frames..."

    !$omp parallel do private(frame_num, trans, f, t, frame, frame_file, success) schedule(dynamic)
    do frame_num = 1, total_frames
        ! Compute which transition and position within transition
        trans = (frame_num - 1) / frames_per_transition + 1
        f = mod(frame_num - 1, frames_per_transition)

        if (trans > num_transitions) then
            trans = num_transitions
            f = frames_per_transition
        end if

        t = real(f) / real(frames_per_transition)

        ! Interpolate between consecutive tiles (512x512)
        allocate(frame(channels, tile_size, tile_size))
        frame = (1.0 - t) * all_outputs(sequence(trans), 1, :, :, :) + &
                t * all_outputs(sequence(trans+1), 1, :, :, :)

        write(frame_file, '(A,I0.6,A)') "/tmp/video_frames/frame_", frame_num, ".bmp"
        call save_image(trim(frame_file), frame, success)
        deallocate(frame)

        if (mod(frame_num, 100) == 0) then
            !$omp critical
            print '(A,I0,A,I0)', "  Frame ", frame_num, "/", total_frames
            !$omp end critical
        end if
    end do
    !$omp end parallel do

    print *, "Total frames generated:", total_frames

    ! Create video with ffmpeg (upscale to 1024x1024, optionally sharpen)
    print *, "Creating video with ffmpeg..."
    if (sharpen_mode == 0) then
        write(cmd, '(A,I0,A)') "ffmpeg -y -framerate ", fps, &
            " -i /tmp/video_frames/frame_%06d.bmp" // &
            " -vf ""scale=1024:1024:flags=lanczos""" // &
            " -c:v libx264 -pix_fmt yuv420p " // trim(output_video)
    else
        write(cmd, '(A,I0,A)') "ffmpeg -y -framerate ", fps, &
            " -i /tmp/video_frames/frame_%06d.bmp" // &
            " -vf ""scale=1024:1024:flags=lanczos,unsharp=7:7:2.5:7:7:0.0""" // &
            " -c:v libx264 -pix_fmt yuv420p " // trim(output_video)
    end if
    call execute_command_line(trim(cmd))

    print *, "Done! Output:", trim(output_video)

end program
