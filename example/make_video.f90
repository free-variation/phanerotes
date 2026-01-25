program make_video
    use image
    use cnn_autoencoder
    use omp_lib
    implicit none

    type(autoencoder) :: net
    real, allocatable :: img(:,:,:)
    real, allocatable :: tile(:,:,:,:), latent(:,:,:,:), output(:,:,:,:)
    real, allocatable :: all_latents(:,:,:,:,:)  ! (num_tiles, batch, channels, w, h)
    real, allocatable :: all_outputs(:,:,:,:,:)  ! (num_tiles, batch, channels, w, h)
    real, allocatable :: latent_flat(:,:)        ! (num_tiles, flattened_size)
    real, allocatable :: norms(:)
    integer, allocatable :: sequence(:)
    logical, allocatable :: used(:)
    type(tensor_cache), allocatable :: tile_acts(:)       ! temporary, one tile
    type(tensor_cache), allocatable :: encoder_acts(:,:)  ! (num_tiles, num_layers-1)

    integer :: width, height, channels
    integer :: tile_width, tile_height, i, j, n, tiles_x, tiles_y, num_tiles
    integer :: x_start, x_end, y_start, y_end
    integer :: latent_size, current, best_next, k
    real :: best_sim, sim

    ! Multi-image support
    integer :: num_images, img_idx, tiles_per_image, tile_offset, unit_num, ios
    character(len=512) :: image_file
    character(len=512), allocatable :: image_files(:)

    ! Video parameters
    integer :: fps, duration, total_frames, frames_per_transition, num_transitions
    integer :: frame_num, trans, f
    real :: t
    real, allocatable :: frame(:,:,:)
    character(len=256) :: frame_file, cmd, input_dir, output_video, model_file, arg
    logical :: success
    integer :: sharpen_mode   ! 0=none, 1=sharpen
    integer :: interp_mode    ! 0=pixel, 1=latent
    integer :: scale_ratio, output_width, output_height
    character(len=16) :: width_str, height_str

    ! Parse arguments
    if (command_argument_count() < 3) then
        print *, "Usage: make_video <model_file> <image_dir> <tile_width> [options]"
        print *, "Options:"
        print *, "  --height N   - tile height (default: same as width)"
        print *, "  --frames N   - total frames (default: 7200)"
        print *, "  --fps N      - frames per second (default: 24)"
        print *, "  --scale N    - upscale ratio (default: 1, no upscaling)"
        print *, "  --sharpen    - apply sharpening (default)"
        print *, "  --no-sharpen - no sharpening"
        print *, "  --pixel      - interpolate pixels (default, fast)"
        print *, "  --latent     - interpolate in latent space (slower, smoother)"
        stop 1
    end if

    call get_command_argument(1, model_file)
    call get_command_argument(2, input_dir)
    call get_command_argument(3, arg)
    read(arg, *) tile_width
    tile_height = tile_width

    sharpen_mode = 1  ! default: sharpen
    interp_mode = -1  ! -1 = not set, 0 = pixel, 1 = latent
    scale_ratio = 1   ! default: no upscaling
    total_frames = 7200
    fps = 24

    i = 4
    do while (i <= command_argument_count())
        call get_command_argument(i, arg)
        if (trim(arg) == "--no-sharpen") then
            sharpen_mode = 0
        else if (trim(arg) == "--sharpen") then
            sharpen_mode = 1
        else if (trim(arg) == "--pixel") then
            if (interp_mode == 1) then
                print *, "Error: --pixel and --latent are mutually exclusive"
                stop 1
            end if
            interp_mode = 0
        else if (trim(arg) == "--latent") then
            if (interp_mode == 0) then
                print *, "Error: --pixel and --latent are mutually exclusive"
                stop 1
            end if
            interp_mode = 1
        else if (trim(arg) == "--scale") then
            i = i + 1
            call get_command_argument(i, arg)
            read(arg, *) scale_ratio
        else if (trim(arg) == "--frames") then
            i = i + 1
            call get_command_argument(i, arg)
            read(arg, *) total_frames
        else if (trim(arg) == "--fps") then
            i = i + 1
            call get_command_argument(i, arg)
            read(arg, *) fps
        else if (trim(arg) == "--height") then
            i = i + 1
            call get_command_argument(i, arg)
            read(arg, *) tile_height
        end if
        i = i + 1
    end do

    ! Default to pixel interpolation if not specified
    if (interp_mode == -1) interp_mode = 0

    output_width = tile_width * scale_ratio
    output_height = tile_height * scale_ratio
    write(output_video, '(A,I0,A,I0,A)') "video_", output_width, "x", output_height, ".mp4"

    print *, "Model:", trim(model_file)
    print *, "Input dir:", trim(input_dir)
    print *, "Output:", trim(output_video)
    print *, "Sharpening:", merge("enabled ", "disabled", sharpen_mode == 1)
    print *, "Interpolation:", merge("latent", "pixel ", interp_mode == 1)

    print *, "Loading autoencoder..."
    net = load_autoencoder(trim(model_file))
    call set_training(net, .false.)

    ! List images in directory
    call execute_command_line("ls " // trim(input_dir) // " > /tmp/video_image_list.txt")

    ! Count images
    num_images = 0
    open(newunit=unit_num, file="/tmp/video_image_list.txt", status="old")
    do
        read(unit_num, '(A)', iostat=ios) image_file
        if (ios /= 0) exit
        num_images = num_images + 1
    end do
    close(unit_num)
    print *, "Found", num_images, "images"

    ! Read image filenames
    allocate(image_files(num_images))
    open(newunit=unit_num, file="/tmp/video_image_list.txt", status="old")
    do i = 1, num_images
        read(unit_num, '(A)') image_files(i)
        image_files(i) = trim(input_dir) // "/" // trim(image_files(i))
    end do
    close(unit_num)

    ! Load first image to get dimensions
    print *, "Loading first image to get dimensions..."
    img = load_image(trim(image_files(1)))
    channels = size(img, 1)
    height = size(img, 2)
    width = size(img, 3)
    print *, "Image size:", channels, "x", height, "x", width

    tiles_x = width / tile_width
    tiles_y = height / tile_height
    tiles_per_image = tiles_x * tiles_y
    num_tiles = tiles_per_image * num_images
    print *, "Tiles per image:", tiles_x, "x", tiles_y, "=", tiles_per_image
    print *, "Total tiles:", num_tiles

    ! Number of transitions is num_tiles - 1 (for pairs: 1+2, 2+3, ..., (n-1)+n)
    num_transitions = num_tiles - 1
    frames_per_transition = total_frames / num_transitions
    print *, "Transitions:", num_transitions
    print *, "Frames per transition:", frames_per_transition

    ! Extract tiles and compute latents
    print *, "Processing tiles..."

    ! First pass to get latent dimensions (using already-loaded first image)
    ! Tile layout: (channels, tile_height, tile_width, batch)
    allocate(tile(channels, tile_height, tile_width, 1))
    tile(:, :, :, 1) = img(:, 1:tile_height, 1:tile_width)
    call autoencoder_forward(net, tile, 0.0, latent, output, tile_acts)
    ! Latent layout: (channels, h, w, batch)
    latent_size = size(latent, 1) * size(latent, 2) * size(latent, 3)

    allocate(all_latents(num_tiles, size(latent,1), size(latent,2), size(latent,3), size(latent,4)))
    allocate(all_outputs(num_tiles, size(output,1), size(output,2), size(output,3), size(output,4)))
    allocate(latent_flat(num_tiles, latent_size))
    allocate(norms(num_tiles))
    if (interp_mode == 1) then
        allocate(encoder_acts(num_tiles, net%config%num_layers - 1))
    end if
    deallocate(tile)

    ! Process all images
    do img_idx = 1, num_images
        if (img_idx > 1) then
            img = load_image(trim(image_files(img_idx)))
        end if
        print '(A,I0,A,I0,A,A)', "  Image ", img_idx, "/", num_images, ": ", trim(image_files(img_idx))

        tile_offset = (img_idx - 1) * tiles_per_image

        !$omp parallel do private(n, i, j, k, x_start, x_end, y_start, y_end, tile, latent, output, tile_acts) schedule(dynamic)
        do n = 1, tiles_per_image
            i = mod(n - 1, tiles_x) + 1
            j = (n - 1) / tiles_x + 1

            x_start = (i-1) * tile_width + 1
            x_end = i * tile_width
            y_start = (j-1) * tile_height + 1
            y_end = j * tile_height

            allocate(tile(channels, tile_height, tile_width, 1))
            ! img is (channels, height, width), extract tile
            tile(:, :, :, 1) = img(:, y_start:y_end, x_start:x_end)
            call autoencoder_forward(net, tile, 0.0, latent, output, tile_acts)

            all_latents(tile_offset + n, :, :, :, :) = latent
            all_outputs(tile_offset + n, :, :, :, :) = output
            latent_flat(tile_offset + n, :) = reshape(latent(:,:,:,1), [latent_size])
            norms(tile_offset + n) = sqrt(sum(latent_flat(tile_offset + n, :)**2))

            if (interp_mode == 1) then
                do k = 1, net%config%num_layers - 1
                    encoder_acts(tile_offset + n, k)%tensor = tile_acts(k)%tensor
                end do
            end if

            deallocate(tile)
        end do
        !$omp end parallel do
    end do
    print *, "Processed", num_tiles, "tiles total"

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
        print '(A,I0,A,I0,A,F6.3,A)', "  Step ", n, ": tile ", best_next, " (sim=", best_sim, ")"
    end do

    ! Create frames directory
    call execute_command_line("mkdir -p /tmp/video_frames")

    ! Generate frames
    ! Each frame shows pair of tiles: sequence(trans) + sequence(trans+1)
    ! Interpolate from pair (trans, trans+1) to pair (trans+1, trans+2)
    print *, "Generating frames..."

    !$omp parallel do private(frame_num, trans, f, t, frame, frame_file, success, output) schedule(dynamic)
    do frame_num = 1, total_frames
        ! Compute which transition and position within transition
        trans = (frame_num - 1) / frames_per_transition + 1
        f = mod(frame_num - 1, frames_per_transition)

        if (trans > num_transitions) then
            trans = num_transitions
            f = frames_per_transition
        end if

        t = real(f) / real(frames_per_transition)

        if (interp_mode == 0) then
            ! Pixel interpolation (fast)
            ! Output layout: (channels, height, width, batch)
            allocate(frame(channels, tile_height, tile_width))
            frame = (1.0 - t) * all_outputs(sequence(trans), :, :, :, 1) + &
                    t * all_outputs(sequence(trans+1), :, :, :, 1)
        else
            ! Latent space interpolation (smoother)
            call decode_latent_interpolated(net, &
                all_latents(sequence(trans), :, :, :, :), &
                all_latents(sequence(trans+1), :, :, :, :), &
                encoder_acts(sequence(trans), :), &
                encoder_acts(sequence(trans+1), :), &
                1.0 - t, output)
            allocate(frame(channels, tile_height, tile_width))
            frame = output(:, :, :, 1)
        end if

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

    ! Create video with ffmpeg
    print *, "Creating video with ffmpeg..."
    write(width_str, '(I0)') output_width
    write(height_str, '(I0)') output_height
    if (sharpen_mode == 0) then
        write(cmd, '(A,I0,A)') "ffmpeg -y -framerate ", fps, &
            " -i /tmp/video_frames/frame_%06d.bmp" // &
            " -vf ""scale=" // trim(width_str) // ":" // trim(height_str) // ":flags=lanczos""" // &
            " -c:v libx264 -pix_fmt yuv420p " // trim(output_video)
    else
        write(cmd, '(A,I0,A)') "ffmpeg -y -framerate ", fps, &
            " -i /tmp/video_frames/frame_%06d.bmp" // &
            " -vf ""scale=" // trim(width_str) // ":" // trim(height_str) // ":flags=lanczos,unsharp=7:7:2.5:7:7:0.0""" // &
            " -c:v libx264 -pix_fmt yuv420p " // trim(output_video)
    end if
    call execute_command_line(trim(cmd))

    print *, "Done! Output:", trim(output_video)

end program
