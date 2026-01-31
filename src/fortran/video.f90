module video
    use :: image
    use :: cnn_autoencoder
    use :: command
    use :: utilities
    use :: omp_lib

    implicit none

    real, parameter :: SMOOTHING_ALPHA = 0.25

    character(MAX_STRING_LENGTH) :: project_dir

    type(autoencoder) :: net
    type(image_entry), allocatable :: images(:)
    real, allocatable :: tiles(:,:,:,:), latent_tiles(:,:,:,:)
    real, allocatable :: cosines(:,:)
    type(tensor_cache), allocatable :: encoder_activations(:,:)

    ! Audio: (num_frames x 14 features)
    real, allocatable :: audio_features(:,:)
    real, allocatable :: energy(:)  ! normalized flux+rms for clock modulation
    integer :: num_audio_frames = 0
    real :: audio_time = 0.0

    ! video generation globals
    real :: fps = 0.0
    integer, allocatable :: theme_audio_boundaries(:)
    integer, allocatable :: theme_tiles(:)
    integer :: bpm = 0
    integer :: current_tile = 0, current_frame = 1
    integer, allocatable :: tile_hits(:)

    ! camera motion parameters (sinsoidal)
    real :: camera_pan_x = 0.0, camera_pan_y = 0.0
    real :: camera_rotation = 0.0, camera_zoom = 0.0
    real :: camera_frequency = 1.0

    ! chromatic aberration parameters
    real :: chroma_probability = 0.0
    real :: chroma_max_offset = 0.0
    logical :: chroma_active = .false.
    real :: chroma_angle = 0.0

contains
    subroutine clear_video()
        if (allocated(images)) deallocate(images)
        if (allocated(tiles)) deallocate(tiles)
        if (allocated(latent_tiles)) deallocate(latent_tiles)
        if (allocated(cosines)) deallocate(cosines)
        if (allocated(encoder_activations)) deallocate(encoder_activations)
        if (allocated(theme_audio_boundaries)) deallocate(theme_audio_boundaries)
        if (allocated(theme_tiles)) deallocate(theme_tiles)
        if (allocated(tile_hits)) deallocate(tile_hits)
        if (allocated(energy)) deallocate(energy)

        project_dir = ""
        current_tile = 0
        current_frame = 1
        fps = 0.0
        bpm = 0
        num_audio_frames = 0
        audio_time = 0.0

        camera_pan_x = 0.0
        camera_pan_y = 0.0
        camera_rotation = 0.0
        camera_zoom = 0.0
        camera_frequency = 1.0

        chroma_probability = 0.0
        chroma_max_offset = 0.0
        chroma_active = .false.
        chroma_angle = 0.0
    end subroutine

    ! load-model ( s:filename -- )
    subroutine load_model()
        character(MAX_STRING_LENGTH) :: filename

        filename = pop_string()
        net = load_autoencoder(filename)

        call set_training(net, .false.)
    end subroutine

    ! prepare-tiles ( n:batch-size n:tile-height n:tile-width s:project-dir -- )
    subroutine prepare_tiles()
        character(MAX_STRING_LENGTH), allocatable :: image_filenames(:)
        integer :: i, j, k
        integer :: tile_width, tile_height
        integer :: batch_size
        integer :: num_tiles_x, num_tiles_y, num_tiles, total_tiles
        integer :: num_channels
        real, allocatable :: latent(:,:,:,:)
        integer :: latent_channels, latent_width, latent_height
        integer :: flat_rows
        real, allocatable :: flat_latent(:,:)
        type(tensor_cache), allocatable :: activations(:)
        integer :: layer

        project_dir = pop_string()
        tile_width = int(pop_number())
        tile_height = int(pop_number())


        ! read in all the images from the project directory
        image_filenames = directory_files(project_dir, "*.jpg")

        allocate(images(size(image_filenames)))
        do i = 1, size(image_filenames)
            images(i)%pixels = load_image(trim(project_dir) // "/" // trim(image_filenames(i)))
        end do
        print '(A,I0)', "images: ", size(images)

        ! we're assuming all images have the same size!
        num_channels = size(images(1)%pixels, 1)
        num_tiles_y = size(images(1)%pixels, 2) / tile_height
        num_tiles_x = size(images(1)%pixels, 3) / tile_width
        num_tiles = num_tiles_y * num_tiles_x
        total_tiles = num_tiles * size(images)
        print '(A,I0,A,I0,A,I0,A)', "tiles: ", total_tiles, " (", num_tiles_x, "x", num_tiles_y, ")"

        ! extract all tiles from the images
        allocate(tiles(num_channels, tile_height, tile_width, total_tiles))
        allocate(tile_hits(total_tiles)); tile_hits = 0
        do i = 1, size(images)
            do j = 1, num_tiles_y
                do k = 1, num_tiles_x
                    tiles(:, :, :, (i - 1)*num_tiles + (j - 1)*num_tiles_x + k) = &
                        images(i)%pixels(1:num_channels, &
                        1 + (j - 1) * tile_height: j * tile_height,&
                        1 + (k - 1) * tile_width:  k * tile_width)
                end do

            end do
        end do

        ! run one tile through the encoder to obtain the latent dimensions
        call encoder_forward(net, tiles(:,:,:,1:1), latent, activations)
        latent_channels = size(latent, 1)
        latent_height = size(latent, 2)
        latent_width = size(latent,3)
        print '(A,I0,A,I0,A,I0)', "latent: ", latent_channels, "x", latent_height, "x", latent_width

        ! compute the latent tensors for all the tiles
        allocate(latent_tiles(latent_channels, latent_height, latent_width, total_tiles))
        allocate(encoder_activations(total_tiles, net%config%num_layers - 1))
        batch_size = int(pop_number())
        do i = 1, total_tiles, batch_size
            j = min(i + batch_size - 1, total_tiles)
            call encoder_forward(net, tiles(:,:,:,i:j), latent, activations)
            latent_tiles(:,:,:,i:j) = latent

            ! split batch activations into per-tile storage
            do k = i, j
                do layer = 1, size(activations)
                    encoder_activations(k, layer)%tensor = activations(layer)%tensor(:,:,:,k-i+1:k-i+1)
                end do
            end do
        end do

        ! compute cosine similarity between latent tiles
        flat_rows = latent_channels * latent_height * latent_width
        flat_latent = reshape(latent_tiles, [flat_rows, total_tiles])
        do i = 1, total_tiles
            flat_latent(:, i) = flat_latent(:, i) / norm2(flat_latent(:, i))
        end do

        allocate(cosines(total_tiles, total_tiles))
        call sgemm('T', 'N', total_tiles, total_tiles, flat_rows, 1.0, &
            flat_latent, flat_rows, flat_latent, flat_rows, 0.0, cosines, total_tiles)
        print '(A,I0,A,I0)', "cosines: ", total_tiles, "x", total_tiles

        ! create frames output directory
        call execute_command_line("mkdir -p " // trim(project_dir) // "/frames")
    end subroutine


    ! analyze-audio ( n:bpm n:fps s:audio-file -- )
    subroutine analyze_audio()
        use stdlib_io, only: loadtxt
        character(MAX_STRING_LENGTH) :: audio_filename, tsv_filename, command
        character(MAX_STRING_LENGTH), allocatable :: output(:)
        real, allocatable :: flux_mono(:), rms_mono(:), flux_norm(:), rms_norm(:)
        integer :: i

        audio_filename = pop_string()
        fps = pop_number()
        bpm = int(pop_number())

        write(command, '(A,A,A,I0)') "scripts/audio_features.sh ", trim(audio_filename), " ", int(fps)
        output = run_command(command)
        if (size(output) == 0) error stop "audio_features.sh produced no output"
        tsv_filename = output(1)

        if (allocated(audio_features)) deallocate(audio_features)
        call loadtxt(tsv_filename, audio_features, skiprows=1, fmt='*', delimiter=char(9))

        print '(A,I0,A,I0)', "audio: ", size(audio_features, 1), " x ", size(audio_features, 2)

        num_audio_frames = size(audio_features, 1)
        audio_time = audio_features(num_audio_frames, 2) + audio_features(2, 2)

        ! compute normalized energy (flux + rms) for clock modulation
        flux_mono = (audio_features(:, 7) + audio_features(:, 8)) / 2
        flux_norm = (flux_mono - minval(flux_mono)) / (maxval(flux_mono) - minval(flux_mono))
        rms_mono = (audio_features(:, 13) + audio_features(:, 14)) / 2
        rms_norm = (rms_mono - minval(rms_mono)) / (maxval(rms_mono) - minval(rms_mono))

        if (allocated(energy)) deallocate(energy)
        allocate(energy(num_audio_frames))
        energy(1) = (flux_norm(1) + rms_norm(1)) / 2
        do i = 2, num_audio_frames
            energy(i) = SMOOTHING_ALPHA * (flux_norm(i) + rms_norm(i)) / 2 + (1 - SMOOTHING_ALPHA) * energy(i - 1)
        end do
    end subroutine

    function random_latent() 
        real :: r
        integer :: random_latent

        call random_number(r)
        random_latent = 1 + int(r * size(latent_tiles, 4))
    end function

    ! establish-themes ( n:num-themes -- )
    subroutine establish_themes()
        integer :: num_themes
        integer :: frame_idx, i, i1, i2
        integer :: candidate, best_tile
        real :: min_dist, best_min_dist

        real, allocatable :: centroid_mono(:), flatness_mono(:)
        real, allocatable :: centroid_norm(:), flatness_norm(:), timbre(:)
        real, allocatable :: smoothed(:), temp(:)
        integer :: min_theme_distance

        num_themes = pop_number()

        ! centroid + flatness for timbral character
        centroid_mono = (audio_features(:, 3) + audio_features(:, 4)) / 2
        centroid_norm = (centroid_mono - minval(centroid_mono)) / (maxval(centroid_mono) - minval(centroid_mono))
        flatness_mono = (audio_features(:, 9) + audio_features(:, 10)) / 2
        flatness_norm = (flatness_mono - minval(flatness_mono)) / (maxval(flatness_mono) - minval(flatness_mono))
        timbre = centroid_norm + flatness_norm

        ! smooth the timbre signal
        allocate(smoothed(num_audio_frames))
        smoothed(1) = timbre(1)
        do i = 2, num_audio_frames
            smoothed(i) = SMOOTHING_ALPHA * timbre(i) + (1 - SMOOTHING_ALPHA) * smoothed(i - 1)
        end do

        ! find the top audio boundaries to be matched to frames
        if (allocated(theme_audio_boundaries)) deallocate(theme_audio_boundaries)
        allocate(theme_audio_boundaries(num_themes - 1))

        min_theme_distance = num_audio_frames / (num_themes * 2)
        temp = smoothed
        do i = 1, num_themes - 1
            frame_idx = maxloc(temp, dim = 1)
            theme_audio_boundaries(i) = frame_idx

            ! exclude window around picked peak
            i1 = max(1, frame_idx - min_theme_distance)
            i2 = min(num_audio_frames, frame_idx + min_theme_distance)
            temp(i1:i2) = -huge(0.0)
        end do
        
        ! greedy max-min selection of dispersed theme tiles
        if (allocated(theme_tiles)) deallocate(theme_tiles)
        allocate(theme_tiles(num_themes))

        theme_tiles(1) = random_latent()

        do i = 2, num_themes
            best_min_dist = -huge(0.0)
            do candidate = 1, size(latent_tiles, 4)
                if (any(theme_tiles(1:i-1) == candidate)) cycle
                min_dist = minval(1.0 - cosines(candidate, theme_tiles(1:i-1)))
                if (min_dist > best_min_dist) then
                    best_min_dist = min_dist
                    best_tile = candidate
                end if
            end do
            theme_tiles(i) = best_tile
        end do

        print '(A,I0,A,I0,A)', "themes: ", num_themes, " tiles, ", num_themes - 1, " boundaries"

    end subroutine

    ! generate-transition ( n:theme-weight n:clock-division -- [img s:filename]... n:current-frame n:num-frames )
    subroutine generate_transition()
        integer :: start_tile, end_tile, candidate
        integer :: num_frames, theme_section
        real :: clock_division, theme_weight
        integer :: i
        real :: score, best_score, alpha
        integer, allocatable :: pool(:)
        integer :: min_hits
        real, allocatable :: output(:,:,:,:)
        real, allocatable :: frames(:,:,:,:)
        character(MAX_STRING_LENGTH) :: filename
        integer :: channels, height, width

        clock_division = pop_number()
        num_frames = int(fps * 60.0 / bpm / clock_division)

        if (current_tile == 0) then
            start_tile = random_latent()
            tile_hits(start_tile) = tile_hits(start_tile) + 1
        else
            start_tile = current_tile
        end if

        theme_weight = pop_number()

        ! find the current theme section
        theme_section = 1
        do i = 1, size(theme_audio_boundaries)
            if (current_frame >= theme_audio_boundaries(i)) theme_section = theme_section + 1
        end do

        ! get the best match for next tile from the current one
        min_hits = minval(tile_hits)
        pool = pack([(i, i = 1, size(tile_hits))], tile_hits == min_hits)

        best_score = -huge(0.0)
        do i = 1, size(pool)
            candidate = pool(i)
            if (candidate == start_tile) cycle

            score = theme_weight * cosines(candidate, theme_tiles(theme_section)) +&
                (1.0 - theme_weight) * cosines(start_tile, candidate)

            if (score > best_score) then
                best_score = score
                end_tile = candidate
            end if
        end do

        ! allocate frame buffer
        channels = size(tiles, 1)
        height = size(tiles, 2)
        width = size(tiles, 3)
        allocate(frames(channels, height, width, num_frames))

        ! decode frames in parallel
        !$omp parallel do private(i, alpha, output) schedule(dynamic)
        do i = 1, num_frames
            alpha = real(num_frames - i) / real(max(num_frames - 1, 1))
            call decode_latent_interpolated(net,&
                latent_tiles(:,:,:, start_tile:start_tile), latent_tiles(:,:,:, end_tile:end_tile),&
                encoder_activations(start_tile, :), encoder_activations(end_tile, :),&
                alpha, output)
            frames(:,:,:,i) = output(:,:,:,1)
        end do
        !$omp end parallel do

        ! push frames and filenames to stacks (reverse order so first frame ends on top)
        do i = num_frames, 1, -1
            call push_image(frames(:,:,:,i))
            write(filename, '(A,A,I0.5,A)') trim(project_dir), "/frames/frame_", current_frame + i - 1, ".bmp"
            call push_string(filename)
        end do

        current_frame = current_frame + num_frames
        tile_hits(end_tile) = tile_hits(end_tile) + 1
        current_tile = end_tile

        call push_number(real(current_frame))
        call push_number(real(num_frames))

    end subroutine

    ! set-camera-motion ( n:pan-x n:pan-y n:rotation n:zoom n:frequency -- )
    subroutine set_camera_motion()
        camera_frequency = pop_number()
        camera_zoom = pop_number()
        camera_rotation = pop_number()
        camera_pan_y = pop_number()
        camera_pan_x = pop_number()
    end subroutine

    ! set-chroma ( n:probability n:max-offset -- )
    subroutine set_chroma()
        chroma_max_offset = pop_number()
        chroma_probability = pop_number()
    end subroutine

    ! roll-chroma ( -- )
    ! rolls dice for this transition, sets chroma_active and chroma_angle
    subroutine roll_chroma()
        real :: roll

        call random_number(roll)
        if (roll < chroma_probability) then
            chroma_active = .true.
            call random_number(chroma_angle)
            chroma_angle = chroma_angle * 2.0 * PI
        else
            chroma_active = .false.
        end if
    end subroutine

    ! load-frames ( s:directory -- [img s:filename]... n:count )
    ! loads all frame_*.bmp files in sorted order
    subroutine load_frames()
        character(MAX_STRING_LENGTH) :: dir
        character(MAX_STRING_LENGTH), allocatable :: filenames(:)
        character(MAX_STRING_LENGTH) :: fullpath
        real, allocatable :: pixels(:,:,:)
        integer :: i, n

        dir = pop_string()
        filenames = directory_files(dir, "frame_*.bmp")
        n = size(filenames)

        ! directory_files returns sorted, so frame_00001 < frame_00002 etc.
        ! push in reverse order so first frame ends up on top
        do i = n, 1, -1
            fullpath = trim(dir) // "/" // trim(filenames(i))
            pixels = load_image(fullpath)
            call push_image(pixels)
            call push_string(fullpath)
        end do

        call push_number(real(n))
    end subroutine

    ! finalize-video ( n:upscale n:fade-in n:fade-out s:fade-in-color s:fade-out-color s:frame-dir s:output-file s:audio-file -- )
    subroutine finalize_video()
        character(MAX_STRING_LENGTH) :: audio_file, output_file, frame_dir
        character(1024) :: command
        character(MAX_STRING_LENGTH) :: fade_in_color, fade_out_color
        character(512) :: frame_pattern, filter_chain, scale_filter
        real :: fade_in, fade_out, fade_out_start
        integer :: upscale

        audio_file = pop_string()
        output_file = pop_string()
        frame_dir = pop_string()
        fade_out_color = pop_string()
        fade_in_color = pop_string()
        fade_out = pop_number()
        fade_in = pop_number()
        upscale = int(pop_number())

        fade_out_start = audio_time - fade_out

        write(frame_pattern, '(A,A)') trim(frame_dir), "/frame_%05d.bmp"

        if (upscale > 1) then
            write(scale_filter, '(A,I0,A,I0,A)') "scale=iw*", upscale, ":ih*", upscale, ":flags=lanczos,"
        else
            scale_filter = ""
        end if

        write(filter_chain, '(A,A,F0.2,A,A,A,F0.2,A,F0.2,A,A)') &
            trim(scale_filter), &
            "nlmeans=s=3:p=7:r=15,unsharp=5:5:2.0,fade=t=in:st=0:d=", fade_in, &
            ":c=", trim(fade_in_color), &
            ",fade=t=out:st=", fade_out_start, ":d=", fade_out, &
            ":c=", trim(fade_out_color)

        write(command, '(A,I0,A,A,A,A,A,A,A,A,A)') &
            "ffmpeg -nostdin -y -framerate ", int(fps), &
            " -i '", trim(frame_pattern), &
            "' -i '", trim(audio_file), &
            "' -vf '", trim(filter_chain), &
            "' -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest '", &
            trim(output_file), "'"

        call execute_command_line(command)
    end subroutine

    ! wobble ( n:frame img -- img )
    subroutine wobble()
        real, allocatable :: pixels(:,:,:)
        integer :: frame, height, width
        real :: t, cx, cy, tx, ty, angle, s
        real :: M(3,3)

        frame = int(pop_number())
        pixels = pop_image()

        if (camera_pan_x == 0.0 .and. camera_pan_y == 0.0 .and. &
            camera_rotation == 0.0 .and. camera_zoom == 0.0) then
            call push_image(pixels)
            return
        end if

        height = size(pixels, 2)
        width = size(pixels, 3)

        t = 2.0 * PI * camera_frequency * real(frame) / real(num_audio_frames)
        cx = real(width) / 2.0
        cy = real(height) / 2.0

        tx = camera_pan_x * sin(t)
        ty = camera_pan_y * cos(t)
        angle = camera_rotation * sin(t + PI/4.0)
        s = 1.0 + camera_zoom * sin(t + PI/2.0)

        M = matmul(translation_matrix(tx, ty), &
            matmul(rotation_matrix(cx, cy, angle), &
                scale_matrix(cx, cy, s, s)))

        call push_image(affine_transform(pixels, M))
    end subroutine

    ! aberrate ( n:frame img -- img )
    subroutine aberrate()
        real, allocatable :: pixels(:,:,:)
        real, allocatable :: r_channel(:,:,:), b_channel(:,:,:)
        integer :: frame
        real :: e, intensity, dx, dy
        real :: M_r(3,3), M_b(3,3)

        frame = int(pop_number())
        pixels = pop_image()

        if (.not. chroma_active) then
            call push_image(pixels)
            return
        end if

        e = energy(min(frame, num_audio_frames))
        intensity = e * chroma_max_offset

        dx = intensity * cos(chroma_angle)
        dy = intensity * sin(chroma_angle)

        M_r = translation_matrix(dx, dy)
        M_b = translation_matrix(-dx, -dy)

        r_channel = affine_transform(pixels(1:1,:,:), M_r)
        b_channel = affine_transform(pixels(3:3,:,:), M_b)

        pixels(1,:,:) = r_channel(1,:,:)
        pixels(3,:,:) = b_channel(1,:,:)

        call push_image(pixels)
    end subroutine

end module
