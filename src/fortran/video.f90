module video
    use image
    use cnn_autoencoder
    use command
    use utilities

    implicit none

    real, parameter :: SMOOTHING_ALPHA = 0.25

    type(autoencoder) :: net
    type(image_entry), allocatable :: images(:)
    real, allocatable :: tiles(:,:,:,:), latent_tiles(:,:,:,:)
    real, allocatable :: cosines(:,:)

    ! Audio: (num_frames x 14 features)
    real, allocatable :: audio_features(:,:)
    real, allocatable :: energy(:)  ! normalized flux+rms for clock modulation
    integer :: num_audio_frames
    real :: audio_time

    ! video generation globals
    real :: fps
    integer, allocatable :: theme_audio_boundaries(:)
    integer, allocatable :: theme_tiles(:)
    integer :: bpm
    integer :: current_tile, current_frame
    integer, allocatable :: tile_hits(:)

contains
    subroutine clear_video()
        if (allocated(images)) deallocate(images)
        if (allocated(tiles)) deallocate(tiles)
        if (allocated(latent_tiles)) deallocate(latent_tiles)
        if (allocated(cosines)) deallocate(cosines)
        if (allocated(theme_audio_boundaries)) deallocate(theme_audio_boundaries)
        if (allocated(theme_tiles)) deallocate(theme_tiles)
        if (allocated(tile_hits)) deallocate(tile_hits)
        if (allocated(energy)) deallocate(energy)

        current_tile = 0
        current_frame = 1
        fps = 0
        bpm = 0
        num_audio_frames = 0
        audio_time = 0.0
    end subroutine

    ! load-model ( s:filename -- )
    subroutine load_model()
        character(MAX_STRING_LENGTH) :: filename

        filename = pop_string()
        net = load_autoencoder(filename)

        call set_training(net, .false.)
    end subroutine

    ! prepare_tiles ( s:project-dir n:tile-width n:tile-height n:batch-size -- )
    subroutine prepare_tiles()
        character(MAX_STRING_LENGTH) :: project_dir
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
        call encoder_forward(net, tiles(:,:,:,1:1), latent)
        latent_channels = size(latent, 1)
        latent_height = size(latent, 2)
        latent_width = size(latent,3)
        print '(A,I0,A,I0,A,I0)', "latent: ", latent_channels, "x", latent_height, "x", latent_width

        ! compute the latent tensors for all the tiles
        allocate(latent_tiles(latent_channels, latent_height, latent_width, total_tiles))
        batch_size = int(pop_number())
        do i = 1, total_tiles, batch_size
            j = min(i + batch_size - 1, total_tiles)
            call encoder_forward(net, tiles(:,:,:,i:j), latent)
            latent_tiles(:,:,:,i:j) = latent
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

    end subroutine


    ! analyze-audio ( s:audio-file n:fps -- )
    subroutine analyze_audio()
        use stdlib_io, only: loadtxt
        character(MAX_STRING_LENGTH) :: audio_filename, tsv_filename, command
        character(MAX_STRING_LENGTH), allocatable :: output(:)
        real, allocatable :: flux_mono(:), rms_mono(:), flux_norm(:), rms_norm(:)
        integer :: i

        audio_filename = pop_string()
        fps = pop_number()

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

        print '(A,I0,A,I0)', "themes: ", num_themes, " tiles, ", num_themes - 1, " boundaries"

    end subroutine

    subroutine generate_transition()
        integer :: start_tile, end_tile, candidate
        integer :: num_frames, theme_section
        real :: clock_division, theme_weight
        integer :: i
        real :: score, best_score
        integer, allocatable :: pool(:)
        integer :: min_hits

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

        current_frame = current_frame + num_frames
        tile_hits(end_tile) = tile_hits(end_tile) + 1
        current_tile = end_tile

        call push_number(int(current_frame))

    end subroutine


end module
