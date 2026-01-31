module test_video
    use testdrive, only: new_unittest, unittest_type, error_type, check
    use video
    use cnn_autoencoder, only: tensor_cache
    use command
    implicit none
    private

    public :: collect_video_tests

    logical :: setup_done = .false.
    logical :: audio_done = .false.

contains

    subroutine collect_video_tests(testsuite)
        type(unittest_type), allocatable, intent(out) :: testsuite(:)

        testsuite = [ &
            new_unittest("prepare_tiles_dimensions", test_prepare_tiles_dimensions), &
            new_unittest("prepare_tiles_encoder_activations", test_prepare_tiles_encoder_activations), &
            new_unittest("prepare_tiles_cosines_diagonal", test_prepare_tiles_cosines_diagonal), &
            new_unittest("prepare_tiles_cosines_symmetric", test_prepare_tiles_cosines_symmetric), &
            new_unittest("analyze_audio", test_analyze_audio), &
            new_unittest("establish_themes", test_establish_themes), &
            new_unittest("generate_transition", test_generate_transition), &
            new_unittest("set_camera_motion", test_set_camera_motion), &
            new_unittest("set_chroma", test_set_chroma), &
            new_unittest("roll_chroma", test_roll_chroma), &
            new_unittest("wobble", test_wobble), &
            new_unittest("aberrate", test_aberrate) &
        ]
    end subroutine


    subroutine setup()
        if (setup_done) return

        call push_string("models/ae-weights-576x384-add.bin")
        call load_model()

        call push_number(4.0)    ! batch_size
        call push_number(384.0)  ! tile_height
        call push_number(576.0)  ! tile_width
        call push_string("projects/fort-greene-2")

        call prepare_tiles()
        setup_done = .true.
    end subroutine


    subroutine test_prepare_tiles_dimensions(error)
        type(error_type), allocatable, intent(out) :: error
        integer :: total_tiles

        call setup()

        total_tiles = size(tiles, 4)

        call check(error, size(tiles, 1) == 3, "tiles should have 3 channels")
        if (allocated(error)) return
        call check(error, size(tiles, 2) == 384, "tiles height should be 384")
        if (allocated(error)) return
        call check(error, size(tiles, 3) == 576, "tiles width should be 576")
        if (allocated(error)) return
        call check(error, total_tiles > 0, "should have at least one tile")
        if (allocated(error)) return
        call check(error, size(cosines, 1) == total_tiles, "cosines rows should match total tiles")
        if (allocated(error)) return
        call check(error, size(cosines, 2) == total_tiles, "cosines cols should match total tiles")
    end subroutine


    subroutine test_prepare_tiles_encoder_activations(error)
        type(error_type), allocatable, intent(out) :: error
        integer :: total_tiles, num_layers, i, layer

        call setup()

        total_tiles = size(tiles, 4)
        num_layers = net%config%num_layers - 1

        call check(error, size(encoder_activations, 1) == total_tiles, "encoder_activations should have total_tiles rows")
        if (allocated(error)) return
        call check(error, size(encoder_activations, 2) == num_layers, "encoder_activations should have num_layers-1 cols")
        if (allocated(error)) return

        ! check each activation tensor is allocated
        do i = 1, total_tiles
            do layer = 1, num_layers
                call check(error, allocated(encoder_activations(i, layer)%tensor), "activation tensor should be allocated")
                if (allocated(error)) return
            end do
        end do
    end subroutine


    subroutine test_prepare_tiles_cosines_diagonal(error)
        type(error_type), allocatable, intent(out) :: error
        integer :: i, total_tiles

        call setup()

        total_tiles = size(cosines, 1)

        ! Diagonal should be ~1.0 (self-similarity)
        do i = 1, total_tiles
            if (abs(cosines(i, i) - 1.0) > 1.0e-3) then
                print '(A,I0,A,I0,A,E12.5)', "cosines(", i, ",", i, ") = ", cosines(i, i)
                call check(error, .false., "diagonal elements should be 1.0")
                return
            end if
        end do
        call check(error, .true., "all diagonal elements are 1.0")
    end subroutine


    subroutine test_prepare_tiles_cosines_symmetric(error)
        type(error_type), allocatable, intent(out) :: error
        integer :: i, j, total_tiles

        call setup()

        total_tiles = size(cosines, 1)

        ! Matrix should be symmetric
        do i = 1, total_tiles
            do j = i + 1, total_tiles
                if (abs(cosines(i, j) - cosines(j, i)) > 1.0e-5) then
                    call check(error, .false., "cosine matrix should be symmetric")
                    return
                end if
            end do
        end do
        call check(error, .true., "cosine matrix is symmetric")
    end subroutine


    subroutine setup_audio()
        if (audio_done) return

        call push_number(120.0)  ! bpm
        call push_number(24.0)   ! fps
        call push_string("projects/fort-greene-2/fort-greene-2.mp3")
        call analyze_audio()
        audio_done = .true.
    end subroutine


    subroutine test_analyze_audio(error)
        type(error_type), allocatable, intent(out) :: error

        call setup_audio()

        call check(error, size(audio_features, 2) == 16, "should have 16 columns")
        if (allocated(error)) return
        call check(error, num_audio_frames > 0, "should have frames")
        if (allocated(error)) return
        call check(error, audio_time > 0.0, "should have positive duration")
    end subroutine


    subroutine test_establish_themes(error)
        type(error_type), allocatable, intent(out) :: error
        integer :: i, j, num_themes
        real :: max_similarity

        call setup()
        call setup_audio()

        num_themes = 4
        call push_number(real(num_themes))
        call establish_themes()

        ! check allocations
        call check(error, size(theme_tiles) == num_themes, "should have num_themes tiles")
        if (allocated(error)) return
        call check(error, size(theme_audio_boundaries) == num_themes - 1, "should have num_themes-1 boundaries")
        if (allocated(error)) return

        ! check tiles are valid indices
        do i = 1, num_themes
            call check(error, theme_tiles(i) >= 1 .and. theme_tiles(i) <= size(cosines, 1), "tile index in range")
            if (allocated(error)) return
        end do

        ! check tiles are dispersed (max similarity between any pair < 0.9)
        max_similarity = 0.0
        do i = 1, num_themes
            do j = i + 1, num_themes
                max_similarity = max(max_similarity, cosines(theme_tiles(i), theme_tiles(j)))
            end do
        end do
        call check(error, max_similarity < 0.9, "theme tiles should be dispersed")
    end subroutine


    subroutine test_generate_transition(error)
        type(error_type), allocatable, intent(out) :: error
        integer :: expected_frames, i
        real :: popped_frame
        real, allocatable :: img(:,:,:)
        character(MAX_STRING_LENGTH) :: filename

        call setup()
        call setup_audio()

        call push_number(4.0)  ! num_themes
        call establish_themes()

        current_tile = 0
        current_frame = 1

        ! fps=24, bpm=120, clock_division=2 -> 24*60/120/2 = 6 frames
        call push_number(0.5)  ! theme_weight
        call push_number(2.0)  ! clock_division
        call generate_transition()

        expected_frames = 6

        ! check num_frames and current_frame were pushed
        call check(error, int(pop_number()) == expected_frames, "num_frames should be on stack")
        if (allocated(error)) return
        call check(error, int(pop_number()) == 1 + expected_frames, "current_frame should be updated")
        if (allocated(error)) return

        ! check we got expected_frames images and filenames
        do i = 1, expected_frames
            filename = pop_string()
            call check(error, index(filename, "/frames/frame_") > 0, "filename should contain /frames/frame_")
            if (allocated(error)) return

            img = pop_image()
            call check(error, size(img, 1) == 3, "image should have 3 channels")
            if (allocated(error)) return
        end do
    end subroutine


    subroutine test_set_camera_motion(error)
        type(error_type), allocatable, intent(out) :: error

        call push_number(10.0)   ! pan_x
        call push_number(5.0)    ! pan_y
        call push_number(2.0)    ! rotation
        call push_number(0.02)   ! zoom
        call push_number(1.5)    ! frequency
        call set_camera_motion()

        call check(error, abs(camera_pan_x - 10.0) < 1e-5, "pan_x should be 10.0")
        if (allocated(error)) return
        call check(error, abs(camera_pan_y - 5.0) < 1e-5, "pan_y should be 5.0")
        if (allocated(error)) return
        call check(error, abs(camera_rotation - 2.0) < 1e-5, "rotation should be 2.0")
        if (allocated(error)) return
        call check(error, abs(camera_zoom - 0.02) < 1e-5, "zoom should be 0.02")
        if (allocated(error)) return
        call check(error, abs(camera_frequency - 1.5) < 1e-5, "frequency should be 1.5")
    end subroutine


    subroutine test_set_chroma(error)
        type(error_type), allocatable, intent(out) :: error

        call push_number(0.3)   ! probability
        call push_number(8.0)   ! max_offset
        call set_chroma()

        call check(error, abs(chroma_probability - 0.3) < 1e-5, "probability should be 0.3")
        if (allocated(error)) return
        call check(error, abs(chroma_max_offset - 8.0) < 1e-5, "max_offset should be 8.0")
    end subroutine


    subroutine test_roll_chroma(error)
        type(error_type), allocatable, intent(out) :: error
        integer :: i, active_count

        ! set probability to 1.0 - should always activate
        call push_number(1.0)
        call push_number(5.0)
        call set_chroma()

        call roll_chroma()
        call check(error, chroma_active, "chroma should be active with probability 1.0")
        if (allocated(error)) return
        call check(error, chroma_angle >= 0.0 .and. chroma_angle < 6.3, "angle should be in [0, 2*pi)")
        if (allocated(error)) return

        ! set probability to 0.0 - should never activate
        call push_number(0.0)
        call push_number(5.0)
        call set_chroma()

        call roll_chroma()
        call check(error, .not. chroma_active, "chroma should be inactive with probability 0.0")
    end subroutine


    subroutine test_wobble(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        integer :: h, w

        call setup()
        call setup_audio()

        h = 100
        w = 150

        ! set up camera motion
        call push_number(5.0)    ! pan_x
        call push_number(3.0)    ! pan_y
        call push_number(1.0)    ! rotation
        call push_number(0.01)   ! zoom
        call push_number(1.0)    ! frequency
        call set_camera_motion()

        ! create test image
        allocate(img(3, h, w))
        img = 0.5

        ! push frame number and image
        call push_number(100.0)
        call push_image(img)
        call wobble()

        result = pop_image()

        call check(error, size(result, 1) == 3, "result should have 3 channels")
        if (allocated(error)) return
        call check(error, size(result, 2) == h, "result height should match")
        if (allocated(error)) return
        call check(error, size(result, 3) == w, "result width should match")
    end subroutine


    subroutine test_aberrate(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        integer :: h, w

        call setup()
        call setup_audio()

        h = 100
        w = 150

        ! set up chroma with probability 1.0
        call push_number(1.0)
        call push_number(10.0)
        call set_chroma()
        call roll_chroma()

        ! create test image with distinct channels
        allocate(img(3, h, w))
        img(1,:,:) = 1.0  ! red
        img(2,:,:) = 0.5  ! green
        img(3,:,:) = 0.0  ! blue

        ! push frame number and image
        call push_number(100.0)
        call push_image(img)
        call aberrate()

        result = pop_image()

        call check(error, size(result, 1) == 3, "result should have 3 channels")
        if (allocated(error)) return
        call check(error, size(result, 2) == h, "result height should match")
        if (allocated(error)) return
        call check(error, size(result, 3) == w, "result width should match")
        if (allocated(error)) return

        ! green channel should be unchanged
        call check(error, all(abs(result(2,:,:) - 0.5) < 1e-5), "green channel should be unchanged")
    end subroutine

end module test_video
