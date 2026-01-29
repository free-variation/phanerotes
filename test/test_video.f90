module test_video
    use testdrive, only: new_unittest, unittest_type, error_type, check
    use video
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
            new_unittest("prepare_tiles_cosines_diagonal", test_prepare_tiles_cosines_diagonal), &
            new_unittest("prepare_tiles_cosines_symmetric", test_prepare_tiles_cosines_symmetric), &
            new_unittest("analyze_audio", test_analyze_audio), &
            new_unittest("establish_themes", test_establish_themes) &
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

        call push_number(24.0)  ! fps
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

end module test_video
