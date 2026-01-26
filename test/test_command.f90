module test_command
    use testdrive, only: new_unittest, unittest_type, error_type, check
    use command
    implicit none
    private

    public :: collect_command_tests

contains

    subroutine collect_command_tests(testsuite)
        type(unittest_type), allocatable, intent(out) :: testsuite(:)

        testsuite = [ &
            new_unittest("string_push_pop", test_string_push_pop), &
            new_unittest("string_lifo_order", test_string_lifo_order), &
            new_unittest("string_with_spaces", test_string_with_spaces), &
            new_unittest("number_push_pop", test_number_push_pop), &
            new_unittest("number_lifo_order", test_number_lifo_order), &
            new_unittest("number_precision", test_number_precision), &
            new_unittest("image_push_pop", test_image_push_pop), &
            new_unittest("image_preserves_data", test_image_preserves_data), &
            new_unittest("image_lifo_order", test_image_lifo_order), &
            new_unittest("image_dup", test_image_dup), &
            new_unittest("image_drop", test_image_drop), &
            new_unittest("image_swap", test_image_swap), &
            new_unittest("image_over", test_image_over), &
            new_unittest("cmd_transpose", test_cmd_transpose), &
            new_unittest("cmd_fliph", test_cmd_fliph), &
            new_unittest("cmd_flipv", test_cmd_flipv), &
            new_unittest("cmd_transform_identity", test_cmd_transform_identity), &
            new_unittest("cmd_transform_rotate", test_cmd_transform_rotate), &
            new_unittest("cmd_split_rgb", test_cmd_split_rgb), &
            new_unittest("cmd_split_greyscale_noop", test_cmd_split_greyscale_noop), &
            new_unittest("cmd_merge_rgb", test_cmd_merge_rgb), &
            new_unittest("cmd_split_merge_roundtrip", test_cmd_split_merge_roundtrip), &
            new_unittest("cmd_fill_rgb", test_cmd_fill_rgb), &
            new_unittest("cmd_fill_greyscale", test_cmd_fill_greyscale), &
            new_unittest("cmd_interpolate_blend", test_cmd_interpolate_blend), &
            new_unittest("cmd_interpolate_alpha_one", test_cmd_interpolate_alpha_one), &
            new_unittest("cmd_interpolate_alpha_zero", test_cmd_interpolate_alpha_zero), &
            new_unittest("cmd_interpolate_frames_basic", test_cmd_interpolate_frames_basic), &
            new_unittest("cmd_interpolate_frames_single", test_cmd_interpolate_frames_single), &
            new_unittest("cmd_interpolate_frames_zero", test_cmd_interpolate_frames_zero) &
        ]
    end subroutine


    subroutine test_string_push_pop(error)
        type(error_type), allocatable, intent(out) :: error
        character(MAX_STRING_LENGTH) :: result

        call push_string("hello")
        result = pop_string()
        call check(error, trim(result) == "hello")
    end subroutine


    subroutine test_string_lifo_order(error)
        type(error_type), allocatable, intent(out) :: error
        character(MAX_STRING_LENGTH) :: result

        call push_string("first")
        call push_string("second")
        call push_string("third")

        result = pop_string()
        call check(error, trim(result) == "third", "expected 'third'")
        if (allocated(error)) return

        result = pop_string()
        call check(error, trim(result) == "second", "expected 'second'")
        if (allocated(error)) return

        result = pop_string()
        call check(error, trim(result) == "first", "expected 'first'")
    end subroutine


    subroutine test_string_with_spaces(error)
        type(error_type), allocatable, intent(out) :: error
        character(MAX_STRING_LENGTH) :: result

        call push_string("hello world")
        result = pop_string()
        call check(error, trim(result) == "hello world")
    end subroutine


    subroutine test_number_push_pop(error)
        type(error_type), allocatable, intent(out) :: error
        real :: result

        call push_number(42.0)
        result = pop_number()
        call check(error, result, 42.0)
    end subroutine


    subroutine test_number_lifo_order(error)
        type(error_type), allocatable, intent(out) :: error
        real :: result

        call push_number(1.0)
        call push_number(2.0)
        call push_number(3.0)

        result = pop_number()
        call check(error, result, 3.0, "expected 3.0")
        if (allocated(error)) return

        result = pop_number()
        call check(error, result, 2.0, "expected 2.0")
        if (allocated(error)) return

        result = pop_number()
        call check(error, result, 1.0, "expected 1.0")
    end subroutine


    subroutine test_number_precision(error)
        type(error_type), allocatable, intent(out) :: error
        real :: result

        call push_number(3.14159265)
        result = pop_number()
        call check(error, abs(result - 3.14159265) < 1.0e-6)
    end subroutine


    subroutine test_image_push_pop(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: pixels(:,:,:), result(:,:,:)

        allocate(pixels(3, 10, 10))
        pixels = 0.5

        call push_image(pixels)
        result = pop_image()

        call check(error, size(result, 1) == 3, "channels mismatch")
        if (allocated(error)) return
        call check(error, size(result, 2) == 10, "width mismatch")
        if (allocated(error)) return
        call check(error, size(result, 3) == 10, "height mismatch")
    end subroutine


    subroutine test_image_preserves_data(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: pixels(:,:,:), result(:,:,:)
        integer :: c, x, y

        allocate(pixels(3, 4, 4))
        do c = 1, 3
            do x = 1, 4
                do y = 1, 4
                    pixels(c, x, y) = real(c * 100 + x * 10 + y) / 1000.0
                end do
            end do
        end do

        call push_image(pixels)
        result = pop_image()

        call check(error, all(abs(result - pixels) < 1.0e-6), "pixel data not preserved")
    end subroutine


    subroutine test_image_lifo_order(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img1(:,:,:), img2(:,:,:), result(:,:,:)

        allocate(img1(3, 2, 2), img2(1, 5, 5))
        img1 = 0.1
        img2 = 0.9

        call push_image(img1)
        call push_image(img2)

        result = pop_image()
        call check(error, size(result, 1) == 1, "expected 1-channel image first")
        if (allocated(error)) return
        call check(error, size(result, 2) == 5, "expected width 5")
        if (allocated(error)) return

        result = pop_image()
        call check(error, size(result, 1) == 3, "expected 3-channel image second")
        if (allocated(error)) return
        call check(error, size(result, 2) == 2, "expected width 2")
    end subroutine


    ! dup: ( a -- a a )
    subroutine test_image_dup(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), r1(:,:,:), r2(:,:,:)

        allocate(img(3, 2, 2))
        img = 0.5

        call push_image(img)
        call dup_image()

        r1 = pop_image()
        r2 = pop_image()

        call check(error, all(abs(r1 - img) < 1.0e-6), "first copy mismatch")
        if (allocated(error)) return
        call check(error, all(abs(r2 - img) < 1.0e-6), "second copy mismatch")
    end subroutine


    ! drop: ( a -- )
    subroutine test_image_drop(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img1(:,:,:), img2(:,:,:), result(:,:,:)

        allocate(img1(3, 2, 2), img2(1, 4, 4))
        img1 = 0.1
        img2 = 0.9

        call push_image(img1)
        call push_image(img2)
        call drop_image()

        result = pop_image()
        call check(error, size(result, 1) == 3, "expected img1 after drop")
    end subroutine


    ! swap: ( a b -- b a )
    subroutine test_image_swap(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img1(:,:,:), img2(:,:,:), r1(:,:,:), r2(:,:,:)

        allocate(img1(3, 2, 2), img2(1, 4, 4))
        img1 = 0.1
        img2 = 0.9

        call push_image(img1)
        call push_image(img2)
        call swap_image()

        r1 = pop_image()  ! should be img1
        r2 = pop_image()  ! should be img2

        call check(error, size(r1, 1) == 3, "top should be 3-channel after swap")
        if (allocated(error)) return
        call check(error, size(r2, 1) == 1, "second should be 1-channel after swap")
    end subroutine


    ! over: ( a b -- a b a )
    subroutine test_image_over(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img1(:,:,:), img2(:,:,:)
        real, allocatable :: r1(:,:,:), r2(:,:,:), r3(:,:,:)

        allocate(img1(3, 2, 2), img2(1, 4, 4))
        img1 = 0.1
        img2 = 0.9

        call push_image(img1)
        call push_image(img2)
        call over_image()

        r1 = pop_image()  ! should be img1 (copy of second)
        r2 = pop_image()  ! should be img2
        r3 = pop_image()  ! should be img1

        call check(error, size(r1, 1) == 3, "top should be 3-channel (copy of a)")
        if (allocated(error)) return
        call check(error, size(r2, 1) == 1, "middle should be 1-channel (b)")
        if (allocated(error)) return
        call check(error, size(r3, 1) == 3, "bottom should be 3-channel (a)")
    end subroutine


    subroutine test_cmd_transpose(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)

        allocate(img(1, 4, 3))  ! 4 wide, 3 tall
        img = 0.0
        img(1, 2, 1) = 1.0  ! mark position

        call push_image(img)
        call transpose()
        result = pop_image()

        call check(error, size(result, 2) == 3, "width should be old height")
        if (allocated(error)) return
        call check(error, size(result, 3) == 4, "height should be old width")
        if (allocated(error)) return
        call check(error, result(1, 1, 2) > 0.9, "pixel should move to transposed position")
    end subroutine


    subroutine test_cmd_fliph(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)

        ! Layout: (channels, height, width)
        ! Horizontal flip reverses width (dimension 3)
        allocate(img(1, 2, 3))
        img(1, :, 1) = 0.1  ! column 1 (width=1) = 0.1
        img(1, :, 2) = 0.5  ! column 2 (width=2) = 0.5
        img(1, :, 3) = 0.9  ! column 3 (width=3) = 0.9

        call push_image(img)
        call fliph()
        result = pop_image()

        call check(error, all(abs(result(1, :, 1) - 0.9) < 1.0e-6), "left should be old right")
        if (allocated(error)) return
        call check(error, all(abs(result(1, :, 3) - 0.1) < 1.0e-6), "right should be old left")
    end subroutine


    subroutine test_cmd_flipv(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)

        ! Layout: (channels, height, width)
        ! Vertical flip reverses height (dimension 2)
        allocate(img(1, 3, 2))
        img(1, 1, :) = 0.1  ! row 1 (height=1) = 0.1
        img(1, 2, :) = 0.5  ! row 2 (height=2) = 0.5
        img(1, 3, :) = 0.9  ! row 3 (height=3) = 0.9

        call push_image(img)
        call flipv()
        result = pop_image()

        call check(error, all(abs(result(1, 1, :) - 0.9) < 1.0e-6), "top should be old bottom")
        if (allocated(error)) return
        call check(error, all(abs(result(1, 3, :) - 0.1) < 1.0e-6), "bottom should be old top")
    end subroutine


    ! transform with identity: cx cy 0.0 1.0 1.0 0.0 0.0
    subroutine test_cmd_transform_identity(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        integer :: x, y

        allocate(img(1, 5, 5))
        do x = 1, 5
            do y = 1, 5
                img(1, x, y) = real(x * 10 + y) / 100.0
            end do
        end do

        call push_image(img)
        ! push: cx cy angle sx sy tx ty
        call push_number(3.0)   ! cx
        call push_number(3.0)   ! cy
        call push_number(0.0)   ! angle
        call push_number(1.0)   ! sx
        call push_number(1.0)   ! sy
        call push_number(0.0)   ! tx
        call push_number(0.0)   ! ty
        call transform()
        result = pop_image()

        call check(error, all(abs(result - img) < 1.0e-5), "identity transform should preserve image")
    end subroutine


    ! transform with 180 degree rotation
    subroutine test_cmd_transform_rotate(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)

        allocate(img(1, 5, 5))
        img = 0.0
        img(1, 2, 2) = 1.0  ! off-center pixel

        call push_image(img)
        ! push: cx cy angle sx sy tx ty
        call push_number(3.0)     ! cx (center)
        call push_number(3.0)     ! cy (center)
        call push_number(180.0)   ! angle
        call push_number(1.0)     ! sx
        call push_number(1.0)     ! sy
        call push_number(0.0)     ! tx
        call push_number(0.0)     ! ty
        call transform()
        result = pop_image()

        ! pixel at (2,2) should rotate to (4,4)
        call check(error, result(1, 4, 4) > 0.5, "pixel should rotate to opposite corner")
        if (allocated(error)) return
        call check(error, result(1, 2, 2) < 0.1, "original position should be empty")
    end subroutine


    subroutine test_cmd_split_rgb(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), r(:,:,:), g(:,:,:), b(:,:,:)

        allocate(img(3, 4, 4))
        img(1,:,:) = 0.2  ! red
        img(2,:,:) = 0.5  ! green
        img(3,:,:) = 0.8  ! blue

        call push_image(img)
        call split()

        b = pop_image()
        g = pop_image()
        r = pop_image()

        call check(error, size(r, 1) == 1, "red should be 1 channel")
        if (allocated(error)) return
        call check(error, size(g, 1) == 1, "green should be 1 channel")
        if (allocated(error)) return
        call check(error, size(b, 1) == 1, "blue should be 1 channel")
        if (allocated(error)) return
        call check(error, all(abs(r(1,:,:) - 0.2) < 1.0e-6), "red values wrong")
        if (allocated(error)) return
        call check(error, all(abs(g(1,:,:) - 0.5) < 1.0e-6), "green values wrong")
        if (allocated(error)) return
        call check(error, all(abs(b(1,:,:) - 0.8) < 1.0e-6), "blue values wrong")
    end subroutine


    subroutine test_cmd_split_greyscale_noop(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)

        allocate(img(1, 3, 3))
        img = 0.5

        call push_image(img)
        call split()

        result = pop_image()

        call check(error, size(result, 1) == 1, "should still be 1 channel")
        if (allocated(error)) return
        call check(error, all(abs(result - img) < 1.0e-6), "should be unchanged")
    end subroutine


    subroutine test_cmd_merge_rgb(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: r(:,:,:), g(:,:,:), b(:,:,:), result(:,:,:)

        allocate(r(1, 3, 3), g(1, 3, 3), b(1, 3, 3))
        r = 0.1
        g = 0.5
        b = 0.9

        call push_image(r)
        call push_image(g)
        call push_image(b)
        call push_number(3.0)
        call merge()

        result = pop_image()

        call check(error, size(result, 1) == 3, "should be 3 channels")
        if (allocated(error)) return
        call check(error, all(abs(result(1,:,:) - 0.1) < 1.0e-6), "red channel wrong")
        if (allocated(error)) return
        call check(error, all(abs(result(2,:,:) - 0.5) < 1.0e-6), "green channel wrong")
        if (allocated(error)) return
        call check(error, all(abs(result(3,:,:) - 0.9) < 1.0e-6), "blue channel wrong")
    end subroutine


    subroutine test_cmd_split_merge_roundtrip(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)
        integer :: c, x, y

        allocate(img(3, 5, 5))
        do c = 1, 3
            do x = 1, 5
                do y = 1, 5
                    img(c, x, y) = real(c * 100 + x * 10 + y) / 1000.0
                end do
            end do
        end do

        call push_image(img)
        call split()
        call push_number(3.0)
        call merge()

        result = pop_image()

        call check(error, all(abs(result - img) < 1.0e-6), "roundtrip should preserve image")
    end subroutine


    subroutine test_cmd_fill_rgb(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)

        allocate(img(3, 4, 4))
        img = 0.0

        call push_image(img)
        ! Stack order: r g b (natural order)
        call push_number(1.0)   ! red
        call push_number(0.5)   ! green
        call push_number(0.25)  ! blue
        call fill()

        result = pop_image()

        call check(error, all(abs(result(1,:,:) - 1.0) < 1.0e-6), "red channel should be 1.0")
        if (allocated(error)) return
        call check(error, all(abs(result(2,:,:) - 0.5) < 1.0e-6), "green channel should be 0.5")
        if (allocated(error)) return
        call check(error, all(abs(result(3,:,:) - 0.25) < 1.0e-6), "blue channel should be 0.25")
    end subroutine


    subroutine test_cmd_fill_greyscale(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img(:,:,:), result(:,:,:)

        allocate(img(1, 3, 3))
        img = 0.0

        call push_image(img)
        call push_number(0.7)
        call fill()

        result = pop_image()

        call check(error, all(abs(result(1,:,:) - 0.7) < 1.0e-6), "greyscale should be 0.7")
    end subroutine


    subroutine test_cmd_interpolate_blend(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img1(:,:,:), img2(:,:,:), result(:,:,:)

        ! Layout: (channels, height, width)
        allocate(img1(1, 3, 3), img2(1, 3, 3))
        img1 = 1.0
        img2 = 0.0

        ! Stack: img2 (bottom), img1 (top), then alpha
        ! interpolate pops: alpha, pixels1, pixels2
        ! result = alpha*pixels1 + (1-alpha)*pixels2
        call push_image(img2)
        call push_image(img1)
        call push_number(0.5)
        call interpolate()

        result = pop_image()

        ! 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        call check(error, all(abs(result - 0.5) < 1.0e-6), "50/50 blend should be 0.5")
    end subroutine


    subroutine test_cmd_interpolate_alpha_one(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img1(:,:,:), img2(:,:,:), result(:,:,:)

        allocate(img1(1, 3, 3), img2(1, 3, 3))
        img1 = 0.8
        img2 = 0.2

        call push_image(img2)
        call push_image(img1)
        call push_number(1.0)  ! alpha=1.0 means 100% pixels1
        call interpolate()

        result = pop_image()

        ! 1.0 * 0.8 + 0.0 * 0.2 = 0.8
        call check(error, all(abs(result - 0.8) < 1.0e-6), "alpha=1.0 should return first image")
    end subroutine


    subroutine test_cmd_interpolate_alpha_zero(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img1(:,:,:), img2(:,:,:), result(:,:,:)

        allocate(img1(1, 3, 3), img2(1, 3, 3))
        img1 = 0.8
        img2 = 0.2

        call push_image(img2)
        call push_image(img1)
        call push_number(0.0)  ! alpha=0.0 means 100% pixels2
        call interpolate()

        result = pop_image()

        ! 0.0 * 0.8 + 1.0 * 0.2 = 0.2
        call check(error, all(abs(result - 0.2) < 1.0e-6), "alpha=0.0 should return second image")
    end subroutine


    subroutine test_cmd_interpolate_frames_basic(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img1(:,:,:), img2(:,:,:)
        real, allocatable :: r1(:,:,:), r2(:,:,:), r3(:,:,:)

        allocate(img1(1, 2, 2), img2(1, 2, 2))
        img1 = 1.0
        img2 = 0.0

        ! Stack: img2 (bottom), img1 (top), then num_frames
        ! interpolate_frames pops: num_frames, pixels1, pixels2
        call push_image(img2)
        call push_image(img1)
        call push_number(3.0)  ! 3 frames
        call interpolate_frames()

        ! num_frames = 3 - 1 = 2, loop i = 0 to 2
        ! i=0: alpha=0.0, result = 0*img1 + 1*img2 = 0.0
        ! i=1: alpha=0.5, result = 0.5*img1 + 0.5*img2 = 0.5
        ! i=2: alpha=1.0, result = 1*img1 + 0*img2 = 1.0
        ! Pushed in order: 0.0, 0.5, 1.0 (top)

        r1 = pop_image()  ! 1.0
        r2 = pop_image()  ! 0.5
        r3 = pop_image()  ! 0.0

        call check(error, all(abs(r1 - 1.0) < 1.0e-6), "first pop should be img1 (1.0)")
        if (allocated(error)) return
        call check(error, all(abs(r2 - 0.5) < 1.0e-6), "second pop should be blend (0.5)")
        if (allocated(error)) return
        call check(error, all(abs(r3 - 0.0) < 1.0e-6), "third pop should be img2 (0.0)")
    end subroutine


    subroutine test_cmd_interpolate_frames_single(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img1(:,:,:), img2(:,:,:), result(:,:,:)

        allocate(img1(1, 2, 2), img2(1, 2, 2))
        img1 = 0.8
        img2 = 0.2

        call push_image(img2)
        call push_image(img1)
        call push_number(1.0)  ! 1 frame
        call interpolate_frames()

        ! num_frames = 1 - 1 = 0, early return with just pixels1
        result = pop_image()

        call check(error, all(abs(result - 0.8) < 1.0e-6), "single frame should return pixels1")
    end subroutine


    subroutine test_cmd_interpolate_frames_zero(error)
        type(error_type), allocatable, intent(out) :: error
        real, allocatable :: img1(:,:,:), img2(:,:,:), result(:,:,:)

        allocate(img1(1, 2, 2), img2(1, 2, 2))
        img1 = 0.7
        img2 = 0.3

        call push_image(img2)
        call push_image(img1)
        call push_number(0.0)  ! 0 frames
        call interpolate_frames()

        ! num_frames = 0 - 1 = -1, early return with just pixels1
        result = pop_image()

        call check(error, all(abs(result - 0.7) < 1.0e-6), "zero frames should return pixels1")
    end subroutine

end module test_command
