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
            new_unittest("image_over", test_image_over) &
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

end module test_command
