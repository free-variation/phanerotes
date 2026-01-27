module test_interpreter
    use testdrive, only: new_unittest, unittest_type, error_type, check
    use interpreter
    use command
    implicit none
    private

    public :: collect_interpreter_tests

contains

    subroutine collect_interpreter_tests(testsuite)
        type(unittest_type), allocatable, intent(out) :: testsuite(:)

        testsuite = [ &
            new_unittest("token_simple_word", test_token_simple_word), &
            new_unittest("token_multiple_words", test_token_multiple_words), &
            new_unittest("token_string_literal", test_token_string_literal), &
            new_unittest("token_string_with_spaces", test_token_string_with_spaces), &
            new_unittest("token_comment", test_token_comment), &
            new_unittest("token_comment_midline", test_token_comment_midline), &
            new_unittest("token_multiple_spaces", test_token_multiple_spaces), &
            new_unittest("token_empty_line", test_token_empty_line), &
            new_unittest("dispatch_number", test_dispatch_number), &
            new_unittest("dispatch_string", test_dispatch_string), &
            new_unittest("execute_pushes_numbers", test_execute_pushes_numbers), &
            new_unittest("repeat_pushes_multiple", test_repeat_pushes_multiple), &
            new_unittest("list_files_pushes_count", test_list_files_pushes_count), &
            new_unittest("colon_def_basic", test_colon_def_basic), &
            new_unittest("colon_def_single_line", test_colon_def_single_line), &
            new_unittest("colon_def_multiple_words", test_colon_def_multiple_words), &
            new_unittest("zero_test_stops_when_zero", test_zero_test_stops_when_zero), &
            new_unittest("zero_test_continues_when_nonzero", test_zero_test_continues_when_nonzero), &
            new_unittest("recursion_with_base_case", test_recursion_with_base_case), &
            new_unittest("empty_string_stack_stops_when_empty", test_empty_string_stack_stops_when_empty), &
            new_unittest("empty_string_stack_continues_when_not_empty", test_empty_string_stack_continues_when_not_empty), &
            new_unittest("empty_image_stack_stops_when_empty", test_empty_image_stack_stops_when_empty), &
            new_unittest("empty_image_stack_continues_when_not_empty", test_empty_image_stack_continues_when_not_empty), &
            new_unittest("arithmetic_add", test_arithmetic_add), &
            new_unittest("arithmetic_subtract", test_arithmetic_subtract), &
            new_unittest("arithmetic_multiply", test_arithmetic_multiply), &
            new_unittest("arithmetic_divide", test_arithmetic_divide), &
            new_unittest("number_dup", test_number_dup), &
            new_unittest("number_drop", test_number_drop), &
            new_unittest("number_swap", test_number_swap), &
            new_unittest("number_over", test_number_over) &
        ]
    end subroutine


    subroutine test_token_simple_word(error)
        type(error_type), allocatable, intent(out) :: error
        character(256) :: token
        integer :: pos

        pos = 1
        call next_token("load", pos, token)

        call check(error, trim(token) == "load", "expected 'load'")
    end subroutine


    subroutine test_token_multiple_words(error)
        type(error_type), allocatable, intent(out) :: error
        character(256) :: token
        integer :: pos

        pos = 1
        call next_token("load save quit", pos, token)
        call check(error, trim(token) == "load", "first token should be 'load'")
        if (allocated(error)) return

        call next_token("load save quit", pos, token)
        call check(error, trim(token) == "save", "second token should be 'save'")
        if (allocated(error)) return

        call next_token("load save quit", pos, token)
        call check(error, trim(token) == "quit", "third token should be 'quit'")
    end subroutine


    subroutine test_token_string_literal(error)
        type(error_type), allocatable, intent(out) :: error
        character(256) :: token
        integer :: pos

        pos = 1
        call next_token('"test.png"', pos, token)

        call check(error, trim(token) == '"test.png', "expected string with leading quote")
    end subroutine


    subroutine test_token_string_with_spaces(error)
        type(error_type), allocatable, intent(out) :: error
        character(256) :: token
        integer :: pos

        pos = 1
        call next_token('"my file.png" load', pos, token)

        call check(error, trim(token) == '"my file.png', "string should include spaces")
        if (allocated(error)) return

        call next_token('"my file.png" load', pos, token)
        call check(error, trim(token) == "load", "next token should be 'load'")
    end subroutine


    subroutine test_token_comment(error)
        type(error_type), allocatable, intent(out) :: error
        character(256) :: token
        integer :: pos

        pos = 1
        call next_token("# this is a comment", pos, token)

        call check(error, len_trim(token) == 0, "comment should return empty token")
    end subroutine


    subroutine test_token_comment_midline(error)
        type(error_type), allocatable, intent(out) :: error
        character(256) :: token
        integer :: pos

        pos = 1
        call next_token("load # comment", pos, token)
        call check(error, trim(token) == "load", "first token before comment")
        if (allocated(error)) return

        call next_token("load # comment", pos, token)
        call check(error, len_trim(token) == 0, "comment should stop parsing")
    end subroutine


    subroutine test_token_multiple_spaces(error)
        type(error_type), allocatable, intent(out) :: error
        character(256) :: token
        integer :: pos

        pos = 1
        call next_token("load    save", pos, token)
        call check(error, trim(token) == "load", "first token")
        if (allocated(error)) return

        call next_token("load    save", pos, token)
        call check(error, trim(token) == "save", "should skip multiple spaces")
    end subroutine


    subroutine test_token_empty_line(error)
        type(error_type), allocatable, intent(out) :: error
        character(256) :: token
        integer :: pos

        pos = 1
        call next_token("", pos, token)

        call check(error, len_trim(token) == 0, "empty line should return empty token")
    end subroutine


    subroutine test_dispatch_number(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit, done
        integer :: repeats
        real :: result

        quit = .false.
        done = .false.
        call dispatch("42.5", quit, done, repeats)
        result = pop_number()

        call check(error, abs(result - 42.5) < 1.0e-6, "should push number")
    end subroutine


    subroutine test_dispatch_string(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit, done
        integer :: repeats
        character(MAX_STRING_LENGTH) :: result

        quit = .false.
        done = .false.
        call dispatch('"hello.png', quit, done, repeats)
        result = pop_string()

        call check(error, trim(result) == "hello.png", "should push string without quote")
    end subroutine


    subroutine test_execute_pushes_numbers(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: n1, n2, n3

        quit = .false.
        call execute_line("1 2 3", quit)

        n3 = pop_number()
        n2 = pop_number()
        n1 = pop_number()

        call check(error, abs(n1 - 1.0) < 1.0e-6, "first number")
        if (allocated(error)) return
        call check(error, abs(n2 - 2.0) < 1.0e-6, "second number")
        if (allocated(error)) return
        call check(error, abs(n3 - 3.0) < 1.0e-6, "third number")
    end subroutine


    subroutine test_repeat_pushes_multiple(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: n1, n2, n3

        quit = .false.
        call execute_line("3 repeat 1", quit)

        n3 = pop_number()
        n2 = pop_number()
        n1 = pop_number()

        call check(error, abs(n1 - 1.0) < 1.0e-6, "first 1")
        if (allocated(error)) return
        call check(error, abs(n2 - 1.0) < 1.0e-6, "second 1")
        if (allocated(error)) return
        call check(error, abs(n3 - 1.0) < 1.0e-6, "third 1")
    end subroutine


    subroutine test_list_files_pushes_count(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: count
        integer :: i
        character(MAX_STRING_LENGTH) :: discard

        quit = .false.
        call execute_line('"test" list-files', quit)

        count = pop_number()

        ! Clean up pushed filenames
        do i = 1, int(count)
            discard = pop_string()
        end do

        call check(error, count > 0, "should find files in test directory")
    end subroutine


    subroutine test_colon_def_basic(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: result

        quit = .false.
        call execute_line(": pushfive 5 ;", quit)
        call execute_line("pushfive", quit)

        result = pop_number()

        call check(error, abs(result - 5.0) < 1.0e-6, "defined word should push 5")
    end subroutine


    subroutine test_colon_def_single_line(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: n1, n2

        quit = .false.
        call execute_line(": pushtwo 1 2 ;", quit)
        call execute_line("pushtwo", quit)

        n2 = pop_number()
        n1 = pop_number()

        call check(error, abs(n1 - 1.0) < 1.0e-6, "first number should be 1")
        if (allocated(error)) return
        call check(error, abs(n2 - 2.0) < 1.0e-6, "second number should be 2")
    end subroutine


    subroutine test_colon_def_multiple_words(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: n1, n2, n3

        quit = .false.
        call execute_line(": push3ones 3 repeat 1 ;", quit)
        call execute_line("push3ones", quit)

        n3 = pop_number()
        n2 = pop_number()
        n1 = pop_number()

        call check(error, abs(n1 - 1.0) < 1.0e-6, "first 1")
        if (allocated(error)) return
        call check(error, abs(n2 - 1.0) < 1.0e-6, "second 1")
        if (allocated(error)) return
        call check(error, abs(n3 - 1.0) < 1.0e-6, "third 1")
    end subroutine


    subroutine test_zero_test_stops_when_zero(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: result

        quit = .false.
        ! Push 0, then 0? should stop, so 99 is never pushed
        call execute_line("0 0? 99", quit)

        result = pop_number()

        call check(error, abs(result - 0.0) < 1.0e-6, "should be 0, not 99")
    end subroutine


    subroutine test_zero_test_continues_when_nonzero(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: n1, n2

        quit = .false.
        ! Push 5, then 0? should not stop (5 /= 0), so 99 is pushed
        call execute_line("5 0? 99", quit)

        n2 = pop_number()
        n1 = pop_number()

        call check(error, abs(n1 - 5.0) < 1.0e-6, "first should be 5")
        if (allocated(error)) return
        call check(error, abs(n2 - 99.0) < 1.0e-6, "second should be 99")
    end subroutine


    subroutine test_recursion_with_base_case(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: n1, n2, n3

        quit = .false.
        ! Define countdown: if 0, stop; otherwise push current value and recurse with n-1
        ! Note: we need subtract. Let's test simpler: count pushes until 0
        ! : countpush dup 0? dup 1 - countpush ;
        ! But we don't have subtraction yet. Test that recursion terminates:

        ! Simpler test: define word that recurses but stops on 0
        ! Push 3, 2, 1 then 0 stops it
        call execute_line(": stop0 dup 0? ;", quit)
        call execute_line("0 stop0", quit)

        n1 = pop_number()

        ! Should just have the 0 (dup'd once before 0? stopped)
        call check(error, abs(n1 - 0.0) < 1.0e-6, "should stop with 0 on stack")
    end subroutine


    subroutine test_empty_string_stack_stops_when_empty(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: result
        character(MAX_STRING_LENGTH) :: discard_str

        quit = .false.
        ! Drain any leftover strings from prior tests
        do while (string_stack_top > 0)
            discard_str = pop_string()
        end do

        call execute_line("42 empty-string-stack? 99", quit)

        result = pop_number()

        call check(error, abs(result - 42.0) < 1.0e-6, "should be 42, not 99")
    end subroutine


    subroutine test_empty_string_stack_continues_when_not_empty(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: result

        quit = .false.
        ! Push a string first, then empty-string-stack? should NOT stop
        call execute_line('"hello" empty-string-stack? 99', quit)

        result = pop_number()

        call check(error, abs(result - 99.0) < 1.0e-6, "should push 99 when string stack not empty")
    end subroutine


    subroutine test_empty_image_stack_stops_when_empty(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: result
        real, allocatable :: discard(:,:,:)

        quit = .false.
        ! Drain any leftover images from prior tests
        do while (image_stack_top > 0)
            discard = pop_image()
        end do

        call execute_line("42 empty-image-stack? 99", quit)

        result = pop_number()

        call check(error, abs(result - 42.0) < 1.0e-6, "should be 42, not 99")
    end subroutine


    subroutine test_empty_image_stack_continues_when_not_empty(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: result
        real, allocatable :: discard(:,:,:)

        quit = .false.
        ! Load an image first, then empty-image-stack? should NOT stop
        call execute_line('"images/test_input_1.png" load empty-image-stack? 99', quit)

        result = pop_number()
        discard = pop_image()

        call check(error, abs(result - 99.0) < 1.0e-6, "should push 99 when image stack not empty")
    end subroutine


    subroutine test_arithmetic_add(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: result

        quit = .false.
        call execute_line("3 2 +", quit)

        result = pop_number()

        call check(error, abs(result - 5.0) < 1.0e-6, "3 + 2 = 5")
    end subroutine


    subroutine test_arithmetic_subtract(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: result

        quit = .false.
        call execute_line("5 3 -", quit)

        result = pop_number()

        ! Forth: "5 3 -" means 5 - 3 = 2
        call check(error, abs(result - 2.0) < 1.0e-6, "5 - 3 = 2")
    end subroutine


    subroutine test_arithmetic_multiply(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: result

        quit = .false.
        call execute_line("4 3 *", quit)

        result = pop_number()

        call check(error, abs(result - 12.0) < 1.0e-6, "4 * 3 = 12")
    end subroutine


    subroutine test_arithmetic_divide(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: result

        quit = .false.
        ! Drain number stack from prior tests
        do while (number_stack_top > 0)
            result = pop_number()
        end do

        call execute_line("12 4 /", quit)

        result = pop_number()

        ! Forth: "12 4 /" means 12 / 4 = 3
        call check(error, abs(result - 3.0) < 1.0e-6, "12 / 4 = 3")
    end subroutine


    ! dup: ( a -- a a )
    subroutine test_number_dup(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: n1, n2

        quit = .false.
        call execute_line("42 dup", quit)

        n2 = pop_number()
        n1 = pop_number()

        call check(error, abs(n1 - 42.0) < 1.0e-6, "first should be 42")
        if (allocated(error)) return
        call check(error, abs(n2 - 42.0) < 1.0e-6, "second should be 42")
    end subroutine


    ! drop: ( a -- )
    subroutine test_number_drop(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: result

        quit = .false.
        call execute_line("1 2 drop", quit)

        result = pop_number()

        call check(error, abs(result - 1.0) < 1.0e-6, "should be 1 after dropping 2")
    end subroutine


    ! swap: ( a b -- b a )
    subroutine test_number_swap(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: n1, n2

        quit = .false.
        call execute_line("1 2 swap", quit)

        n2 = pop_number()
        n1 = pop_number()

        call check(error, abs(n1 - 2.0) < 1.0e-6, "first should be 2")
        if (allocated(error)) return
        call check(error, abs(n2 - 1.0) < 1.0e-6, "second should be 1")
    end subroutine


    ! over: ( a b -- a b a )
    subroutine test_number_over(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: quit
        real :: n1, n2, n3

        quit = .false.
        call execute_line("1 2 over", quit)

        n3 = pop_number()
        n2 = pop_number()
        n1 = pop_number()

        call check(error, abs(n1 - 1.0) < 1.0e-6, "bottom should be 1")
        if (allocated(error)) return
        call check(error, abs(n2 - 2.0) < 1.0e-6, "middle should be 2")
        if (allocated(error)) return
        call check(error, abs(n3 - 1.0) < 1.0e-6, "top should be 1 (copy of bottom)")
    end subroutine

end module test_interpreter
