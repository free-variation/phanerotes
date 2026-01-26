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
            new_unittest("colon_def_multiple_words", test_colon_def_multiple_words) &
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
        logical :: running
        integer :: repeats
        real :: result

        running = .true.
        call dispatch("42.5", running, repeats)
        result = pop_number()

        call check(error, abs(result - 42.5) < 1.0e-6, "should push number")
    end subroutine


    subroutine test_dispatch_string(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: running
        integer :: repeats
        character(MAX_STRING_LENGTH) :: result

        running = .true.
        call dispatch('"hello.png', running, repeats)
        result = pop_string()

        call check(error, trim(result) == "hello.png", "should push string without quote")
    end subroutine


    subroutine test_execute_pushes_numbers(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: running
        real :: n1, n2, n3

        running = .true.
        call execute_line("1 2 3", running)

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
        logical :: running
        real :: n1, n2, n3

        running = .true.
        call execute_line("3 repeat 1", running)

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
        logical :: running
        real :: count

        running = .true.
        call execute_line('"test" list_files', running)

        count = pop_number()

        call check(error, count > 0, "should find files in test directory")
    end subroutine


    subroutine test_colon_def_basic(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: running
        real :: result

        running = .true.
        call execute_line(": pushfive 5 ;", running)
        call execute_line("pushfive", running)

        result = pop_number()

        call check(error, abs(result - 5.0) < 1.0e-6, "defined word should push 5")
    end subroutine


    subroutine test_colon_def_single_line(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: running
        real :: n1, n2

        running = .true.
        call execute_line(": pushtwo 1 2 ;", running)
        call execute_line("pushtwo", running)

        n2 = pop_number()
        n1 = pop_number()

        call check(error, abs(n1 - 1.0) < 1.0e-6, "first number should be 1")
        if (allocated(error)) return
        call check(error, abs(n2 - 2.0) < 1.0e-6, "second number should be 2")
    end subroutine


    subroutine test_colon_def_multiple_words(error)
        type(error_type), allocatable, intent(out) :: error
        logical :: running
        real :: n1, n2, n3

        running = .true.
        call execute_line(": push3ones 3 repeat 1 ;", running)
        call execute_line("push3ones", running)

        n3 = pop_number()
        n2 = pop_number()
        n1 = pop_number()

        call check(error, abs(n1 - 1.0) < 1.0e-6, "first 1")
        if (allocated(error)) return
        call check(error, abs(n2 - 1.0) < 1.0e-6, "second 1")
        if (allocated(error)) return
        call check(error, abs(n3 - 1.0) < 1.0e-6, "third 1")
    end subroutine

end module test_interpreter
