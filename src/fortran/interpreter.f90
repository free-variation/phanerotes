module interpreter
    use :: command
    implicit none

    type :: word_definition
        character(MAX_STRING_LENGTH) :: name
        character(MAX_STRING_LENGTH) :: tokens(128)
        integer :: num_tokens
    end type

    integer, parameter :: MAX_WORDS = 256
    type(word_definition) :: dictionary(MAX_WORDS)
    integer :: compiling = 0
    integer :: num_words = 0

    contains
        subroutine execute_line(line, quit)
            character(*), intent(in) :: line
            logical, intent(inout) :: quit

            character(MAX_STRING_LENGTH), allocatable :: tokens(:)

            tokens = tokenize(line)
            call dispatch_tokens(tokens, quit)
        end subroutine

        recursive subroutine dispatch_tokens(tokens, quit)
            character(*), intent(in) :: tokens(:)
            logical, intent(inout) :: quit

            integer :: i, repeats
            logical :: done

            repeats = 1
            done = .false.

            if (size(tokens) == 0) return

            if (compiling == 1) then
                dictionary(num_words)%name = tokens(1)
                dictionary(num_words)%num_tokens = 0
                compiling = 2
            else if (compiling == 2) then
                if (tokens(1) == ";") then
                    compiling = 0
                else
                    dictionary(num_words)%num_tokens = dictionary(num_words)%num_tokens + 1
                    dictionary(num_words)%tokens(dictionary(num_words)%num_tokens) = tokens(1)
                end if
            else
                call dispatch(tokens(1), quit, done, repeats)
                if (quit .or. done) return
            end if

            do i = 1, repeats
                call dispatch_tokens(tokens(2:), quit)
                if (quit) exit
            end do

        end subroutine

        function tokenize(line)
            character(*), intent(in) :: line
            character(MAX_STRING_LENGTH), allocatable :: tokenize(:)

            character(MAX_STRING_LENGTH) :: token
            integer :: pos, num_tokens, i

            pos = 1
            num_tokens = 0
            do while (pos <= len_trim(line))
                call next_token(line, pos, token)
                if (len_trim(token) == 0) exit

                num_tokens = num_tokens + 1
            end do

            allocate(tokenize(num_tokens))

            pos = 1
            do i = 1, num_tokens
                call next_token(line, pos, tokenize(i))
            end do
        end function

        subroutine next_token(line, pos, token)
            character(*), intent(in) :: line
            integer, intent(inout) :: pos
            character(*), intent(out) :: token
            integer :: start, line_len

            token = ""
            line_len = len_trim(line)

            ! skip whitespace
            do while (pos <= line_len)
                if (line(pos:pos) /= ' ') exit
                pos = pos + 1
            end do

            if (pos > line_len) return

            ! comment - rest of line ignored
            if (line(pos:pos) == "#") return

            ! string literal
            if (line(pos:pos) == '"') then
                pos = pos + 1
                start = pos
                do while (pos <= line_len)
                    if (line(pos:pos) == '"') exit
                    pos = pos + 1
                end do

                token = '"' // line(start:pos-1)
                if (pos <= line_len) pos = pos + 1
                return
            end if

            ! regular token
            start = pos
            do while (pos <= line_len)
                if (line(pos:pos) == ' ') exit
                pos = pos + 1
            end do
            token = line(start:pos-1)
        end subroutine

        subroutine dispatch(token, quit, done, repeats)
            character(*), intent(in) :: token
            logical, intent(inout) :: quit
            logical, intent(inout) :: done
            integer, intent(out) :: repeats

            real :: num, b
            integer :: iostat
            integer :: i

            repeats = 1

            if (len_trim(token) == 0) return

            ! string literal
            if (token(1:1) == '"') then
                call push_string(token(2:))
                return
            end if

            ! try as a number (but "/" is a Fortran list-directed terminator, skip it)
            if (trim(token) /= "/") then
                read(token, *, iostat = iostat) num
                if (iostat == 0) then
                    call push_number(num)
                    return
                end if
            end if

            ! if the token is a colon definition, run that and return
            do i = 1, num_words
                if (dictionary(i)%name == token) then
                    call dispatch_tokens(dictionary(i)%tokens(1:dictionary(i)%num_tokens), quit)
                    return
                end if
            end do

            ! words
            select case (trim(token))
            case ("quit")
                quit = .true.
            case ("idup")
                call dup_image()
            case ("idrop")
                call drop_image()
            case ("iswap")
                call swap_image()
            case ("iover")
                call over_image()

            case (".")
                call dot()
            case ("s.")
                call sdot()
            case ("cr")
                print *
            
            case ("dup")
                call dup_number()
            case ("drop")
                call drop_number()
            case ("swap")
                call swap_number()
            case ("over")
                call over_number()
            case ("+")
                call push_number(pop_number() + pop_number())
            case ("-")
                b = pop_number()
                call push_number(pop_number() - b)
            case ("*")
                call push_number(pop_number() * pop_number())
            case ("/")
                b = pop_number()
                call push_number(pop_number() / b)

            case ("repeat")
                repeats = int(pop_number())
            case ("0?")
                if (peek_number() == 0) done = .true.
            case ("empty-string-stack?")
                if (string_stack_top == 0) done = .true.
            case ("empty-image-stack?")
                if (image_stack_top == 0) done = .true.
            case (":")
                compiling = 1
                num_words = num_words + 1

            case ("load")
                call load()
            case ("save")
                call save()
            case ("transpose")
                call transpose()
            case ("fliph")
                call fliph()
            case ("flipv")
                call flipv()
            case ("transform")
                call transform()
            case ("split")
                call split()
            case ("merge")
                call merge()
            case ("list-files")
                call list_files()
            case ("interpolate")
                call interpolate()
            case ("interpolate-frames")
                call interpolate_frames()
            case default
                print *, "unknown word: ", trim(token)
            end select
        end subroutine

    end module interpreter
