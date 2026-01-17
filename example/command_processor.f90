program command_processor
    use :: command

    character(MAX_STRING_LENGTH) :: s

    call push_string("test")
    s = pop_string()
    print *, s

end program
