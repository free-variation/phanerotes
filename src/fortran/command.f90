module command
    implicit none
    
    integer, parameter :: MAX_STACK = 100
    integer, parameter :: MAX_STRING_LENGTH = 256

    type image_entry 
        real, allocatable :: pixels(:,:,:)
    end type
    
    type(image_entry):: image_stack(MAX_STACK)
    real :: number_stack(MAX_STACK)
    character(MAX_STRING_LENGTH) :: string_stack(MAX_STACK)

    integer :: image_stack_top = 0
    integer :: number_stack_top = 0
    integer :: string_stack_top = 0

    contains

        ! ---------- Stack management ----------
        subroutine push_string(s)
            character(*), intent(in) :: s

            if (string_stack_top >= MAX_STACK) error stop "string stack overflow"

            string_stack_top = string_stack_top + 1
            string_stack(string_stack_top) = s
        end subroutine push_string
        
        subroutine push_number(n)
            real, intent(in) :: n

            if (number_stack_top >= MAX_STACK) error stop "number stack overflow"

            number_stack_top = number_stack_top + 1
            number_stack(number_stack_top) = n
        end subroutine push_number

        subroutine push_image(pixels)
            real, intent(in) :: pixels(:,:,:)

            if (image_stack_top >= MAX_STACK) error stop "image stack overflow"            

            image_stack_top = image_stack_top + 1
            image_stack(image_stack_top)%pixels = pixels
        end subroutine push_image

        function pop_string()
            character(MAX_STRING_LENGTH) :: pop_string

            if (string_stack_top == 0) error stop "string stack underflow"
            
            pop_string = string_stack(string_stack_top)
            string_stack_top = string_stack_top - 1
        end function
        
        function pop_number()
            real :: pop_number

            if (number_stack_top == 0) error stop "number stack underflow"
            
            pop_number = number_stack(number_stack_top)
            number_stack_top = number_stack_top - 1
        end function

 
        function pop_image() result(pixels)
            real, allocatable :: pixels(:,:,:)
            
            if (image_stack_top == 0) error stop "image stack underflow"
            
            pixels = image_stack(image_stack_top)%pixels
            deallocate(image_stack(image_stack_top)%pixels)
            image_stack_top = image_stack_top - 1
        end function

        ! ---------- Stack words ----------
        subroutine dup_image()
            real, allocatable :: pixels(:, :, :) 
            
            pixels = pop_image()
            call push_image(pixels)
            call push_image(pixels)
        end subroutine dup_image

        subroutine drop_image()
            real, allocatable :: discard(:,:,:)

            discard = pop_image()
        end subroutine drop_image

        subroutine swap_image() 
            real, allocatable :: pixels1(:,:,:), pixels2(:,:,:)

            pixels1 = pop_image()
            pixels2 = pop_image()
            call push_image(pixels1)
            call push_image(pixels2)
        end subroutine swap_image

        subroutine over_image()
            real, allocatable :: pixels1(:,:,:), pixels2(:,:,:)

            pixels1 = pop_image()
            pixels2 = pop_image()
            call push_image(pixels2)
            call push_image(pixels1)
            call push_image(pixels2)
        end subroutine over_image
end module
