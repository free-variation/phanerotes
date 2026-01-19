module cnn_autoencoder
    use :: cnn_core
    use :: nn

    implicit none

    type :: autoencoder_config
        integer :: input_channels   ! e.g. 3 for RGB, 1 for greyscale
        integer :: num_layers       
        integer :: base_channels    ! channels after first layer
        integer :: max_channels     ! channel cap (e.g. 512)
        integer :: kernel_width     ! typically 3
        integer :: kernel_height    ! typically 3
        integer :: stride           ! typically 2 for encoder
        integer :: padding          ! typically 1 for 3x3 kernel
    end type

    type :: autoencoder
        type(autoencoder_config) :: config
        type(conv_layer), allocatable :: encoder(:)
        type(conv_layer), allocatable :: decoder(:)
    end type

    contains
        subroutine init_layer(layer, config, in_channels, out_channels, stride)
            type(conv_layer), intent(inout) :: layer
            type(autoencoder_config), intent(in) :: config
            integer, intent(in) :: in_channels, out_channels, stride
            real :: scale

            layer%in_channels = in_channels
            layer%out_channels = out_channels
            layer%kernel_width = config%kernel_width
            layer%kernel_height = config%kernel_height
            layer%stride = stride
            layer%padding = config%padding

            allocate(layer%weights(out_channels, in_channels, layer%kernel_width, layer%kernel_height))
            allocate(layer%bias(out_channels))
            allocate(layer%weights_grad, mold = layer%weights)
            allocate(layer%bias_grad, mold = layer%bias)

            layer%bias = 0.0
            layer%weights_grad = 0.0
            layer%bias_grad = 0.0

            scale = sqrt(2.0 / (layer%in_channels * layer%kernel_width * layer%kernel_height))
            call random_number(layer%weights)
            layer%weights = (layer%weights - 0.5) * 2.0 * scale
        end subroutine


        function autoencoder_init(config) result(net)
            type(autoencoder_config), intent(in) :: config
            type(autoencoder) :: net

            integer :: i, in_channels, out_channels

            net%config = config

            allocate(net%encoder(config%num_layers))
            allocate(net%decoder(config%num_layers))

            do i = 1, config%num_layers
                if (i == 1) then
                    in_channels = config%input_channels
                else
                    in_channels = min(config%base_channels * 2**(i - 1), config%max_channels)
                end if
                out_channels = min(config%base_channels * 2**i, config%max_channels)
                
                call init_layer(net%encoder(i), config, in_channels, out_channels, config%stride)
            end do
                
            do i = 1, config%num_layers
                in_channels = net%encoder(config%num_layers - i + 1)%out_channels
                out_channels = net%encoder(config%num_layers - i + 1)%in_channels

                call init_layer(net%decoder(i), config, in_channels, out_channels, 1)
            end do
        end function

        subroutine autoencoder_forward(net, input, latent, output) 
            type(autoencoder), intent(inout) :: net
            real, intent(in) :: input(:,:,:)
            real, allocatable, intent(out) :: latent(:,:,:)
            real, allocatable, intent(out) :: output(:,:,:)

            real, allocatable :: layer_input(:,:,:), layer_output(:,:,:)
            integer :: i

            layer_input = input
            do i = 1, net%config%num_layers
                call conv_forward(net%encoder(i), layer_input, layer_output)
                layer_output = relu_forward(layer_output)

                layer_input = layer_output
            end do

            latent = layer_output

            do i = 1, net%config%num_layers - 1
                layer_input = upsample(layer_input, net%config%stride)
                call conv_forward(net%decoder(i), layer_input, layer_output)
                layer_output = relu_forward(layer_output)

                layer_input = layer_output
            end do

            layer_input = upsample(layer_input, net%config%stride)
            call conv_forward(net%decoder(net%config%num_layers), layer_input, output)
            output = sigmoid_forward(output)
        end subroutine

end module
