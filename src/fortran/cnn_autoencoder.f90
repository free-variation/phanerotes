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
        real :: beta                ! KL regularization term
    end type

    type :: activation_cache
        real, allocatable :: pre_relu(:, :,:,:)
    end type

    type :: autoencoder
        type(autoencoder_config) :: config
        type(conv_layer), allocatable :: encoder(:)

        type(conv_layer) :: latent_mu
        type(conv_layer) :: latent_log_var

        type(conv_layer), allocatable :: decoder(:)

        type(activation_cache), allocatable :: encoder_cache(:)
        type(activation_cache), allocatable :: decoder_cache(:)
        real, allocatable :: cached_mu(:, :,:,:)
        real, allocatable :: cached_log_var(:, :,:,:)
        real, allocatable :: cached_epsilon(:, :,:,:)
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

            layer%training = .true.
        end subroutine


        function autoencoder_init(config) result(net)
            type(autoencoder_config), intent(in) :: config
            type(autoencoder) :: net

            integer :: i, in_channels, out_channels

            net%config = config

            allocate(net%encoder(config%num_layers))
            allocate(net%decoder(config%num_layers))
            allocate(net%encoder_cache(config%num_layers))
            allocate(net%decoder_cache(config%num_layers))

            do i = 1, config%num_layers
                if (i == 1) then
                    in_channels = config%input_channels
                else
                    in_channels = min(config%base_channels * 2**(i - 1), config%max_channels)
                end if
                out_channels = min(config%base_channels * 2**i, config%max_channels)
                
                call init_layer(net%encoder(i), config, in_channels, out_channels, config%stride)
            end do

            call init_layer(net%latent_mu, config, out_channels, out_channels, 1)
            call init_layer(net%latent_log_var, config, out_channels, out_channels, 1)
            
                
            do i = 1, config%num_layers
                in_channels = net%encoder(config%num_layers - i + 1)%out_channels
                out_channels = net%encoder(config%num_layers - i + 1)%in_channels

                call init_layer(net%decoder(i), config, in_channels, out_channels, 1)
            end do
        end function

        subroutine set_training(net, training)
           type(autoencoder), intent(inout) ::net
           logical, intent(in) :: training 

           integer :: i

           do i = 1, net%config%num_layers
               net%encoder(i)%training = training
               net%decoder(i)%training = training
           end do

           net%latent_mu%training = training
           net%latent_log_var%training = training
           end subroutine

        subroutine autoencoder_forward(net, input, latent_mu, latent_log_var, output) 
            type(autoencoder), intent(inout) :: net
            real, intent(in) :: input(:, :,:,:)
            real, allocatable, intent(out) :: latent_mu(:, :,:,:), latent_log_var(:, :,:,:)
            real, allocatable, intent(out) :: output(:, :,:,:)

            real, allocatable :: layer_input(:, :,:,:), layer_output(:, :,:,:)
            integer :: i
            real, allocatable :: z(:, :,:,:), epsilon(:, :,:,:)

            layer_input = input
            do i = 1, net%config%num_layers
                call conv_forward(net%encoder(i), layer_input, layer_output)
                if (net%encoder(i)%training) net%encoder_cache(i)%pre_relu = layer_output
                layer_output = relu_forward(layer_output)

                layer_input = layer_output
            end do

            call conv_forward(net%latent_mu, layer_output, latent_mu)
            call conv_forward(net%latent_log_var, layer_output, latent_log_var)

            if (net%latent_mu%training) then
                allocate(epsilon, mold = latent_mu)
                block
                    real, allocatable :: u1(:,:,:,:), u2(:,:,:,:)
                    real, parameter :: PI = 3.14159265358979323846
                    allocate(u1, u2, mold = latent_mu)
                    call random_number(u1)
                    call random_number(u2)
                    u1 = max(u1, 1.0e-10)  ! avoid log(0)
                    epsilon = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)
                end block

                net%cached_mu = latent_mu
                net%cached_log_var = latent_log_var
                net%cached_epsilon = epsilon

                z = latent_mu + exp(0.5 * latent_log_var) * epsilon
            else
                z = latent_mu  ! No noise during inference
            end if
            layer_input = z

            do i = 1, net%config%num_layers - 1
                layer_input = upsample(layer_input, net%config%stride)
                call conv_forward(net%decoder(i), layer_input, layer_output)
                if (net%decoder(i)%training) net%decoder_cache(i)%pre_relu = layer_output
                layer_output = relu_forward(layer_output)

                layer_input = layer_output
            end do

            layer_input = upsample(layer_input, net%config%stride)
            call conv_forward(net%decoder(net%config%num_layers), layer_input, output)
            output = sigmoid_forward(output)
        end subroutine

        subroutine decode_latent(net, latent, output)
            type(autoencoder), intent(inout) :: net
            real, intent(in) :: latent(:, :,:,:)
            real, allocatable, intent(out) :: output(:, :,:,:)

            real, allocatable :: layer_input(:, :,:,:), layer_output(:, :,:,:)
            integer :: i

            layer_input = latent
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

        subroutine autoencoder_backward(net, output, grad_loss) 
            type(autoencoder), intent(inout) :: net
            real, intent(in) :: output(:, :,:,:)
            real, intent(in) :: grad_loss(:, :,:,:)
            
            real, allocatable :: grad_input(:, :,:,:), grad_output(:, :,:,:)
            real, allocatable :: grad_mu(:, :,:,:), grad_log_var(:, :,:,:)
            real, allocatable :: grad_input_mu(:, :,:,:), grad_input_log_var(:, :,:,:)
            integer :: i

            grad_output = sigmoid_backward(output, grad_loss)
            call conv_backward(net%decoder(net%config%num_layers), grad_output, grad_input)
            do i = net%config%num_layers - 1, 1, -1
                grad_output = upsample_backward(grad_input, net%config%stride)
                grad_output = relu_backward(net%decoder_cache(i)%pre_relu, grad_output)
                call conv_backward(net%decoder(i), grad_output, grad_input)
            end do

            grad_input = upsample_backward(grad_input, net%config%stride)

            grad_mu = grad_input
            grad_log_var = grad_input * net%cached_epsilon * 0.5 * exp(0.5 * net%cached_log_var)
            grad_mu = grad_mu + net%config%beta * net%cached_mu
            grad_log_var = grad_log_var + net%config%beta * 0.5 *(exp(net%cached_log_var) - 1.0)

            call conv_backward(net%latent_mu, grad_mu, grad_input_mu)
            call conv_backward(net%latent_log_var, grad_log_var, grad_input_log_var)

            grad_input = grad_input_mu + grad_input_log_var

            do i = net%config%num_layers, 1, -1
                grad_output = relu_backward(net%encoder_cache(i)%pre_relu, grad_input)
                call conv_backward(net%encoder(i), grad_output, grad_input)
            end do
        end subroutine

        subroutine save_weights(net, filename)
            type(autoencoder), intent(in) :: net
            character(*), intent(in) :: filename

            integer :: i, unit

            open(newunit=unit, file = filename, form = "unformatted", access = "stream")
            do i = 1, net%config%num_layers
                write(unit) net%encoder(i)%weights
                write(unit) net%encoder(i)%bias
            end do

            write(unit) net%latent_mu%weights
            write(unit) net%latent_mu%bias
            write(unit) net%latent_log_var%weights
            write(unit) net%latent_log_var%bias

            do i = 1, net%config%num_layers
                write(unit) net%decoder(i)%weights
                write(unit) net%decoder(i)%bias
            end do

            close(unit)
        end subroutine

        subroutine load_weights(net, filename)
            type(autoencoder), intent(inout) :: net
            character(*), intent(in) :: filename

            integer :: i, unit

            open(newunit=unit, file=filename, form="unformatted", access="stream", status="old")
            do i = 1, net%config%num_layers
                read(unit) net%encoder(i)%weights
                read(unit) net%encoder(i)%bias
            end do

            read(unit) net%latent_mu%weights
            read(unit) net%latent_mu%bias
            read(unit) net%latent_log_var%weights
            read(unit) net%latent_log_var%bias

            do i = 1, net%config%num_layers
                read(unit) net%decoder(i)%weights
                read(unit) net%decoder(i)%bias
            end do

            close(unit)
        end subroutine

end module
