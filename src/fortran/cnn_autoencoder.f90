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

    type :: activation_cache
        real, allocatable :: activations(:, :,:,:)
    end type

    type :: gradient_cache
        real, allocatable :: gradients(:, :,:,:)
    end type

    type :: autoencoder
        type(autoencoder_config) :: config
        type(conv_layer), allocatable :: encoder(:)
        type(conv_layer), allocatable :: decoder(:)

        type(activation_cache), allocatable :: encoder_cache(:)
        type(activation_cache), allocatable :: decoder_cache(:)
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

            do i = 1, config%num_layers
                in_channels = net%encoder(config%num_layers - i + 1)%out_channels
                if (i < config%num_layers) then
                    in_channels = in_channels + net%encoder(config%num_layers - i)%out_channels
                end if
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
       end subroutine

       pure function concatenate_channels(a, b)
           real, intent(in) :: a(:, :,:,:), b(:, :,:,:)
           real, allocatable :: concatenate_channels(:, :,:,:)
           integer :: num_channels1, num_channels2

           num_channels1 = size(a, 2)
           num_channels2 = size(b, 2)

           allocate(concatenate_channels(size(a, 1), num_channels1 + num_channels2, size(a, 3), size(a, 4)))
           concatenate_channels(:, 1:num_channels1, :, :) = a
           concatenate_channels(:, num_channels1 + 1:num_channels1 + num_channels2, :, :) = b
       end function

       function dropout_channels(a, dropout)
           real, intent(in) :: a(:, :,:,:)
           real, intent(in) :: dropout
           real, allocatable :: dropout_channels(:, :,:,:)

           real :: rand
           integer :: c, num_channels
           real :: scale

           dropout_channels = a
           if (dropout == 0.0) return

           scale = 1.0 / (1.0 - dropout)

           do c = 1, size(a, 2) ! number of channels
               call random_number(rand)
               if (rand < dropout) then
                   dropout_channels(:, c, :, :) = 0.0
               else
                   dropout_channels(:, c, :, :) = dropout_channels(:, c, :, :) * scale
               end if
           end do
       end function

       subroutine autoencoder_forward(net, input, dropout, latent, output, encoder_acts_out)
           type(autoencoder), intent(inout) :: net
           real, intent(in) :: dropout
           real, intent(in) :: input(:, :,:,:)
           real, allocatable, intent(out) :: latent(:, :,:,:)
           real, allocatable, intent(out) :: output(:, :,:,:)
           type(activation_cache), allocatable, intent(out), optional :: encoder_acts_out(:)

           real, allocatable :: layer_input(:, :,:,:), layer_output(:, :,:,:), concatenated_inputs(:, :,:,:)
           type(activation_cache), allocatable :: encoder_activations(:)
           integer :: i

           allocate(encoder_activations(net%config%num_layers - 1))

           layer_input = input
           do i = 1, net%config%num_layers
               call conv_forward(net%encoder(i), layer_input, layer_output)
               if (net%encoder(i)%training) net%encoder_cache(i)%activations = layer_output

               layer_output = relu_forward(layer_output)
               if (i < net%config%num_layers) encoder_activations(i)%activations = layer_output

               layer_input = layer_output
           end do

           latent = layer_output
           if (present(encoder_acts_out)) encoder_acts_out = encoder_activations
           layer_input = layer_output

           do i = 1, net%config%num_layers - 1
               layer_input = upsample(layer_input, net%config%stride)
               concatenated_inputs = concatenate_channels(layer_input, &
                   dropout_channels(encoder_activations(net%config%num_layers - i)%activations, dropout))

               call conv_forward(net%decoder(i), concatenated_inputs, layer_output)
               if (net%decoder(i)%training) net%decoder_cache(i)%activations = layer_output
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
           type(gradient_cache), allocatable :: skip_gradients(:)
           integer :: i, num_skip_channels

           allocate(skip_gradients(net%config%num_layers - 1))

           grad_output = sigmoid_backward(output, grad_loss)
           call conv_backward(net%decoder(net%config%num_layers), grad_output, grad_input)
           do i = net%config%num_layers - 1, 1, -1
               grad_output = upsample_backward(grad_input, net%config%stride)
               grad_output = relu_backward(net%decoder_cache(i)%activations, grad_output)

               call conv_backward(net%decoder(i), grad_output, grad_input)

               num_skip_channels = net%encoder(net%config%num_layers - i)%out_channels
               skip_gradients(i)%gradients = grad_input(:, size(grad_input, 2) - num_skip_channels + 1:, :, :)
               grad_input = grad_input(:, 1:size(grad_input, 2) - num_skip_channels, :, :) 
           end do

           grad_input = upsample_backward(grad_input, net%config%stride)

           do i = net%config%num_layers, 1, -1
               if (i < net%config%num_layers) then
                   grad_input = grad_input + skip_gradients(net%config%num_layers - i)%gradients
               end if
               grad_output = relu_backward(net%encoder_cache(i)%activations, grad_input)
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

           do i = 1, net%config%num_layers
               read(unit) net%decoder(i)%weights
               read(unit) net%decoder(i)%bias
           end do

           close(unit)
       end subroutine

       subroutine decode_latent(net, latent, output)
           type(autoencoder), intent(inout) :: net
           real, intent(in) :: latent(:, :,:,:)
           real, allocatable, intent(out) :: output(:, :,:,:)

           real, allocatable :: layer_input(:, :,:,:), layer_output(:, :,:,:), padded_input(:, :,:,:)
           integer :: i, batch_size, decoder_channels, skip_channels, width, height

           layer_input = latent
           do i = 1, net%config%num_layers - 1
               layer_input = upsample(layer_input, net%config%stride)

               batch_size = size(layer_input, 1)
               decoder_channels = size(layer_input, 2)
               width = size(layer_input, 3)
               height = size(layer_input, 4)
               skip_channels = net%encoder(net%config%num_layers - i)%out_channels

               allocate(padded_input(batch_size, decoder_channels + skip_channels, width, height))
               padded_input(:, 1:decoder_channels, :, :) = layer_input
               padded_input(:, decoder_channels + 1:, :, :) = 0.0

               call conv_forward(net%decoder(i), padded_input, layer_output)
               deallocate(padded_input)

               layer_output = relu_forward(layer_output)
               layer_input = layer_output
           end do

           layer_input = upsample(layer_input, net%config%stride)
           call conv_forward(net%decoder(net%config%num_layers), layer_input, output)
           output = sigmoid_forward(output)
       end subroutine

       subroutine decode_latent_interpolated(net, latent_a, latent_b, &
               encoder_acts_a, encoder_acts_b, alpha, output)
           type(autoencoder), intent(inout) :: net
           real, intent(in) :: latent_a(:, :,:,:), latent_b(:, :,:,:)
           type(activation_cache), intent(in) :: encoder_acts_a(:), encoder_acts_b(:)
           real, intent(in) :: alpha
           real, allocatable, intent(out) :: output(:, :,:,:)

           real, allocatable :: layer_input(:, :,:,:), layer_output(:, :,:,:)
           real, allocatable :: skip_interp(:, :,:,:), concatenated_inputs(:, :,:,:)
           integer :: i, skip_idx

           layer_input = alpha * latent_a + (1.0 - alpha) * latent_b

           do i = 1, net%config%num_layers - 1
               layer_input = upsample(layer_input, net%config%stride)

               skip_idx = net%config%num_layers - i
               skip_interp = alpha * encoder_acts_a(skip_idx)%activations + &
                   (1.0 - alpha) * encoder_acts_b(skip_idx)%activations

               concatenated_inputs = concatenate_channels(layer_input, skip_interp)

               call conv_forward(net%decoder(i), concatenated_inputs, layer_output)
               layer_output = relu_forward(layer_output)
               layer_input = layer_output
           end do

           layer_input = upsample(layer_input, net%config%stride)
           call conv_forward(net%decoder(net%config%num_layers), layer_input, output)
           output = sigmoid_forward(output)
       end subroutine

   end module
