package nn_odinray

// Built-in demo architectures: CNN, SimpleRNN, LSTM, GRU.
// Each builder constructs an Architecture, lays it out, and frames the camera.

import rl "vendor:raylib"

Demo_Kind :: enum {
	CNN,
	RNN,
	LSTM,
	GRU,
}

demo_name :: proc(k: Demo_Kind) -> cstring {
	switch k {
	case .CNN:  return "CNN"
	case .RNN:  return "SimpleRNN"
	case .LSTM: return "LSTM"
	case .GRU:  return "GRU"
	}
	return "?"
}

build_demo :: proc(kind: Demo_Kind) -> Architecture {
	switch kind {
	case .CNN:  return build_demo_cnn()
	case .RNN:  return build_demo_rnn()
	case .LSTM: return build_demo_lstm()
	case .GRU:  return build_demo_gru()
	}
	return build_demo_cnn()
}

// build the demo CNN: input -> conv -> pool -> conv -> pool -> flatten -> dense -> output
build_demo_cnn :: proc() -> Architecture {
	arch := Architecture{}
	arch.anim_speed = 0.6

	// Input 3x32x32
	in_id := add_layer(
		&arch,
		Layer{
			type = .Input,
			input_shape = {3, 32, 32},
			output_shape = {3, 32, 32},
			params = Input_Params{channels = 3, height = 32, width = 32},
		},
	)

	// Conv2D filters=16, k=3, stride=1, pad=1  -> 16x32x32
	c1_id := add_layer(
		&arch,
		Layer{
			type = .Conv2D,
			input_shape = {3, 32, 32},
			output_shape = {16, 32, 32},
			params = Conv_Params{filters = 16, kernel = 3, stride = 1, padding = 1},
		},
	)

	// MaxPool 2x2 stride 2 -> 16x16x16
	p1_id := add_layer(
		&arch,
		Layer{
			type = .MaxPool,
			input_shape = {16, 32, 32},
			output_shape = {16, 16, 16},
			params = Pool_Params{pool_size = 2, stride = 2},
		},
	)

	// Conv2D filters=32 -> 32x16x16
	c2_id := add_layer(
		&arch,
		Layer{
			type = .Conv2D,
			input_shape = {16, 16, 16},
			output_shape = {32, 16, 16},
			params = Conv_Params{filters = 32, kernel = 3, stride = 1, padding = 1},
		},
	)

	// MaxPool 2x2 -> 32x8x8
	p2_id := add_layer(
		&arch,
		Layer{
			type = .MaxPool,
			input_shape = {32, 16, 16},
			output_shape = {32, 8, 8},
			params = Pool_Params{pool_size = 2, stride = 2},
		},
	)

	// Flatten -> 2048
	flat_id := add_layer(
		&arch,
		Layer{
			type = .Flatten,
			input_shape = {32, 8, 8},
			output_shape = {2048, 1, 1},
		},
	)

	// Dense 64
	d1_id := add_layer(
		&arch,
		Layer{
			type = .Dense,
			input_shape = {2048, 1, 1},
			output_shape = {64, 1, 1},
			params = Dense_Params{units = 64, activation = "relu"},
		},
	)

	// Output 10
	out_id := add_layer(
		&arch,
		Layer{
			type = .Output,
			input_shape = {64, 1, 1},
			output_shape = {10, 1, 1},
			params = Output_Params{classes = 10},
		},
	)

	// implicit sequential wiring -> explicit Connection entries
	ids := []int{in_id, c1_id, p1_id, c2_id, p2_id, flat_id, d1_id, out_id}
	for i in 0 ..< len(ids) - 1 {
		append(&arch.connections, Connection{from_id = ids[i], to_id = ids[i + 1], weight = 1.0})
	}

	auto_layout(&arch)

	arch.camera = rl.Camera2D{
		offset = rl.Vector2{640, 360},
		target = network_center(&arch),
		rotation = 0,
		zoom = 1.0,
	}
	return arch
}

build_demo_rnn :: proc() -> Architecture {
	arch := Architecture{}
	arch.anim_speed = 0.6
	in_id := add_layer(&arch, Layer{
		type = .Input,
		input_shape = {16, 1, 1},
		output_shape = {16, 1, 1},
		params = Input_Params{channels = 16, height = 1, width = 1},
	})
	r1_id := add_layer(&arch, Layer{
		type = .SimpleRNN,
		input_shape = {16, 1, 1},
		output_shape = {64, 1, 1},
		params = RNN_Params{units = 64, activation = "tanh", return_sequences = true},
	})
	r2_id := add_layer(&arch, Layer{
		type = .SimpleRNN,
		input_shape = {64, 1, 1},
		output_shape = {32, 1, 1},
		params = RNN_Params{units = 32, activation = "tanh", return_sequences = false},
	})
	d_id := add_layer(&arch, Layer{
		type = .Dense,
		input_shape = {32, 1, 1},
		output_shape = {16, 1, 1},
		params = Dense_Params{units = 16, activation = "relu"},
	})
	o_id := add_layer(&arch, Layer{
		type = .Output,
		input_shape = {16, 1, 1},
		output_shape = {3, 1, 1},
		params = Output_Params{classes = 3},
	})
	ids := []int{in_id, r1_id, r2_id, d_id, o_id}
	for i in 0 ..< len(ids) - 1 {
		append(&arch.connections, Connection{from_id = ids[i], to_id = ids[i + 1], weight = 1.0})
	}
	auto_layout(&arch)
	arch.camera = rl.Camera2D{offset = rl.Vector2{640, 360}, target = network_center(&arch), zoom = 1.0}
	return arch
}

build_demo_lstm :: proc() -> Architecture {
	arch := Architecture{}
	arch.anim_speed = 0.6
	in_id := add_layer(&arch, Layer{
		type = .Input,
		input_shape = {32, 1, 1},
		output_shape = {32, 1, 1},
		params = Input_Params{channels = 32, height = 1, width = 1},
	})
	l1_id := add_layer(&arch, Layer{
		type = .LSTM,
		input_shape = {32, 1, 1},
		output_shape = {128, 1, 1},
		params = LSTM_Params{units = 128, activation = "tanh", recurrent_activation = "sigmoid"},
	})
	l2_id := add_layer(&arch, Layer{
		type = .LSTM,
		input_shape = {128, 1, 1},
		output_shape = {64, 1, 1},
		params = LSTM_Params{units = 64, activation = "tanh", recurrent_activation = "sigmoid"},
	})
	d_id := add_layer(&arch, Layer{
		type = .Dense,
		input_shape = {64, 1, 1},
		output_shape = {32, 1, 1},
		params = Dense_Params{units = 32, activation = "relu"},
	})
	o_id := add_layer(&arch, Layer{
		type = .Output,
		input_shape = {32, 1, 1},
		output_shape = {10, 1, 1},
		params = Output_Params{classes = 10},
	})
	ids := []int{in_id, l1_id, l2_id, d_id, o_id}
	for i in 0 ..< len(ids) - 1 {
		append(&arch.connections, Connection{from_id = ids[i], to_id = ids[i + 1], weight = 1.0})
	}
	auto_layout(&arch)
	arch.camera = rl.Camera2D{offset = rl.Vector2{640, 360}, target = network_center(&arch), zoom = 1.0}
	return arch
}

build_demo_gru :: proc() -> Architecture {
	arch := Architecture{}
	arch.anim_speed = 0.6
	in_id := add_layer(&arch, Layer{
		type = .Input,
		input_shape = {64, 1, 1},
		output_shape = {64, 1, 1},
		params = Input_Params{channels = 64, height = 1, width = 1},
	})
	g1_id := add_layer(&arch, Layer{
		type = .GRU,
		input_shape = {64, 1, 1},
		output_shape = {96, 1, 1},
		params = GRU_Params{units = 96, activation = "tanh", recurrent_activation = "sigmoid"},
	})
	g2_id := add_layer(&arch, Layer{
		type = .GRU,
		input_shape = {96, 1, 1},
		output_shape = {48, 1, 1},
		params = GRU_Params{units = 48, activation = "tanh", recurrent_activation = "sigmoid"},
	})
	d_id := add_layer(&arch, Layer{
		type = .Dense,
		input_shape = {48, 1, 1},
		output_shape = {24, 1, 1},
		params = Dense_Params{units = 24, activation = "relu"},
	})
	o_id := add_layer(&arch, Layer{
		type = .Output,
		input_shape = {24, 1, 1},
		output_shape = {5, 1, 1},
		params = Output_Params{classes = 5},
	})
	ids := []int{in_id, g1_id, g2_id, d_id, o_id}
	for i in 0 ..< len(ids) - 1 {
		append(&arch.connections, Connection{from_id = ids[i], to_id = ids[i + 1], weight = 1.0})
	}
	auto_layout(&arch)
	arch.camera = rl.Camera2D{offset = rl.Vector2{640, 360}, target = network_center(&arch), zoom = 1.0}
	return arch
}
