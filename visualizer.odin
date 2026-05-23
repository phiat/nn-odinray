package pp1

import "core:encoding/json"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
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

Layer_Type :: enum {
	Input,
	Conv2D,
	MaxPool,
	AveragePool,
	Flatten,
	Dense,
	SimpleRNN,
	LSTM,
	GRU,
	Output,
}

Conv_Params :: struct {
	filters: int,
	kernel:  int,
	stride:  int,
	padding: int,
}

Pool_Params :: struct {
	pool_size: int,
	stride:    int,
}

Dense_Params :: struct {
	units:      int,
	activation: string,
}

RNN_Params :: struct {
	units:            int,
	activation:       string,
	return_sequences: bool,
}

LSTM_Params :: struct {
	units:                int,
	activation:           string,
	recurrent_activation: string,
}

GRU_Params :: struct {
	units:                int,
	activation:           string,
	recurrent_activation: string,
}

Input_Params :: struct {
	channels: int,
	height:   int,
	width:    int,
}

Output_Params :: struct {
	classes: int,
}

Layer_Params :: union {
	Conv_Params,
	Pool_Params,
	Dense_Params,
	RNN_Params,
	LSTM_Params,
	GRU_Params,
	Input_Params,
	Output_Params,
}

Layer :: struct {
	id:           int,
	type:         Layer_Type,
	position:     rl.Vector2, // top-left corner in world space
	size:         rl.Vector2,
	input_shape:  [3]int, // channels, height, width
	output_shape: [3]int,
	params:       Layer_Params,
	color:        rl.Color,
}

Connection :: struct {
	from_id: int,
	to_id:   int,
	weight:  f32,
}

Pulse :: struct {
	conn_idx: int,
	t:        f32, // 0..1 along connection
	color:    rl.Color,
}

Architecture :: struct {
	layers:            [dynamic]Layer,
	connections:       [dynamic]Connection,
	camera:            rl.Camera2D,
	anim_speed:        f32,
	pulses:            [dynamic]Pulse,
	pulse_spawn_timer: f32,
}

// ----- construction ------------------------------------------------------

destroy_architecture :: proc(arch: ^Architecture) {
	delete(arch.layers)
	delete(arch.connections)
	delete(arch.pulses)
}

add_layer :: proc(arch: ^Architecture, l: Layer) -> int {
	id := 0
	for ll in arch.layers do if ll.id >= id do id = ll.id + 1
	layer := l
	layer.id = id
	layer.color = layer_color(layer.type)
	append(&arch.layers, layer)
	return id
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

build_demo :: proc(kind: Demo_Kind) -> Architecture {
	switch kind {
	case .CNN:  return build_demo_cnn()
	case .RNN:  return build_demo_rnn()
	case .LSTM: return build_demo_lstm()
	case .GRU:  return build_demo_gru()
	}
	return build_demo_cnn()
}

is_recurrent :: proc(t: Layer_Type) -> bool {
	return t == .SimpleRNN || t == .LSTM || t == .GRU
}

// ----- shape propagation & mutation -------------------------------------

ACTIVATIONS := []string{"relu", "tanh", "sigmoid", "linear", "softmax"}
RECURRENT_ACTIVATIONS := []string{"sigmoid", "tanh", "relu"}

cycle_string :: proc(current: string, choices: []string, dir: int) -> string {
	idx := 0
	for c, i in choices {
		if c == current {
			idx = i
			break
		}
	}
	n := len(choices)
	idx = (idx + dir + n) % n
	return choices[idx]
}

derive_output_shape :: proc(l: ^Layer) {
	switch p in l.params {
	case Input_Params:
		l.output_shape = [3]int{max(p.channels, 1), max(p.height, 1), max(p.width, 1)}
	case Conv_Params:
		k := max(p.kernel, 1)
		st := max(p.stride, 1)
		pd := p.padding
		out_h := (l.input_shape[1] - k + 2 * pd) / st + 1
		out_w := (l.input_shape[2] - k + 2 * pd) / st + 1
		if out_h < 1 do out_h = 1
		if out_w < 1 do out_w = 1
		l.output_shape = [3]int{max(p.filters, 1), out_h, out_w}
	case Pool_Params:
		ps := max(p.pool_size, 1)
		st := max(p.stride, ps)
		out_h := l.input_shape[1] / st
		out_w := l.input_shape[2] / st
		if out_h < 1 do out_h = 1
		if out_w < 1 do out_w = 1
		l.output_shape = [3]int{l.input_shape[0], out_h, out_w}
	case Dense_Params:
		l.output_shape = [3]int{max(p.units, 1), 1, 1}
	case RNN_Params:
		l.output_shape = [3]int{max(p.units, 1), 1, 1}
	case LSTM_Params:
		l.output_shape = [3]int{max(p.units, 1), 1, 1}
	case GRU_Params:
		l.output_shape = [3]int{max(p.units, 1), 1, 1}
	case Output_Params:
		l.output_shape = [3]int{max(p.classes, 1), 1, 1}
	case:
		if l.type == .Flatten {
			total := l.input_shape[0] * l.input_shape[1] * l.input_shape[2]
			if total < 1 do total = 1
			l.output_shape = [3]int{total, 1, 1}
		}
	}
}

recompute_shapes :: proc(arch: ^Architecture) {
	n := len(arch.layers)
	if n == 0 do return

	in_count := make([]int, n, context.temp_allocator)
	id_to_idx := make(map[int]int, n, context.temp_allocator)
	for l, i in arch.layers do id_to_idx[l.id] = i
	for c in arch.connections {
		if idx, ok := id_to_idx[c.to_id]; ok do in_count[idx] += 1
	}

	queue := make([dynamic]int, 0, n, context.temp_allocator)
	for i in 0 ..< n do if in_count[i] == 0 do append(&queue, i)

	q_head := 0
	for q_head < len(queue) {
		idx := queue[q_head]
		q_head += 1
		derive_output_shape(&arch.layers[idx])
		from_id := arch.layers[idx].id
		for c in arch.connections {
			if c.from_id == from_id {
				if cidx, ok := id_to_idx[c.to_id]; ok {
					arch.layers[cidx].input_shape = arch.layers[idx].output_shape
					in_count[cidx] -= 1
					if in_count[cidx] == 0 do append(&queue, cidx)
				}
			}
		}
	}

	for i in 0 ..< n do compute_layer_size(&arch.layers[i])
}

delete_layer_at :: proc(arch: ^Architecture, idx: int) {
	if idx < 0 || idx >= len(arch.layers) do return
	deleted_id := arch.layers[idx].id

	incoming := make([dynamic]int, 0, 4, context.temp_allocator)
	outgoing := make([dynamic]int, 0, 4, context.temp_allocator)
	for c in arch.connections {
		if c.to_id == deleted_id do append(&incoming, c.from_id)
		if c.from_id == deleted_id do append(&outgoing, c.to_id)
	}

	new_conns := make([dynamic]Connection, 0, len(arch.connections))
	for c in arch.connections {
		if c.from_id != deleted_id && c.to_id != deleted_id {
			append(&new_conns, c)
		}
	}
	for from in incoming {
		for to in outgoing {
			exists := false
			for c in new_conns {
				if c.from_id == from && c.to_id == to {
					exists = true
					break
				}
			}
			if !exists do append(&new_conns, Connection{from_id = from, to_id = to, weight = 1.0})
		}
	}
	delete(arch.connections)
	arch.connections = new_conns

	ordered_remove(&arch.layers, idx)
	recompute_shapes(arch)
}

new_layer_of_type :: proc(t: Layer_Type, input_shape: [3]int) -> Layer {
	l := Layer{type = t, input_shape = input_shape}
	switch t {
	case .Conv2D:
		l.params = Conv_Params{filters = 16, kernel = 3, stride = 1, padding = 1}
	case .MaxPool, .AveragePool:
		l.params = Pool_Params{pool_size = 2, stride = 2}
	case .Flatten:
	// no params
	case .Dense:
		l.params = Dense_Params{units = 32, activation = "relu"}
	case .SimpleRNN:
		l.params = RNN_Params{units = 32, activation = "tanh", return_sequences = false}
	case .LSTM:
		l.params = LSTM_Params{units = 32, activation = "tanh", recurrent_activation = "sigmoid"}
	case .GRU:
		l.params = GRU_Params{units = 32, activation = "tanh", recurrent_activation = "sigmoid"}
	case .Input:
		l.params = Input_Params{channels = max(input_shape[0], 1), height = max(input_shape[1], 1), width = max(input_shape[2], 1)}
	case .Output:
		l.params = Output_Params{classes = 10}
	}
	derive_output_shape(&l)
	return l
}

// add a layer without wiring it up; positions it at the camera target
add_layer_standalone :: proc(arch: ^Architecture, t: Layer_Type) -> int {
	input_shape := [3]int{1, 1, 1}
	if t == .Input do input_shape = [3]int{3, 32, 32}
	new_layer := new_layer_of_type(t, input_shape)
	compute_layer_size(&new_layer)
	new_layer.position = rl.Vector2{
		arch.camera.target[0] - new_layer.size[0] / 2,
		arch.camera.target[1] - new_layer.size[1] / 2,
	}
	return add_layer(arch, new_layer)
}

// returns the id of the last layer in topo order (the "tail" of the network), or -1 if none.
tail_layer_id :: proc(arch: ^Architecture) -> int {
	if len(arch.layers) == 0 do return -1
	has_out := make(map[int]bool, len(arch.layers), context.temp_allocator)
	for c in arch.connections do has_out[c.from_id] = true
	for i := len(arch.layers) - 1; i >= 0; i -= 1 {
		if !has_out[arch.layers[i].id] do return arch.layers[i].id
	}
	return arch.layers[len(arch.layers) - 1].id
}

add_layer_after :: proc(arch: ^Architecture, after_idx: int, t: Layer_Type) -> int {
	if after_idx < 0 || after_idx >= len(arch.layers) do return -1
	src_id := arch.layers[after_idx].id
	src_size := arch.layers[after_idx].size
	src_pos := arch.layers[after_idx].position
	src_out := arch.layers[after_idx].output_shape

	new_layer := new_layer_of_type(t, src_out)
	new_layer.position = rl.Vector2{src_pos[0] + src_size[0] + 90, src_pos[1]}
	compute_layer_size(&new_layer)

	// shift downstream layers right to make room
	insert_x := src_pos[0] + src_size[0] + 45
	push := new_layer.size[0] + 90
	for i in 0 ..< len(arch.layers) {
		if arch.layers[i].id == src_id do continue
		if arch.layers[i].position[0] >= insert_x {
			arch.layers[i].position[0] += push
		}
	}

	new_id := add_layer(arch, new_layer)

	outgoing := make([dynamic]int, 0, 4, context.temp_allocator)
	for c in arch.connections {
		if c.from_id == src_id do append(&outgoing, c.to_id)
	}

	new_conns := make([dynamic]Connection, 0, len(arch.connections) + len(outgoing) + 1)
	for c in arch.connections {
		if c.from_id != src_id do append(&new_conns, c)
	}
	append(&new_conns, Connection{from_id = src_id, to_id = new_id, weight = 1.0})
	for to in outgoing {
		append(&new_conns, Connection{from_id = new_id, to_id = to, weight = 1.0})
	}
	delete(arch.connections)
	arch.connections = new_conns

	recompute_shapes(arch)
	return new_id
}

ADDABLE_TYPES := [?]Layer_Type{
	.Conv2D, .MaxPool, .AveragePool, .Flatten,
	.Dense, .SimpleRNN, .LSTM, .GRU, .Output,
}

point_segment_distance :: proc(p, a, b: rl.Vector2) -> f32 {
	abx := b[0] - a[0]
	aby := b[1] - a[1]
	d2 := abx * abx + aby * aby
	if d2 < 1e-6 {
		dx := p[0] - a[0]
		dy := p[1] - a[1]
		return math.sqrt_f32(dx * dx + dy * dy)
	}
	t := ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / d2
	if t < 0 do t = 0
	if t > 1 do t = 1
	cx := a[0] + t * abx
	cy := a[1] + t * aby
	dx := p[0] - cx
	dy := p[1] - cy
	return math.sqrt_f32(dx * dx + dy * dy)
}

find_connection_at :: proc(arch: ^Architecture, world: rl.Vector2, threshold_world: f32) -> int {
	for c, i in arch.connections {
		from := layer_by_id(arch, c.from_id)
		to := layer_by_id(arch, c.to_id)
		if from == nil || to == nil do continue
		a := layer_right_anchor(from)
		b := layer_left_anchor(to)
		if point_segment_distance(world, a, b) <= threshold_world do return i
	}
	return -1
}

find_output_handle :: proc(arch: ^Architecture, world: rl.Vector2, radius_world: f32) -> int {
	for i in 0 ..< len(arch.layers) {
		anchor := layer_right_anchor(&arch.layers[i])
		dx := world[0] - anchor[0]
		dy := world[1] - anchor[1]
		if dx * dx + dy * dy <= radius_world * radius_world do return i
	}
	return -1
}

add_connection :: proc(arch: ^Architecture, from_id, to_id: int) -> bool {
	if from_id == to_id do return false
	if layer_by_id(arch, from_id) == nil || layer_by_id(arch, to_id) == nil do return false
	for c in arch.connections {
		if c.from_id == from_id && c.to_id == to_id do return false
	}
	append(&arch.connections, Connection{from_id = from_id, to_id = to_id, weight = 1.0})
	recompute_shapes(arch)
	return true
}

// ----- undo / redo history -----------------------------------------------

History :: struct {
	undo:      [dynamic]Architecture,
	redo:      [dynamic]Architecture,
	max_depth: int,
}

destroy_history :: proc(h: ^History) {
	for &s in h.undo do destroy_architecture(&s)
	for &s in h.redo do destroy_architecture(&s)
	delete(h.undo)
	delete(h.redo)
}

clear_history :: proc(h: ^History) {
	for &s in h.undo do destroy_architecture(&s)
	for &s in h.redo do destroy_architecture(&s)
	clear(&h.undo)
	clear(&h.redo)
}

clone_architecture :: proc(src: ^Architecture) -> Architecture {
	dst := Architecture{}
	dst.anim_speed = src.anim_speed
	dst.camera = src.camera
	dst.layers = make([dynamic]Layer, 0, len(src.layers))
	for l in src.layers do append(&dst.layers, l)
	dst.connections = make([dynamic]Connection, 0, len(src.connections))
	for c in src.connections do append(&dst.connections, c)
	dst.pulses = make([dynamic]Pulse, 0)
	return dst
}

push_undo :: proc(h: ^History, arch: ^Architecture) {
	if h.max_depth == 0 do h.max_depth = 64
	snap := clone_architecture(arch)
	append(&h.undo, snap)
	for len(h.undo) > h.max_depth {
		destroy_architecture(&h.undo[0])
		ordered_remove(&h.undo, 0)
	}
	// any prior redo states are invalidated by a fresh action
	for &s in h.redo do destroy_architecture(&s)
	clear(&h.redo)
}

apply_snapshot :: proc(arch: ^Architecture, snap: Architecture) {
	// preserve camera (less jarring than yanking the view)
	cam := arch.camera
	destroy_architecture(arch)
	arch.layers = snap.layers
	arch.connections = snap.connections
	arch.anim_speed = snap.anim_speed
	arch.pulses = make([dynamic]Pulse, 0)
	arch.camera = cam
}

undo :: proc(h: ^History, arch: ^Architecture) -> bool {
	if len(h.undo) == 0 do return false
	append(&h.redo, clone_architecture(arch))
	snap := pop(&h.undo)
	apply_snapshot(arch, snap)
	return true
}

redo :: proc(h: ^History, arch: ^Architecture) -> bool {
	if len(h.redo) == 0 do return false
	append(&h.undo, clone_architecture(arch))
	snap := pop(&h.redo)
	apply_snapshot(arch, snap)
	return true
}

// ----- layout & sizing ---------------------------------------------------

compute_layer_size :: proc(l: ^Layer) {
	switch l.type {
	case .Input, .Conv2D, .MaxPool, .AveragePool:
		spatial := f32(max(l.output_shape[1], l.output_shape[2]))
		side := math.clamp(spatial * 4.0, 60.0, 180.0)
		channels := f32(l.output_shape[0])
		extra := math.clamp(math.log10_f32(channels + 1.0) * 16.0, 0.0, 40.0)
		l.size = rl.Vector2{side + 30, side + 20 + extra}
	case .Flatten, .Dense, .Output, .SimpleRNN, .LSTM, .GRU:
		units := f32(l.output_shape[0])
		h := math.clamp(math.sqrt_f32(units) * 8.0, 64.0, 220.0)
		l.size = rl.Vector2{120, h}
	}
}

auto_layout :: proc(arch: ^Architecture) {
	n := len(arch.layers)
	if n == 0 do return

	in_count := make([]int, n, context.temp_allocator)
	id_to_idx := make(map[int]int, n, context.temp_allocator)
	for l, i in arch.layers do id_to_idx[l.id] = i
	for c in arch.connections {
		if idx, ok := id_to_idx[c.to_id]; ok do in_count[idx] += 1
	}

	order := make([dynamic]int, 0, n, context.temp_allocator)
	for i in 0 ..< n do if in_count[i] == 0 do append(&order, i)
	head := 0
	for head < len(order) {
		idx := order[head]
		head += 1
		from_id := arch.layers[idx].id
		for c in arch.connections {
			if c.from_id == from_id {
				if cidx, ok := id_to_idx[c.to_id]; ok {
					in_count[cidx] -= 1
					if in_count[cidx] == 0 do append(&order, cidx)
				}
			}
		}
	}
	// any disconnected layers fall through at the end
	for i in 0 ..< n {
		found := false
		for k in order do if k == i {found = true; break}
		if !found do append(&order, i)
	}

	x: f32 = 80
	center_y: f32 = 360
	for idx in order {
		l := &arch.layers[idx]
		compute_layer_size(l)
		l.position = rl.Vector2{x, center_y - l.size.y / 2}
		x += l.size.x + 90
	}
}

network_center :: proc(arch: ^Architecture) -> rl.Vector2 {
	if len(arch.layers) == 0 do return rl.Vector2{0, 0}
	min_x, min_y := arch.layers[0].position.x, arch.layers[0].position.y
	max_x, max_y := min_x, min_y
	for &l in arch.layers {
		min_x = min(min_x, l.position.x)
		min_y = min(min_y, l.position.y)
		max_x = max(max_x, l.position.x + l.size.x)
		max_y = max(max_y, l.position.y + l.size.y)
	}
	return rl.Vector2{(min_x + max_x) / 2, (min_y + max_y) / 2}
}

network_bounds :: proc(arch: ^Architecture) -> rl.Vector2 {
	if len(arch.layers) == 0 do return rl.Vector2{0, 0}
	min_x, min_y := arch.layers[0].position.x, arch.layers[0].position.y
	max_x, max_y := min_x, min_y
	for &l in arch.layers {
		min_x = min(min_x, l.position.x)
		min_y = min(min_y, l.position.y)
		max_x = max(max_x, l.position.x + l.size.x)
		max_y = max(max_y, l.position.y + l.size.y)
	}
	return rl.Vector2{max_x - min_x, max_y - min_y}
}

fit_camera :: proc(arch: ^Architecture, screen_w, screen_h: f32, margin: f32 = 80) {
	size := network_bounds(arch)
	if size[0] <= 0 || size[1] <= 0 {
		arch.camera.zoom = 1.0
		arch.camera.target = network_center(arch)
		return
	}
	zx := (screen_w - margin) / size[0]
	zy := (screen_h - margin) / size[1]
	z := min(zx, zy)
	if z > 2.0 do z = 2.0
	if z < 0.2 do z = 0.2
	arch.camera.zoom = z
	arch.camera.target = network_center(arch)
}

layer_rect :: proc(l: ^Layer) -> rl.Rectangle {
	return rl.Rectangle{l.position.x, l.position.y, l.size.x, l.size.y}
}

layer_left_anchor :: proc(l: ^Layer) -> rl.Vector2 {
	return rl.Vector2{l.position.x, l.position.y + l.size.y / 2}
}

layer_right_anchor :: proc(l: ^Layer) -> rl.Vector2 {
	return rl.Vector2{l.position.x + l.size.x, l.position.y + l.size.y / 2}
}

layer_by_id :: proc(arch: ^Architecture, id: int) -> ^Layer {
	for i in 0 ..< len(arch.layers) {
		if arch.layers[i].id == id do return &arch.layers[i]
	}
	return nil
}

// ----- visuals -----------------------------------------------------------

layer_label :: proc(t: Layer_Type) -> cstring {
	switch t {
	case .Input:        return "Input"
	case .Conv2D:       return "Conv2D"
	case .MaxPool:      return "MaxPool"
	case .AveragePool:  return "AvgPool"
	case .Flatten:      return "Flatten"
	case .Dense:        return "Dense"
	case .SimpleRNN:    return "SimpleRNN"
	case .LSTM:         return "LSTM"
	case .GRU:          return "GRU"
	case .Output:       return "Output"
	}
	return "Layer"
}

layer_type_name :: proc(t: Layer_Type) -> string {
	switch t {
	case .Input:        return "input"
	case .Conv2D:       return "conv2d"
	case .MaxPool:      return "maxpool"
	case .AveragePool:  return "avgpool"
	case .Flatten:      return "flatten"
	case .Dense:        return "dense"
	case .SimpleRNN:    return "rnn"
	case .LSTM:         return "lstm"
	case .GRU:          return "gru"
	case .Output:       return "output"
	}
	return "unknown"
}

layer_type_from_name :: proc(s: string) -> (Layer_Type, bool) {
	switch s {
	case "input":   return .Input, true
	case "conv2d":  return .Conv2D, true
	case "maxpool": return .MaxPool, true
	case "avgpool": return .AveragePool, true
	case "flatten": return .Flatten, true
	case "dense":   return .Dense, true
	case "rnn":     return .SimpleRNN, true
	case "lstm":    return .LSTM, true
	case "gru":     return .GRU, true
	case "output":  return .Output, true
	}
	return .Input, false
}

layer_color :: proc(t: Layer_Type) -> rl.Color {
	switch t {
	case .Input:                                  return rl.Color{ 80, 160, 240, 255}
	case .Conv2D:                                 return rl.Color{ 90, 200, 130, 255}
	case .MaxPool:                                return rl.Color{170, 110, 220, 255}
	case .AveragePool:                            return rl.Color{150, 100, 200, 255}
	case .Flatten:                                return rl.Color{180, 180, 190, 255}
	case .Dense:                                  return rl.Color{240, 160,  70, 255}
	case .SimpleRNN, .LSTM, .GRU:                 return rl.Color{220, 100, 150, 255}
	case .Output:                                 return rl.Color{230,  80,  95, 255}
	}
	return rl.WHITE
}

darken :: proc(c: rl.Color, factor: f32) -> rl.Color {
	return rl.Color{
		u8(f32(c.r) * factor),
		u8(f32(c.g) * factor),
		u8(f32(c.b) * factor),
		c.a,
	}
}

shape_label :: proc(shape: [3]int) -> cstring {
	if shape[1] == 1 && shape[2] == 1 {
		return fmt.ctprintf("%d", shape[0])
	}
	return fmt.ctprintf("%dx%dx%d", shape[0], shape[1], shape[2])
}

params_label :: proc(l: ^Layer) -> cstring {
	switch p in l.params {
	case Conv_Params:   return fmt.ctprintf("k=%d s=%d f=%d", p.kernel, p.stride, p.filters)
	case Pool_Params:   return fmt.ctprintf("p=%d s=%d", p.pool_size, p.stride)
	case Dense_Params:  return fmt.ctprintf("u=%d %s", p.units, p.activation)
	case RNN_Params:    return fmt.ctprintf("u=%d %s", p.units, p.activation)
	case LSTM_Params:   return fmt.ctprintf("u=%d", p.units)
	case GRU_Params:    return fmt.ctprintf("u=%d", p.units)
	case Input_Params:  return fmt.ctprintf("%dx%dx%d", p.channels, p.height, p.width)
	case Output_Params: return fmt.ctprintf("classes=%d", p.classes)
	}
	return ""
}

// ----- drawing -----------------------------------------------------------

draw_connections :: proc(arch: ^Architecture, selected_conn: int = -1, hover_conn: int = -1) {
	for c, i in arch.connections {
		from := layer_by_id(arch, c.from_id)
		to := layer_by_id(arch, c.to_id)
		if from == nil || to == nil do continue
		a := layer_right_anchor(from)
		b := layer_left_anchor(to)

		col: rl.Color
		thick: f32 = 2.0
		switch {
		case i == selected_conn:
			col = rl.Color{255, 230, 100, 240}
			thick = 3.0
		case i == hover_conn:
			col = rl.Color{220, 230, 255, 230}
			thick = 2.6
		case:
			col = rl.Color{180, 195, 220, 200}
		}
		rl.DrawLineEx(a, b, thick, col)
		draw_arrowhead(a, b, col)
	}
}

draw_output_handles :: proc(arch: ^Architecture, highlight_idx: int) {
	for i in 0 ..< len(arch.layers) {
		anchor := layer_right_anchor(&arch.layers[i])
		hot := i == highlight_idx
		ring := hot ? rl.Color{255, 230, 120, 255} : rl.Color{120, 160, 200, 180}
		fill := hot ? rl.Color{255, 250, 200, 255} : rl.Color{40, 60, 90, 220}
		rl.DrawCircleV(anchor, hot ? 7 : 5, ring)
		rl.DrawCircleV(anchor, hot ? 4 : 3, fill)
	}
}

draw_in_flight_connection :: proc(arch: ^Architecture, from_id: int, mouse_world: rl.Vector2) {
	src := layer_by_id(arch, from_id)
	if src == nil do return
	a := layer_right_anchor(src)
	col := rl.Color{220, 240, 255, 220}
	rl.DrawLineEx(a, mouse_world, 2.5, col)
	rl.DrawCircleV(mouse_world, 6, rl.Color{255, 240, 180, 200})
	rl.DrawCircleV(mouse_world, 3, rl.Color{255, 255, 255, 255})
}

draw_self_loops :: proc(arch: ^Architecture) {
	for i in 0 ..< len(arch.layers) {
		l := &arch.layers[i]
		if !is_recurrent(l.type) do continue
		draw_self_loop(l)
	}
}

draw_self_loop :: proc(l: ^Layer) {
	// arc from top-right of layer, sweeping up and over, ending at top-mid pointing down
	top_right := rl.Vector2{l.position.x + l.size.x * 0.9, l.position.y + 4}
	top_mid := rl.Vector2{l.position.x + l.size.x * 0.4, l.position.y + 4}
	bulge := f32(40)
	c1 := rl.Vector2{top_right[0] + bulge, top_right[1] - bulge * 1.4}
	c2 := rl.Vector2{top_mid[0] + bulge * 0.2, top_mid[1] - bulge * 1.4}

	col := rl.Color{200, 160, 240, 220}
	segs := 24
	prev := top_right
	for i in 1 ..= segs {
		t := f32(i) / f32(segs)
		u := 1 - t
		p := rl.Vector2{
			u * u * u * top_right[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t * t * t * top_mid[0],
			u * u * u * top_right[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t * t * t * top_mid[1],
		}
		rl.DrawLineEx(prev, p, 2.0, col)
		prev = p
	}
	// arrow head pointing down into the layer at top_mid
	rl.DrawTriangle(
		rl.Vector2{top_mid[0], top_mid[1] + 6},
		rl.Vector2{top_mid[0] - 5, top_mid[1] - 4},
		rl.Vector2{top_mid[0] + 5, top_mid[1] - 4},
		col,
	)
	// small "t-1" tag
	rl.DrawText("t-1", i32(c1[0] - 8), i32(c1[1] - 18), 11, rl.Color{220, 200, 250, 220})
}

draw_arrowhead :: proc(a, b: rl.Vector2, col: rl.Color) {
	dir := rl.Vector2{b[0] - a[0], b[1] - a[1]}
	len_ := math.sqrt_f32(dir[0] * dir[0] + dir[1] * dir[1])
	if len_ < 0.001 do return
	dir = rl.Vector2{dir[0] / len_, dir[1] / len_}
	perp := rl.Vector2{-dir[1], dir[0]}
	tip := b
	base := rl.Vector2{b[0] - dir[0] * 10, b[1] - dir[1] * 10}
	left := rl.Vector2{base[0] + perp[0] * 5, base[1] + perp[1] * 5}
	right := rl.Vector2{base[0] - perp[0] * 5, base[1] - perp[1] * 5}
	rl.DrawTriangle(tip, left, right, col)
}

draw_layers :: proc(arch: ^Architecture, hovered: int) {
	for i in 0 ..< len(arch.layers) {
		l := &arch.layers[i]
		rect := layer_rect(l)

		// multi-channel: draw offset shadow "stack" behind the main card
		ch := l.output_shape[0]
		if ch > 1 && (l.type == .Input || l.type == .Conv2D || l.type == .MaxPool || l.type == .AveragePool) {
			n := int(math.log10_f32(f32(ch))) + 1
			if n > 4 do n = 4
			if n < 1 do n = 1
			shadow := darken(l.color, 0.22)
			shadow.a = 220
			outline := darken(l.color, 0.55)
			for k := n; k >= 1; k -= 1 {
				off := f32(k) * 4
				r := rl.Rectangle{rect.x + off, rect.y - off, rect.width, rect.height}
				rl.DrawRectangleRounded(r, 0.18, 8, shadow)
				rl.DrawRectangleRoundedLinesEx(r, 0.18, 8, 1.0, outline)
			}
		}

		bg := darken(l.color, 0.30)
		bg.a = 235
		rl.DrawRectangleRounded(rect, 0.18, 10, bg)

		thick: f32 = i == hovered ? 3.5 : 2.0
		border_col := l.color
		if i == hovered {
			border_col = rl.Color{255, 255, 255, 255}
		}
		rl.DrawRectangleRoundedLinesEx(rect, 0.18, 10, thick, border_col)

		// header strip on top
		header := rl.Rectangle{rect.x, rect.y, rect.width, 22}
		rl.DrawRectangleRec(header, l.color)

		// name text
		name := layer_label(l.type)
		name_w := rl.MeasureText(name, 14)
		rl.DrawText(
			name,
			i32(rect.x + (rect.width - f32(name_w)) / 2),
			i32(rect.y + 4),
			14,
			rl.Color{15, 18, 25, 255},
		)

		// output shape (centered)
		out_str := shape_label(l.output_shape)
		out_w := rl.MeasureText(out_str, 16)
		rl.DrawText(
			out_str,
			i32(rect.x + (rect.width - f32(out_w)) / 2),
			i32(rect.y + rect.height / 2 - 8),
			16,
			rl.Color{235, 240, 245, 255},
		)

		// params footer
		params := params_label(l)
		if params != "" {
			pw := rl.MeasureText(params, 11)
			rl.DrawText(
				params,
				i32(rect.x + (rect.width - f32(pw)) / 2),
				i32(rect.y + rect.height - 16),
				11,
				rl.Color{200, 210, 220, 220},
			)
		}
	}
}

// ----- animation ---------------------------------------------------------

PULSE_SPAWN_INTERVAL :: 0.35

update_animation :: proc(arch: ^Architecture, dt: f32) {
	arch.pulse_spawn_timer += dt
	if arch.pulse_spawn_timer >= PULSE_SPAWN_INTERVAL && len(arch.connections) > 0 {
		arch.pulse_spawn_timer = 0
		// spawn a wave on the first connection; the first connection feeds the next layer,
		// pulses propagate by spawning when reaching the destination.
		hue_t := rand.float32()
		col := pulse_color(hue_t)
		append(&arch.pulses, Pulse{conn_idx = 0, t = 0, color = col})
	}

	// advance pulses
	for i := 0; i < len(arch.pulses); {
		p := &arch.pulses[i]
		p.t += dt * arch.anim_speed
		if p.t >= 1.0 {
			// continue along subsequent connections if the destination has outgoing
			next_conn := next_connection(arch, p.conn_idx)
			if next_conn >= 0 {
				new_p := Pulse{conn_idx = next_conn, t = p.t - 1.0, color = p.color}
				unordered_remove(&arch.pulses, i)
				append(&arch.pulses, new_p)
				continue
			} else {
				unordered_remove(&arch.pulses, i)
				continue
			}
		}
		i += 1
	}
}

next_connection :: proc(arch: ^Architecture, conn_idx: int) -> int {
	if conn_idx < 0 || conn_idx >= len(arch.connections) do return -1
	dest := arch.connections[conn_idx].to_id
	for c, idx in arch.connections {
		if c.from_id == dest do return idx
	}
	return -1
}

pulse_color :: proc(t: f32) -> rl.Color {
	// simple HSV-ish cycle through a few presets
	colors := [?]rl.Color{
		{120, 220, 255, 255},
		{255, 220, 110, 255},
		{160, 255, 170, 255},
		{255, 160, 240, 255},
		{255, 255, 255, 255},
	}
	idx := int(t * f32(len(colors))) % len(colors)
	return colors[idx]
}

draw_pulses :: proc(arch: ^Architecture) {
	for p in arch.pulses {
		if p.conn_idx < 0 || p.conn_idx >= len(arch.connections) do continue
		c := arch.connections[p.conn_idx]
		from := layer_by_id(arch, c.from_id)
		to := layer_by_id(arch, c.to_id)
		if from == nil || to == nil do continue
		a := layer_right_anchor(from)
		b := layer_left_anchor(to)
		pos := rl.Vector2{a[0] + (b[0] - a[0]) * p.t, a[1] + (b[1] - a[1]) * p.t}
		// halo
		halo := p.color
		halo.a = 70
		rl.DrawCircleV(pos, 10, halo)
		rl.DrawCircleV(pos, 5, p.color)
	}
}

// ----- JSON persistence --------------------------------------------------

JSON_Params :: struct {
	filters:              int,
	kernel:               int,
	stride:               int,
	padding:              int,
	pool_size:            int,
	units:                int,
	activation:           string,
	recurrent_activation: string,
	return_sequences:     bool,
	channels:             int,
	height:               int,
	width:                int,
	classes:              int,
}

JSON_Layer :: struct {
	id:           int,
	type:         string,
	pos:          [2]f32,
	input_shape:  [3]int,
	output_shape: [3]int,
	params:       JSON_Params,
}

JSON_Connection :: struct {
	from:   int,
	to:     int,
	weight: f32,
}

JSON_Arch :: struct {
	version:     int,
	anim_speed:  f32,
	layers:      []JSON_Layer,
	connections: []JSON_Connection,
}

params_to_json :: proc(p: Layer_Params) -> JSON_Params {
	out := JSON_Params{}
	switch v in p {
	case Conv_Params:
		out.filters = v.filters
		out.kernel = v.kernel
		out.stride = v.stride
		out.padding = v.padding
	case Pool_Params:
		out.pool_size = v.pool_size
		out.stride = v.stride
	case Dense_Params:
		out.units = v.units
		out.activation = v.activation
	case RNN_Params:
		out.units = v.units
		out.activation = v.activation
		out.return_sequences = v.return_sequences
	case LSTM_Params:
		out.units = v.units
		out.activation = v.activation
		out.recurrent_activation = v.recurrent_activation
	case GRU_Params:
		out.units = v.units
		out.activation = v.activation
		out.recurrent_activation = v.recurrent_activation
	case Input_Params:
		out.channels = v.channels
		out.height = v.height
		out.width = v.width
	case Output_Params:
		out.classes = v.classes
	}
	return out
}

params_from_json :: proc(t: Layer_Type, p: JSON_Params) -> Layer_Params {
	switch t {
	case .Conv2D:
		return Conv_Params{filters = p.filters, kernel = p.kernel, stride = p.stride, padding = p.padding}
	case .MaxPool, .AveragePool:
		return Pool_Params{pool_size = p.pool_size, stride = p.stride}
	case .Dense:
		return Dense_Params{units = p.units, activation = p.activation}
	case .SimpleRNN:
		return RNN_Params{units = p.units, activation = p.activation, return_sequences = p.return_sequences}
	case .LSTM:
		return LSTM_Params{units = p.units, activation = p.activation, recurrent_activation = p.recurrent_activation}
	case .GRU:
		return GRU_Params{units = p.units, activation = p.activation, recurrent_activation = p.recurrent_activation}
	case .Input:
		return Input_Params{channels = p.channels, height = p.height, width = p.width}
	case .Output:
		return Output_Params{classes = p.classes}
	case .Flatten:
		// flatten has no params
		return nil
	}
	return nil
}

save_architecture :: proc(arch: ^Architecture, path: string) -> bool {
	j := JSON_Arch{}
	j.version = 1
	j.anim_speed = arch.anim_speed
	j.layers = make([]JSON_Layer, len(arch.layers))
	defer delete(j.layers)
	for l, i in arch.layers {
		j.layers[i] = JSON_Layer{
			id           = l.id,
			type         = layer_type_name(l.type),
			pos          = [2]f32{l.position[0], l.position[1]},
			input_shape  = l.input_shape,
			output_shape = l.output_shape,
			params       = params_to_json(l.params),
		}
	}
	j.connections = make([]JSON_Connection, len(arch.connections))
	defer delete(j.connections)
	for c, i in arch.connections {
		j.connections[i] = JSON_Connection{from = c.from_id, to = c.to_id, weight = c.weight}
	}

	opts := json.Marshal_Options{spec = .JSON, pretty = true, use_spaces = true, spaces = 2}
	data, err := json.marshal(j, opts)
	if err != nil {
		fmt.eprintln("save: marshal error:", err)
		return false
	}
	defer delete(data)
	if werr := os.write_entire_file(path, data); werr != nil {
		fmt.eprintln("save: write failed for", path, werr)
		return false
	}
	return true
}

load_architecture :: proc(path: string) -> (Architecture, bool) {
	data, rerr := os.read_entire_file_from_path(path, context.allocator)
	if rerr != nil {
		fmt.eprintln("load: read failed for", path, rerr)
		return Architecture{}, false
	}
	defer delete(data)

	j := JSON_Arch{}
	if err := json.unmarshal(data, &j); err != nil {
		fmt.eprintln("load: unmarshal error:", err)
		return Architecture{}, false
	}
	defer {
		delete(j.layers)
		delete(j.connections)
	}

	arch := Architecture{}
	arch.anim_speed = j.anim_speed if j.anim_speed > 0 else 0.6

	for jl in j.layers {
		t, ok := layer_type_from_name(jl.type)
		if !ok do continue
		l := Layer{
			id           = jl.id,
			type         = t,
			position     = rl.Vector2{jl.pos[0], jl.pos[1]},
			input_shape  = jl.input_shape,
			output_shape = jl.output_shape,
			params       = params_from_json(t, jl.params),
			color        = layer_color(t),
		}
		compute_layer_size(&l)
		append(&arch.layers, l)
	}
	for jc in j.connections {
		append(&arch.connections, Connection{from_id = jc.from, to_id = jc.to, weight = jc.weight})
	}
	arch.camera = rl.Camera2D{offset = rl.Vector2{640, 360}, target = network_center(&arch), zoom = 1.0}
	return arch, true
}
