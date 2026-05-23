package nn_odinray

// Core data model: layer/connection/architecture types, parameters,
// shape propagation, analysis (param count + FLOPs), hit testing, mutations.
// No drawing or I/O lives here.

import "core:fmt"
import "core:math"
import rl "vendor:raylib"

// ----- types -------------------------------------------------------------

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
	id:              int,
	type:            Layer_Type,
	position:        rl.Vector2, // top-left corner in world space
	size:            rl.Vector2,
	input_shape:     [3]int, // channels, height, width
	output_shape:    [3]int,
	params:          Layer_Params,
	color:           rl.Color,
	// transient render-only fields, recomputed each frame
	render_x_offset: f32,
	render_w_extra:  f32, // extra width when this layer is unrolled into N copies
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
	unrolled:          bool, // render-only: expand recurrent layers into N copies
}

UNROLL_STEPS :: 4
UNROLL_GAP :: f32(30)

ADDABLE_TYPES := [?]Layer_Type{
	.Conv2D, .MaxPool, .AveragePool, .Flatten,
	.Dense, .SimpleRNN, .LSTM, .GRU, .Output,
}

ACTIVATIONS := []string{"relu", "tanh", "sigmoid", "linear", "softmax"}
RECURRENT_ACTIVATIONS := []string{"sigmoid", "tanh", "relu"}

// ----- destruction / classification -------------------------------------

destroy_architecture :: proc(arch: ^Architecture) {
	delete(arch.layers)
	delete(arch.connections)
	delete(arch.pulses)
}

is_recurrent :: proc(t: Layer_Type) -> bool {
	return t == .SimpleRNN || t == .LSTM || t == .GRU
}

// ----- labels / colors --------------------------------------------------

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

// ----- geometry ---------------------------------------------------------

layer_by_id :: proc(arch: ^Architecture, id: int) -> ^Layer {
	for i in 0 ..< len(arch.layers) {
		if arch.layers[i].id == id do return &arch.layers[i]
	}
	return nil
}

// All geometry helpers return RENDER-space coords (i.e. apply render_x_offset / render_w_extra).
// During non-unrolled rendering both transients are zero and these reduce to the base layer geometry.

layer_rect :: proc(l: ^Layer) -> rl.Rectangle {
	return rl.Rectangle{
		l.position.x + l.render_x_offset,
		l.position.y,
		l.size.x + l.render_w_extra,
		l.size.y,
	}
}

layer_left_anchor :: proc(l: ^Layer) -> rl.Vector2 {
	return rl.Vector2{l.position.x + l.render_x_offset, l.position.y + l.size.y / 2}
}

layer_right_anchor :: proc(l: ^Layer) -> rl.Vector2 {
	return rl.Vector2{
		l.position.x + l.render_x_offset + l.size.x + l.render_w_extra,
		l.position.y + l.size.y / 2,
	}
}

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

// ----- shape propagation ------------------------------------------------

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

// ----- analysis: param count & FLOPs ------------------------------------

// Forward-pass MAC counted as 2 FLOPs (standard convention).
// For recurrent layers, FLOPs are reported per single time step.
layer_param_count :: proc(l: ^Layer) -> int {
	in_ch := l.input_shape[0]
	flat_in := l.input_shape[0] * l.input_shape[1] * l.input_shape[2]
	switch p in l.params {
	case Conv_Params:
		return p.filters * (p.kernel * p.kernel * in_ch + 1)
	case Pool_Params:
		return 0
	case Dense_Params:
		return p.units * (flat_in + 1)
	case RNN_Params:
		return p.units * (flat_in + p.units + 1)
	case LSTM_Params:
		return 4 * p.units * (flat_in + p.units + 1)
	case GRU_Params:
		return 3 * p.units * (flat_in + p.units + 1)
	case Input_Params:
		return 0
	case Output_Params:
		return p.classes * (flat_in + 1)
	}
	return 0
}

layer_flops :: proc(l: ^Layer) -> int {
	flat_in := l.input_shape[0] * l.input_shape[1] * l.input_shape[2]
	switch p in l.params {
	case Conv_Params:
		return 2 * (p.kernel * p.kernel * l.input_shape[0]) * l.output_shape[1] * l.output_shape[2] * p.filters
	case Pool_Params:
		return p.pool_size * p.pool_size * l.output_shape[1] * l.output_shape[2] * l.input_shape[0]
	case Dense_Params:
		return 2 * flat_in * p.units
	case RNN_Params:
		return 2 * p.units * (flat_in + p.units)
	case LSTM_Params:
		return 8 * p.units * (flat_in + p.units)
	case GRU_Params:
		return 6 * p.units * (flat_in + p.units)
	case Output_Params:
		return 2 * flat_in * p.classes
	case Input_Params:
		return 0
	}
	return 0
}

total_params :: proc(arch: ^Architecture) -> int {
	total := 0
	for i in 0 ..< len(arch.layers) do total += layer_param_count(&arch.layers[i])
	return total
}

total_flops :: proc(arch: ^Architecture) -> int {
	total := 0
	for i in 0 ..< len(arch.layers) do total += layer_flops(&arch.layers[i])
	return total
}

format_count :: proc(n: int) -> cstring {
	if n < 1_000 do return fmt.ctprintf("%d", n)
	if n < 1_000_000 do return fmt.ctprintf("%.1fK", f64(n) / 1000.0)
	if n < 1_000_000_000 do return fmt.ctprintf("%.2fM", f64(n) / 1_000_000.0)
	if n < 1_000_000_000_000 do return fmt.ctprintf("%.2fG", f64(n) / 1_000_000_000.0)
	return fmt.ctprintf("%.2fT", f64(n) / 1_000_000_000_000.0)
}

// ----- mutations --------------------------------------------------------

add_layer :: proc(arch: ^Architecture, l: Layer) -> int {
	id := 0
	for ll in arch.layers do if ll.id >= id do id = ll.id + 1
	layer := l
	layer.id = id
	layer.color = layer_color(layer.type)
	append(&arch.layers, layer)
	return id
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

// ----- hit testing ------------------------------------------------------

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
