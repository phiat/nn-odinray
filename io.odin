package pp1

// JSON persistence and the undo/redo history stack.
// Both manage snapshots of the data model defined in model.odin.

import "core:encoding/json"
import "core:fmt"
import "core:os"
import rl "vendor:raylib"

// ----- JSON --------------------------------------------------------------

// Flat union-of-all-fields. Fields unused by a given layer type are written
// as zero and ignored on load via params_from_json dispatching on Layer_Type.
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
