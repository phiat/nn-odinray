package nn_odinray

// Visualization layer: graph layout, drawing the network (cards, connections,
// self-loops, output handles, unrolled time steps) and the forward-pass pulse
// animation. Reads from the model (model.odin); no mutation of state.

import "core:math"
import "core:math/rand"
import rl "vendor:raylib"

// ----- layout ------------------------------------------------------------

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
	r0 := layer_rect(&arch.layers[0])
	min_x, min_y := r0.x, r0.y
	max_x, max_y := r0.x + r0.width, r0.y + r0.height
	for i in 0 ..< len(arch.layers) {
		r := layer_rect(&arch.layers[i])
		min_x = min(min_x, r.x)
		min_y = min(min_y, r.y)
		max_x = max(max_x, r.x + r.width)
		max_y = max(max_y, r.y + r.height)
	}
	return rl.Vector2{(min_x + max_x) / 2, (min_y + max_y) / 2}
}

network_bounds :: proc(arch: ^Architecture) -> rl.Vector2 {
	if len(arch.layers) == 0 do return rl.Vector2{0, 0}
	r0 := layer_rect(&arch.layers[0])
	min_x, min_y := r0.x, r0.y
	max_x, max_y := r0.x + r0.width, r0.y + r0.height
	for i in 0 ..< len(arch.layers) {
		r := layer_rect(&arch.layers[i])
		min_x = min(min_x, r.x)
		min_y = min(min_y, r.y)
		max_x = max(max_x, r.x + r.width)
		max_y = max(max_y, r.y + r.height)
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

// Recompute every layer's transient render_x_offset / render_w_extra based on arch.unrolled.
// Call once per frame before handle_input and drawing so hit-testing and rendering agree.
compute_render_offsets :: proc(arch: ^Architecture) {
	for i in 0 ..< len(arch.layers) {
		arch.layers[i].render_x_offset = 0
		arch.layers[i].render_w_extra = 0
	}
	if !arch.unrolled do return

	for i in 0 ..< len(arch.layers) {
		l := &arch.layers[i]
		if !is_recurrent(l.type) do continue
		shift := f32(UNROLL_STEPS - 1) * (l.size.x + UNROLL_GAP)
		l.render_w_extra = shift
		boundary := l.position.x + l.size.x / 2
		for j in 0 ..< len(arch.layers) {
			if i == j do continue
			if arch.layers[j].position.x > boundary {
				arch.layers[j].render_x_offset += shift
			}
		}
	}
}

// ----- drawing -----------------------------------------------------------

darken :: proc(c: rl.Color, factor: f32) -> rl.Color {
	return rl.Color{
		u8(f32(c.r) * factor),
		u8(f32(c.g) * factor),
		u8(f32(c.b) * factor),
		c.a,
	}
}

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
	if arch.unrolled do return // time-unrolled view replaces self-loops with explicit step arrows
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

// Draw a single layer card at the given rect with given highlight state and optional step suffix.
// step_label is non-empty in unrolled mode (e.g. "t-1", "t", "t+1") and is rendered above the card.
draw_layer_card :: proc(l: ^Layer, rect: rl.Rectangle, highlighted: bool, step_label: cstring = "") {
	// multi-channel stack shadow
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

	thick: f32 = highlighted ? 3.5 : 2.0
	border_col := highlighted ? rl.Color{255, 255, 255, 255} : l.color
	rl.DrawRectangleRoundedLinesEx(rect, 0.18, 10, thick, border_col)

	header := rl.Rectangle{rect.x, rect.y, rect.width, 22}
	rl.DrawRectangleRec(header, l.color)

	name := layer_label(l.type)
	name_w := rl.MeasureText(name, 14)
	rl.DrawText(
		name,
		i32(rect.x + (rect.width - f32(name_w)) / 2),
		i32(rect.y + 4),
		14,
		rl.Color{15, 18, 25, 255},
	)

	out_str := shape_label(l.output_shape)
	out_w := rl.MeasureText(out_str, 16)
	rl.DrawText(
		out_str,
		i32(rect.x + (rect.width - f32(out_w)) / 2),
		i32(rect.y + rect.height / 2 - 8),
		16,
		rl.Color{235, 240, 245, 255},
	)

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

	if step_label != "" {
		sw := rl.MeasureText(step_label, 12)
		rl.DrawText(
			step_label,
			i32(rect.x + (rect.width - f32(sw)) / 2),
			i32(rect.y - 16),
			12,
			rl.Color{220, 200, 250, 240},
		)
	}
}

UNROLL_STEP_LABELS := [?]cstring{"t-1", "t", "t+1", "t+2", "t+3", "t+4", "t+5", "t+6"}

draw_layers :: proc(arch: ^Architecture, hovered: int) {
	for i in 0 ..< len(arch.layers) {
		l := &arch.layers[i]
		if arch.unrolled && is_recurrent(l.type) {
			// draw N copies side by side with time-step arrows between them
			base_x := l.position.x + l.render_x_offset
			single_w := l.size.x
			for step in 0 ..< UNROLL_STEPS {
				x := base_x + f32(step) * (single_w + UNROLL_GAP)
				r := rl.Rectangle{x, l.position.y, single_w, l.size.y}
				label: cstring
				if step < len(UNROLL_STEP_LABELS) {
					label = UNROLL_STEP_LABELS[step]
				}
				draw_layer_card(l, r, i == hovered, label)
			}
			// time-step arrows between consecutive copies
			arrow_col := rl.Color{200, 160, 240, 220}
			for step in 0 ..< UNROLL_STEPS - 1 {
				a := rl.Vector2{base_x + f32(step) * (single_w + UNROLL_GAP) + single_w, l.position.y + l.size.y / 2}
				b := rl.Vector2{base_x + f32(step + 1) * (single_w + UNROLL_GAP), l.position.y + l.size.y / 2}
				rl.DrawLineEx(a, b, 2.0, arrow_col)
				draw_arrowhead(a, b, arrow_col)
			}
		} else {
			draw_layer_card(l, layer_rect(l), i == hovered)
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
