package pp1

import "core:fmt"
import "core:strings"
import rl "vendor:raylib"

HELP_LINES := []cstring{
	"CNN/RNN Visualizer  -  MVP",
	"",
	"Left drag layer   -  move layer",
	"Drag from o handle-  create connection",
	"Click connection  -  select (Delete removes)",
	"Left drag empty   -  pan",
	"Middle drag       -  pan",
	"Wheel             -  zoom (around cursor)",
	"Right click layer -  open property panel",
	"N                 -  add layer (picker)",
	"Delete            -  remove selected conn/layer",
	"Tab               -  cycle demo (CNN/RNN/LSTM/GRU)",
	"Ctrl+Z / Ctrl+Y  -  undo / redo",
	"S / L             -  save / load architecture.json",
	"R                 -  reset layout & camera",
	"+ / -             -  animation speed",
	"P                 -  screenshot PNG",
	"H / Esc           -  toggle help / cancel",
}

toolbar_button :: proc(rect: rl.Rectangle, label: cstring, enabled: bool, ui: ^UI_State) -> bool {
	mouse := rl.GetMousePosition()
	hovered := enabled && rl.CheckCollisionPointRec(mouse, rect)
	bg: rl.Color
	border: rl.Color
	fg: rl.Color
	if !enabled {
		bg = rl.Color{30, 36, 48, 255}
		border = rl.Color{60, 72, 90, 255}
		fg = rl.Color{120, 130, 150, 220}
	} else if hovered {
		bg = rl.Color{55, 80, 120, 255}
		border = rl.Color{160, 200, 240, 255}
		fg = rl.Color{230, 240, 255, 255}
	} else {
		bg = rl.Color{30, 50, 80, 255}
		border = rl.Color{90, 130, 170, 255}
		fg = rl.Color{200, 220, 240, 255}
	}
	rl.DrawRectangleRec(rect, bg)
	rl.DrawRectangleLinesEx(rect, 1.0, border)
	tw := rl.MeasureText(label, 12)
	rl.DrawText(label, i32(rect.x + (rect.width - f32(tw)) / 2), i32(rect.y + 6), 12, fg)
	if hovered && rl.IsMouseButtonPressed(.LEFT) {
		ui.consumed_click = true
		return true
	}
	return false
}

ui_button :: proc(rect: rl.Rectangle, label: cstring, ui: ^UI_State) -> bool {
	mouse := rl.GetMousePosition()
	hovered := rl.CheckCollisionPointRec(mouse, rect)
	bg := hovered ? rl.Color{60, 80, 110, 255} : rl.Color{35, 45, 65, 255}
	border := hovered ? rl.Color{160, 200, 240, 255} : rl.Color{80, 100, 130, 255}
	rl.DrawRectangleRec(rect, bg)
	rl.DrawRectangleLinesEx(rect, 1.0, border)
	tw := rl.MeasureText(label, 13)
	rl.DrawText(label, i32(rect.x + (rect.width - f32(tw)) / 2), i32(rect.y + (rect.height - 13) / 2), 13, rl.Color{225, 235, 245, 255})
	if hovered && rl.IsMouseButtonPressed(.LEFT) {
		ui.consumed_click = true
		return true
	}
	return false
}

ui_danger_button :: proc(rect: rl.Rectangle, label: cstring, ui: ^UI_State) -> bool {
	mouse := rl.GetMousePosition()
	hovered := rl.CheckCollisionPointRec(mouse, rect)
	bg := hovered ? rl.Color{160, 60, 70, 255} : rl.Color{80, 30, 36, 255}
	border := hovered ? rl.Color{240, 140, 150, 255} : rl.Color{160, 70, 80, 255}
	rl.DrawRectangleRec(rect, bg)
	rl.DrawRectangleLinesEx(rect, 1.0, border)
	tw := rl.MeasureText(label, 13)
	rl.DrawText(label, i32(rect.x + (rect.width - f32(tw)) / 2), i32(rect.y + (rect.height - 13) / 2), 13, rl.Color{255, 230, 230, 255})
	if hovered && rl.IsMouseButtonPressed(.LEFT) {
		ui.consumed_click = true
		return true
	}
	return false
}

draw_ui_overlay :: proc(arch: ^Architecture, ui: ^UI_State) {
	sw := i32(rl.GetScreenWidth())
	sh := i32(rl.GetScreenHeight())

	// top status bar
	rl.DrawRectangle(0, 0, sw, 30, rl.Color{12, 14, 20, 220})
	status := fmt.ctprintf(
		"demo:%s  layers:%d  conns:%d  pulses:%d  zoom:%.2f  speed:%.1fx  fps:%d",
		demo_name(ui.current_demo),
		len(arch.layers),
		len(arch.connections),
		len(arch.pulses),
		arch.camera.zoom,
		arch.anim_speed,
		rl.GetFPS(),
	)
	rl.DrawText(status, 10, 8, 14, rl.Color{200, 220, 240, 255})

	// Undo / Redo buttons
	can_undo := len(ui.history.undo) > 0
	can_redo := len(ui.history.redo) > 0
	undo_btn := rl.Rectangle{f32(sw) - 350, 3, 55, 24}
	redo_btn := rl.Rectangle{f32(sw) - 290, 3, 55, 24}
	if toolbar_button(undo_btn, fmt.ctprintf("Undo %d", len(ui.history.undo)), can_undo, ui) {
		if undo(ui.history, arch) {
			ui.hovered_layer = -1
			ui.dragging_layer = -1
			show_toast(ui, "undo")
		}
	}
	if toolbar_button(redo_btn, fmt.ctprintf("Redo %d", len(ui.history.redo)), can_redo, ui) {
		if redo(ui.history, arch) {
			ui.hovered_layer = -1
			ui.dragging_layer = -1
			show_toast(ui, "redo")
		}
	}

	// "+ Add Layer" toolbar button near top-right
	add_btn := rl.Rectangle{f32(sw) - 230, 3, 110, 24}
	add_hovered := rl.CheckCollisionPointRec(rl.GetMousePosition(), add_btn)
	add_bg := add_hovered ? rl.Color{70, 130, 90, 255} : rl.Color{40, 90, 60, 255}
	rl.DrawRectangleRec(add_btn, add_bg)
	rl.DrawRectangleLinesEx(add_btn, 1.0, rl.Color{130, 200, 150, 255})
	add_label := cstring("+ Add Layer (N)")
	alw := rl.MeasureText(add_label, 12)
	rl.DrawText(add_label, i32(add_btn.x + (add_btn.width - f32(alw)) / 2), i32(add_btn.y + 6), 12, rl.Color{225, 245, 230, 255})
	if add_hovered && rl.IsMouseButtonPressed(.LEFT) {
		ui.consumed_click = true
		// target = selected, else tail of the graph, else -1 for standalone
		picker_target := -1
		sel := selected_layer_index(arch, ui)
		if sel >= 0 {
			picker_target = arch.layers[sel].id
		} else {
			picker_target = tail_layer_id(arch)
		}
		ui.picker_after_id = picker_target
		ui.picker_open = true
	}

	hint := cstring("press H for help")
	hw := rl.MeasureText(hint, 12)
	rl.DrawText(hint, sw - hw - 12, 9, 12, rl.Color{160, 180, 200, 200})

	if ui.show_help {
		draw_help_panel()
	}

	if ui.hovered_layer >= 0 && ui.hovered_layer < len(arch.layers) {
		draw_layer_inspector(&arch.layers[ui.hovered_layer])
	}

	if ui.screenshot_flash > 0 {
		a: u8 = u8(min(ui.screenshot_flash, 1.0) * 220)
		fname := fmt.ctprintf("saved visualizer_%03d.png", ui.screenshot_seq)
		w := rl.MeasureText(fname, 16)
		rl.DrawRectangle(
			(sw - w - 24) / 2,
			60,
			w + 24,
			28,
			rl.Color{0, 0, 0, a},
		)
		rl.DrawText(fname, (sw - w) / 2, 66, 16, rl.Color{220, 255, 220, a})
	}

	if ui.panel_open {
		draw_property_panel(arch, ui)
	}

	if ui.picker_open {
		draw_layer_picker(arch, ui)
	}

	if ui.toast_remaining > 0 && len(ui.toast_text) > 0 {
		a: u8 = u8(min(ui.toast_remaining / 2.4, 1.0) * 230)
		txt := strings.clone_to_cstring(ui.toast_text, context.temp_allocator)
		w := rl.MeasureText(txt, 16)
		bx := (sw - w - 28) / 2
		by := sh - 50
		rl.DrawRectangle(bx, by, w + 28, 30, rl.Color{20, 30, 40, a})
		rl.DrawRectangleLinesEx(rl.Rectangle{f32(bx), f32(by), f32(w + 28), 30}, 1.0, rl.Color{120, 180, 220, a})
		rl.DrawText(txt, bx + 14, by + 7, 16, rl.Color{220, 240, 255, a})
	}
}

draw_help_panel :: proc() {
	x: i32 = 10
	y: i32 = 40
	w: i32 = 280
	h: i32 = i32(len(HELP_LINES)) * 18 + 16
	rl.DrawRectangle(x, y, w, h, rl.Color{12, 14, 20, 210})
	rl.DrawRectangleLinesEx(rl.Rectangle{f32(x), f32(y), f32(w), f32(h)}, 1.0, rl.Color{80, 100, 120, 255})
	for line, i in HELP_LINES {
		size: i32 = i == 0 ? 14 : 12
		col := i == 0 ? rl.Color{210, 230, 255, 255} : rl.Color{180, 195, 215, 255}
		rl.DrawText(line, x + 10, y + 8 + i32(i) * 18, size, col)
	}
}

int_row :: proc(ui: ^UI_State, panel_x: f32, y: ^f32, label: cstring, val: ^int, min_v, max_v: int) -> bool {
	rl.DrawText(label, i32(panel_x + 14), i32(y^ + 7), 13, rl.Color{180, 200, 220, 255})

	minus := rl.Rectangle{panel_x + 132, y^, 26, 24}
	plus := rl.Rectangle{panel_x + PANEL_WIDTH - 40, y^, 26, 24}
	changed := false
	if ui_button(minus, "-", ui) {
		if val^ > min_v {val^ -= 1; changed = true}
	}
	if ui_button(plus, "+", ui) {
		if val^ < max_v {val^ += 1; changed = true}
	}
	val_str := fmt.ctprintf("%d", val^)
	val_w := rl.MeasureText(val_str, 14)
	center_x := panel_x + 132 + 26 + (PANEL_WIDTH - 40 - 132 - 26) / 2
	rl.DrawText(val_str, i32(center_x - f32(val_w) / 2), i32(y^ + 7), 14, rl.Color{230, 240, 250, 255})
	y^ += 30
	return changed
}

string_row :: proc(ui: ^UI_State, panel_x: f32, y: ^f32, label: cstring, val: ^string, choices: []string) -> bool {
	rl.DrawText(label, i32(panel_x + 14), i32(y^ + 7), 13, rl.Color{180, 200, 220, 255})

	left := rl.Rectangle{panel_x + 132, y^, 22, 24}
	right := rl.Rectangle{panel_x + PANEL_WIDTH - 36, y^, 22, 24}
	changed := false
	if ui_button(left, "<", ui) {
		val^ = cycle_string(val^, choices, -1)
		changed = true
	}
	if ui_button(right, ">", ui) {
		val^ = cycle_string(val^, choices, +1)
		changed = true
	}
	cval := strings.clone_to_cstring(val^, context.temp_allocator)
	val_w := rl.MeasureText(cval, 13)
	center_x := panel_x + 132 + 22 + (PANEL_WIDTH - 36 - 132 - 22) / 2
	rl.DrawText(cval, i32(center_x - f32(val_w) / 2), i32(y^ + 7), 13, rl.Color{200, 230, 250, 255})
	y^ += 30
	return changed
}

bool_row :: proc(ui: ^UI_State, panel_x: f32, y: ^f32, label: cstring, val: ^bool) -> bool {
	rl.DrawText(label, i32(panel_x + 14), i32(y^ + 7), 13, rl.Color{180, 200, 220, 255})
	btn := rl.Rectangle{panel_x + 132, y^, PANEL_WIDTH - 132 - 14, 24}
	lbl: cstring = val^ ? "true" : "false"
	changed := false
	if ui_button(btn, lbl, ui) {
		val^ = !val^
		changed = true
	}
	y^ += 30
	return changed
}

draw_property_panel :: proc(arch: ^Architecture, ui: ^UI_State) {
	rect := panel_rect()
	rl.DrawRectangleRec(rect, rl.Color{15, 20, 30, 235})
	rl.DrawRectangleLinesEx(rect, 1.0, rl.Color{80, 110, 140, 255})

	idx := selected_layer_index(arch, ui)
	if idx < 0 {
		rl.DrawText("no layer selected", i32(rect.x + 14), i32(rect.y + 14), 13, rl.Color{180, 200, 220, 220})
		return
	}
	l := &arch.layers[idx]

	// header
	header := rl.Rectangle{rect.x, rect.y, rect.width, 28}
	rl.DrawRectangleRec(header, l.color)
	title := fmt.ctprintf("%s #%d", layer_label(l.type), l.id)
	rl.DrawText(title, i32(rect.x + 12), i32(rect.y + 7), 16, rl.Color{15, 18, 25, 255})
	close_rect := rl.Rectangle{rect.x + rect.width - 28, rect.y + 3, 22, 22}
	if ui_button(close_rect, "x", ui) {
		ui.panel_open = false
		ui.selected_layer = -1
		return
	}

	y := rect.y + 40

	// shape rows
	in_str := fmt.ctprintf("in:  %dx%dx%d", l.input_shape[0], l.input_shape[1], l.input_shape[2])
	out_str := fmt.ctprintf("out: %dx%dx%d", l.output_shape[0], l.output_shape[1], l.output_shape[2])
	rl.DrawText(in_str, i32(rect.x + 14), i32(y), 13, rl.Color{170, 200, 230, 255})
	y += 18
	rl.DrawText(out_str, i32(rect.x + 14), i32(y), 13, rl.Color{160, 220, 200, 255})
	y += 24
	rl.DrawLine(i32(rect.x + 10), i32(y), i32(rect.x + rect.width - 10), i32(y), rl.Color{60, 80, 100, 255})
	y += 10

	changed := false
	switch &p in l.params {
	case Conv_Params:
		if int_row(ui, rect.x, &y, "Filters", &p.filters, 1, 1024) do changed = true
		if int_row(ui, rect.x, &y, "Kernel",  &p.kernel,  1, 9)    do changed = true
		if int_row(ui, rect.x, &y, "Stride",  &p.stride,  1, 8)    do changed = true
		if int_row(ui, rect.x, &y, "Padding", &p.padding, 0, 8)    do changed = true
	case Pool_Params:
		if int_row(ui, rect.x, &y, "PoolSize", &p.pool_size, 1, 8) do changed = true
		if int_row(ui, rect.x, &y, "Stride",   &p.stride,    1, 8) do changed = true
	case Dense_Params:
		if int_row(ui, rect.x, &y, "Units", &p.units, 1, 4096) do changed = true
		if string_row(ui, rect.x, &y, "Activation", &p.activation, ACTIVATIONS) do changed = true
	case RNN_Params:
		if int_row(ui, rect.x, &y, "Units", &p.units, 1, 1024) do changed = true
		if string_row(ui, rect.x, &y, "Activation", &p.activation, ACTIVATIONS) do changed = true
		if bool_row(ui, rect.x, &y, "ReturnSeq", &p.return_sequences) do changed = true
	case LSTM_Params:
		if int_row(ui, rect.x, &y, "Units", &p.units, 1, 1024) do changed = true
		if string_row(ui, rect.x, &y, "Activation", &p.activation, ACTIVATIONS) do changed = true
		if string_row(ui, rect.x, &y, "RecurrentAct", &p.recurrent_activation, RECURRENT_ACTIVATIONS) do changed = true
	case GRU_Params:
		if int_row(ui, rect.x, &y, "Units", &p.units, 1, 1024) do changed = true
		if string_row(ui, rect.x, &y, "Activation", &p.activation, ACTIVATIONS) do changed = true
		if string_row(ui, rect.x, &y, "RecurrentAct", &p.recurrent_activation, RECURRENT_ACTIVATIONS) do changed = true
	case Input_Params:
		if int_row(ui, rect.x, &y, "Channels", &p.channels, 1, 1024) do changed = true
		if int_row(ui, rect.x, &y, "Height",   &p.height,   1, 512)  do changed = true
		if int_row(ui, rect.x, &y, "Width",    &p.width,    1, 512)  do changed = true
	case Output_Params:
		if int_row(ui, rect.x, &y, "Classes", &p.classes, 1, 1024) do changed = true
	}

	if changed do recompute_shapes(arch)

	y += 6
	rl.DrawLine(i32(rect.x + 10), i32(y), i32(rect.x + rect.width - 10), i32(y), rl.Color{60, 80, 100, 255})
	y += 12

	if ui_button(rl.Rectangle{rect.x + 14, y, rect.width - 28, 28}, "+ Add layer after this", ui) {
		ui.picker_after_id = l.id
		ui.picker_open = true
		return
	}
	y += 36

	if ui_danger_button(rl.Rectangle{rect.x + 14, y, rect.width - 28, 28}, "Delete layer", ui) {
		push_undo(ui.history, arch)
		delete_layer_at(arch, idx)
		ui.panel_open = false
		ui.selected_layer = -1
		return
	}
}

draw_layer_picker :: proc(arch: ^Architecture, ui: ^UI_State) {
	sw := f32(rl.GetScreenWidth())
	sh := f32(rl.GetScreenHeight())

	// dim background
	rl.DrawRectangle(0, 0, i32(sw), i32(sh), rl.Color{0, 0, 0, 140})

	w: f32 = 320
	cols := 2
	bh: f32 = 30
	gap_y: f32 = 8
	rows := (len(ADDABLE_TYPES) + cols - 1) / cols
	h: f32 = 50 + f32(rows) * (bh + gap_y) + 18 + 38
	x := (sw - w) / 2
	y := (sh - h) / 2

	rl.DrawRectangleRec(rl.Rectangle{x, y, w, h}, rl.Color{20, 30, 45, 245})
	rl.DrawRectangleLinesEx(rl.Rectangle{x, y, w, h}, 1.5, rl.Color{120, 160, 200, 255})

	title: cstring
	if ui.picker_after_id >= 0 {
		title = fmt.ctprintf("Add layer after #%d", ui.picker_after_id)
	} else if len(arch.layers) == 0 {
		title = "Add first layer"
	} else {
		title = "Add new layer"
	}
	rl.DrawText(title, i32(x + 14), i32(y + 14), 16, rl.Color{230, 240, 250, 255})

	bw := (w - 30 - f32(cols - 1) * 10) / f32(cols)
	bx0 := x + 15
	by0 := y + 46

	for t, i in ADDABLE_TYPES {
		col := i % cols
		row := i / cols
		bx := bx0 + f32(col) * (bw + 10)
		by := by0 + f32(row) * (bh + gap_y)
		if ui_button(rl.Rectangle{bx, by, bw, bh}, layer_label(t), ui) {
			after_idx := -1
			if ui.picker_after_id >= 0 {
				for l, idx in arch.layers {
					if l.id == ui.picker_after_id {
						after_idx = idx
						break
					}
				}
			}
			push_undo(ui.history, arch)
			new_id := -1
			if after_idx >= 0 {
				new_id = add_layer_after(arch, after_idx, t)
			} else {
				new_id = add_layer_standalone(arch, t)
				recompute_shapes(arch)
			}
			if new_id >= 0 {
				ui.selected_layer = new_id
				ui.panel_open = true
				show_toast(ui, fmt.tprintf("added %s", string(layer_label(t))))
			}
			ui.picker_open = false
			return
		}
	}

	cancel_y := by0 + f32(rows) * (bh + gap_y) + 10
	if ui_button(rl.Rectangle{x + w / 2 - 60, cancel_y, 120, 28}, "Cancel (Esc)", ui) {
		ui.picker_open = false
	}
}

draw_layer_inspector :: proc(l: ^Layer) {
	sw := i32(rl.GetScreenWidth())
	sh := i32(rl.GetScreenHeight())
	w: i32 = 240
	h: i32 = 120
	x := sw - w - 10
	y := sh - h - 10
	rl.DrawRectangle(x, y, w, h, rl.Color{12, 14, 20, 220})
	rl.DrawRectangleLinesEx(rl.Rectangle{f32(x), f32(y), f32(w), f32(h)}, 1.0, rl.Color{80, 100, 120, 255})

	rl.DrawText(layer_label(l.type), x + 10, y + 8, 16, l.color)

	in_str := fmt.ctprintf("in : %dx%dx%d", l.input_shape[0], l.input_shape[1], l.input_shape[2])
	out_str := fmt.ctprintf("out: %dx%dx%d", l.output_shape[0], l.output_shape[1], l.output_shape[2])
	rl.DrawText(in_str, x + 10, y + 32, 13, rl.Color{210, 220, 230, 255})
	rl.DrawText(out_str, x + 10, y + 50, 13, rl.Color{210, 220, 230, 255})

	p := params_label(l)
	if p != "" {
		rl.DrawText(p, x + 10, y + 70, 13, rl.Color{180, 220, 255, 255})
	}

	id_str := fmt.ctprintf("id #%d", l.id)
	rl.DrawText(id_str, x + 10, y + h - 22, 11, rl.Color{140, 160, 180, 220})
}
