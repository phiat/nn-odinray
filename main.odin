package pp1

import "core:fmt"
import "core:os"
import "core:strings"
import rl "vendor:raylib"

WINDOW_WIDTH :: 1280
WINDOW_HEIGHT :: 720

UI_State :: struct {
	dragging_layer:    int, // index in arch.layers, -1 if none
	drag_offset:       rl.Vector2, // mouse-world minus layer top-left
	hovered_layer:     int,
	selected_layer:    int, // id, not index, so it survives mutations
	panel_open:        bool,
	panning:           bool,
	show_help:         bool,
	screenshot_seq:    int,
	screenshot_flash:  f32,
	current_demo:      Demo_Kind,
	toast_text:        string,
	toast_remaining:   f32,
	consumed_click:    bool,

	// connection editing
	hover_output_idx:    int, // layer index whose output handle is under cursor
	connecting_from_id:  int, // -1 if not actively dragging a new connection
	selected_connection: int, // index in arch.connections, -1 if none

	// layer-type picker popup
	picker_open:     bool,
	picker_after_id: int,

	// undo/redo (owned by main; ui carries a pointer for buttons in draw_ui_overlay)
	history: ^History,
}

PANEL_WIDTH :: 300

panel_rect :: proc() -> rl.Rectangle {
	sw := f32(rl.GetScreenWidth())
	sh := f32(rl.GetScreenHeight())
	return rl.Rectangle{sw - PANEL_WIDTH - 10, 40, PANEL_WIDTH, sh - 60}
}

mouse_over_panel :: proc(ui: ^UI_State) -> bool {
	if !ui.panel_open do return false
	return rl.CheckCollisionPointRec(rl.GetMousePosition(), panel_rect())
}

mouse_over_chrome :: proc() -> bool {
	// reserve the top status bar; clicks there belong to chrome, not the world.
	return rl.GetMousePosition()[1] < 30
}

selected_layer_index :: proc(arch: ^Architecture, ui: ^UI_State) -> int {
	if !ui.panel_open do return -1
	for l, i in arch.layers do if l.id == ui.selected_layer do return i
	return -1
}

SAVE_PATH :: "architecture.json"

show_toast :: proc(ui: ^UI_State, msg: string) {
	ui.toast_text = msg
	ui.toast_remaining = 2.4
}

screen_to_world :: proc(arch: ^Architecture, pos: rl.Vector2) -> rl.Vector2 {
	return rl.GetScreenToWorld2D(pos, arch.camera)
}

find_layer_at :: proc(arch: ^Architecture, world_pos: rl.Vector2) -> int {
	// iterate in reverse so layers drawn later (visually on top) hit first
	for i := len(arch.layers) - 1; i >= 0; i -= 1 {
		l := &arch.layers[i]
		if rl.CheckCollisionPointRec(world_pos, layer_rect(l)) do return i
	}
	return -1
}

handle_input :: proc(arch: ^Architecture, ui: ^UI_State, dt: f32) {
	mouse := rl.GetMousePosition()
	world := screen_to_world(arch, mouse)
	over_panel := mouse_over_panel(ui)
	over_chrome := mouse_over_chrome()
	picker_active := ui.picker_open
	suppress := over_panel || over_chrome || picker_active

	// per-frame reset: consumed_click is set by UI buttons during draw to mark
	// "the click was handled" but it only applies to the frame it was set in.
	ui.consumed_click = false

	// hover (world only)
	if suppress {
		ui.hovered_layer = -1
		ui.hover_output_idx = -1
	} else {
		ui.hover_output_idx = find_output_handle(arch, world, 12 / arch.camera.zoom)
		ui.hovered_layer = ui.hover_output_idx >= 0 ? -1 : find_layer_at(arch, world)
	}

	// zoom (around cursor); skip if mouse over panel/picker
	if !suppress {
		wheel := rl.GetMouseWheelMove()
		if wheel != 0 {
			before := screen_to_world(arch, mouse)
			arch.camera.zoom *= (1.0 + wheel * 0.1)
			if arch.camera.zoom < 0.15 do arch.camera.zoom = 0.15
			if arch.camera.zoom > 4.0 do arch.camera.zoom = 4.0
			after := screen_to_world(arch, mouse)
			arch.camera.target[0] += before[0] - after[0]
			arch.camera.target[1] += before[1] - after[1]
		}
	}

	// right-click on a world layer: open the property panel
	if !suppress && rl.IsMouseButtonPressed(.RIGHT) {
		idx := find_layer_at(arch, world)
		if idx >= 0 {
			ui.selected_layer = arch.layers[idx].id
			ui.panel_open = true
		}
	}

	// left press priority: output handle (start connection) > layer body (drag) > connection (select) > pan
	if !suppress && !ui.consumed_click && rl.IsMouseButtonPressed(.LEFT) {
		if ui.hover_output_idx >= 0 {
			ui.connecting_from_id = arch.layers[ui.hover_output_idx].id
			ui.selected_connection = -1
		} else {
			lidx := find_layer_at(arch, world)
			if lidx >= 0 {
				ui.dragging_layer = lidx
				l := &arch.layers[lidx]
				ui.drag_offset = rl.Vector2{world[0] - l.position.x, world[1] - l.position.y}
				ui.selected_connection = -1
			} else {
				cidx := find_connection_at(arch, world, 6 / arch.camera.zoom)
				if cidx >= 0 {
					ui.selected_connection = cidx
				} else {
					ui.selected_connection = -1
					ui.panning = true
				}
			}
		}
	}

	// drop the in-flight connection on left release
	if ui.connecting_from_id >= 0 && rl.IsMouseButtonReleased(.LEFT) {
		target := find_layer_at(arch, world)
		if target >= 0 {
			target_id := arch.layers[target].id
			push_undo(ui.history, arch)
			if add_connection(arch, ui.connecting_from_id, target_id) {
				show_toast(ui, fmt.tprintf("connected #%d -> #%d", ui.connecting_from_id, target_id))
			} else {
				// rejected — discard the speculative snapshot
				if len(ui.history.undo) > 0 {
					destroy_architecture(&ui.history.undo[len(ui.history.undo) - 1])
					ordered_remove(&ui.history.undo, len(ui.history.undo) - 1)
				}
			}
		}
		ui.connecting_from_id = -1
	}

	// middle mouse always pans
	if !suppress && rl.IsMouseButtonPressed(.MIDDLE) do ui.panning = true

	if ui.dragging_layer >= 0 && rl.IsMouseButtonDown(.LEFT) {
		l := &arch.layers[ui.dragging_layer]
		l.position[0] = world[0] - ui.drag_offset[0]
		l.position[1] = world[1] - ui.drag_offset[1]
	}

	if ui.panning && (rl.IsMouseButtonDown(.LEFT) || rl.IsMouseButtonDown(.MIDDLE)) {
		delta := rl.GetMouseDelta()
		arch.camera.target[0] -= delta[0] / arch.camera.zoom
		arch.camera.target[1] -= delta[1] / arch.camera.zoom
	}

	if !rl.IsMouseButtonDown(.LEFT) && !rl.IsMouseButtonDown(.MIDDLE) {
		ui.dragging_layer = -1
		ui.panning = false
	}

	// keyboard
	if rl.IsKeyPressed(.H) do ui.show_help = !ui.show_help
	if rl.IsKeyPressed(.R) {
		auto_layout(arch)
		fit_camera(arch, f32(rl.GetScreenWidth()), f32(rl.GetScreenHeight()))
	}
	if rl.IsKeyPressed(.EQUAL) || rl.IsKeyPressed(.KP_ADD) {
		arch.anim_speed = min(arch.anim_speed + 0.2, 4.0)
	}
	if rl.IsKeyPressed(.MINUS) || rl.IsKeyPressed(.KP_SUBTRACT) {
		arch.anim_speed = max(arch.anim_speed - 0.2, 0.1)
	}
	if rl.IsKeyPressed(.P) {
		ui.screenshot_seq += 1
		fname := fmt.ctprintf("visualizer_%03d.png", ui.screenshot_seq)
		rl.TakeScreenshot(fname)
		ui.screenshot_flash = 1.2
	}
	if rl.IsKeyPressed(.TAB) {
		next := Demo_Kind((int(ui.current_demo) + 1) % len(Demo_Kind))
		switch_demo(arch, ui, next)
	}
	if rl.IsKeyPressed(.S) {
		if save_architecture(arch, SAVE_PATH) {
			show_toast(ui, fmt.tprintf("saved → %s", SAVE_PATH))
		} else {
			show_toast(ui, fmt.tprintf("save failed: %s", SAVE_PATH))
		}
	}
	if rl.IsKeyPressed(.L) {
		new_arch, ok := load_architecture(SAVE_PATH)
		if ok {
			destroy_architecture(arch)
			arch^ = new_arch
			clear_history(ui.history)
			ui.panel_open = false
			ui.selected_layer = -1
			fit_camera(arch, f32(rl.GetScreenWidth()), f32(rl.GetScreenHeight()))
			show_toast(ui, fmt.tprintf("loaded ← %s", SAVE_PATH))
		} else {
			show_toast(ui, fmt.tprintf("load failed: %s", SAVE_PATH))
		}
	}

	// N: open the layer-type picker; insert after hovered/selected if any, else after the tail, else standalone
	if rl.IsKeyPressed(.N) {
		target_idx := ui.hovered_layer
		if target_idx < 0 do target_idx = selected_layer_index(arch, ui)
		if target_idx >= 0 {
			ui.picker_after_id = arch.layers[target_idx].id
		} else {
			ui.picker_after_id = tail_layer_id(arch) // -1 if empty
		}
		ui.picker_open = true
	}

	// Delete: selected connection takes priority, then hovered/selected layer
	if rl.IsKeyPressed(.DELETE) || rl.IsKeyPressed(.BACKSPACE) {
		if ui.selected_connection >= 0 && ui.selected_connection < len(arch.connections) {
			push_undo(ui.history, arch)
			ordered_remove(&arch.connections, ui.selected_connection)
			ui.selected_connection = -1
			recompute_shapes(arch)
			show_toast(ui, "deleted connection")
		} else {
			target_idx := ui.hovered_layer
			if target_idx < 0 do target_idx = selected_layer_index(arch, ui)
			if target_idx >= 0 {
				push_undo(ui.history, arch)
				delete_layer_at(arch, target_idx)
				ui.hovered_layer = -1
				ui.dragging_layer = -1
				if selected_layer_index(arch, ui) < 0 {
					ui.panel_open = false
					ui.selected_layer = -1
				}
				show_toast(ui, "deleted layer")
			}
		}
	}

	// Ctrl+Z = undo; Ctrl+Y or Ctrl+Shift+Z = redo
	ctrl := rl.IsKeyDown(.LEFT_CONTROL) || rl.IsKeyDown(.RIGHT_CONTROL)
	shift := rl.IsKeyDown(.LEFT_SHIFT) || rl.IsKeyDown(.RIGHT_SHIFT)
	if ctrl && rl.IsKeyPressed(.Z) {
		if shift {
			if redo(ui.history, arch) {
				ui.hovered_layer = -1; ui.dragging_layer = -1; ui.connecting_from_id = -1
				show_toast(ui, "redo")
			}
		} else {
			if undo(ui.history, arch) {
				ui.hovered_layer = -1; ui.dragging_layer = -1; ui.connecting_from_id = -1
				show_toast(ui, "undo")
			}
		}
	}
	if ctrl && rl.IsKeyPressed(.Y) {
		if redo(ui.history, arch) {
			ui.hovered_layer = -1; ui.dragging_layer = -1; ui.connecting_from_id = -1
			show_toast(ui, "redo")
		}
	}

	if rl.IsKeyPressed(.ESCAPE) {
		if ui.picker_open {
			ui.picker_open = false
		} else if ui.connecting_from_id >= 0 {
			ui.connecting_from_id = -1
		} else if ui.selected_connection >= 0 {
			ui.selected_connection = -1
		} else {
			ui.panel_open = false
			ui.selected_layer = -1
		}
	}

	if ui.screenshot_flash > 0 do ui.screenshot_flash -= dt
	if ui.toast_remaining > 0 do ui.toast_remaining -= dt
}

run_shape_test :: proc() {
	arch := build_demo_cnn()
	defer destroy_architecture(&arch)

	fmt.println("== initial shapes ==")
	for l in arch.layers {
		fmt.printfln("  %s #%d: in=%v out=%v", layer_label(l.type), l.id, l.input_shape, l.output_shape)
	}

	// bump Conv2D #1 filters from 16 to 64
	for &l in arch.layers {
		if l.type == .Conv2D {
			switch &p in l.params {
			case Conv_Params:
				if p.filters == 16 do p.filters = 64
			case Pool_Params, Dense_Params, RNN_Params, LSTM_Params, GRU_Params, Input_Params, Output_Params:
			}
		}
	}
	recompute_shapes(&arch)

	fmt.println("\n== after bumping first Conv2D filters 16->64 ==")
	for l in arch.layers {
		fmt.printfln("  %s #%d: in=%v out=%v", layer_label(l.type), l.id, l.input_shape, l.output_shape)
	}

	// delete the first MaxPool and verify shapes still resolve
	for l, i in arch.layers {
		if l.type == .MaxPool {
			delete_layer_at(&arch, i)
			break
		}
	}

	fmt.println("\n== after deleting first MaxPool ==")
	for l in arch.layers {
		fmt.printfln("  %s #%d: in=%v out=%v", layer_label(l.type), l.id, l.input_shape, l.output_shape)
	}

	// add a Dense after Flatten
	for l, i in arch.layers {
		if l.type == .Flatten {
			new_id := add_layer_after(&arch, i, .Dense)
			fmt.println("\nadded Dense #", new_id, "after Flatten")
			break
		}
	}
	fmt.println("\n== after inserting Dense after Flatten ==")
	for l in arch.layers {
		fmt.printfln("  %s #%d: in=%v out=%v", layer_label(l.type), l.id, l.input_shape, l.output_shape)
	}
}

switch_demo :: proc(arch: ^Architecture, ui: ^UI_State, kind: Demo_Kind) {
	destroy_architecture(arch)
	arch^ = build_demo(kind)
	clear_history(ui.history)
	ui.panel_open = false
	ui.selected_layer = -1
	ui.current_demo = kind
	fit_camera(arch, f32(rl.GetScreenWidth()), f32(rl.GetScreenHeight()))
	show_toast(ui, fmt.tprintf("demo: %s", string(demo_name(kind))))
}

main :: proc() {
	// optional CLI:
	//   --shot <path>     render a few frames then save a PNG and exit
	//   --demo <name>     start in CNN/RNN/LSTM/GRU (case-insensitive)
	shot_path: string = ""
	save_test_path: string = ""
	start_demo := Demo_Kind.CNN
	open_panel_idx := -1
	open_picker_idx := -1
	{
		args := os.args
		for i := 1; i < len(args); i += 1 {
			if args[i] == "--shot" && i + 1 < len(args) {
				shot_path = args[i + 1]
				i += 1
			} else if args[i] == "--demo" && i + 1 < len(args) {
				switch args[i + 1] {
				case "cnn", "CNN":   start_demo = .CNN
				case "rnn", "RNN":   start_demo = .RNN
				case "lstm", "LSTM": start_demo = .LSTM
				case "gru", "GRU":   start_demo = .GRU
				}
				i += 1
			} else if args[i] == "--save-test" && i + 1 < len(args) {
				save_test_path = args[i + 1]
				i += 1
			} else if args[i] == "--panel" && i + 1 < len(args) {
				v := 0
				for c in args[i + 1] {
					if c >= '0' && c <= '9' do v = v * 10 + int(c - '0')
				}
				open_panel_idx = v
				i += 1
			} else if args[i] == "--shape-test" {
				run_shape_test()
				return
			} else if args[i] == "--picker" && i + 1 < len(args) {
				v := 0
				for c in args[i + 1] {
					if c >= '0' && c <= '9' do v = v * 10 + int(c - '0')
				}
				open_picker_idx = v
				i += 1
			}
		}
	}

	// non-interactive: save the chosen demo to JSON and exit
	if save_test_path != "" {
		tmp := build_demo(start_demo)
		defer destroy_architecture(&tmp)
		if !save_architecture(&tmp, save_test_path) {
			fmt.eprintln("save-test: failed")
			os.exit(1)
		}
		loaded, ok := load_architecture(save_test_path)
		if !ok {
			fmt.eprintln("save-test: re-load failed")
			os.exit(2)
		}
		defer destroy_architecture(&loaded)
		fmt.printfln(
			"save-test ok: demo=%s wrote=%s layers=%d/%d conns=%d/%d",
			demo_name(start_demo),
			save_test_path,
			len(tmp.layers),
			len(loaded.layers),
			len(tmp.connections),
			len(loaded.connections),
		)
		return
	}

	rl.SetConfigFlags({.WINDOW_RESIZABLE, .MSAA_4X_HINT})
	rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CNN/RNN Visualizer — MVP")
	defer rl.CloseWindow()
	rl.SetTargetFPS(60)

	arch := build_demo(start_demo)
	defer destroy_architecture(&arch)

	history := History{max_depth = 64}
	defer destroy_history(&history)

	ui := UI_State{
		dragging_layer      = -1,
		hovered_layer       = -1,
		selected_layer      = -1,
		hover_output_idx    = -1,
		connecting_from_id  = -1,
		selected_connection = -1,
		show_help           = true,
		current_demo        = start_demo,
		history             = &history,
	}

	if open_panel_idx >= 0 && open_panel_idx < len(arch.layers) {
		ui.selected_layer = arch.layers[open_panel_idx].id
		ui.panel_open = true
	}
	if open_picker_idx >= 0 && open_picker_idx < len(arch.layers) {
		ui.picker_after_id = arch.layers[open_picker_idx].id
		ui.picker_open = true
	}

	shot_frame := 0
	fitted := false

	for !rl.WindowShouldClose() {
		dt := rl.GetFrameTime()

		sw := f32(rl.GetScreenWidth())
		sh := f32(rl.GetScreenHeight())

		// keep camera offset centered on window even when resized
		arch.camera.offset = rl.Vector2{sw / 2, sh / 2}

		if !fitted {
			fit_camera(&arch, sw, sh)
			fitted = true
		}

		handle_input(&arch, &ui, dt)
		update_animation(&arch, dt)

		rl.BeginDrawing()
		rl.ClearBackground(rl.Color{18, 22, 30, 255})

		draw_background_grid(&arch)

		mouse := rl.GetMousePosition()
		world := screen_to_world(&arch, mouse)
		hover_conn := -1
		over_panel := mouse_over_panel(&ui)
		if !over_panel && !ui.picker_open && ui.hover_output_idx < 0 && ui.connecting_from_id < 0 && ui.hovered_layer < 0 {
			hover_conn = find_connection_at(&arch, world, 6 / arch.camera.zoom)
		}

		rl.BeginMode2D(arch.camera)
		draw_connections(&arch, ui.selected_connection, hover_conn)
		draw_self_loops(&arch)
		draw_pulses(&arch)
		draw_layers(&arch, ui.hovered_layer)
		draw_output_handles(&arch, ui.hover_output_idx)
		if ui.connecting_from_id >= 0 {
			draw_in_flight_connection(&arch, ui.connecting_from_id, world)
		}
		rl.EndMode2D()

		draw_ui_overlay(&arch, &ui)

		rl.EndDrawing()

		if shot_path != "" {
			shot_frame += 1
			if shot_frame == 30 {
				cpath := strings.clone_to_cstring(shot_path)
				defer delete(cpath)
				rl.TakeScreenshot(cpath)
				break
			}
		}
	}
}

draw_background_grid :: proc(arch: ^Architecture) {
	// faint grid lines anchored in world space
	rl.BeginMode2D(arch.camera)
	defer rl.EndMode2D()

	screen_w := f32(rl.GetScreenWidth())
	screen_h := f32(rl.GetScreenHeight())

	tl := rl.GetScreenToWorld2D(rl.Vector2{0, 0}, arch.camera)
	br := rl.GetScreenToWorld2D(rl.Vector2{screen_w, screen_h}, arch.camera)

	step: f32 = 80
	col := rl.Color{40, 48, 60, 255}

	start_x := math_floor_to(tl[0], step)
	for x := start_x; x < br[0]; x += step {
		rl.DrawLineEx(rl.Vector2{x, tl[1]}, rl.Vector2{x, br[1]}, 1.0 / arch.camera.zoom, col)
	}
	start_y := math_floor_to(tl[1], step)
	for y := start_y; y < br[1]; y += step {
		rl.DrawLineEx(rl.Vector2{tl[0], y}, rl.Vector2{br[0], y}, 1.0 / arch.camera.zoom, col)
	}
}

math_floor_to :: proc(v, step: f32) -> f32 {
	return f32(int(v / step)) * step - step
}
