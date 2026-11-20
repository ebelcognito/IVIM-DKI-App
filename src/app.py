"""IVIM-DKI Analysis app: graphical interface for the IDTV-AD method."""
import sys
import os
import queue
import threading
import types

import numpy as np
import nibabel as nib

import engine

if '--selftest' in sys.argv:          # headless check of the fitting engine (used to verify builds)
    import selftest
    sys.exit(selftest.main(sys.argv[sys.argv.index('--selftest') + 1:]))

import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
import os
import time
import pandas as pd
from tkinter import messagebox
from scipy import stats
import string

# Analysis state shared by the tabs (input data, settings, results)
analysis = types.SimpleNamespace(inp=None, affine=np.eye(4), spacing=(1.0, 1.0, 1.0), loaded_mask=None)
fit_queue = queue.Queue()
# --autorun <file.nii.gz>: load the file, run with the current (simulation) settings, save and close (build check)
AUTORUN = sys.argv[sys.argv.index('--autorun') + 1] if '--autorun' in sys.argv else None
fit_cancel = threading.Event()
current_angle = 0
run_counter = 1  # Counter for unique run identification



# Zoom state variables
output_zoom_mode = "pan"  # "pan", "rect_zoom", "click_zoom"
stats_zoom_mode = "pan"   # "pan", "rect_zoom", "click_zoom"
output_zoom_counter = 0
stats_zoom_counter = 0
output_rect_selector = None
stats_rect_selector = None
output_zoom_history = []  # Store zoom states for undo
stats_zoom_history = []   # Store zoom states for undo

def select_data():
    file_path = filedialog.askopenfilename(
        title="Choose IVIM-DKI NIfTI file",
        filetypes=[
            ("NIfTI files", "*.nii *.nii.gz"),
            ("All files", "*.*")
        ]
    )
    load_data_file(file_path)


def load_data_file(file_path):
    global img_data, slices, input_file_path, current_angle, current_nifti_path
    if file_path:
        try:
            current_nifti_path = file_path
            img = nib.load(file_path)
            img_data = img.get_fdata()
            print(f"Loaded data with shape: {img_data.shape}")
            
            # Handle different data dimensions
            if img_data.ndim == 4:
                slices = img_data.shape[2]  # 4D data (x,y,z,b)
            elif img_data.ndim == 3:
                slices = img_data.shape[2]  # 3D data (x,y,z)
            else:
                messagebox.showerror("Error", f"Unsupported data dimensions: {img_data.shape}")
                return
                
            analysis.inp = img_data  # Store the input data in the analysis object
            analysis.affine = img.affine
            analysis.spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
            analysis.loaded_mask = None
            mask_file_label.config(text="No mask file loaded")
            current_angle = 0  # Reset the rotation angle
            input_file_path = os.path.basename(file_path)
            load_bval_sidecar(file_path)
            data_info_label.config(text=f"{input_file_path}: {img_data.shape}, voxel "
                                        f"{' x '.join('%.2f' % v for v in analysis.spacing)} mm")
            
            # Configure slider
            slice_slider.config(to=slices-1, from_=0)
            slice_slider.set(0)
            
            # Update preview
            print("About to call update_preview(0)")
            update_preview(0)  # Show the first slice
            
            print(f"Successfully loaded {input_file_path} with {slices} slices")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load NIfTI file:\n{e}")
            print(f"Error loading file: {e}")

def load_bval_sidecar(nifti_path):
    """Fill the b-value field from <name>.bval or <name>.txt next to the NIfTI file, if present."""
    stem = nifti_path[:-7] if nifti_path.endswith('.nii.gz') else os.path.splitext(nifti_path)[0]
    for ext in ('.bval', '.txt'):
        if os.path.exists(stem + ext):
            values = np.loadtxt(stem + ext).ravel()
            b_values_entry.delete(0, tk.END)
            b_values_entry.insert(0, ', '.join('%g' % v for v in values))
            print(f"b-values read from {os.path.basename(stem + ext)}")
            return


def select_mask_file():
    if analysis.inp is None:
        messagebox.showwarning("No data", "Load the IVIM-DKI data before its mask.")
        return
    path = filedialog.askopenfilename(title="Choose fitting mask (NIfTI, same grid as the data)",
                                      filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")])
    if not path:
        return
    try:
        values = nib.load(path).get_fdata()
        if values.shape[:3] != analysis.inp.shape[:3]:
            raise ValueError(f"Mask shape {values.shape} does not match the data {analysis.inp.shape[:3]}.")
        analysis.loaded_mask = values[..., 0] > 0 if values.ndim == 4 else values > 0
        if not analysis.loaded_mask.any():
            raise ValueError("The mask is empty.")
    except Exception as e:
        analysis.loaded_mask = None
        messagebox.showerror("Mask error", str(e))
        return
    mask_mode_var.set("file")
    mask_file_label.config(text=f"{os.path.basename(path)} ({int(analysis.loaded_mask.sum()):,} voxels)")


def apply_preset(*_):
    initial_guess_entry.delete(0, tk.END)
    initial_guess_entry.insert(0, ", ".join('%g' % v for v in engine.INITIAL[tissue_var.get()]))
    lower_bounds_entry.delete(0, tk.END)
    lower_bounds_entry.insert(0, ", ".join('%g' % v for v in engine.LOWER[bounds_var.get()]))


def update_preview(val):
    global current_angle
    print(f"update_preview called with val={val}")
    # Check if img_data exists
    if 'img_data' not in globals() or img_data is None:
        print("No img_data available")
        return
        
    try:
        slice_idx = int(float(val))
        if img_data.ndim == 4:
            slice_data = img_data[:, :, slice_idx, 0]  # First 3D scan (4D data)
        elif img_data.ndim == 3:
            slice_data = img_data[:, :, slice_idx]  # 3D data
        else:
            print(f"Unexpected data dimensions: {img_data.shape}")
            return
            
        slice_data_np = np.array(slice_data)  # Ensure compatibility with Matplotlib
        
        # Rotate the image by the current angle
        slice_data_np = np.rot90(slice_data_np, k=current_angle // 90)

        # Normalize the data to range 0-1 for better visualization
        if np.max(slice_data_np) > np.min(slice_data_np):
            slice_data_np = (slice_data_np - np.min(slice_data_np)) / (np.max(slice_data_np) - np.min(slice_data_np))
        
        ax.clear()
        ax.imshow(slice_data_np, cmap='viridis')
        ax.axis('off')
        canvas.draw()
        
        # Update slice number label
        slice_number_label.config(text=f"Slice: {slice_idx + 1}/{slices}")
    except Exception as e:
        print(f"Error updating preview: {e}")
        ax.clear()
        ax.text(0.5, 0.5, f'Error loading slice {slice_idx + 1}', 
               ha='center', va='center', transform=ax.transAxes)
        canvas.draw()

def flip_preview():
    global current_angle
    current_angle = (current_angle + 90) % 360
    update_preview(slice_slider.get())

def store_parameters():
    def four(entry, name):
        values = np.fromstring(entry.get(), sep=',')
        if len(values) != 4:
            raise ValueError(f"{name} needs four comma-separated values (D, D*, f, k).")
        return values
    analysis.b_values = np.fromstring(b_values_entry.get(), sep=',')
    analysis.initial_guess = four(initial_guess_entry, "Initial guess")
    analysis.lower_bounds = four(lower_bounds_entry, "Lower bounds")
    analysis.upper_bounds = four(upper_bounds_entry, "Upper bounds")
    analysis.tv_weights = four(tv_weights_entry, "TV weights")
    analysis.iterations = int(iterations_entry.get())
    analysis.voxel_units = bool(voxel_units_var.get())
    analysis.precision = precision_var.get()
    analysis.mask_mode = mask_mode_var.get()
    analysis.slice_indices = parse_slice_string(slice_index_entry.get(), slices)


def build_fit_mask():
    data = analysis.inp
    if analysis.mask_mode == "file":
        if analysis.loaded_mask is None:
            raise ValueError("Choose a mask file first, or select another mask option.")
        mask = analysis.loaded_mask.copy()
    elif analysis.mask_mode == "auto":
        mask = engine.body_mask(data, analysis.spacing)
    else:
        mask = np.ones(data.shape[:3], bool)
    if analysis.slice_indices is not None:
        keep = np.zeros(data.shape[2], bool)
        keep[analysis.slice_indices] = True
        mask &= keep[None, None, :]
    return mask


def parse_slice_string(s, max_slice):
    """
    Turn a user string like  '5:14, 18, 21'
    into a sorted unique *zero-based* list   [4,5,…,13,17,20]
    """
    if not s.strip():                       # empty → process all slices
        return None

    idx = []
    for part in s.split(','):
        part = part.strip()
        if ':' in part:                     # range "m:n"
            lo, hi = map(int, part.split(':'))
            idx.extend(range(lo-1, hi))     # user counts from 1 → convert to 0-based
        else:                               # single integer
            idx.append(int(part)-1)
    # keep in-bounds, unique, sorted
    idx = sorted({i for i in idx if 0 <= i < max_slice})
    return idx or None

def dict_to_paramcube(maps):
    """Convert parameter dict to 4D array (H,W,Z,4) for UI compatibility."""
    return np.stack([maps['D'], maps['D_star'], 
                     maps['f'], maps['K']], axis=-1)

def run_analysis():
    if analysis.inp is None or analysis.inp.ndim != 4:
        messagebox.showwarning("No data", "Load a 4D IVIM-DKI NIfTI file (x, y, z, b-value) first.")
        return
    try:
        store_parameters()
        mask = build_fit_mask()
    except Exception as e:
        messagebox.showerror("Settings error", str(e))
        return
    analysis.fit_mask = mask
    fit_cancel.clear()
    run_button.config(state='disabled')
    stop_button.config(state='normal')
    progress_bar['value'] = 0
    status_label.config(text=f"Fitting {int(mask.sum()):,} voxels ... (first block includes compilation)")

    def worker():
        try:
            maps, info = engine.fit(
                analysis.inp, analysis.b_values, mask, analysis.spacing, analysis.initial_guess,
                analysis.lower_bounds, analysis.upper_bounds, weights=analysis.tv_weights,
                iterations=analysis.iterations, voxel_units=analysis.voxel_units, precision=analysis.precision,
                progress=lambda done, total, cost: fit_queue.put(('progress', done, total, cost)),
                cancel=fit_cancel)
            fit_queue.put(('done', maps, info))
        except Exception as e:
            fit_queue.put(('error', e))

    threading.Thread(target=worker, daemon=True).start()
    root.after(200, poll_fit)


def poll_fit():
    global run_counter
    try:
        while True:
            msg = fit_queue.get_nowait()
            if msg[0] == 'progress':
                _, done, total, cost = msg
                progress_bar['value'] = 100 * done / total
                status_label.config(text=f"Iteration {done}/{total}, cost {cost:.4g}")
            elif msg[0] == 'error':
                run_button.config(state='normal'); stop_button.config(state='disabled')
                status_label.config(text="Fit failed")
                messagebox.showerror("Fit failed", str(msg[1]))
                return
            elif msg[0] == 'done':
                _, maps, info = msg
                run_button.config(state='normal'); stop_button.config(state='disabled')
                analysis.fit_info = info
                analysis.Para = np.nan_to_num(maps, nan=0.0)      # 0 outside the fitting mask
                progress_bar['value'] = 100
                time_label.config(text=f"Time taken: {info['seconds']:.2f} s")
                status_label.config(text=f"{info['status']} after {info['iterations']} iterations "
                                         f"({info['voxels']:,} voxels, {info['backend'].upper()})")
                save_parameter_maps_to_folder(analysis.Para, info['seconds'])
                run_counter += 1
                return
    except queue.Empty:
        pass
    root.after(200, poll_fit)


def stop_analysis():
    fit_cancel.set()
    status_label.config(text="Stopping after the current block of iterations ...")


def save_parameter_maps_to_folder(para_data, time_taken):
    """Automatically save parameter maps to a uniquely named folder"""
    global run_counter
    
    # Get base filename without extension
    base_name = os.path.splitext(input_file_path)[0] if 'input_file_path' in globals() else "unknown"
    
    base_name = base_name[:-4] if base_name.endswith('.nii') else base_name
    iterations = analysis.fit_info['iterations']
    import datetime
    folder_name = f"{base_name}_IDTV-AD_{iterations}its_{datetime.datetime.now():%Y%m%d-%H%M%S}_{run_counter:03d}"

    # Create main results directory and specific folder for this run
    results_dir = "Analysis_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    output_folder = os.path.join(results_dir, folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save parameter maps
    param_names = ['D', 'D_star', 'f', 'K']
    saved_files = []
    
    for i, param in enumerate(param_names):
        map_data = para_data[:, :, :, i]
        nii_img = nib.Nifti1Image(map_data.astype(np.float32), affine=analysis.affine)
        filename = f"{param}_map.nii"
        filepath = os.path.join(output_folder, filename)
        nib.save(nii_img, filepath)
        saved_files.append(filepath)
    
    nib.save(nib.Nifti1Image(analysis.fit_mask.astype(np.uint8), affine=analysis.affine),
             os.path.join(output_folder, "fit_mask.nii"))

    # Save composite PNG image of middle slices
    save_composite_parameter_image(para_data, output_folder, folder_name)
    
    # Save detailed run information
    info_file = os.path.join(output_folder, "analysis_info.txt")
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(info_file, 'w') as f:
        f.write(f"IVIM-DKI Analysis Run Information\n")
        f.write(f"=================================\n")
        f.write(f"Run ID: {run_counter:03d}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Folder Name: {folder_name}\n")
        f.write(f"\nMethod: IDTV-AD (bounded damped Gauss-Newton, free S0, per-parameter log-TV)\n")
        f.write(f"Status: {analysis.fit_info['status']}\n")
        f.write(f"Iterations: {analysis.fit_info['iterations']} of {analysis.iterations}\n")
        f.write(f"TV weights (D, D*, f, k): {[float(v) for v in analysis.tv_weights]}\n")
        f.write(f"TV units: {'voxel (through-plane scaled by slice/pixel ratio)' if analysis.voxel_units else 'mm'}\n")
        f.write(f"Precision: {analysis.precision}; device: {analysis.fit_info['backend']}\n")
        f.write(f"Fitting mask: {analysis.mask_mode} ({analysis.fit_info['voxels']} voxels)\n")
        f.write(f"Time Taken: {time_taken:.2f} seconds\n")
        f.write(f"\nInput Data:\n")
        f.write(f"Input File: {input_file_path if 'input_file_path' in globals() else 'unknown'}\n")
        f.write(f"Voxel size (mm): {list(analysis.spacing)}\n")
        f.write(f"Slice Indices: {analysis.slice_indices}\n")
        f.write(f"B-values: {[float(v) for v in analysis.b_values]}\n")
        f.write(f"Initial Guess: {[float(v) for v in analysis.initial_guess]}\n")
        f.write(f"Lower Bounds: {[float(v) for v in analysis.lower_bounds]}\n")
        f.write(f"Upper Bounds: {[float(v) for v in analysis.upper_bounds]}\n")
        f.write(f"\nSaved Parameter Maps:\n")
        for i, param in enumerate(param_names):
            f.write(f"  - {param}_map.nii\n")
        f.write(f"  - fit_mask.nii\n")
        f.write(f"  - {folder_name}_parameter_maps_slice_{para_data.shape[2] // 2 + 1}.png (composite image)\n")
    
    print(f"Parameter maps automatically saved to folder: {folder_name}")
    print(f"  Location: {output_folder}")
    print(f"  Files saved: {len(saved_files)} parameter maps + analysis info")
    
    # Show success message to user
    if AUTORUN:
        print(f"AUTORUN saved {output_folder}", flush=True)
        root.after(500, root.destroy)
        return
    messagebox.showinfo("Analysis Complete", 
                       f"Analysis completed successfully!\n\n"
                       f"Parameter maps saved to:\n{output_folder}\n\n"
                       f"Time taken: {time_taken:.2f} seconds")

def save_composite_parameter_image(para_data, output_folder, folder_name):
    """Save composite PNG image showing middle slices of all 4 parameter maps"""
    import matplotlib.pyplot as plt
    
    # Get middle slice index
    middle_slice = para_data.shape[2] // 2
    
    # Parameter information
    param_info = [
        {'name': 'D', 'data': para_data[:, :, middle_slice, 0], 'vmin': 0, 'vmax': 0.003, 'units': 'mm²/s', 'cmap': 'viridis'},
        {'name': 'D*', 'data': para_data[:, :, middle_slice, 1], 'vmin': 0, 'vmax': 0.03, 'units': 'mm²/s', 'cmap': 'viridis'},
        {'name': 'f', 'data': para_data[:, :, middle_slice, 2], 'vmin': 0, 'vmax': 0.8, 'units': 'unitless', 'cmap': 'viridis'},
        {'name': 'K', 'data': para_data[:, :, middle_slice, 3], 'vmin': 0, 'vmax': 3, 'units': 'unitless', 'cmap': 'viridis'}
    ]
    
    # Create figure with 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'IVIM-DKI Parameter Maps - Middle Slice ({middle_slice + 1})\n{folder_name}', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, param in enumerate(param_info):
        ax = axes_flat[i]
        
        # Apply rotation (same as in UI)
        slice_data = np.rot90(param['data'], k=current_angle // 90)
        
        # Handle NaN or zero data
        if np.all(np.isnan(slice_data)) or np.all(slice_data == 0):
            ax.text(0.5, 0.5, f'No valid data\nfor {param["name"]} parameter', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{param["name"]} map (No Data)', fontsize=14, fontweight='bold')
            ax.axis('off')
            continue
        
        # Plot the parameter map
        im = ax.imshow(slice_data, cmap=param['cmap'], vmin=param['vmin'], vmax=param['vmax'], 
                      aspect='equal', interpolation='nearest')
        ax.set_title(f'{param["name"]} map', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label(f'{param["name"]} ({param["units"]})', rotation=270, labelpad=15)
        
        # Add data range information
        with np.errstate(invalid='ignore'):
            data_min, data_max = np.nanmin(slice_data), np.nanmax(slice_data)
        
        if not (np.isnan(data_min) or np.isnan(data_max)):
            ax.text(0.02, 0.98, f'Range: [{data_min:.4f}, {data_max:.4f}]', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the composite image
    composite_filename = f"{folder_name}_parameter_maps_slice_{middle_slice + 1}.png"
    composite_filepath = os.path.join(output_folder, composite_filename)
    
    plt.savefig(composite_filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()  # Close the figure to free memory
    
    print(f"Composite parameter map image saved: {composite_filename}")

def save_parameter_maps_auto(para_data, time_taken):
    """Legacy function - redirects to new folder-based saving"""
    save_parameter_maps_to_folder(para_data, time_taken)

def reset_colorbar_ranges():
    """Reset colorbar ranges to default values"""
    d_min_entry.delete(0, tk.END)
    d_min_entry.insert(0, "0")
    d_max_entry.delete(0, tk.END)
    d_max_entry.insert(0, "0.003")
    
    dstar_min_entry.delete(0, tk.END)
    dstar_min_entry.insert(0, "0")
    dstar_max_entry.delete(0, tk.END)
    dstar_max_entry.insert(0, "0.03")
    
    f_min_entry.delete(0, tk.END)
    f_min_entry.insert(0, "0")
    f_max_entry.delete(0, tk.END)
    f_max_entry.insert(0, "0.8")
    
    k_min_entry.delete(0, tk.END)
    k_min_entry.insert(0, "0")
    k_max_entry.delete(0, tk.END)
    k_max_entry.insert(0, "3")
    
    # Update the display
    update_output_preview(output_slice_slider.get())

def get_colorbar_range(selected_map):
    """Get the colorbar range from UI entries"""
    try:
        if selected_map == 'D':
            vmin = float(d_min_entry.get())
            vmax = float(d_max_entry.get())
            units = 'mm²/s'
        elif selected_map == 'D*':
            vmin = float(dstar_min_entry.get())
            vmax = float(dstar_max_entry.get())
            units = 'mm²/s'
        elif selected_map == 'f':
            vmin = float(f_min_entry.get())
            vmax = float(f_max_entry.get())
            units = 'unitless'
        elif selected_map == 'k':
            vmin = float(k_min_entry.get())
            vmax = float(k_max_entry.get())
            units = 'unitless'
        else:
            return 0, 1, 'unitless'
        
        # Ensure vmin < vmax
        if vmin >= vmax:
            vmax = vmin + 0.001
            
        return vmin, vmax, units
    except ValueError:
        # If there's an error in parsing, use defaults
        if selected_map == 'D':
            return 0, 0.003, 'mm²/s'
        elif selected_map == 'D*':
            return 0, 0.03, 'mm²/s'
        elif selected_map == 'f':
            return 0, 0.8, 'unitless'
        elif selected_map == 'k':
            return 0, 3, 'unitless'
        else:
            return 0, 1, 'unitless'

def update_output_preview(val):
    global current_angle
    slice_idx = int(float(val))
    selected_map = output_map_var.get()
    if selected_map == 'D':
        map_data = analysis.Para[:, :, :, 0]
    elif selected_map == 'D*':
        map_data = analysis.Para[:, :, :, 1]
    elif selected_map == 'f':
        map_data = analysis.Para[:, :, :, 2]
    elif selected_map == 'k':
        map_data = analysis.Para[:, :, :, 3]
    else:
        return

    # Get colorbar range from UI
    vmin, vmax, units = get_colorbar_range(selected_map)
    cmap_name = 'viridis'

    slice_data = map_data[:, :, slice_idx]
    slice_data_np = np.array(slice_data)  # Convert JAX array to NumPy array
    
    # Check if slice has valid data
    if np.all(np.isnan(slice_data_np)) or np.all(slice_data_np == 0):
        # Handle all-NaN or all-zero slices
        ax_output.clear()
        ax_output.text(0.5, 0.5, f'No valid data in slice {slice_idx + 1}\nfor {selected_map} parameter', 
                      ha='center', va='center', transform=ax_output.transAxes, fontsize=12)
        ax_output.set_title(f'{selected_map} map - Slice {slice_idx + 1} (No Data)', fontsize=12, pad=10)
        ax_output.axis('off')
        
        # Remove existing colorbar safely
        if hasattr(update_output_preview, 'colorbar') and update_output_preview.colorbar is not None:
            try:
                update_output_preview.colorbar.remove()
                update_output_preview.colorbar = None
            except (AttributeError, ValueError):
                pass
        
        # Maintain subplot positioning
        fig_output.subplots_adjust(right=0.85)
        canvas_output.draw()
        output_slice_number_label.config(text=f"Slice: {slice_idx + 1}/{slices}")
        return
    
    # Rotate the image by the current angle
    slice_data_np = np.rot90(slice_data_np, k=current_angle // 90)

    # Store current axis limits before clearing (for zoom preservation)
    if hasattr(ax_output, '_xlim') and hasattr(ax_output, '_ylim'):
        current_xlim, current_ylim = ax_output._xlim, ax_output._ylim
    else:
        current_xlim, current_ylim = None, None
    
    # Clear only the plot content, not the axes
    ax_output.clear()
    
    # Plot with specified colormap and range
    im = ax_output.imshow(slice_data_np, cmap=cmap_name, vmin=vmin, vmax=vmax, aspect='equal')
    ax_output.axis('off')
    ax_output.set_title(f'{selected_map} map - Slice {slice_idx + 1}', fontsize=12, pad=10)
    
    # Restore zoom if it was set
    if current_xlim is not None and current_ylim is not None:
        ax_output.set_xlim(current_xlim)
        ax_output.set_ylim(current_ylim)
        ax_output._xlim, ax_output._ylim = current_xlim, current_ylim
    else:
        # Store default limits
        ax_output._xlim = ax_output.get_xlim()
        ax_output._ylim = ax_output.get_ylim()
    
    # Handle colorbar more carefully
    if hasattr(update_output_preview, 'colorbar') and update_output_preview.colorbar is not None:
        try:
            # Update existing colorbar instead of removing/recreating
            update_output_preview.colorbar.mappable.set_array(slice_data_np)
            update_output_preview.colorbar.mappable.set_clim(vmin, vmax)
            update_output_preview.colorbar.set_label(f'{selected_map} ({units})', rotation=270, labelpad=15)
            update_output_preview.colorbar.draw_all()
        except (AttributeError, ValueError):
            # If updating fails, remove and recreate
            try:
                update_output_preview.colorbar.remove()
            except:
                pass
            update_output_preview.colorbar = None
    
    # Create new colorbar if none exists
    if update_output_preview.colorbar is None:
        try:
            # Create colorbar in a specific position to maintain layout
            cbar_ax = fig_output.add_axes([0.87, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
            update_output_preview.colorbar = fig_output.colorbar(im, cax=cbar_ax)
            update_output_preview.colorbar.set_label(f'{selected_map} ({units})', rotation=270, labelpad=15)
        except Exception as e:
            print(f"Warning: Could not create colorbar: {e}")
            update_output_preview.colorbar = None
    
    # Add data range information to the plot (with NaN handling)
    with np.errstate(invalid='ignore'):
        data_min, data_max = np.nanmin(slice_data_np), np.nanmax(slice_data_np)
    
    if not (np.isnan(data_min) or np.isnan(data_max)):
        ax_output.text(0.02, 0.98, f'Data range: [{data_min:.4f}, {data_max:.4f}]', 
                       transform=ax_output.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Maintain consistent subplot positioning - DO NOT call tight_layout()
    fig_output.subplots_adjust(right=0.85)
    canvas_output.draw()
    
    # Update slice number label
    output_slice_number_label.config(text=f"Slice: {slice_idx + 1}/{slices}")

# Initialize colorbar attribute
update_output_preview.colorbar = None

def on_input_slider_move(val):
    update_preview(val)

def on_output_slider_move(val):
    update_output_preview(val)

def display_map(map_type):
    if not hasattr(analysis, 'Para'):
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")])
        if file_path:
            img = nib.load(file_path)
            map_data = img.get_fdata()
            # Create a 4D array to store all parameter maps
            analysis.Para = np.zeros((*map_data.shape, 4), dtype=np.float32)
            if map_type == 'D':
                analysis.Para[:, :, :, 0] = map_data
            elif map_type == 'D*':
                analysis.Para[:, :, :, 1] = map_data
            elif map_type == 'f':
                analysis.Para[:, :, :, 2] = map_data
            elif map_type == 'k':
                analysis.Para[:, :, :, 3] = map_data
        else:
            return

    output_map_var.set(map_type)
    slices = analysis.Para.shape[2]  # Assuming 3D data
    output_slice_slider.config(to=slices-1, from_=0)
    output_slice_slider.set(0)  # Reset to first slice
    
    # Clear any existing colorbar and reset zoom state
    if hasattr(update_output_preview, 'colorbar') and update_output_preview.colorbar is not None:
        try:
            update_output_preview.colorbar.remove()
            update_output_preview.colorbar = None
        except:
            pass
    
    # Reset zoom state
    if hasattr(ax_output, '_xlim'):
        delattr(ax_output, '_xlim')
    if hasattr(ax_output, '_ylim'):
        delattr(ax_output, '_ylim')
    
    # Reset subplot layout
    fig_output.subplots_adjust(right=0.85)
    
    update_output_preview(0)

def save_maps():
    if hasattr(analysis, 'Para'):
        base_name = os.path.splitext(input_file_path)[0]
        param_names = ['D', 'D_star', 'f', 'k']
        output_dir = "Output_Parameter_Maps"  # Corrected directory name

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, param in enumerate(param_names):
            map_data = analysis.Para[:, :, :, i]
            nii_img = nib.Nifti1Image(map_data.astype(np.float32), affine=analysis.affine)
            nib.save(nii_img, os.path.join(output_dir, f"{base_name}_{param}.nii"))
        print("Maps saved successfully")
    else:
        print("No parameter maps to save")

# New functions for statistics
def select_parameter_map():
    global parameter_map, parameter_slices, parameter_map_path
    file_path = filedialog.askopenfilename(
        filetypes=[("NIfTI files", "*.*")],
        initialdir="Output_Parameter_Maps"
    )
    if file_path:
        img = nib.load(file_path)
        parameter_map = img.get_fdata()
        parameter_slices = parameter_map.shape[2]  # Assuming 3D data
        parameter_map_path = file_path  # Store the path for later use
        update_statistics_preview(0)  # Show the first slice
        statistics_slice_slider.config(to=parameter_slices-1, from_=0)
        statistics_slice_slider.set(0)

def select_roi():
    global roi_mask
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.*")])
    if file_path:
        img = nib.load(file_path)
        roi_mask = img.get_fdata()
        update_statistics_preview(statistics_slice_slider.get())  # Update preview with ROI mask applied

def update_statistics_preview(val):
    slice_idx = int(float(val))
    if 'parameter_map' in globals() and 'roi_mask' in globals():
        slice_data = parameter_map[:, :, slice_idx] * roi_mask[:, :, slice_idx]  # Apply mask
    elif 'parameter_map' in globals():
        slice_data = parameter_map[:, :, slice_idx]
    else:
        return
    slice_data_np = np.array(slice_data)  # Ensure compatibility with Matplotlib
    
    # Normalize the data to range 0-1 for better visualization
    slice_data_np = (slice_data_np - np.min(slice_data_np)) / (np.max(slice_data_np) - np.min(slice_data_np))
    
    ax_statistics.clear()
    ax_statistics.imshow(slice_data_np, cmap='viridis')
    ax_statistics.axis('off')
    canvas_statistics.draw()
    
    # Update slice number label
    statistics_slice_number_label.config(text=f"Slice: {slice_idx + 1}/{parameter_slices}")


def calculate_and_save_statistics():
    if 'parameter_map' in globals() and 'roi_mask' in globals() and 'parameter_map_path' in globals():
        try:
            # Extract masked data
            masked_data = parameter_map[roi_mask > 0]

            if masked_data.size == 0:
                messagebox.showwarning("Warning", "The ROI mask does not select any data.")
                print("The ROI mask does not select any data.")
                return

            # Calculate Mean and Standard Deviation
            mean_value = np.mean(masked_data)
            std_value = np.std(masked_data)

            # Calculate Skewness and Kurtosis
            skewness = stats.skew(masked_data)
            kurtosis = stats.kurtosis(masked_data)

            # Calculate Entropy
            # Compute histogram with 256 bins
            histogram, bin_edges = np.histogram(masked_data, bins=256, density=True)
            # Avoid zero probabilities
            histogram = histogram + 1e-12
            entropy = stats.entropy(histogram, base=2)  # Using base 2 for entropy in bits

            # Prepare statistics data
            statistics_data = {
                "Mean": [mean_value],
                "Standard Deviation": [std_value],
                "Skewness": [skewness],
                "Kurtosis": [kurtosis],
                "Entropy (bits)": [entropy]
            }

            df = pd.DataFrame(statistics_data)

            # Determine the Output_Statistics directory relative to the parameter_map_path
            parameter_map_dir = os.path.dirname(parameter_map_path)
            parent_dir = os.path.dirname(parameter_map_dir)
            output_dir = os.path.join(parent_dir, "Output_Statistics")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            base_name = os.path.splitext(os.path.basename(parameter_map_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}_statistics.xlsx")

            # Save to Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Statistics')

            print(f"Statistics saved to {output_file}")
            messagebox.showinfo("Success", f"Statistics saved to {output_file}")

        except Exception as e:
            print(f"Error saving statistics: {e}")
            messagebox.showerror("Error", f"Failed to save statistics:\n{e}")
    else:
        print("Parameter map or ROI mask not selected")
        messagebox.showwarning("Warning",
                               "Please select both a parameter map and an ROI mask before calculating statistics.")

# ==================== ZOOM FUNCTIONALITY ====================

def get_unique_zoom_id(counter):
    """Generate unique zoom identifier (A, B, C, ..., Z, AA, AB, ...)"""
    if counter < 26:
        return string.ascii_uppercase[counter]
    else:
        first = string.ascii_uppercase[(counter - 26) // 26]
        second = string.ascii_uppercase[(counter - 26) % 26]
        return first + second

def save_current_zoom_state(ax, history_list):
    """Save current zoom state to history"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    history_list.append((xlim, ylim))

def restore_zoom_state(ax, xlim, ylim):
    """Restore zoom state"""
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# ==================== OUTPUT TAB ZOOM FUNCTIONS ====================

def toggle_output_zoom_mode(mode):
    """Toggle zoom mode for output tab"""
    global output_zoom_mode, output_rect_selector
    output_zoom_mode = mode
    
    # Update button states
    output_pan_btn.config(relief='sunken' if mode == 'pan' else 'raised')
    output_rect_zoom_btn.config(relief='sunken' if mode == 'rect_zoom' else 'raised')
    output_click_zoom_btn.config(relief='sunken' if mode == 'click_zoom' else 'raised')
    
    # Handle rectangle selector
    if mode == 'rect_zoom':
        if output_rect_selector is None:
            output_rect_selector = RectangleSelector(
                ax_output, on_output_rect_select,
                useblit=True, button=[1], minspanx=5, minspany=5,
                spancoords='pixels', interactive=False
            )
        output_rect_selector.set_active(True)
    else:
        if output_rect_selector is not None:
            output_rect_selector.set_active(False)

def on_output_rect_select(eclick, erelease):
    """Handle rectangle selection for zoom"""
    if output_zoom_mode == 'rect_zoom':
        save_current_zoom_state(ax_output, output_zoom_history)
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])
        ax_output.set_xlim(x1, x2)
        ax_output.set_ylim(y1, y2)
        # Store the new limits
        ax_output._xlim = ax_output.get_xlim()
        ax_output._ylim = ax_output.get_ylim()
        canvas_output.draw()

def on_output_click(event):
    """Handle click events for zoom"""
    global output_zoom_mode
    if event.inaxes != ax_output or event.button != 1:
        return
        
    if output_zoom_mode == 'click_zoom':
        if event.dblclick:  # Double-click to zoom in
            save_current_zoom_state(ax_output, output_zoom_history)
            xlim = ax_output.get_xlim()
            ylim = ax_output.get_ylim()
            
            # Zoom in by factor of 2 around click point
            zoom_factor = 0.5
            x_center, y_center = event.xdata, event.ydata
            x_range = (xlim[1] - xlim[0]) * zoom_factor / 2
            y_range = (ylim[1] - ylim[0]) * zoom_factor / 2
            
            ax_output.set_xlim(x_center - x_range, x_center + x_range)
            ax_output.set_ylim(y_center - y_range, y_center + y_range)
            # Store the new limits
            ax_output._xlim = ax_output.get_xlim()
            ax_output._ylim = ax_output.get_ylim()
            canvas_output.draw()

def reset_output_zoom():
    """Reset zoom to original view"""
    global output_zoom_history
    if hasattr(analysis, 'Para') and analysis.Para is not None:
        slice_idx = int(output_slice_slider.get())
        selected_map = output_map_var.get()
        if selected_map in ['D', 'D*', 'f', 'k']:
            # Reset to full image extent
            try:
                height, width = analysis.Para.shape[:2]
                ax_output.set_xlim(0, width)
                ax_output.set_ylim(height, 0)  # Inverted for image display
                # Store the reset limits
                ax_output._xlim = ax_output.get_xlim()
                ax_output._ylim = ax_output.get_ylim()
                canvas_output.draw()
                output_zoom_history.clear()
            except (AttributeError, IndexError):
                print("Warning: Could not reset zoom - invalid data shape")
    else:
        print("Warning: No parameter maps available for zoom reset")

def undo_output_zoom():
    """Undo last zoom operation"""
    global output_zoom_history
    if output_zoom_history:
        xlim, ylim = output_zoom_history.pop()
        restore_zoom_state(ax_output, xlim, ylim)
        # Store the restored limits
        ax_output._xlim = xlim
        ax_output._ylim = ylim
        canvas_output.draw()

def save_output_zoom():
    """Save current zoomed view"""
    global output_zoom_counter
    if not hasattr(analysis, 'Para'):
        messagebox.showwarning("Warning", "No parameter maps available to save.")
        return
        
    selected_map = output_map_var.get()
    if not selected_map:
        messagebox.showwarning("Warning", "Please select a parameter map first.")
        return
    
    # Generate unique filename
    base_name = os.path.splitext(input_file_path)[0] if 'input_file_path' in globals() else "unknown"
    zoom_id = get_unique_zoom_id(output_zoom_counter)
    filename = f"{base_name}_{selected_map}_map_zoom_{zoom_id}.png"
    
    # Create output directory
    zoom_dir = "Zoomed_Views"
    if not os.path.exists(zoom_dir):
        os.makedirs(zoom_dir)
    
    filepath = os.path.join(zoom_dir, filename)
    
    # Save with consistent layout
    fig_output.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    output_zoom_counter += 1
    
    messagebox.showinfo("Success", f"Zoomed view saved as:\n{filepath}")
    print(f"Saved zoomed view: {filepath}")

# ==================== STATISTICS TAB ZOOM FUNCTIONS ====================

def toggle_stats_zoom_mode(mode):
    """Toggle zoom mode for statistics tab"""
    global stats_zoom_mode, stats_rect_selector
    stats_zoom_mode = mode
    
    # Update button states
    stats_pan_btn.config(relief='sunken' if mode == 'pan' else 'raised')
    stats_rect_zoom_btn.config(relief='sunken' if mode == 'rect_zoom' else 'raised')
    stats_click_zoom_btn.config(relief='sunken' if mode == 'click_zoom' else 'raised')
    
    # Handle rectangle selector
    if mode == 'rect_zoom':
        if stats_rect_selector is None:
            stats_rect_selector = RectangleSelector(
                ax_statistics, on_stats_rect_select,
                useblit=True, button=[1], minspanx=5, minspany=5,
                spancoords='pixels', interactive=False
            )
        stats_rect_selector.set_active(True)
    else:
        if stats_rect_selector is not None:
            stats_rect_selector.set_active(False)

def on_stats_rect_select(eclick, erelease):
    """Handle rectangle selection for zoom"""
    if stats_zoom_mode == 'rect_zoom':
        save_current_zoom_state(ax_statistics, stats_zoom_history)
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])
        ax_statistics.set_xlim(x1, x2)
        ax_statistics.set_ylim(y1, y2)
        # Store the new limits
        ax_statistics._xlim = ax_statistics.get_xlim()
        ax_statistics._ylim = ax_statistics.get_ylim()
        canvas_statistics.draw()

def on_stats_click(event):
    """Handle click events for zoom"""
    global stats_zoom_mode
    if event.inaxes != ax_statistics or event.button != 1:
        return
        
    if stats_zoom_mode == 'click_zoom':
        if event.dblclick:  # Double-click to zoom in
            save_current_zoom_state(ax_statistics, stats_zoom_history)
            xlim = ax_statistics.get_xlim()
            ylim = ax_statistics.get_ylim()
            
            # Zoom in by factor of 2 around click point
            zoom_factor = 0.5
            x_center, y_center = event.xdata, event.ydata
            x_range = (xlim[1] - xlim[0]) * zoom_factor / 2
            y_range = (ylim[1] - ylim[0]) * zoom_factor / 2
            
            ax_statistics.set_xlim(x_center - x_range, x_center + x_range)
            ax_statistics.set_ylim(y_center - y_range, y_center + y_range)
            # Store the new limits
            ax_statistics._xlim = ax_statistics.get_xlim()
            ax_statistics._ylim = ax_statistics.get_ylim()
            canvas_statistics.draw()

def reset_stats_zoom():
    """Reset zoom to original view"""
    global stats_zoom_history
    if 'parameter_map' in globals():
        # Reset to full image extent
        ax_statistics.set_xlim(0, parameter_map.shape[1])
        ax_statistics.set_ylim(parameter_map.shape[0], 0)  # Inverted for image display
        canvas_statistics.draw()
        stats_zoom_history.clear()

def undo_stats_zoom():
    """Undo last zoom operation"""
    global stats_zoom_history
    if stats_zoom_history:
        xlim, ylim = stats_zoom_history.pop()
        restore_zoom_state(ax_statistics, xlim, ylim)
        # Store the restored limits
        ax_statistics._xlim = xlim
        ax_statistics._ylim = ylim
        canvas_statistics.draw()

def save_stats_zoom():
    """Save current zoomed view"""
    global stats_zoom_counter
    if 'parameter_map_path' not in globals():
        messagebox.showwarning("Warning", "No parameter map loaded.")
        return
    
    # Generate unique filename
    base_name = os.path.splitext(os.path.basename(parameter_map_path))[0]
    zoom_id = get_unique_zoom_id(stats_zoom_counter)
    filename = f"{base_name}_zoom_{zoom_id}.png"
    
    # Create output directory
    zoom_dir = "Zoomed_Views"
    if not os.path.exists(zoom_dir):
        os.makedirs(zoom_dir)
    
    filepath = os.path.join(zoom_dir, filename)
    
    # Save the current view
    fig_statistics.savefig(filepath, dpi=300, bbox_inches='tight')
    stats_zoom_counter += 1
    
    messagebox.showinfo("Success", f"Zoomed view saved as:\n{filepath}")
    print(f"Saved zoomed view: {filepath}")

# ==================== END ZOOM FUNCTIONALITY ====================

# ------------------ Statistics Tab Content ------------------

root = tk.Tk()                       # ➜ create the default root
root.title("IVIM-DKI Analysis (IDTV-AD)")
root.geometry("1150x930")

# Create a notebook (tabs)
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')

# Create frames for each tab
input_frame = ttk.Frame(notebook)
output_frame = ttk.Frame(notebook)
statistics_frame = ttk.Frame(notebook)
about_frame = ttk.Frame(notebook)

# Add frames to notebook
notebook.add(input_frame, text="Input")
notebook.add(output_frame, text="Output")
notebook.add(statistics_frame, text="Statistics")
notebook.add(about_frame, text="About")

# ------------------ Input Tab Content ------------------
input_label_frame = tk.LabelFrame(input_frame, text="Input data and Analysis")
input_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

preview_frame = tk.Frame(input_label_frame)
preview_frame.pack(side=tk.LEFT, padx=10, pady=10, fill='both', expand=True)

controls_frame = tk.Frame(input_label_frame)
controls_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill='y')  # Changed to fill 'y' to prevent expansion horizontally

select_data_button = tk.Button(controls_frame, text="Select IVIM-DKI data", command=select_data)
select_data_button.pack(pady=5, fill='x')
data_info_label = tk.Label(controls_frame, text="No data loaded", fg='gray', wraplength=320, justify='left')
data_info_label.pack(anchor='w')

# Set up Matplotlib figure and canvas for input preview
fig = Figure(figsize=(4, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=preview_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

slice_slider = ttk.Scale(controls_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=on_input_slider_move)
slice_slider.pack(pady=5, fill='x')

# Label to display the current slice number for input preview
slice_number_label = tk.Label(preview_frame, text="Slice: 0/0")
slice_number_label.pack(pady=5)

# Flip button
flip_button = tk.Button(preview_frame, text="Flip 90°", command=flip_preview)
flip_button.pack(pady=5)

slice_index_label = tk.Label(controls_frame,
    text="Slice numbers  (e.g. 5:14  or 5,7,9):")
slice_index_label.pack(anchor='w')

slice_index_entry = tk.Entry(controls_frame)
slice_index_entry.pack(fill='x')

# Input fields for parameters
b_values_label = tk.Label(controls_frame, text="B-values (comma separated):")
b_values_label.pack(anchor='w')
b_values_entry = tk.Entry(controls_frame)
b_values_entry.pack(fill='x')
b_values_entry.insert(0, "0, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1250, 1500, 2000, 2500")

preset_frame = tk.Frame(controls_frame)
preset_frame.pack(fill='x', pady=(6, 2))
tk.Label(preset_frame, text="Tissue preset:").pack(side='left')
tissue_var = tk.StringVar(value="Simulation")
ttk.Combobox(preset_frame, textvariable=tissue_var, values=list(engine.INITIAL), state='readonly', width=16).pack(side='left', padx=4)
bounds_var = tk.StringVar(value="Simulation (D* floor 0.0001)")
ttk.Combobox(controls_frame, textvariable=bounds_var, values=list(engine.LOWER), state='readonly').pack(fill='x', pady=(0, 4))
tissue_var.trace_add('write', apply_preset)
bounds_var.trace_add('write', apply_preset)

initial_guess_label = tk.Label(controls_frame, text="Initial guesses D, D*, f, k (comma separated):")
initial_guess_label.pack(anchor='w')
initial_guess_entry = tk.Entry(controls_frame)
initial_guess_entry.pack(fill='x')
initial_guess_entry.insert(0, "0.0013,0.013,0.23,1.1")

lower_bounds_label = tk.Label(controls_frame, text="Lower bounds (comma separated):")
lower_bounds_label.pack(anchor='w')
lower_bounds_entry = tk.Entry(controls_frame)
lower_bounds_entry.pack(fill='x')
lower_bounds_entry.insert(0, "0.0001, 0.0001, 0.001, 0.01")

upper_bounds_label = tk.Label(controls_frame, text="Upper bounds (comma separated):")
upper_bounds_label.pack(anchor='w')
upper_bounds_entry = tk.Entry(controls_frame)
upper_bounds_entry.pack(fill='x')
upper_bounds_entry.insert(0, "0.005, 0.3, 1, 3")

tv_weights_label = tk.Label(controls_frame, text="TV weights D, D*, f, k:")
tv_weights_label.pack(anchor='w')
tv_weights_entry = tk.Entry(controls_frame)
tv_weights_entry.pack(fill='x')
tv_weights_entry.insert(0, ", ".join('%g' % v for v in engine.PAPER_WEIGHTS))

iterations_label = tk.Label(controls_frame, text="Iterations:")
iterations_label.pack(anchor='w')
iterations_entry = tk.Entry(controls_frame)
iterations_entry.pack(fill='x')
iterations_entry.insert(0, str(engine.PAPER_ITERATIONS))

voxel_units_var = tk.BooleanVar(value=True)
tk.Checkbutton(controls_frame, text="TV in voxel units (as in the paper)", variable=voxel_units_var).pack(anchor='w')
precision_var = tk.StringVar(value="float32")
precision_frame = tk.Frame(controls_frame)
precision_frame.pack(fill='x')
tk.Label(precision_frame, text="Precision:").pack(side='left')
tk.Radiobutton(precision_frame, text="single", variable=precision_var, value="float32").pack(side='left')
tk.Radiobutton(precision_frame, text="double", variable=precision_var, value="float64").pack(side='left')

mask_frame = tk.LabelFrame(controls_frame, text="Fitting mask", font=('Arial', 9, 'bold'))
mask_frame.pack(fill='x', pady=(8, 4))
mask_mode_var = tk.StringVar(value="whole")
tk.Radiobutton(mask_frame, text="Automatic body mask (patient data)", variable=mask_mode_var, value="auto").pack(anchor='w')
tk.Radiobutton(mask_frame, text="Whole volume (simulations)", variable=mask_mode_var, value="whole").pack(anchor='w')
mask_file_row = tk.Frame(mask_frame)
mask_file_row.pack(fill='x')
tk.Radiobutton(mask_file_row, text="Mask file:", variable=mask_mode_var, value="file").pack(side='left')
tk.Button(mask_file_row, text="Choose...", command=select_mask_file).pack(side='left', padx=4)
mask_file_label = tk.Label(mask_frame, text="No mask file loaded", fg='gray')
mask_file_label.pack(anchor='w', padx=20)

run_row = tk.Frame(controls_frame)
run_row.pack(pady=(10, 4), fill='x')
run_button = tk.Button(run_row, text="Run", command=run_analysis, bg='lightgreen')
run_button.pack(side='left', expand=True, fill='x')
stop_button = tk.Button(run_row, text="Stop", command=stop_analysis, state='disabled')
stop_button.pack(side='left', padx=(4, 0))

# Progress bar
progress_bar = ttk.Progressbar(controls_frame, orient="horizontal", length=200, mode="determinate")
progress_bar.pack(pady=5, fill='x')

# Time taken label
time_label = tk.Label(controls_frame, text="Time taken: 0.00 seconds")
time_label.pack(pady=2)
status_label = tk.Label(controls_frame, text="", fg='gray', wraplength=320, justify='left')
status_label.pack(pady=2)

# ------------------ Output Tab Content ------------------
output_label_frame = tk.LabelFrame(output_frame, text="IVIM-DKI with TV maps")
output_label_frame.pack(fill='both', expand=True, padx=10, pady=10)

output_note_label = tk.Label(output_label_frame, text="Note: You can either view parameter map after analysis is complete or select the desired parameter map from Output_ParameterMaps folder by clicking any buttons below.")
output_note_label.pack(pady=5)

output_controls_frame = tk.Frame(output_label_frame)
output_controls_frame.pack(pady=5)

output_map_var = tk.StringVar()

d_map_button = tk.Button(output_controls_frame, text="D map", command=lambda: display_map('D'))
d_map_button.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

d_star_map_button = tk.Button(output_controls_frame, text="D* map", command=lambda: display_map('D*'))
d_star_map_button.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

f_map_button = tk.Button(output_controls_frame, text="f map", command=lambda: display_map('f'))
f_map_button.grid(row=1, column=0, padx=5, pady=5, sticky='ew')

k_map_button = tk.Button(output_controls_frame, text="k map", command=lambda: display_map('k'))
k_map_button.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

# Make buttons expand equally
output_controls_frame.columnconfigure(0, weight=1)
output_controls_frame.columnconfigure(1, weight=1)

# Zoom controls for Output tab
output_zoom_frame = tk.Frame(output_label_frame)
output_zoom_frame.pack(pady=5, fill='x')

output_zoom_label = tk.Label(output_zoom_frame, text="Zoom Controls:")
output_zoom_label.pack(side='left', padx=(0, 10))

output_pan_btn = tk.Button(output_zoom_frame, text="Pan", relief='sunken',
                          command=lambda: toggle_output_zoom_mode('pan'))
output_pan_btn.pack(side='left', padx=2)

output_rect_zoom_btn = tk.Button(output_zoom_frame, text="Rect Zoom",
                                command=lambda: toggle_output_zoom_mode('rect_zoom'))
output_rect_zoom_btn.pack(side='left', padx=2)

output_click_zoom_btn = tk.Button(output_zoom_frame, text="Click Zoom",
                                 command=lambda: toggle_output_zoom_mode('click_zoom'))
output_click_zoom_btn.pack(side='left', padx=2)

output_reset_btn = tk.Button(output_zoom_frame, text="Reset",
                            command=reset_output_zoom)
output_reset_btn.pack(side='left', padx=2)

output_undo_btn = tk.Button(output_zoom_frame, text="Undo",
                           command=undo_output_zoom)
output_undo_btn.pack(side='left', padx=2)

output_save_zoom_btn = tk.Button(output_zoom_frame, text="Save Zoom", bg='lightgreen',
                                command=save_output_zoom)
output_save_zoom_btn.pack(side='right', padx=2)

# Colorbar controls for Output tab
output_colorbar_frame = tk.Frame(output_label_frame)
output_colorbar_frame.pack(pady=5, fill='x')

output_colorbar_label = tk.Label(output_colorbar_frame, text="Colorbar Ranges:")
output_colorbar_label.pack(side='left', padx=(0, 10))

# D range controls
d_range_frame = tk.Frame(output_colorbar_frame)
d_range_frame.pack(side='left', padx=5)
tk.Label(d_range_frame, text="D:", font=('Arial', 8)).pack(side='left')
d_min_entry = tk.Entry(d_range_frame, width=6, font=('Arial', 8))
d_min_entry.pack(side='left', padx=1)
d_min_entry.insert(0, "0")
tk.Label(d_range_frame, text="-", font=('Arial', 8)).pack(side='left')
d_max_entry = tk.Entry(d_range_frame, width=6, font=('Arial', 8))
d_max_entry.pack(side='left', padx=1)
d_max_entry.insert(0, "0.003")

# D* range controls
dstar_range_frame = tk.Frame(output_colorbar_frame)
dstar_range_frame.pack(side='left', padx=5)
tk.Label(dstar_range_frame, text="D*:", font=('Arial', 8)).pack(side='left')
dstar_min_entry = tk.Entry(dstar_range_frame, width=6, font=('Arial', 8))
dstar_min_entry.pack(side='left', padx=1)
dstar_min_entry.insert(0, "0")
tk.Label(dstar_range_frame, text="-", font=('Arial', 8)).pack(side='left')
dstar_max_entry = tk.Entry(dstar_range_frame, width=6, font=('Arial', 8))
dstar_max_entry.pack(side='left', padx=1)
dstar_max_entry.insert(0, "0.03")

# f range controls
f_range_frame = tk.Frame(output_colorbar_frame)
f_range_frame.pack(side='left', padx=5)
tk.Label(f_range_frame, text="f:", font=('Arial', 8)).pack(side='left')
f_min_entry = tk.Entry(f_range_frame, width=6, font=('Arial', 8))
f_min_entry.pack(side='left', padx=1)
f_min_entry.insert(0, "0")
tk.Label(f_range_frame, text="-", font=('Arial', 8)).pack(side='left')
f_max_entry = tk.Entry(f_range_frame, width=6, font=('Arial', 8))
f_max_entry.pack(side='left', padx=1)
f_max_entry.insert(0, "0.8")

# K range controls
k_range_frame = tk.Frame(output_colorbar_frame)
k_range_frame.pack(side='left', padx=5)
tk.Label(k_range_frame, text="K:", font=('Arial', 8)).pack(side='left')
k_min_entry = tk.Entry(k_range_frame, width=6, font=('Arial', 8))
k_min_entry.pack(side='left', padx=1)
k_min_entry.insert(0, "0")
tk.Label(k_range_frame, text="-", font=('Arial', 8)).pack(side='left')
k_max_entry = tk.Entry(k_range_frame, width=6, font=('Arial', 8))
k_max_entry.pack(side='left', padx=1)
k_max_entry.insert(0, "3")

# Apply colorbar settings button
apply_colorbar_btn = tk.Button(output_colorbar_frame, text="Apply", bg='lightblue',
                              command=lambda: update_output_preview(output_slice_slider.get()))
apply_colorbar_btn.pack(side='right', padx=5)

# Reset colorbar button
reset_colorbar_btn = tk.Button(output_colorbar_frame, text="Reset", 
                              command=reset_colorbar_ranges)
reset_colorbar_btn.pack(side='right', padx=2)

save_maps_button = tk.Button(output_label_frame, text="Save parameter maps", command=save_maps)
save_maps_button.pack(pady=10, fill='x')

output_preview_frame = tk.Frame(output_label_frame)
output_preview_frame.pack(pady=5, fill='both', expand=True)

# Set up Matplotlib figure and canvas for output preview
fig_output = Figure(figsize=(6, 4), dpi=100)
# Create subplot with space for colorbar
ax_output = fig_output.add_subplot(111)
fig_output.subplots_adjust(right=0.85)  # Leave space for colorbar
canvas_output = FigureCanvasTkAgg(fig_output, master=output_preview_frame)
canvas_output.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Connect zoom event handlers
canvas_output.mpl_connect('button_press_event', on_output_click)

output_slice_slider = ttk.Scale(output_label_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=on_output_slider_move)
output_slice_slider.pack(pady=5, fill='x')

# Label to display the current slice number for output preview
output_slice_number_label = tk.Label(output_preview_frame, text="Slice: 0/0")
output_slice_number_label.pack(pady=5)

# ------------------ Statistics Tab Content ------------------
statistics_label_frame = tk.LabelFrame(statistics_frame, text="Select Parameter Maps")
# Adjust pack options: fill horizontally, do not expand vertically
statistics_label_frame.pack(fill='x', padx=10, pady=5)

select_parameter_map_button = tk.Button(statistics_label_frame, text="Select Parameter Map", command=select_parameter_map)
# Make the button fill horizontally within the LabelFrame
select_parameter_map_button.pack(pady=5, padx=5, fill='x')

roi_label_frame = tk.LabelFrame(statistics_frame, text="Select ROI Masks")
# Adjust pack options: fill horizontally, do not expand vertically
roi_label_frame.pack(fill='x', padx=10, pady=5)

# Create a container frame within roi_label_frame for buttons
buttons_container = tk.Frame(roi_label_frame)
buttons_container.pack(pady=5, padx=5, fill='x')

# Select ROI Button
select_roi_button = tk.Button(buttons_container, text="Select ROI", command=select_roi)
select_roi_button.pack(side='left', expand=True, fill='x', padx=(0, 5))

# Calculate Stats Button
calculate_stats_button = tk.Button(buttons_container, text="Calculate ROI Stats & Save Statistics", command=calculate_and_save_statistics)
calculate_stats_button.pack(side='left', expand=True, fill='x', padx=(5, 0))

# Zoom controls for Statistics tab
stats_zoom_frame = tk.Frame(roi_label_frame)
stats_zoom_frame.pack(pady=5, padx=5, fill='x')

stats_zoom_label = tk.Label(stats_zoom_frame, text="Zoom Controls:")
stats_zoom_label.pack(side='left', padx=(0, 10))

stats_pan_btn = tk.Button(stats_zoom_frame, text="Pan", relief='sunken',
                         command=lambda: toggle_stats_zoom_mode('pan'))
stats_pan_btn.pack(side='left', padx=2)

stats_rect_zoom_btn = tk.Button(stats_zoom_frame, text="Rect Zoom",
                               command=lambda: toggle_stats_zoom_mode('rect_zoom'))
stats_rect_zoom_btn.pack(side='left', padx=2)

stats_click_zoom_btn = tk.Button(stats_zoom_frame, text="Click Zoom",
                                command=lambda: toggle_stats_zoom_mode('click_zoom'))
stats_click_zoom_btn.pack(side='left', padx=2)

stats_reset_btn = tk.Button(stats_zoom_frame, text="Reset",
                           command=reset_stats_zoom)
stats_reset_btn.pack(side='left', padx=2)

stats_undo_btn = tk.Button(stats_zoom_frame, text="Undo",
                          command=undo_stats_zoom)
stats_undo_btn.pack(side='left', padx=2)

stats_save_zoom_btn = tk.Button(stats_zoom_frame, text="Save Zoom", bg='lightgreen',
                               command=save_stats_zoom)
stats_save_zoom_btn.pack(side='right', padx=2)

# Set up Matplotlib figure and canvas for statistics preview
fig_statistics = Figure(figsize=(4, 4), dpi=100)
ax_statistics = fig_statistics.add_subplot(111)
canvas_statistics = FigureCanvasTkAgg(fig_statistics, master=roi_label_frame)
# Adjust packing to fill horizontally and allow some expansion
canvas_statistics.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

# Connect zoom event handlers
canvas_statistics.mpl_connect('button_press_event', on_stats_click)

statistics_slice_slider = ttk.Scale(roi_label_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=update_statistics_preview)
# Make the slider fill horizontally
statistics_slice_slider.pack(pady=5, padx=5, fill='x')

# Label to display the current slice number for statistics preview
statistics_slice_number_label = tk.Label(roi_label_frame, text="Slice: 0/0")
statistics_slice_number_label.pack(pady=5, padx=5, anchor='w')



# ------------------ About Tab Content ------------------
about_label = tk.Label(about_frame, text="IVIM-DKI Analysis Tool\nVersion 3.0 (IDTV-AD)\nDeveloped by Hisham Hanif and Himansu Maurya\n\n"
                       "Fitting: bounded damped Gauss-Newton with free voxel-wise S0 and parameter-specific\n"
                       "total-variation regularization on log-parameters (IDTV-AD).", justify='center')
about_label.pack(expand=True)

if AUTORUN:
    root.after(1000, lambda: (load_data_file(AUTORUN), run_analysis()))
root.mainloop()
