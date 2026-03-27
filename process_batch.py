#!/usr/bin/env python3
"""
process_batch.py — Hunyuan3D 2.1 → 3D Print Batch Pipeline

Fully automated: reference images → shape generation → PBR texturing →
mesh repair → texture upscale → decimation → .3mf export.

Usage:
    python process_batch.py --input_dir ./test_images --output_dir ./output_3mf --low_vram_mode

Requirements beyond existing env:
    pip install manifold3d
"""

import argparse
import gc
import glob
import logging
import os
import sys

# Match gradio_app.py lines 18-19: add subproject dirs to Python path
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import time
import traceback
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("process_batch")

# ---------------------------------------------------------------------------
# Supported input image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

# ===========================================================================
# 0. PIPELINE INITIALIZATION
# ===========================================================================

def init_pipeline(args):
    """
    Load shape model, texture pipeline, background remover, and post-processors.
    Mirrors the model-loading block from gradio_app.py with identical mmgp offloading.
    """
    log.info("Initializing pipeline...")

    # ---- Background removal ------------------------------------------------
    log.info("Loading background removal model...")
    from hy3dshape.rembg import BackgroundRemover
    rmbg_worker = BackgroundRemover()

    # ---- Shape model -------------------------------------------------------
    log.info("Loading shape model (Hunyuan3DDiTFlowMatchingPipeline)...")
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=False,
        device=args.device,
    )

    # ---- Post-processors (existing) ----------------------------------------
    log.info("Loading mesh post-processors...")
    from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover, FaceReducer

    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()

    # ---- Texture pipeline --------------------------------------------------
    log.info("Loading texture pipeline (Hunyuan3DPaintPipeline)...")
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=512)
    conf.device = "cpu"  # mmgp manages GPU placement
    conf.render_size = 1024
    conf.texture_size = 2048
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    tex_pipeline = Hunyuan3DPaintPipeline(conf)

    # ---- mmgp offloading on the multiview diffusion sub-pipeline -----------
    if args.low_vram_mode:
        log.info("Applying mmgp HighRAM_LowVRAM offloading to multiview model...")
        from mmgp import offload, profile_type
        core_pipe = tex_pipeline.models["multiview_model"].pipeline
        offload.profile(core_pipe, profile_type.HighRAM_LowVRAM)

    log.info("Pipeline initialization complete.")
    return {
        "i23d_worker": i23d_worker,
        "tex_pipeline": tex_pipeline,
        "rmbg_worker": rmbg_worker,
        "floater_remove_worker": floater_remove_worker,
        "degenerate_face_remove_worker": degenerate_face_remove_worker,
        "face_reduce_worker": face_reduce_worker,
    }


# ===========================================================================
# VRAM SWAP FUNCTIONS (copied verbatim from gradio_app.py)
# ===========================================================================

def offload_shape_to_cpu(i23d_worker):
    """Move shape model to CPU to free VRAM."""
    if hasattr(i23d_worker, "model") and i23d_worker.model is not None:
        i23d_worker.model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()


def restore_shape_to_gpu(i23d_worker):
    """Move shape model back to GPU."""
    if hasattr(i23d_worker, "model") and i23d_worker.model is not None:
        i23d_worker.model.to("cuda")


def move_texture_to_gpu(tex_pipeline):
    """[MMGP] mmgp handles GPU placement. Only move dino_v2 manually."""
    mv = tex_pipeline.models.get("multiview_model")
    if mv is not None:
        mv.device = "cuda"
        if hasattr(mv, "dino_v2") and mv.dino_v2 is not None:
            mv.dino_v2 = mv.dino_v2.to("cuda")


def move_texture_to_cpu(tex_pipeline):
    """[MMGP] Minimal cleanup: just handle dino_v2."""
    mv = tex_pipeline.models.get("multiview_model")
    if mv is not None:
        mv.device = "cpu"
        if hasattr(mv, "dino_v2") and mv.dino_v2 is not None:
            mv.dino_v2 = mv.dino_v2.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()


def offload_all_to_cpu(i23d_worker, tex_pipeline):
    """Offload both shape and texture models, freeing all VRAM for post-processing."""
    offload_shape_to_cpu(i23d_worker)
    move_texture_to_cpu(tex_pipeline)
    torch.cuda.empty_cache()
    gc.collect()
    log.info("All models offloaded to CPU. VRAM free for post-processing.")


# ===========================================================================
# 1. GENERATION (reuses existing pipeline)
# ===========================================================================

def remove_background(image: Image.Image, rmbg_worker) -> Image.Image:
    """Remove background from input image using hy3dshape BackgroundRemover."""
    return rmbg_worker(image.convert("RGB"))


def generate_3d(image_path, models, save_dir, args):
    """
    Run full shape + texture generation for one image.
    Returns: path to the textured mesh output folder.
    """
    i23d_worker = models["i23d_worker"]
    tex_pipeline = models["tex_pipeline"]
    rmbg_worker = models["rmbg_worker"]
    face_reduce_worker = models["face_reduce_worker"]
    floater_remove_worker = models["floater_remove_worker"]
    degenerate_face_remove_worker = models["degenerate_face_remove_worker"]

    os.makedirs(save_dir, exist_ok=True)

    # ---- Load and preprocess image ----------------------------------------
    log.info(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGBA")

    log.info("Removing background...")
    image = remove_background(image, rmbg_worker)

    # ---- Shape generation --------------------------------------------------
    log.info("Running shape generation...")
    mesh_output = i23d_worker(
        image=image,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
    )

    if isinstance(mesh_output, list):
        mesh = mesh_output[0]
    else:
        mesh = mesh_output

    # ---- Built-in post-processors -----------------------------------------
    log.info("Running floater removal + degenerate face removal...")
    mesh = floater_remove_worker(mesh)
    mesh = degenerate_face_remove_worker(mesh)

    if args.gen_face_reduce > 0:
        log.info(f"Reducing faces to {args.gen_face_reduce} (generation stage)...")
        mesh = face_reduce_worker(mesh, args.gen_face_reduce)

    # Save raw shape OBJ for texture pipeline input
    raw_obj_path = os.path.join(save_dir, "raw_shape.obj")
    mesh.export(raw_obj_path)
    log.info(f"Raw shape saved: {raw_obj_path}")

    # ---- VRAM swap: shape → CPU, texture → GPU ----------------------------
    log.info("Swapping VRAM: shape → CPU, texture → GPU...")
    offload_shape_to_cpu(i23d_worker)
    move_texture_to_gpu(tex_pipeline)

    # ---- Texture generation ------------------------------------------------
    log.info("Running PBR texture generation...")
    textured_dir = os.path.join(save_dir, "textured")
    os.makedirs(textured_dir, exist_ok=True)
    textured_mesh_path = os.path.join(textured_dir, "textured_mesh.obj")

    # Save input image for texture pipeline
    input_img_path = os.path.join(save_dir, "input_rgba.png")
    image.save(input_img_path)

    tex_pipeline(
        mesh_path=raw_obj_path,
        image_path=input_img_path,
        output_mesh_path=textured_mesh_path,
    )

    # ---- VRAM swap: texture → CPU, shape → GPU ----------------------------
    log.info("Swapping VRAM: texture → CPU, shape → GPU...")
    move_texture_to_cpu(tex_pipeline)
    restore_shape_to_gpu(i23d_worker)

    log.info(f"Textured mesh saved to: {textured_dir}")
    return textured_dir


# ===========================================================================
# 2. MESH REPAIR (trimesh + PyMeshLab + Manifold3D)
# ===========================================================================

def repair_mesh(obj_path, output_path=None):
    """
    Full mesh repair chain:
      1. PyMeshLab: load OBJ directly (preserves UVs), repair non-manifold, close holes
      2. Back-side selective Laplacian smoothing (normal Z < -0.3)
      3. Manifold3D watertight check (geometry only, with validation)

    Returns: repaired trimesh.Trimesh object (also saved to output_path if given).
    """
    log.info(f"Repairing mesh: {obj_path}")
    import pymeshlab

    # ---- Load OBJ directly in PyMeshLab (preserves UVs + texture refs) ----
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    log.info(f"  Loaded: {ms.current_mesh().vertex_number()} verts, {ms.current_mesh().face_number()} faces")

    # ---- Step 1: PyMeshLab repairs ----------------------------------------
    log.info("  PyMeshLab: removing duplicates, repairing non-manifold...")
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()

    # Run non-manifold repair iteratively (some edges need multiple passes)
    for i in range(5):
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()

    # Attempt hole closing
    try:
        ms.meshing_close_holes(maxholesize=30)
        log.info("  Holes closed successfully.")
    except Exception as e:
        log.warning(f"  Could not close holes ({e}). Trying smaller size...")
        try:
            ms.meshing_close_holes(maxholesize=10)
            log.info("  Holes closed with maxholesize=10.")
        except Exception:
            log.warning("  Hole closing skipped entirely.")

    # ---- Step 2: Back-side selective Laplacian smoothing -------------------
    log.info("  Back-side smoothing: targeting faces with normal Z < -0.3...")
    current_mesh = ms.current_mesh()
    face_normals = current_mesh.face_normal_matrix()

    if face_normals is not None and len(face_normals) > 0:
        back_mask = face_normals[:, 2] < -0.3
        num_back = int(np.sum(back_mask))
        log.info(f"  Found {num_back}/{len(face_normals)} back-facing faces")

        if num_back > 0:
            ms.set_selection_none()
            ms.compute_selection_by_condition_per_vertex(
                condselect="(nx*0 + ny*0 + nz) < -0.3"
            )
            ms.apply_coord_laplacian_smoothing(
                stepsmoothnum=3,
                selected=True,
            )

    # ---- Save repaired OBJ via PyMeshLab (preserves UVs) ------------------
    if output_path is None:
        output_path = obj_path.replace(".obj", "_repaired.obj")

    ms.save_current_mesh(output_path)
    log.info(f"  Repaired mesh saved (with UVs): {output_path}")

    # ---- Reload in trimesh (picks up UVs from the saved OBJ) --------------
    repaired = trimesh.load(output_path, process=False)
    if isinstance(repaired, trimesh.Scene):
        repaired = repaired.dump(concatenate=True)

    # Fix normals in trimesh
    trimesh.repair.fix_normals(repaired)
    trimesh.repair.fix_inversion(repaired)

    log.info(f"  Final repaired mesh: {len(repaired.vertices)} verts, {len(repaired.faces)} faces")

    # ---- Step 3: Manifold3D watertight check (optional, geometry only) ----
    # Manifold3D strips UVs, so we only use it to VALIDATE, not to replace
    try:
        import manifold3d
        test_mesh = manifold3d.Manifold(
            mesh=manifold3d.Mesh(
                vert_properties=np.array(repaired.vertices, dtype=np.float64),
                tri_verts=np.array(repaired.faces, dtype=np.uint32),
            )
        )
        result = test_mesh.to_mesh()
        if len(result.tri_verts) > 0:
            log.info("  Manifold3D: mesh validated as watertight.")
        else:
            log.warning("  Manifold3D: validation returned empty mesh — using PyMeshLab output.")
    except Exception as e:
        log.warning(f"  Manifold3D validation skipped ({e}) — using PyMeshLab output.")

    return repaired


# ===========================================================================
# 3. TEXTURE ENHANCEMENT (Real-ESRGAN second pass, albedo only)
# ===========================================================================

def enhance_texture(texture_dir, esrgan_ckpt="hy3dpaint/ckpt/RealESRGAN_x4plus.pth"):
    """
    Run a standalone Real-ESRGAN 4× upscale pass on the baked albedo atlas.
    Called AFTER all shape+texture models are offloaded to CPU.
    Uses ~1-2GB VRAM.
    """
    albedo_path = os.path.join(texture_dir, "textured_mesh.jpg")
    if not os.path.exists(albedo_path):
        # Try .png
        albedo_path = os.path.join(texture_dir, "textured_mesh.png")
    if not os.path.exists(albedo_path):
        log.warning(f"No albedo texture found in {texture_dir}, skipping enhancement.")
        return None

    log.info(f"Enhancing albedo texture: {albedo_path}")

    import cv2
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    # Build the ESRGAN model (small: ~60MB on GPU)
    rrdb_model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=4,
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path=esrgan_ckpt,
        model=rrdb_model,
        tile=512,       # tile to keep VRAM low on large atlases
        tile_pad=10,
        half=True,      # FP16 for speed + VRAM savings
        device="cuda",
    )

    # Load albedo
    img = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
    log.info(f"  Input albedo size: {img.shape[1]}×{img.shape[0]}")

    # Upscale 4×
    output, _ = upsampler.enhance(img, outscale=4)
    log.info(f"  Output albedo size: {output.shape[1]}×{output.shape[0]}")

    # Save enhanced albedo (overwrite original)
    enhanced_path = os.path.join(texture_dir, "textured_mesh_enhanced.jpg")
    cv2.imwrite(enhanced_path, output, [cv2.IMWRITE_JPEG_QUALITY, 98])
    log.info(f"  Enhanced albedo saved: {enhanced_path}")

    # Also overwrite the original so downstream export picks it up
    cv2.imwrite(albedo_path, output, [cv2.IMWRITE_JPEG_QUALITY, 98])

    # Cleanup ESRGAN from GPU
    del upsampler, rrdb_model
    torch.cuda.empty_cache()
    gc.collect()

    return enhanced_path


# ===========================================================================
# 4. DECIMATION (PyMeshLab quadric edge collapse)
# ===========================================================================

def decimate_mesh(obj_path, target_faces=80000, output_path=None):
    """
    Decimate mesh to target face count using PyMeshLab quadric edge collapse.
    Loads OBJ directly through PyMeshLab to preserve UVs.
    Returns: decimated trimesh.Trimesh (with UVs), also saves to output_path.
    """
    import pymeshlab

    # Load to check face count
    mesh = trimesh.load(obj_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    current_faces = len(mesh.faces)
    if current_faces <= target_faces:
        log.info(f"Mesh already has {current_faces} faces (≤ {target_faces}), skipping decimation.")
        if output_path:
            import shutil
            shutil.copy2(obj_path, output_path)
            # Also copy MTL and texture files
            obj_dir = os.path.dirname(obj_path)
            out_dir = os.path.dirname(output_path)
            for ext in [".mtl", ".jpg", ".png"]:
                src = obj_path.replace(".obj", ext)
                if os.path.exists(src) and obj_dir != out_dir:
                    shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))
        return mesh

    log.info(f"Decimating mesh: {current_faces} → {target_faces} faces...")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preserveboundary=True,
        preservetopology=True,
        qualitythr=0.5,
    )

    if output_path is None:
        output_path = obj_path.replace(".obj", "_decimated.obj")

    ms.save_current_mesh(output_path)
    log.info(f"  Decimated and saved (with UVs): {output_path}")

    # Reload in trimesh to pick up UVs
    decimated = trimesh.load(output_path, process=False)
    if isinstance(decimated, trimesh.Scene):
        decimated = decimated.dump(concatenate=True)

    log.info(f"  Decimated: {len(decimated.faces)} faces")
    return decimated


# ===========================================================================
# 5. EXPORT TO .3MF
# ===========================================================================

def bake_vertex_colors(mesh, albedo_path):
    """
    Sample the albedo texture at each vertex's UV coordinate to produce
    vertex colors. Handles both per-vertex and per-face-vertex UV layouts.
    """
    if not os.path.exists(albedo_path):
        log.warning(f"Albedo not found at {albedo_path}, exporting without color.")
        return mesh

    texture = Image.open(albedo_path).convert("RGB")
    tex_array = np.array(texture)
    h, w = tex_array.shape[:2]

    uv = None

    # Try to extract UVs — trimesh stores them differently depending on visual type
    if hasattr(mesh, "visual") and mesh.visual is not None:
        visual = mesh.visual

        # Case 1: TextureVisuals (loaded from OBJ with material)
        if hasattr(visual, "uv") and visual.uv is not None:
            raw_uv = visual.uv
            if len(raw_uv) == len(mesh.vertices):
                uv = raw_uv
            else:
                # Per-face-vertex UVs: average to per-vertex
                log.info(f"  Converting {len(raw_uv)} face-vertex UVs to {len(mesh.vertices)} vertex UVs...")
                uv = np.zeros((len(mesh.vertices), 2), dtype=np.float64)
                counts = np.zeros(len(mesh.vertices), dtype=np.int32)
                for fi, face in enumerate(mesh.faces):
                    for vi in range(3):
                        uv_idx = fi * 3 + vi
                        if uv_idx < len(raw_uv):
                            uv[face[vi]] += raw_uv[uv_idx]
                            counts[face[vi]] += 1
                mask = counts > 0
                uv[mask] /= counts[mask, np.newaxis]

    if uv is not None and len(uv) > 0:
        uv = uv % 1.0
        px = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
        py = np.clip(((1.0 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)

        colors = tex_array[py, px]
        alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
        vertex_colors = np.hstack([colors, alpha])

        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=vertex_colors,
        )
        log.info(f"  Baked vertex colors from albedo ({w}×{h}) onto {len(mesh.vertices)} vertices.")
    else:
        log.warning("  No UV coordinates found on mesh, exporting without color.")

    return mesh


def export_3mf(mesh, albedo_path, output_path):
    """
    Export mesh to .3mf format with baked vertex colors from albedo texture.
    """
    log.info(f"Exporting .3mf: {output_path}")

    # Bake texture into vertex colors for reliable 3D print color
    mesh = bake_vertex_colors(mesh, albedo_path)

    # trimesh exports .3mf natively
    mesh.export(output_path, file_type="3mf")
    log.info(f"  .3mf exported: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    return output_path


# ===========================================================================
# 6. FULL SINGLE-IMAGE PIPELINE
# ===========================================================================

def process_single(image_path, models, output_dir, args):
    """
    Full end-to-end pipeline for a single image:
      generation → repair → texture enhance → decimate → export .3mf
    """
    stem = Path(image_path).stem
    work_dir = os.path.join(output_dir, stem)
    os.makedirs(work_dir, exist_ok=True)

    log.info(f"{'='*60}")
    log.info(f"Processing: {stem}")
    log.info(f"{'='*60}")

    # ---- Stage 1: Generation ----------------------------------------------
    textured_dir = generate_3d(image_path, models, work_dir, args)

    # ---- Offload all GPU models before post-processing --------------------
    offload_all_to_cpu(models["i23d_worker"], models["tex_pipeline"])

    # ---- Stage 2: Mesh Repair ---------------------------------------------
    textured_obj = os.path.join(textured_dir, "textured_mesh.obj")
    repaired_obj = os.path.join(work_dir, "repaired_mesh.obj")

    if os.path.exists(textured_obj):
        repaired_mesh = repair_mesh(textured_obj, output_path=repaired_obj)
    else:
        # Fallback: try to find any OBJ in textured_dir
        obj_files = glob.glob(os.path.join(textured_dir, "*.obj"))
        if obj_files:
            repaired_mesh = repair_mesh(obj_files[0], output_path=repaired_obj)
        else:
            raise FileNotFoundError(f"No OBJ found in {textured_dir}")

    # ---- Stage 3: Texture Enhancement (Real-ESRGAN on albedo) -------------
    enhance_texture(textured_dir)

    # ---- Stage 4: Decimation ----------------------------------------------
    decimated_obj = os.path.join(work_dir, "decimated_mesh.obj")
    decimated_mesh = decimate_mesh(repaired_obj, target_faces=args.target_faces, output_path=decimated_obj)

    # ---- Stage 5: Export .3mf ---------------------------------------------
    albedo_path = os.path.join(textured_dir, "textured_mesh.jpg")
    if not os.path.exists(albedo_path):
        albedo_path = os.path.join(textured_dir, "textured_mesh.png")

    output_3mf = os.path.join(work_dir, f"{stem}.3mf")
    export_3mf(decimated_mesh, albedo_path, output_3mf)

    # ---- Restore shape model to GPU for next image ------------------------
    restore_shape_to_gpu(models["i23d_worker"])

    return output_3mf


# ===========================================================================
# 7. BATCH ORCHESTRATOR
# ===========================================================================

def process_batch(args):
    """Main batch loop: init pipeline, iterate images, process each end-to-end."""

    # Discover input images
    input_dir = Path(args.input_dir)
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS and f.is_file()
    ])

    if not image_files:
        log.error(f"No images found in {input_dir} (supported: {IMAGE_EXTS})")
        sys.exit(1)

    log.info(f"Found {len(image_files)} images in {input_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models once
    models = init_pipeline(args)

    # Process each image sequentially (end-to-end before moving to next)
    results = {"success": [], "failed": []}

    for idx, img_path in enumerate(image_files, 1):
        log.info(f"\n{'#'*60}")
        log.info(f"# IMAGE {idx}/{len(image_files)}: {img_path.name}")
        log.info(f"{'#'*60}")
        t0 = time.time()

        try:
            output_path = process_single(str(img_path), models, str(output_dir), args)
            elapsed = time.time() - t0
            log.info(f"SUCCESS: {img_path.name} → {output_path} ({elapsed:.0f}s)")
            results["success"].append(img_path.name)
        except Exception as e:
            elapsed = time.time() - t0
            log.error(f"FAILED: {img_path.name} after {elapsed:.0f}s — {e}")
            log.error(traceback.format_exc())
            results["failed"].append((img_path.name, str(e)))

            # Try to restore GPU state for next image
            try:
                restore_shape_to_gpu(models["i23d_worker"])
            except Exception:
                pass

    # ---- Summary -----------------------------------------------------------
    log.info(f"\n{'='*60}")
    log.info(f"BATCH COMPLETE")
    log.info(f"  Succeeded: {len(results['success'])}/{len(image_files)}")
    for name in results["success"]:
        log.info(f"    ✓ {name}")
    if results["failed"]:
        log.info(f"  Failed: {len(results['failed'])}/{len(image_files)}")
        for name, err in results["failed"]:
            log.info(f"    ✗ {name}: {err}")
    log.info(f"{'='*60}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hunyuan3D 2.1 → 3D Print Batch Pipeline",
    )

    # I/O
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Folder containing reference images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output folder for .3mf files and intermediates")

    # Model paths (defaults match gradio_app.py)
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2.1",
                        help="HuggingFace repo or local path for shape model")
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-1",
                        help="Subfolder for shape model weights")
    parser.add_argument("--texgen_model_path", type=str, default="tencent/Hunyuan3D-2.1",
                        help="HuggingFace repo or local path for texture model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for shape model (default: cuda)")
    parser.add_argument("--low_vram_mode", action="store_true",
                        help="Enable mmgp VRAM offloading (recommended for 16GB)")

    # Generation params
    parser.add_argument("--num_steps", type=int, default=30,
                        help="Shape generation inference steps (default: 30)")
    parser.add_argument("--guidance_scale", type=float, default=5.5,
                        help="Shape generation guidance scale (default: 5.5)")
    parser.add_argument("--octree_resolution", type=int, default=380,
                        help="Octree resolution for marching cubes (default: 380)")
    parser.add_argument("--num_chunks", type=int, default=200000,
                        help="Number of chunks for shape generation (default: 200000)")
    parser.add_argument("--gen_face_reduce", type=int, default=90000,
                        help="Face reduction during generation stage (0 to skip, default: 90000)")

    # Post-processing params
    parser.add_argument("--target_faces", type=int, default=80000,
                        help="Target face count for final decimation (default: 80000)")

    args = parser.parse_args()
    process_batch(args)


if __name__ == "__main__":
    main()
