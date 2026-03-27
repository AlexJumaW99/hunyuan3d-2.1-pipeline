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

# Reduce CUDA memory fragmentation (helps prevent OOM on 16GB cards)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import traceback
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

# ---------------------------------------------------------------------------
# Logging — 90s HACKER ROBOTIC EDITION
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("process_batch")

# Suppress verbose third-party logging
logging.getLogger("trimesh").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

def _ts():
    """Timestamp in hacker format."""
    return time.strftime("%H:%M:%S")

def _banner():
    """Print the boot banner."""
    print("\n")
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║                                                                  ║")
    print("  ║   ███╗   ███╗███████╗██████╗  ██████╗ ███████╗                  ║")
    print("  ║   ████╗ ████║██╔════╝██╔══██╗██╔═══██╗██╔════╝                  ║")
    print("  ║   ██╔████╔██║█████╗  ██████╔╝██║   ██║█████╗                    ║")
    print("  ║   ██║╚██╔╝██║██╔══╝  ██╔══██╗██║   ██║██╔══╝                    ║")
    print("  ║   ██║ ╚═╝ ██║███████╗██║  ██║╚██████╔╝███████╗                  ║")
    print("  ║   ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝                  ║")
    print("  ║                                                                  ║")
    print("  ║                          /\\                                      ║")
    print("  ║                         /  \\                                     ║")
    print("  ║                        /    \\                                    ║")
    print("  ║                       /  /\\  \\                                   ║")
    print("  ║                      /  /  \\  \\                                  ║")
    print("  ║                     /  /    \\  \\                                 ║")
    print("  ║                    /  /  /\\  \\  \\                                ║")
    print("  ║                   /__/__/  \\__\\__\\                               ║")
    print("  ║                                                                  ║")
    print("  ║                  ═══ 3 D   E N G I N E ═══                       ║")
    print("  ║                        v1.0 // 2026                              ║")
    print("  ║                                                                  ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")
    print("")
    print(f"  🤖 [SYSTEM] Boot sequence initiated @ {_ts()}")
    print(f"  🖥️  [SYSTEM] GPU: NVIDIA RTX 5070 Ti // 16GB VRAM")
    print(f"  ⚡ [SYSTEM] CUDA: {torch.version.cuda} // PyTorch: {torch.__version__}")
    print("")

def hlog(emoji, tag, msg, indent=0):
    """Themed log output."""
    pad = "    " * indent
    print(f"  {pad}{emoji} [{_ts()}] [{tag}] {msg}")

def hlog_stage(stage_num, stage_name):
    """Print a major stage header."""
    print("")
    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  ⚙️  STAGE {stage_num} » {stage_name:<47}│")
    print(f"  └─────────────────────────────────────────────────────────┘")
    print("")

def hlog_substep(emoji, msg):
    """Print an indented substep."""
    print(f"      {emoji} {msg}")

def hlog_result(emoji, msg):
    """Print a result line."""
    print(f"      {emoji} {msg}")

def hlog_separator():
    """Print a thin separator."""
    print(f"  {'·' * 60}")

def hlog_image_header(idx, total, name):
    """Print image processing header."""
    print("\n")
    print(f"  ╔{'═' * 60}╗")
    print(f"  ║  🎯 TARGET [{idx}/{total}]: {name:<42}║")
    print(f"  ╚{'═' * 60}╝")
    print("")

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
    _banner()
    hlog("🔧", "INIT", "Booting neural subsystems...")

    # ---- Background removal ------------------------------------------------
    hlog("📡", "INIT", "Loading background removal unit...")
    from hy3dshape.rembg import BackgroundRemover
    rmbg_worker = BackgroundRemover()
    hlog_result("✅", "Background remover ─── ONLINE")

    # ---- Shape model -------------------------------------------------------
    hlog("📡", "INIT", "Loading shape generation core (DiT FlowMatching)...")
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=False,
        device=args.device,
    )
    hlog_result("✅", "Shape core ─── ONLINE")

    # ---- Post-processors (existing) ----------------------------------------
    hlog("📡", "INIT", "Loading mesh post-processor array...")
    from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover, FaceReducer

    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()
    hlog_result("✅", "Post-processors ─── ONLINE")

    # ---- Texture pipeline --------------------------------------------------
    hlog("📡", "INIT", "Loading PBR texture synthesis engine...")
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
        hlog("🧠", "MMGP", "Engaging HighRAM_LowVRAM memory offload protocol...")
        from mmgp import offload, profile_type
        core_pipe = tex_pipeline.models["multiview_model"].pipeline
        offload.profile(core_pipe, profile_type.HighRAM_LowVRAM)
        hlog_result("✅", "MMGP offload ─── ENGAGED")

    print("")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  🤖 ALL SYSTEMS NOMINAL ── PIPELINE READY FOR INPUT     │")
    print("  └─────────────────────────────────────────────────────────┘")
    print("")
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
    hlog("💾", "VRAM", "All neural cores offloaded to CPU ── VRAM clear for post-ops")


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
    hlog("📂", "INPUT", f"Ingesting target image: {image_path}")
    image = Image.open(image_path).convert("RGBA")

    hlog("✂️ ", "REMBG", "Stripping background noise...")
    image = remove_background(image, rmbg_worker)
    hlog_result("✅", "Background ─── ELIMINATED")

    # ---- Shape generation --------------------------------------------------
    hlog_stage(1, "SHAPE GENERATION")
    hlog("🔮", "SHAPE", "Initiating 3D diffusion sampling...")
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
    hlog("🧹", "CLEAN", "Purging floating artifacts + degenerate faces...")
    mesh = floater_remove_worker(mesh)
    mesh = degenerate_face_remove_worker(mesh)

    if args.gen_face_reduce > 0:
        hlog("📐", "REDUCE", f"Compressing mesh topology → {args.gen_face_reduce} faces...")
        mesh = face_reduce_worker(mesh, args.gen_face_reduce)

    # Save raw shape OBJ for texture pipeline input
    raw_obj_path = os.path.join(save_dir, "raw_shape.obj")
    mesh.export(raw_obj_path)
    hlog_result("💾", f"Raw shape archived → {raw_obj_path}")

    # ---- VRAM swap: shape → CPU, texture → GPU ----------------------------
    hlog("🔄", "VRAM", "Swapping neural cores: SHAPE→CPU // TEXTURE→GPU...")
    offload_shape_to_cpu(i23d_worker)
    move_texture_to_gpu(tex_pipeline)

    # ---- Texture generation ------------------------------------------------
    hlog_stage(2, "PBR TEXTURE SYNTHESIS")
    hlog("🎨", "PAINT", "Generating albedo + metallic + roughness maps...")
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
    hlog("🔄", "VRAM", "Swapping neural cores: TEXTURE→CPU // SHAPE→GPU...")
    move_texture_to_cpu(tex_pipeline)
    restore_shape_to_gpu(i23d_worker)

    hlog_result("✅", f"Textured mesh archived → {textured_dir}")
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
    hlog_stage(3, "MESH REPAIR PROTOCOL")
    hlog("🔧", "REPAIR", f"Target mesh: {obj_path}")
    import pymeshlab

    # ---- Load OBJ directly in PyMeshLab (preserves UVs + texture refs) ----
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    hlog_substep("📊", f"Mesh loaded: {ms.current_mesh().vertex_number()} verts // {ms.current_mesh().face_number()} faces")

    # ---- Step 1: PyMeshLab repairs ----------------------------------------
    hlog_substep("🔩", "Purging duplicate geometry...")
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()

    hlog_substep("🔩", "Running non-manifold repair [5 iterations]...")
    # Run non-manifold repair iteratively (some edges need multiple passes)
    for i in range(5):
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()

    # Attempt hole closing
    try:
        ms.meshing_close_holes(maxholesize=30)
        hlog_substep("✅", "Hull breach sealed ─── holes closed [maxsize=30]")
    except Exception as e:
        hlog_substep("⚠️ ", f"Primary seal failed ({e})")
        try:
            ms.meshing_close_holes(maxholesize=10)
            hlog_substep("✅", "Secondary seal applied ─── holes closed [maxsize=10]")
        except Exception:
            hlog_substep("⚠️ ", "Hull seal skipped ─── will attempt downstream")

    # ---- Step 2: Back-side selective Laplacian smoothing -------------------
    hlog_substep("🔬", "Scanning for back-face anomalies [normal_Z < -0.3]...")
    current_mesh = ms.current_mesh()
    face_normals = current_mesh.face_normal_matrix()

    if face_normals is not None and len(face_normals) > 0:
        back_mask = face_normals[:, 2] < -0.3
        num_back = int(np.sum(back_mask))
        hlog_substep("🎯", f"Detected {num_back}/{len(face_normals)} back-facing anomalies")

        if num_back > 0:
            ms.set_selection_none()
            ms.compute_selection_by_condition_per_vertex(
                condselect="(nx*0 + ny*0 + nz) < -0.3"
            )
            ms.apply_coord_laplacian_smoothing(
                stepsmoothnum=3,
                selected=True,
            )
            hlog_substep("✅", "Laplacian smoothing applied [3 passes]")

    # ---- Save repaired OBJ via PyMeshLab (preserves UVs) ------------------
    if output_path is None:
        output_path = obj_path.replace(".obj", "_repaired.obj")

    ms.save_current_mesh(output_path)
    hlog_substep("💾", f"Repaired mesh + UVs archived → {os.path.basename(output_path)}")

    # ---- Reload in trimesh (picks up UVs from the saved OBJ) --------------
    repaired = trimesh.load(output_path, process=False)
    if isinstance(repaired, trimesh.Scene):
        repaired = repaired.dump(concatenate=True)

    # Fix normals in trimesh
    trimesh.repair.fix_normals(repaired)
    trimesh.repair.fix_inversion(repaired)

    hlog_substep("📊", f"Final mesh: {len(repaired.vertices)} verts // {len(repaired.faces)} faces")

    # ---- Step 3: Manifold3D watertight check (optional, geometry only) ----
    hlog_substep("🔬", "Running Manifold3D watertight validation...")
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
            hlog_substep("✅", "Manifold3D ─── WATERTIGHT CONFIRMED")
        else:
            hlog_substep("⚠️ ", "Manifold3D returned empty ─── using PyMeshLab output")
    except Exception as e:
        hlog_substep("⚠️ ", f"Manifold3D skipped ({e}) ─── using PyMeshLab output")

    print("")

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
        hlog_substep("⚠️ ", f"No albedo texture found in {texture_dir} ─── skipping")
        return None

    hlog_stage(4, "TEXTURE UPSCALE [REAL-ESRGAN 4×]")
    hlog("🖼️ ", "ESRGAN", f"Target atlas: {albedo_path}")

    import cv2
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    # Build the ESRGAN model (small: ~60MB on GPU)
    hlog_substep("📡", "Loading Real-ESRGAN neural upscaler [~60MB VRAM]...")
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
    hlog_substep("📊", f"Input resolution: {img.shape[1]}×{img.shape[0]}")

    # Upscale 4×
    hlog_substep("⚡", "Neural upscale in progress...")
    output, _ = upsampler.enhance(img, outscale=4)
    hlog_substep("📊", f"Output resolution: {output.shape[1]}×{output.shape[0]}")

    # Save enhanced albedo (overwrite original)
    enhanced_path = os.path.join(texture_dir, "textured_mesh_enhanced.jpg")
    cv2.imwrite(enhanced_path, output, [cv2.IMWRITE_JPEG_QUALITY, 98])
    hlog_substep("💾", f"Enhanced atlas archived → {os.path.basename(enhanced_path)}")

    # Also overwrite the original so downstream export picks it up
    cv2.imwrite(albedo_path, output, [cv2.IMWRITE_JPEG_QUALITY, 98])

    # Cleanup ESRGAN from GPU
    del upsampler, rrdb_model
    torch.cuda.empty_cache()
    gc.collect()
    hlog_substep("🗑️ ", "ESRGAN purged from VRAM")
    print("")

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
        hlog("📐", "DECIMATE", f"Mesh has {current_faces} faces (≤ {target_faces}) ─── skip")
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

    hlog_stage(5, "MESH DECIMATION")
    hlog("📐", "DECIMATE", f"Compressing: {current_faces} → {target_faces} faces...")
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
    hlog_substep("💾", f"Decimated mesh + UVs archived → {os.path.basename(output_path)}")

    # Reload in trimesh to pick up UVs
    decimated = trimesh.load(output_path, process=False)
    if isinstance(decimated, trimesh.Scene):
        decimated = decimated.dump(concatenate=True)

    hlog_substep("📊", f"Final polygon count: {len(decimated.faces)} faces")
    print("")
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
        hlog_substep("⚠️ ", f"Albedo not found at {albedo_path} ─── exporting colorless")
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
                hlog_substep("🔄", f"Remapping {len(raw_uv)} face-vertex UVs → {len(mesh.vertices)} vertex UVs...")
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
        hlog_substep("🎨", f"Vertex colors baked from {w}×{h} atlas → {len(mesh.vertices)} vertices")
    else:
        hlog_substep("⚠️ ", "No UV coordinates found ─── exporting colorless")

    return mesh


def export_3mf(mesh, albedo_path, output_path):
    """
    Export mesh to .3mf format with baked vertex colors from albedo texture.
    """
    hlog_stage(6, "3MF EXPORT [PRINT-READY]")
    hlog("📦", "EXPORT", f"Destination: {output_path}")

    # Bake texture into vertex colors for reliable 3D print color
    hlog_substep("🎨", "Baking vertex colors from albedo texture...")
    mesh = bake_vertex_colors(mesh, albedo_path)

    # trimesh exports .3mf natively
    hlog_substep("📦", "Writing .3mf container...")
    mesh.export(output_path, file_type="3mf")
    size_mb = os.path.getsize(output_path) / 1e6
    hlog_substep("✅", f"Export complete ─── {size_mb:.1f} MB")
    print("")
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
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(output_dir, f"{stem}_{timestamp}")
    os.makedirs(work_dir, exist_ok=True)

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

    output_3mf = os.path.join(work_dir, f"{stem}_{timestamp}.3mf")
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
        hlog("❌", "ERROR", f"No images found in {input_dir} (supported: {IMAGE_EXTS})")
        sys.exit(1)

    hlog("📂", "SCAN", f"Detected {len(image_files)} target(s) in {input_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models once
    models = init_pipeline(args)

    # Process each image sequentially (end-to-end before moving to next)
    results = {"success": [], "failed": []}

    for idx, img_path in enumerate(image_files, 1):
        hlog_image_header(idx, len(image_files), img_path.name)
        t0 = time.time()

        try:
            output_path = process_single(str(img_path), models, str(output_dir), args)
            elapsed = time.time() - t0

            print("")
            print(f"  ┌─────────────────────────────────────────────────────────┐")
            print(f"  │  ✅ MISSION COMPLETE                                     │")
            print(f"  │  📁 {os.path.basename(output_path):<53}│")
            print(f"  │  ⏱️  {elapsed:.0f}s elapsed                                       │")
            print(f"  └─────────────────────────────────────────────────────────┘")

            results["success"].append(img_path.name)
        except Exception as e:
            elapsed = time.time() - t0
            hlog("❌", "FAIL", f"{img_path.name} ── aborted after {elapsed:.0f}s")
            hlog("💀", "FAIL", f"Cause: {e}")
            log.debug(traceback.format_exc())
            results["failed"].append((img_path.name, str(e)))

            # Aggressive VRAM cleanup after failure to prevent cascading OOM
            try:
                hlog("🔄", "RECOVER", "Flushing VRAM + recovering GPU state...")
                offload_all_to_cpu(models["i23d_worker"], models["tex_pipeline"])
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()  # double flush after gc
                restore_shape_to_gpu(models["i23d_worker"])
                hlog("✅", "RECOVER", "GPU state restored ── ready for next target")
            except Exception:
                hlog("⚠️ ", "RECOVER", "GPU recovery failed ── attempting continue")
                torch.cuda.empty_cache()
                gc.collect()

    # ---- Summary -----------------------------------------------------------
    total = len(image_files)
    ok = len(results["success"])
    fail = len(results["failed"])

    print("\n")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║                    BATCH OPERATION SUMMARY                   ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  📊 Processed: {total:<3} targets                                 ║")
    print(f"  ║  ✅ Success:   {ok:<3}                                            ║")
    print(f"  ║  ❌ Failed:    {fail:<3}                                            ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    for name in results["success"]:
        display = name[:50]
        print(f"  ║  ✅ {display:<57}║")
    for name, err in results["failed"]:
        display = f"{name}: {err}"[:50]
        print(f"  ║  ❌ {display:<57}║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    if fail == 0:
        print("  ║  🤖 ALL TARGETS PROCESSED ── SYSTEM NOMINAL                ║")
    else:
        print("  ║  ⚠️  PARTIAL COMPLETION ── CHECK LOGS                       ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print("")


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
    parser.add_argument("--num_chunks", type=int, default=500000,
                        help="Number of chunks for shape generation (default: 500000, higher = less VRAM)")
    parser.add_argument("--gen_face_reduce", type=int, default=90000,
                        help="Face reduction during generation stage (0 to skip, default: 90000)")

    # Post-processing params
    parser.add_argument("--target_faces", type=int, default=80000,
                        help="Target face count for final decimation (default: 80000)")

    args = parser.parse_args()
    process_batch(args)


if __name__ == "__main__":
    main()
