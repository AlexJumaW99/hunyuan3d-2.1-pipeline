#!/usr/bin/env python3
"""
process_batch.py — Hunyuan3D 2.1 → 3D Print Batch Pipeline (Quality-Optimized)

Fully automated: reference images → shape generation → PBR texturing →
mesh repair → decimation → GLB/3MF export.

Optimized for maximum texture/mesh quality on 16GB VRAM GPUs.

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
    print("  ║                  QUALITY-OPTIMIZED v2.0                          ║")
    print("  ║                        v2.0 // 2026                              ║")
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

def _log_vram(label=""):
    """Log current VRAM usage."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1024**3
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        hlog_substep("📊", f"VRAM {label}: {used:.1f}GB used / {free_gb:.1f}GB free / {total_gb:.1f}GB total")

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

    QUALITY OPTIMIZATIONS:
    - Multiview diffusion at 768px (up from 512) for sharper textures
    - 9 candidate views (up from 6-8) for better surface coverage
    - render_size=2048 to preserve ESRGAN upscale detail during baking
    - texture_size=4096 for high-resolution UV atlas output
    """
    _banner()
    hlog("🔧", "INIT", "Booting neural subsystems...")
    _log_vram("at boot")

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
    _log_vram("after shape model")

    # ---- Post-processors (existing) ----------------------------------------
    hlog("📡", "INIT", "Loading mesh post-processor array...")
    from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover, FaceReducer

    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()
    hlog_result("✅", "Post-processors ─── ONLINE")

    # ---- Texture pipeline --------------------------------------------------
    # QUALITY: We load texture to CPU first, then swap via mmgp.
    # Shape model must be offloaded to CPU before texture model loads on GPU.
    hlog("📡", "INIT", "Loading PBR texture synthesis engine (QUALITY MODE)...")
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    # --- QUALITY SETTINGS ---
    # max_num_view: more viewpoints = better surface coverage, less inpainting
    # resolution=512: the max that reliably fits in 16GB VRAM with mmgp offloading.
    #   The UNet processes 3×2×N_views latent images per step — at 768px this needs
    #   ~12GB for activations alone, which OOMs on 16GB cards.
    #   512px with higher render_size/atlas_size is the quality sweet spot for 16GB.
    tex_max_views = args.tex_max_views
    tex_resolution = args.tex_resolution
    hlog_substep("📐", f"Multiview resolution: {tex_resolution}px // Max views: {tex_max_views}")

    conf = Hunyuan3DPaintConfig(max_num_view=tex_max_views, resolution=tex_resolution)
    conf.device = "cpu"  # mmgp manages GPU placement

    # render_size controls the resolution at which views are projected onto UV space.
    # ESRGAN upscales each 512px view to 2048px (4×).
    # The baking code resizes back to render_size before back-projection.
    # Setting render_size=2048 preserves the full ESRGAN enhancement.
    # (Default was 1024 which threw away half the upscaled detail.)
    conf.render_size = args.tex_render_size
    hlog_substep("📐", f"Bake render size: {conf.render_size}px")

    # texture_size is the output UV atlas resolution.
    # 4096 gives excellent detail for 3D printing and real-time rendering.
    conf.texture_size = args.tex_atlas_size
    hlog_substep("📐", f"UV atlas size: {conf.texture_size}px")

    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"

    # More even view weights: prevent back/sides from being low quality.
    # Default was [1, 0.1, 0.5, 0.1, 0.05, 0.05] which heavily favors front.
    # For 3D printing, all sides matter equally.
    # The config __init__ builds candidate_camera_azims with 6 base views + 24
    # elevation candidates (30 total). We need weights for all of them.
    base_weights = [1.0, 0.5, 0.8, 0.5, 0.2, 0.2]  # front, right, back, left, top, bottom
    num_extra = len(conf.candidate_camera_azims) - len(base_weights)
    extra_weights = [0.05] * max(0, num_extra)  # elevation candidates (higher than default 0.01)
    conf.candidate_view_weights = base_weights + extra_weights

    # Bake exponent: lower = more even blending between views (less sharp falloff)
    # Default is 4, which creates harsh boundaries. 3 gives smoother transitions.
    conf.bake_exp = 3

    tex_pipeline = Hunyuan3DPaintPipeline(conf)

    # ---- mmgp offloading on the multiview diffusion sub-pipeline -----------
    if args.low_vram_mode:
        hlog("🧠", "MMGP", "Engaging HighRAM_LowVRAM memory offload protocol...")
        from mmgp import offload, profile_type
        try:
            core_pipe = tex_pipeline.models["multiview_model"].pipeline
            offload.profile(core_pipe, profile_type.HighRAM_LowVRAM)
            hlog_result("✅", "MMGP HighRAM_LowVRAM ─── ENGAGED")
        except Exception as mmgp_err:
            hlog_substep("⚠️ ", f"HighRAM_LowVRAM failed: {mmgp_err}")
            try:
                core_pipe = tex_pipeline.models["multiview_model"].pipeline
                offload.profile(core_pipe, profile_type.LowRAM_LowVRAM)
                hlog_result("✅", "MMGP LowRAM_LowVRAM fallback ─── ENGAGED")
            except Exception as mmgp_err2:
                hlog_substep("❌", f"All MMGP profiles failed: {mmgp_err2}")

    _log_vram("after texture pipeline init")

    print("")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  🤖 ALL SYSTEMS NOMINAL ── PIPELINE READY FOR INPUT     │")
    print("  │  📐 QUALITY MODE: HIGH-RES TEXTURES ENABLED             │")
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
# VRAM SWAP FUNCTIONS — THOROUGH CLEANUP FOR 16GB CARDS
# ===========================================================================

def _force_vram_flush():
    """Nuclear VRAM cleanup — call after any model movement or OOM."""
    # First pass: release Python references
    gc.collect()
    # Clear PyTorch's CUDA memory cache
    torch.cuda.empty_cache()
    # Second pass: catch anything gc freed
    gc.collect()
    torch.cuda.empty_cache()
    # Reset CUDA memory stats to help allocator defragment
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()
    # Synchronize to ensure all CUDA ops are complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def offload_shape_to_cpu(i23d_worker):
    """Move shape model to CPU to free VRAM."""
    try:
        if hasattr(i23d_worker, "model") and i23d_worker.model is not None:
            i23d_worker.model.to("cpu")
        if hasattr(i23d_worker, "conditioner") and i23d_worker.conditioner is not None:
            i23d_worker.conditioner.to("cpu")
        if hasattr(i23d_worker, "vae") and i23d_worker.vae is not None:
            i23d_worker.vae.to("cpu")
    except Exception as e:
        hlog_substep("⚠️ ", f"Warning offloading shape: {e}")
    _force_vram_flush()
    hlog_substep("💾", "Shape model offloaded to CPU")
    _log_vram("after shape offload")


def restore_shape_to_gpu(i23d_worker):
    """Move shape model back to GPU."""
    try:
        if hasattr(i23d_worker, "model") and i23d_worker.model is not None:
            i23d_worker.model.to("cuda")
        if hasattr(i23d_worker, "conditioner") and i23d_worker.conditioner is not None:
            i23d_worker.conditioner.to("cuda")
        if hasattr(i23d_worker, "vae") and i23d_worker.vae is not None:
            i23d_worker.vae.to("cuda")
    except Exception as e:
        hlog_substep("⚠️ ", f"Warning restoring shape: {e}")
    hlog_substep("🔄", "Shape model restored to GPU")
    _log_vram("after shape restore")


def move_texture_to_gpu(tex_pipeline):
    """[MMGP] mmgp handles GPU placement. Only move dino_v2 manually."""
    try:
        mv = tex_pipeline.models.get("multiview_model")
        if mv is not None:
            mv.device = "cuda"
            if hasattr(mv, "dino_v2") and mv.dino_v2 is not None:
                mv.dino_v2 = mv.dino_v2.to("cuda")
        _force_vram_flush()
        hlog_substep("🔄", "Texture pipeline activated (mmgp manages UNet)")
        _log_vram("after texture GPU move")
    except Exception as e:
        hlog_substep("⚠️ ", f"Warning in move_texture_to_gpu: {e}")


def move_texture_to_cpu(tex_pipeline):
    """[MMGP] Thorough cleanup: handle dino_v2 + ESRGAN."""
    try:
        mv = tex_pipeline.models.get("multiview_model")
        if mv is not None:
            mv.device = "cpu"
            if hasattr(mv, "dino_v2") and mv.dino_v2 is not None:
                mv.dino_v2 = mv.dino_v2.to("cpu")

        # Also move the ESRGAN super_model to CPU if it's on GPU
        super_model = tex_pipeline.models.get("super_model")
        if super_model is not None and hasattr(super_model, "upsampler"):
            upsampler = super_model.upsampler
            if hasattr(upsampler, "model") and upsampler.model is not None:
                try:
                    upsampler.model.to("cpu")
                except Exception:
                    pass
            if hasattr(upsampler, "device"):
                upsampler.device = torch.device("cpu")
    except Exception as e:
        hlog_substep("⚠️ ", f"Warning in move_texture_to_cpu: {e}")
    _force_vram_flush()
    hlog_substep("💾", "Texture pipeline offloaded to CPU")
    _log_vram("after texture CPU move")


def offload_all_to_cpu(i23d_worker, tex_pipeline):
    """Offload ALL models, freeing all VRAM for post-processing."""
    offload_shape_to_cpu(i23d_worker)
    move_texture_to_cpu(tex_pipeline)
    _force_vram_flush()
    hlog("💾", "VRAM", "All neural cores offloaded ── VRAM clear for post-ops")
    _log_vram("after full offload")


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

    # ---- Ensure shape model is on GPU (may be on CPU after a failed recovery) --
    if hasattr(i23d_worker, "model") and i23d_worker.model is not None:
        first_param = next(i23d_worker.model.parameters(), None)
        if first_param is not None and not first_param.is_cuda:
            hlog("🔄", "VRAM", "Shape model found on CPU — restoring to GPU...")
            _force_vram_flush()
            restore_shape_to_gpu(i23d_worker)

    # ---- Load and preprocess image ----------------------------------------
    hlog("📂", "INPUT", f"Ingesting target image: {image_path}")
    image = Image.open(image_path).convert("RGBA")

    hlog("✂️ ", "REMBG", "Stripping background noise...")
    image = remove_background(image, rmbg_worker)
    hlog_result("✅", "Background ─── ELIMINATED")

    # ---- Shape generation --------------------------------------------------
    hlog_stage(1, "SHAPE GENERATION")
    hlog("🔮", "SHAPE", "Initiating 3D diffusion sampling...")
    hlog_substep("📐", f"Steps: {args.num_steps} // Guidance: {args.guidance_scale}")
    hlog_substep("📐", f"Octree: {args.octree_resolution} // Chunks: {args.num_chunks}")
    _log_vram("before shape generation")

    t_shape_start = time.time()
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

    t_shape = time.time() - t_shape_start
    hlog_result("✅", f"Shape generated in {t_shape:.0f}s")
    hlog_substep("📊", f"Raw mesh: {len(mesh.vertices)} verts // {len(mesh.faces)} faces")

    # ---- Built-in post-processors (before texturing) ----------------------
    # QUALITY: Only run floater/degenerate removal, NOT face reduction.
    # The texture pipeline internally does its own remesh to ~40K faces.
    # Pre-decimating here would degrade the mesh that gets textured.
    hlog("🧹", "CLEAN", "Purging floating artifacts + degenerate faces...")
    mesh = floater_remove_worker(mesh)
    mesh = degenerate_face_remove_worker(mesh)
    hlog_substep("📊", f"Cleaned mesh: {len(mesh.vertices)} verts // {len(mesh.faces)} faces")

    # NOTE: gen_face_reduce is intentionally skipped by default (set to 0).
    # The texture pipeline's internal remesh_mesh() handles face reduction
    # to its target count. Pre-reducing here causes double decimation and
    # degrades texture baking quality.
    if args.gen_face_reduce > 0:
        hlog("📐", "REDUCE", f"Pre-texture face reduction → {args.gen_face_reduce} faces...")
        mesh = face_reduce_worker(mesh, args.gen_face_reduce)
        hlog_substep("📊", f"Reduced mesh: {len(mesh.vertices)} verts // {len(mesh.faces)} faces")

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
    hlog_substep("📐", f"Resolution: {args.tex_resolution}px // Views: up to {args.tex_max_views}")
    hlog_substep("📐", f"Render→bake: {args.tex_render_size}px // Atlas: {args.tex_atlas_size}px")

    textured_dir = os.path.join(save_dir, "textured")
    os.makedirs(textured_dir, exist_ok=True)
    textured_mesh_path = os.path.join(textured_dir, "textured_mesh.obj")

    # Save input image for texture pipeline
    input_img_path = os.path.join(save_dir, "input_rgba.png")
    image.save(input_img_path)

    t_tex_start = time.time()
    tex_pipeline(
        mesh_path=raw_obj_path,
        image_path=input_img_path,
        output_mesh_path=textured_mesh_path,
    )
    t_tex = time.time() - t_tex_start
    hlog_result("✅", f"Texturing completed in {t_tex:.0f}s")

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
    Mesh repair chain:
      1. PyMeshLab: load OBJ directly (preserves UVs), repair non-manifold, close holes
      2. Gentle selective Laplacian smoothing on severe back-face anomalies only
      3. Manifold3D watertight validation (geometry only, non-destructive)

    QUALITY: Reduced smoothing aggressiveness to preserve texture-mapped detail.
    """
    hlog_stage(3, "MESH REPAIR PROTOCOL")
    hlog("🔧", "REPAIR", f"Target mesh: {obj_path}")
    import pymeshlab

    # ---- Load OBJ directly in PyMeshLab (preserves UVs + texture refs) ----
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    hlog_substep("📊", f"Loaded: {ms.current_mesh().vertex_number()} verts // "
                        f"{ms.current_mesh().face_number()} faces")

    # ---- Step 1: PyMeshLab repairs ----------------------------------------
    hlog_substep("🔩", "Purging duplicate geometry...")
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()

    hlog_substep("🔩", "Running non-manifold repair [3 iterations]...")
    for i in range(3):
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()

    # Attempt hole closing — conservative to avoid UV damage
    try:
        ms.meshing_close_holes(maxholesize=20)
        hlog_substep("✅", "Hull breach sealed ─── holes closed [maxsize=20]")
    except Exception as e:
        hlog_substep("⚠️ ", f"Hole seal skipped ({e})")

    # ---- Step 2: Very gentle back-face smoothing --------------------------
    # QUALITY CHANGE: Only smooth extremely bad back-faces (Z < -0.5 instead of -0.3)
    # and use fewer passes (2 instead of 3) to preserve texture detail.
    hlog_substep("🔬", "Scanning for severe back-face anomalies [normal_Z < -0.5]...")
    current_mesh = ms.current_mesh()
    face_normals = current_mesh.face_normal_matrix()

    if face_normals is not None and len(face_normals) > 0:
        back_mask = face_normals[:, 2] < -0.5
        num_back = int(np.sum(back_mask))
        hlog_substep("🎯", f"Detected {num_back}/{len(face_normals)} severe back-facing faces")

        if num_back > 0 and num_back < len(face_normals) * 0.3:
            # Only smooth if it's a minority of faces — otherwise it's intentional geometry
            ms.set_selection_none()
            ms.compute_selection_by_condition_per_vertex(
                condselect="(nx*0 + ny*0 + nz) < -0.5"
            )
            ms.apply_coord_laplacian_smoothing(
                stepsmoothnum=2,
                selected=True,
            )
            hlog_substep("✅", "Gentle Laplacian smoothing applied [2 passes, selected only]")
        else:
            hlog_substep("⏭️ ", "Skipped smoothing (too many or zero back-faces)")

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

    hlog_substep("📊", f"Final repaired: {len(repaired.vertices)} verts // {len(repaired.faces)} faces")

    # ---- Step 3: Manifold3D watertight validation (non-destructive) -------
    hlog_substep("🔬", "Running Manifold3D watertight validation...")
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
# 3. DECIMATION (PyMeshLab quadric edge collapse)
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
            obj_base = os.path.splitext(os.path.basename(obj_path))[0]
            for ext in [".mtl", ".jpg", ".png", "_metallic.jpg", "_roughness.jpg"]:
                src = os.path.join(obj_dir, obj_base + ext)
                if os.path.exists(src) and obj_dir != out_dir:
                    shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))
        return mesh

    hlog_stage(4, "MESH DECIMATION")
    hlog("📐", "DECIMATE", f"Compressing: {current_faces} → {target_faces} faces...")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)

    # QUALITY: higher qualitythr preserves more geometric detail
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preserveboundary=True,
        preservetopology=True,
        qualitythr=0.7,  # higher = preserve more quality (was 0.5)
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
# 4. EXPORT — GLB (primary, PBR textures) + 3MF (vertex colors for printing)
# ===========================================================================

def export_glb_with_pbr(textured_dir, output_path):
    """
    Export as GLB with full PBR material textures (albedo, metallic, roughness).
    This preserves maximum texture quality — no vertex color degradation.
    """
    from hy3dpaint.convert_utils import create_glb_with_pbr_materials

    obj_path = os.path.join(textured_dir, "textured_mesh.obj")
    if not os.path.exists(obj_path):
        hlog_substep("⚠️ ", f"No textured OBJ found at {obj_path}")
        return None

    obj_base = os.path.splitext(obj_path)[0]
    textures = {
        'albedo': obj_base + '.jpg',
        'metallic': obj_base + '_metallic.jpg',
        'roughness': obj_base + '_roughness.jpg',
    }

    # Filter to only existing textures
    textures = {k: v for k, v in textures.items() if os.path.exists(v)}

    if 'albedo' not in textures:
        # Try .png
        png_albedo = obj_base + '.png'
        if os.path.exists(png_albedo):
            textures['albedo'] = png_albedo

    if not textures:
        hlog_substep("⚠️ ", "No texture files found — skipping GLB export")
        return None

    hlog_substep("📦", f"PBR textures: {', '.join(textures.keys())}")
    create_glb_with_pbr_materials(obj_path, textures, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    hlog_substep("✅", f"GLB exported ─── {size_mb:.1f} MB")
    return output_path


def bake_vertex_colors(mesh, albedo_path):
    """
    Sample the albedo texture at each vertex's UV coordinate to produce
    vertex colors. Used only for .3mf export (3D printing).
    """
    if not os.path.exists(albedo_path):
        hlog_substep("⚠️ ", f"Albedo not found at {albedo_path} ─── exporting colorless")
        return mesh

    texture = Image.open(albedo_path).convert("RGB")
    tex_array = np.array(texture)
    h, w = tex_array.shape[:2]

    uv = None

    if hasattr(mesh, "visual") and mesh.visual is not None:
        visual = mesh.visual
        if hasattr(visual, "uv") and visual.uv is not None:
            raw_uv = visual.uv
            if len(raw_uv) == len(mesh.vertices):
                uv = raw_uv
            else:
                # Per-face-vertex UVs: average to per-vertex
                hlog_substep("🔄", f"Remapping {len(raw_uv)} face-vertex UVs → "
                                    f"{len(mesh.vertices)} vertex UVs...")
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


def export_outputs(mesh, textured_dir, work_dir, stem, timestamp, args):
    """
    Export final outputs in requested formats.
    - GLB with PBR textures (always — this is the high-quality output)
    - 3MF with vertex colors (optional, for 3D printing)
    """
    hlog_stage(5, "FINAL EXPORT")

    # Find the albedo texture path
    albedo_path = os.path.join(textured_dir, "textured_mesh.jpg")
    if not os.path.exists(albedo_path):
        albedo_path = os.path.join(textured_dir, "textured_mesh.png")

    outputs = {}

    # ---- Apply roughness scaling (controls shininess) ---------------------
    roughness_scale = getattr(args, "roughness_scale", 1.0)
    if roughness_scale != 1.0:
        roughness_path = os.path.join(textured_dir, "textured_mesh_roughness.jpg")
        if os.path.exists(roughness_path):
            hlog_substep("✨", f"Adjusting roughness × {roughness_scale:.2f} "
                               f"({'very glossy' if roughness_scale <= 0.3 else 'shiny' if roughness_scale <= 0.5 else 'subtle sheen' if roughness_scale <= 0.7 else 'near default'})")
            rough_img = Image.open(roughness_path)
            rough_arr = np.array(rough_img).astype(np.float32)
            rough_arr = np.clip(rough_arr * roughness_scale, 0, 255).astype(np.uint8)
            Image.fromarray(rough_arr).save(roughness_path, quality=98)
        else:
            hlog_substep("⚠️ ", f"No roughness texture found at {roughness_path} — skipping")

    # ---- GLB export (primary — full PBR quality) --------------------------
    output_glb = os.path.join(work_dir, f"{stem}_{timestamp}.glb")
    hlog("📦", "EXPORT", f"GLB with PBR textures → {os.path.basename(output_glb)}")
    glb_result = export_glb_with_pbr(textured_dir, output_glb)
    if glb_result:
        outputs["glb"] = glb_result
    else:
        # Fallback: export mesh directly as GLB
        hlog_substep("🔄", "Falling back to trimesh GLB export...")
        mesh.export(output_glb)
        outputs["glb"] = output_glb

    # ---- 3MF export (for 3D printing — vertex colors) ---------------------
    if args.export_3mf:
        output_3mf = os.path.join(work_dir, f"{stem}_{timestamp}.3mf")
        hlog("📦", "EXPORT", f"3MF with vertex colors → {os.path.basename(output_3mf)}")

        mesh_for_3mf = mesh.copy()
        mesh_for_3mf = bake_vertex_colors(mesh_for_3mf, albedo_path)
        mesh_for_3mf.export(output_3mf, file_type="3mf")
        size_mb = os.path.getsize(output_3mf) / 1e6
        hlog_substep("✅", f"3MF export complete ─── {size_mb:.1f} MB")
        outputs["3mf"] = output_3mf

    # ---- Also copy the raw textured OBJ + textures for manual use ---------
    import shutil
    final_obj_dir = os.path.join(work_dir, "textured_obj")
    os.makedirs(final_obj_dir, exist_ok=True)
    for f in glob.glob(os.path.join(textured_dir, "textured_mesh*")):
        shutil.copy2(f, final_obj_dir)
    hlog_substep("💾", f"Raw textured OBJ + textures copied → textured_obj/")

    print("")
    return outputs


# ===========================================================================
# 5. FULL SINGLE-IMAGE PIPELINE
# ===========================================================================

def process_single(image_path, models, output_dir, args):
    """
    Full end-to-end pipeline for a single image:
      generation → repair → decimate → export GLB+3MF

    QUALITY CHANGES vs original:
    - No second ESRGAN pass (the texture pipeline already does one internally)
    - No pre-texture face reduction (let the texture pipeline handle it)
    - Higher texture resolution throughout
    - GLB export preserves full PBR textures
    """
    stem = Path(image_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(output_dir, f"{stem}_{timestamp}")
    os.makedirs(work_dir, exist_ok=True)

    t_total_start = time.time()

    # ---- Stage 1-2: Shape Generation + Texturing --------------------------
    textured_dir = generate_3d(image_path, models, work_dir, args)

    # ---- Offload all GPU models before post-processing --------------------
    offload_all_to_cpu(models["i23d_worker"], models["tex_pipeline"])

    # ---- Stage 3: Mesh Repair ---------------------------------------------
    textured_obj = os.path.join(textured_dir, "textured_mesh.obj")
    repaired_obj = os.path.join(work_dir, "repaired_mesh.obj")

    if os.path.exists(textured_obj):
        repaired_mesh = repair_mesh(textured_obj, output_path=repaired_obj)
    else:
        obj_files = glob.glob(os.path.join(textured_dir, "*.obj"))
        if obj_files:
            repaired_mesh = repair_mesh(obj_files[0], output_path=repaired_obj)
        else:
            raise FileNotFoundError(f"No OBJ found in {textured_dir}")

    # ---- REMOVED: Second ESRGAN pass --------------------------------------
    # The texture pipeline already runs ESRGAN 4× on each multiview image
    # BEFORE baking them into the UV atlas (see textureGenPipeline.py,
    # imageSuperNet). Running ESRGAN again on the composite atlas amplifies
    # UV seam artifacts and blending boundaries rather than improving detail.
    # The internal ESRGAN pass is sufficient.

    # ---- Stage 4: Decimation (optional) -----------------------------------
    if args.target_faces > 0:
        decimated_obj = os.path.join(work_dir, "decimated_mesh.obj")
        final_mesh = decimate_mesh(
            repaired_obj, target_faces=args.target_faces, output_path=decimated_obj
        )
    else:
        final_mesh = repaired_mesh
        hlog("📐", "DECIMATE", "Skipped (target_faces=0)")

    # ---- Stage 5: Export --------------------------------------------------
    outputs = export_outputs(
        final_mesh, textured_dir, work_dir, stem, timestamp, args
    )

    # ---- Restore shape model to GPU for next image ------------------------
    restore_shape_to_gpu(models["i23d_worker"])

    t_total = time.time() - t_total_start

    return outputs, t_total


# ===========================================================================
# 6. BATCH ORCHESTRATOR
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

    # Process each image sequentially
    results = {"success": [], "failed": []}

    for idx, img_path in enumerate(image_files, 1):
        hlog_image_header(idx, len(image_files), img_path.name)
        t0 = time.time()

        try:
            outputs, elapsed = process_single(str(img_path), models, str(output_dir), args)

            print("")
            print(f"  ┌─────────────────────────────────────────────────────────┐")
            print(f"  │  ✅ MISSION COMPLETE                                     │")
            for fmt, path in outputs.items():
                basename = os.path.basename(path)
                print(f"  │  📁 [{fmt.upper()}] {basename:<49}│")
            print(f"  │  ⏱️  {elapsed:.0f}s elapsed                                       │")
            print(f"  └─────────────────────────────────────────────────────────┘")

            results["success"].append(img_path.name)
        except Exception as e:
            elapsed = time.time() - t0
            hlog("❌", "FAIL", f"{img_path.name} ── aborted after {elapsed:.0f}s")
            hlog("💀", "FAIL", f"Cause: {e}")
            traceback.print_exc()

            # Nuclear VRAM cleanup after failure — must reclaim ALL GPU memory
            # before next image, otherwise cascading OOMs will kill the batch.
            try:
                hlog("🔄", "RECOVER", "NUCLEAR VRAM RECOVERY — forcing all tensors to CPU...")

                # Step 1: Move all known model components to CPU
                offload_shape_to_cpu(models["i23d_worker"])
                move_texture_to_cpu(models["tex_pipeline"])

                # Step 2: Force ALL parameters of mmgp-managed pipeline to CPU
                # (mmgp hooks can leave layers on GPU that normal .to("cpu") misses)
                try:
                    tex_pipe = models["tex_pipeline"]
                    mv = tex_pipe.models.get("multiview_model")
                    if mv is not None and hasattr(mv, "pipeline"):
                        for param in mv.pipeline.parameters():
                            if param.is_cuda:
                                param.data = param.data.to("cpu")
                        for buf in mv.pipeline.buffers():
                            if buf.is_cuda:
                                buf.data = buf.data.to("cpu")
                except Exception as e2:
                    hlog_substep("⚠️ ", f"mmgp deep cleanup partial: {e2}")

                # Step 3: Nuclear flush
                _force_vram_flush()
                _log_vram("after nuclear cleanup")

                # Step 4: Only restore shape if we have enough VRAM
                if torch.cuda.is_available():
                    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
                    if free_gb >= 8.0:
                        restore_shape_to_gpu(models["i23d_worker"])
                        hlog("✅", "RECOVER", "GPU state restored ── ready for next target")
                    else:
                        hlog("⚠️ ", "RECOVER",
                             f"Only {free_gb:.1f}GB free — shape stays on CPU, "
                             f"will reload at next image start")
            except Exception:
                hlog("⚠️ ", "RECOVER", "GPU recovery failed ── attempting continue")
                _force_vram_flush()

            results["failed"].append((img_path.name, str(e)))

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
        description="Hunyuan3D 2.1 → 3D Asset Batch Pipeline (Quality-Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUALITY PRESET (recommended for 16GB VRAM):
  python process_batch.py \\
    --input_dir ./test_images \\
    --output_dir ./save_dir/batch_output \\
    --low_vram_mode

SHINY MODELS (glossy/reflective surfaces):
  python process_batch.py \\
    --input_dir ./test_images \\
    --output_dir ./save_dir/batch_output \\
    --low_vram_mode \\
    --roughness_scale 0.5

MAXIMUM QUALITY (slower, may need careful VRAM monitoring):
  python process_batch.py \\
    --input_dir ./test_images \\
    --output_dir ./save_dir/batch_output \\
    --low_vram_mode \\
    --tex_resolution 512 \\
    --tex_render_size 2048 \\
    --tex_atlas_size 4096 \\
    --tex_max_views 8 \\
    --octree_resolution 384 \\
    --num_chunks 8000

SAFE MODE (if you get OOM errors):
  python process_batch.py \\
    --input_dir ./test_images \\
    --output_dir ./save_dir/batch_output \\
    --low_vram_mode \\
    --tex_resolution 512 \\
    --tex_render_size 1024 \\
    --tex_atlas_size 2048 \\
    --tex_max_views 6 \\
    --octree_resolution 256 \\
    --num_chunks 4000
        """,
    )

    # I/O
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Folder containing reference images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output folder for generated assets and intermediates")

    # Model paths (defaults match gradio_app.py)
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2.1",
                        help="HuggingFace repo or local path for shape model")
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-1",
                        help="Subfolder for shape model weights")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for shape model (default: cuda)")
    parser.add_argument("--low_vram_mode", action="store_true",
                        help="Enable mmgp VRAM offloading (required for 16GB cards)")

    # --- Shape generation params ---
    parser.add_argument("--num_steps", type=int, default=30,
                        help="Shape diffusion inference steps (default: 30). "
                             "Higher = better quality, slower. Range: 20-50.")
    parser.add_argument("--guidance_scale", type=float, default=5.5,
                        help="Shape generation guidance scale (default: 5.5). "
                             "Higher = more faithful to input, may lose detail.")
    parser.add_argument("--octree_resolution", type=int, default=384,
                        help="Marching cubes grid resolution (default: 384). "
                             "384 is good quality. 512 is max but very VRAM hungry. "
                             "256 is safe fallback.")
    parser.add_argument("--num_chunks", type=int, default=8000,
                        help="VAE decoder batch size (default: 8000). "
                             "CRITICAL VRAM LEVER: lower = less VRAM, slower. "
                             "8000 is safe for 16GB. Never use 200000 on 16GB.")
    parser.add_argument("--gen_face_reduce", type=int, default=0,
                        help="Pre-texture face reduction (default: 0 = skip). "
                             "Set to 0 to let the texture pipeline handle its own "
                             "internal remeshing. Pre-reducing causes double decimation.")

    # --- Texture quality params ---
    parser.add_argument("--tex_resolution", type=int, default=512,
                        help="Multiview diffusion resolution in px (default: 512). "
                             "512 is the max that fits 16GB VRAM reliably. "
                             "768 requires >20GB VRAM (do NOT use on 16GB).")
    parser.add_argument("--tex_render_size", type=int, default=2048,
                        help="Resolution for view→UV baking projection (default: 2048). "
                             "Matches ESRGAN 4× output from 512px views. "
                             "This does NOT affect diffusion VRAM — safe to keep high.")
    parser.add_argument("--tex_atlas_size", type=int, default=4096,
                        help="Output UV atlas resolution (default: 4096). "
                             "4096 = high detail for printing/rendering. "
                             "This does NOT affect diffusion VRAM — safe to keep high.")
    parser.add_argument("--tex_max_views", type=int, default=6,
                        help="Max viewpoints for texture generation (default: 6). "
                             "More views = better coverage but more VRAM. "
                             "6 is safe for 16GB. 8-9 may OOM. Range: 6-12.")

    # --- Post-processing params ---
    parser.add_argument("--target_faces", type=int, default=80000,
                        help="Target face count for final decimation (default: 80000). "
                             "Set to 0 to skip decimation entirely.")

    # --- Export options ---
    parser.add_argument("--export_3mf", action="store_true",
                        help="Also export .3mf with vertex colors (for 3D printing). "
                             "GLB with PBR textures is always exported.")
    parser.add_argument("--roughness_scale", type=float, default=1.0,
                        help="Scale roughness texture to control shininess (default: 1.0). "
                             "Lower = shinier/more reflective. "
                             "0.3 = very glossy, 0.5 = noticeably shiny, "
                             "0.7 = subtle sheen, 1.0 = unmodified (model default).")

    args = parser.parse_args()

    # Validate texture settings for 16GB VRAM
    if args.low_vram_mode:
        if args.tex_resolution > 512:
            print(f"  ⚠️  WARNING: tex_resolution={args.tex_resolution} WILL OOM on 16GB VRAM. "
                  f"Max safe value: 512. Overriding to 512.")
            args.tex_resolution = 512
        if args.tex_max_views > 8:
            print(f"  ⚠️  WARNING: tex_max_views={args.tex_max_views} risky on 16GB. "
                  f"Capping to 8.")
            args.tex_max_views = 8
        if args.num_chunks > 50000:
            print(f"  ⚠️  WARNING: num_chunks={args.num_chunks} may OOM on 16GB. "
                  f"Recommended: 8000.")
        if args.octree_resolution > 384:
            print(f"  ⚠️  WARNING: octree_resolution={args.octree_resolution} with "
                  f"num_chunks={args.num_chunks} may OOM. Consider octree_resolution=384.")

    process_batch(args)


if __name__ == "__main__":
    main()
