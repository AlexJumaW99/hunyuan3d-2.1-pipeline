# ═══════════════════════════════════════════════════════
#  MEROE 3D ENGINE — DOCKER CHEATSHEET
# ═══════════════════════════════════════════════════════

# -------------------------------------------------------
#  1. STARTING & STOPPING
# -------------------------------------------------------

# Start the batch container (after reboot or stop)
docker start hy3d21_batch

# Get a shell inside the running container
docker exec -it hy3d21_batch bash

# Stop the container (models unloaded, data preserved)
docker stop hy3d21_batch

# Check if containers are running
docker ps

# Check ALL containers (including stopped)
docker ps -a

# -------------------------------------------------------
#  2. ADDING TEST IMAGES (run from HOST terminal)
# -------------------------------------------------------

# Clear old test images
docker exec hy3d21_batch rm -f /workspace/Hunyuan3D-2.1/test_images/*

# Copy a single image
docker cp ~/path/to/image.png hy3d21_batch:/workspace/Hunyuan3D-2.1/test_images/

# Copy all images from a folder
for f in ~/Pictures/3d_ai_test_images/*.{png,jpg,jpeg,webp}; do
    [ -f "$f" ] && docker cp "$f" hy3d21_batch:/workspace/Hunyuan3D-2.1/test_images/
done

for f in ~/Pictures/single_image/*.{png,jpg,jpeg,webp}; do
    [ -f "$f" ] && docker cp "$f" hy3d21_batch:/workspace/Hunyuan3D-2.1/test_images/
done

# Verify images are there
docker exec hy3d21_batch ls /workspace/Hunyuan3D-2.1/test_images/

# -------------------------------------------------------
#  3. RUNNING THE PIPELINE (run from INSIDE container)
# -------------------------------------------------------

# First: get inside the container
docker exec -it hy3d21_batch bash

# Then run the pipeline
cd /workspace/Hunyuan3D-2.1
python process_batch.py --input_dir ./test_images --output_dir ./save_dir/batch_output --low_vram_mode

# Lower resolution if you get VRAM errors (trades detail for stability)
python process_batch.py --input_dir ./test_images --output_dir ./save_dir/batch_output --low_vram_mode --octree_resolution 256

# -------------------------------------------------------
#  4. VIEWING OUTPUTS (from HOST terminal)
# -------------------------------------------------------

# List all batch outputs
ls ~/hunyuan3d-outputs/batch_output/

# Open a GLB file in browser for 3D preview
# Drag the .glb file to: https://gltf-viewer.donmccurdy.com

# -------------------------------------------------------
#  5. UPDATING process_batch.py (from HOST terminal)
# -------------------------------------------------------

# Copy updated script into container
docker cp ~/Downloads/process_batch.py hy3d21_batch:/workspace/Hunyuan3D-2.1/process_batch.py

# Also update your git repo
cp ~/Downloads/process_batch.py ~/Hunyuan3D-2.1/process_batch.py
cd ~/Hunyuan3D-2.1
git add -A
git commit -m "description of changes"
git push

# -------------------------------------------------------
#  6. FREEING GPU VRAM (before running batch)
# -------------------------------------------------------

# Check what's using your GPU (from host)
nvidia-smi

# Option A: Close Chrome, extra terminals, file manager
# Option B: Switch to text-only TTY (frees ~3GB)
#   Press Ctrl+Alt+F3 → log in → run docker exec from there
#   Press Ctrl+Alt+F1 to return to desktop when done

# -------------------------------------------------------
#  7. SWITCHING TO GRADIO GUI
# -------------------------------------------------------

# Stop batch container, start original
docker stop hy3d21_batch
docker start hy3d21
docker exec -it hy3d21 bash
cd /workspace/Hunyuan3D-2.1
python gradio_app.py --model_path tencent/Hunyuan3D-2.1 --subfolder hunyuan3d-dit-v2-1 --texgen_model_path tencent/Hunyuan3D-2.1 --low_vram_mode --port 7860
# Then open http://localhost:7860 in browser

# Switch back to batch
docker stop hy3d21
docker start hy3d21_batch
docker exec -it hy3d21_batch bash

# -------------------------------------------------------
#  8. NUCLEAR OPTIONS (if things go wrong)
# -------------------------------------------------------

# Force kill a stuck container
docker kill hy3d21_batch

# Remove a container completely (lose installed packages)
docker rm hy3d21_batch

# Create a fresh container from scratch
docker run --gpus all -it \
  --name hy3d21_batch \
  -v /home/meroe/hunyuan3d-outputs:/workspace/Hunyuan3D-2.1/save_dir \
  hunyuan3d21:latest \
  bash

# After fresh container: reinstall packages + copy files
pip install mmgp==3.7.6 optimum-quanto==0.2.7 manifold3d lxml
# Then from HOST: copy in process_batch.py and patched files
# (see section 5 above)

# -------------------------------------------------------
#  9. INSTALL PACKAGES (after fresh container only)
# -------------------------------------------------------

pip install mmgp==3.7.6 optimum-quanto==0.2.7 manifold3d lxml
