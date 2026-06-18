# Tests

## Debug with stage 2
### Usage Examples

Basic run (defaults):
python tests/debug_stage2_scene.py

Change object shape and mirror distance:
python tests/debug_stage2_scene.py --shape sphere
--mirror_gap_ahead 5.0

Adjust camera to zoom in and look down more:
python tests/debug_stage2_scene.py --camera_dist 0.8
--camera_elevation 30

Stripped-down scene (no walls/floor), high resolution:
python tests/debug_stage2_scene.py --no_walls --no_floor        
--render_size 1024

Front-facing camera with random object rotation:
python tests/debug_stage2_scene.py --camera_method fixed_front  
--random_rotation

Run with --help to see all options and their defaults:
python tests/debug_stage2_scene.py --help
### Recommended Parameters
 python tests/debug_stage2_scene.py --mirror_gap_ahead 1.7 --camera_dist 2.2 --camera_azim_min -10 --camera_azim_max 10 --camera_look_at_height 1.8 --camera_elevation 26

 ## Debug with stage 1 and stage 2
 ### Usage Examples

 Basic run (defaults):
 python tests/debug_stage1_stage2.py

 ### Recommended Parameters
  python tests/debug_stage1_stage2.py -p "A wooden chair" --object_scale 1.5 --mesh_model trellis --mirror_gap_ahead 1.7 --camera_dist 2.2 --camera_azim_min -10 --camera_azim_max 10 --camera_look_at_height 1.8 --camera_elevation 26 --object_base_rotation 180 --random_rotation --include_mirror_wall --render_size 1024

## Debug with stage 3
python tests/debug_stage3_flux_omini.py -d tests/mock_data/depth_map.png -p "A wooden chair in front of a mirror"
