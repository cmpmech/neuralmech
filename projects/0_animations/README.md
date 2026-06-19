# Animations

Turns folders of rendered frames into video files for the slides. The frames are
produced elsewhere: any driver run with `--animate` writes its frames to
`results/animations/animation_frames/<name>/`. This folder only assembles them.

- `frames_to_animation.py`
  ffmpeg wrapper that encodes one frame folder into an H.264 `.mp4`. Takes
  `--frames_dir`, `--output`, and optional `--pattern` / `--fps` / `--crf`
  (quality, lower is better)
- `make_animations.sh`
  the full collection: one `frames_to_animation.py` call per animation,
  grouped by chapter. Run after the frame folders have been generated. Mostly intended as a overview and lookup.
