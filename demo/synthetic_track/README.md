# Synthetic demonstration track (in progress — stages 1-2)

This directory holds the first stage of a fully synthetic demonstration track for
the CARVE pipeline. Nothing here is derived from real footage: every clip and
image was generated from text, and no real road users, vehicles, or locations
appear anywhere in this track.

## Contents

- `videos/src01_V0.mp4` … `src12_V0.mp4` — twelve synthetic accident-free
  CCTV-style source clips (fixed-camera traffic scenes; raw ~5 s generations,
  1280x720). Each clip passed a manual screen: fixed viewpoint, no collision,
  at least two moving vehicles, clear road structure.
- `references/src01_refE.png` … `src12_refE.png` — the stage-one environment
  reference frames produced by an image-editing model from each source's first
  frame, covering three environment compositions (night+rain, day+fog,
  post-rain dusk), four sources each. Each reference passed a manual screen:
  road topology and vehicle layout preserved, requested environment realized,
  no spurious accident content.
- `previews/src01_V0_frames.png` … `src12_V0_frames.png` — six-frame preview
  grids (uniformly sampled) of each source clip, for quick inspection in
  mirrors that do not render video files.
- `quartets/` — the first two complete quartets (src03, src05): all four
  branches per source in the 64-frame / 16 fps protocol format, each frame
  carrying a visible SYNTHETIC watermark, plus one four-row comparison grid
  per quartet (`*_quartet_grid.png`, rows V0 / VE / VA / VAE, six frames per
  row) so the counterfactual structure is inspectable in mirrors that do not
  render video. Branches passed the manual screen: environment realized on VE
  with no spurious accident, mid-clip vehicle contact visible and sustained
  on VA / VAE, fixed viewpoint, source-consistent layout. Candidates that
  failed the screen (for example, a late near-collision without clear
  contact) were excluded, mirroring how the quality gates discard
  borderline generations. Programmatic gate runs and audit numbers ship
  with the completed track.

## What comes next

The completed track ships with the code release: the edited branches (VE, VA,
VAE) for each source, carried through the three quality gates, quartet
construction, and the audit metrics, covering all illumination, weather, and
road-surface levels and all six accident types, with rejected candidates and
gate logs included. Final media are normalized to the 64-frame / 16 fps
protocol format and carry a synthetic-origin provenance note.

## Purpose and scope

This track is an executable demonstration of the pipeline on media that anyone
can download and inspect. A fully generated set covers a different visual
domain from real CCTV, so it demonstrates the machinery rather than
substituting for the real-footage benchmark.
