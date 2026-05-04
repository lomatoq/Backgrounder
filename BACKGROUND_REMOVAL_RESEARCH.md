# Background Removal Research Notes

Last checked: 2026-05-04.

## Practical default

For a one-click background remover, the best default path is a routed
multi-expert system, not a single universal model or one global post-process.
The May 2026 research direction is: cheap base alpha first, measure image/edge
failure mode, then send only the right cases to the expensive expert.

- `BiRefNet_HR-matting` is a strong local default for detailed image cutouts.
- `BEN2` is useful as a second opinion and provides a confidence signal.
- Uncertainty-gated sharpening improves hard edges without adding another large
  model. Depth Anything stays opt-in because it can over-expand trimaps and
  hurt normal cutouts.
- `SAM 3.1` / SAM-style segmenters are useful as object priors and for
  promptable video/object selection. SAM 3.1 mainly improves multi-object video
  tracking throughput with Object Multiplex; it still produces segmentation
  masks, not production alpha mattes.
- SDMatte/LiteSDMatte-style diffusion matting is a heavy expert for fine detail,
  not something to run blindly on every cartoon/logo/product.

Implemented product rule:

- clean solid border + flat asset type -> connected chroma cleanup + despill;
- busy/natural border -> preserve the neural alpha, no chroma cleanup;
- fur/plants/transparent/complex -> allow SDMatte auto route when enabled;
- cartoon/logo snap is disabled unless a future route explicitly enables it.

Primary notes used:

- Local PDF: `C:/Users/nirrt/Downloads/Катынг-эдж рашэнні для выдалення фону і матынгу на май 2026.pdf`
- Meta SAM 3.1 blog/update: https://ai.meta.com/blog/segment-anything-model-3/
- Meta SAM 3 publication: https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- Meta SAM 3 GitHub release notes: https://github.com/facebookresearch/sam3
- ZIM paper page: https://huggingface.co/papers/2411.00626
- SDMatte paper page: https://huggingface.co/papers/2508.00443
- BiRefNet paper page: https://www.sciopen.com/article/10.26599/AIR.2024.9150038
- BRIA RMBG-2.0 repository/model notes: https://github.com/Bria-AI/RMBG-2.0

## SAM 3.1

SAM 3.1 is attractive for video, multi-object tracking, and prompt-driven object
selection. It is not the best first stage for automatic background removal
because the app usually needs a full alpha matte for "the subject", not a
prompted instance mask. The better product shape is:

- Default: automatic BiRefNet/BEN2 alpha matte.
- Advanced: SAM-style model when the user wants a specific object/concept or a
  hard multi-object scene.
- Future video mode: SAM 3.1-style multiplexed tracking for faster multi-object
  processing across frames.

Sources:

- Meta SAM 3 / 3.1 blog and publication pages:
  https://ai.meta.com/blog/segment-anything-model-3/
  https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- BRIA RMBG-2.0 model card:
  https://huggingface.co/briaai/RMBG-2.0
- ComfyUI-RMBG model comparison and integration notes:
  https://github.com/1038lab/ComfyUI-RMBG

## UI decision

The app should expose a small simple surface:

- `Smart Auto`: best default speed/quality balance.
- `Fast`: single-model path for quick work.
- `Max Quality`: slower, opt-in refiners for hard images.

Developer controls remain available under the accordion so experiments do not
pollute the normal workflow.

## Flat Background / CG Cutout Fix

The screenshot failure mode is a classic keying problem, not a model-size
problem: the neural mask keeps a semi-transparent ring of the original blue
backdrop. Film/VFX keyers handle this with clean-plate estimation, black/white
matte clipping, edge shrink, antialiasing, and despill.

Implementation decision:

- Detect near-solid backgrounds from border pixels.
- Estimate the background color as a clean plate.
- Build a combined RGB-distance + hue/dominance key matte.
- Treat keyed pixels as background only when they are connected to the image
  border. This prevents blue costume/logo details or dark cartoon outlines from
  becoming holes just because they share the screen hue.
- Apply a tiny connected-background halo shell for antialiased spill, then
  black-clip true background and white-clip confident foreground.
- Keep regression tests for: flat blue halo removal, dark outline preservation,
  enclosed blue foreground detail preservation, and busy-border skip.

Relevant references:

- Notch Chroma Key docs, especially clean plate, fully keyed areas, shrink edge,
  and spill suppression:
  https://manual.notch.one/2026.1/en/docs/reference/nodes/post-fx/image-processing/chroma-key/
- Blender Keying node docs, one-stop keying plus despill/matte tuning:
  https://docs.blender.org/manual/en/latest/compositing/types/keying/keying.html
- PyMatting trimap workflow:
  https://pymatting.github.io/
- Rembg alpha matting options and erosion-based edge cleanup:
  https://stackoverflow.com/questions/70520673/when-using-rembg-in-python-to-remove-image-background-how-can-i-turn-on-the-alp

## CorridorKey / Corridor Crew Keying

Best match for the blue/green rim problem: CorridorKey.

CorridorKey is not just a mask generator. It solves the screen-color unmixing
problem: given an RGB green-screen frame plus a coarse alpha hint, it predicts:

- clean linear alpha;
- straight foreground RGB as if the green screen was never there;
- processed RGBA / comp outputs.

This matters for our screenshots because SDMatte and BiRefNet can improve alpha
but still leave original blue RGB in semi-transparent edge pixels. CorridorKey is
designed specifically for that edge-contamination failure mode.

Practical integration plan for Backgrounder:

- keep BiRefNet/BEN2 as the coarse AlphaHint generator;
- detect near-solid blue/green border color;
- for green screen, pass RGB + coarse alpha directly to CorridorKey;
- for blue screen, run a blue-to-green adapter before CorridorKey, then swap the
  recovered foreground channels back after inference;
- use CorridorKey output foreground + alpha as final RGBA;
- keep SDMatte as a generic matting refiner, not as the keying solution.

Useful repos / forks:

- Upstream CorridorKey:
  https://github.com/nikopueringer/CorridorKey
- CorridorKey Python API docs:
  https://github.com/nikopueringer/CorridorKey/blob/main/CorridorKeyModule/README.md
- EZ-CorridorKey GUI fork:
  https://github.com/edenaion/EZ-CorridorKey
- CorridorKey OpenVINO community extension:
  https://github.com/daniil-lyakhov/CorridorKeyOpenVINO
- ComfyUI-CorridorKey:
  https://github.com/SeanBRVFX/ComfyUI-CorridorKey
- CorridorKey Runtime / Resolve OFX:
  https://github.com/alexandremendoncaalvaro/CorridorKey-Runtime
- CorridorKey ONNX / MLX model ladder:
  https://huggingface.co/alexandrealvaro/CorridorKey
- CorridorKey AE:
  https://github.com/iamjoshuadavies/corridorkey-ae
- CorridorKey OFX Windows wrapper:
  https://github.com/gitcapoom/corridorkey_ofx
- Sorceress True Pixel public tool page (closed source, exposes "Pick Chroma
  Key Color" beside "CorridorKey"):
  https://sorceress.games/pixel-art

Constraints:

- License is effectively CC BY-NC-SA 4.0 with extra restrictions; do not bake it
  into commercial/API workflows without reviewing terms.
- Upstream is green-screen-first. Its `color_utils.py` despill is green-specific.
- The strongest blue-screen implementation found is CorridorKey-Runtime. It has
  `ScreenColorMode { Green, Blue }`, estimates the blue screen from the image,
  maps it into the green-domain before inference, and restores RGB after. Its
  tests verify this beats a naive blue/green channel swap for off-axis blue.
- CorridorKey AE has a useful HSV hue-rotation auto-hint keyer, but the current
  source restricts auto-detect to the green hue range.
- I did not find a verified open-source CorridorKey fork with true arbitrary
  key-color model inference. The practical all-color fix for Backgrounder is
  therefore a classic clean-plate/hue/dominance keyer after the neural matte,
  plus a future optional CorridorKey-Runtime style green/blue backend.
- It wants a coarse alpha hint; our existing pipeline can provide that.
- Model checkpoint is much smaller than SDMatte (~300-400 MB), but runtime still
  wants a decent GPU.
