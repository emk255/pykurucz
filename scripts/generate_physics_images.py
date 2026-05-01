"""Generate hand-drawn-style physics infographics for the docs site
using Google's Gemini 3 Pro Image Preview model.

The intent is *not* to produce technical schematics — those are better
done as inline SVGs (see ``docs/physics/*.md`` for examples). This
script generates a small number of *cartoon-like, scientist-sketched*
illustrations that help a reader build intuition for a concept (Voigt
profile geometry, Stark microfield distribution, atmosphere layers,
etc.).

Style direction (lifted from ``~/Annotify_with_GPT_Image_2.0`` so it
stays consistent with related projects):

  - Hand-drawn, scientist-sketched aesthetic — not polished cartoon, not
    AI-generated-looking.
  - Muted, neutral palette: soft blues, beiges, light earth tones.
  - White / off-white background — never tinted.
  - Annotations and labels in legible, real English (no gibberish).
  - Concept-specific: every illustration calls out the *one* idea
    you're trying to teach.

Each generated PNG is written under ``docs/assets/physics/`` so the
markdown pages can reference them with a stable relative path. The
generation specs are kept inline below for reproducibility — if a
figure needs tweaking, edit the prompt here and re-run.

Usage
-----

::

    # Set your Google API key in ~/.env (key name "GOOGLE") then:
    python scripts/generate_physics_images.py            # generate all
    python scripts/generate_physics_images.py voigt      # one figure
    python scripts/generate_physics_images.py --list     # list specs
    python scripts/generate_physics_images.py --force    # regenerate

Requires::

    pip install google-genai pillow python-dotenv

The model used is ``gemini-3-pro-image-preview`` (matching the API call
shape in ``~/transform_paper.py``).

Notes
-----

The model takes 15–60 s per image. We rate-limit to one in flight at a
time. We only have 8 CPU threads available locally so we are deliberate
about not running this in a tight loop.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# --- Project layout ---------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "assets" / "physics"

# --- Generation specifications ---------------------------------------------
#
# Each entry is one figure. ``id`` is the filename stem. ``size`` should be
# one of the gpt-image-2 supported sizes (1024x1024, 1024x1536, 1536x1024).
# ``prompt`` is a self-contained brief — no shared template — so each
# figure can be re-run without context from the others.
#
# Prompts deliberately avoid asking for technical accuracy in mathematical
# detail (e.g. exact ratios in a Voigt profile). The model is prompted
# instead to *signal* the qualitative geometry (narrow core, wide wings,
# etc.) and to label clearly. Reviewers should still verify each output.


@dataclass
class FigureSpec:
    id: str
    page: str  # markdown page that will reference this figure
    title: str  # short caption / alt text
    aspect: str  # rough aspect-ratio hint included in the prompt
    prompt: str


_STYLE_BLOCK = """
STYLE — match exactly:
- Hand-drawn, scientist-sketched feel; like a chalkboard or notebook
  diagram drawn during a lecture.
- Pure white background (#FFFFFF). No coloured fill, no gradient.
- Muted palette only: soft slate blue (#5b6068 / #80858d), warm beige
  (#a8acb3 / #d2d1cd), occasional accent in deep navy (#2a3550). NO
  bright reds / yellows / pinks.
- Strokes look hand-drawn: slight wobble, varying line weight, no
  perfectly straight rules.
- Labels are in plain English (legible, no gibberish, no typos).
  Spell every word correctly. Use sans-serif handwriting-like font.
- Empty space is fine — do NOT fill the canvas with extra ornament.
- NO photorealistic objects, NO 3D rendering, NO drop shadows,
  NO logos, NO watermarks, NO emoji.
- The image must be the *concept illustration only* — no title bar,
  no caption strip, no border frame.
"""

FIGURES: list[FigureSpec] = [
    FigureSpec(
        id="voigt-profile",
        page="docs/physics/line-broadening.md",
        title="Voigt = Gaussian core × Lorentzian wings",
        aspect="3:2 landscape",
        prompt=f"""
A simple sketch illustrating that the Voigt profile is the convolution
of a Gaussian (Doppler core) and a Lorentzian (damping wings).

Composition:
- Three small panels arranged left-to-right with a "*" (convolution
  symbol) between the first two and an "=" sign between the second
  and third.
- Panel 1: a Gaussian curve labelled "Doppler (Gaussian)" — narrow,
  tall, symmetric, no wings.
- Panel 2: a Lorentzian curve labelled "damping (Lorentzian)" —
  shorter peak, long slowly-falling wings.
- Panel 3: the resulting Voigt profile labelled "Voigt", with a
  Gaussian-ish core and visible wings reaching further than the
  Gaussian alone.
- Below all three, a tiny shared horizontal axis with the symbol
  "Δλ / Δλ_D" (displacement in Doppler units).
- A small pencil-style annotation pointing at the Voigt wings reads
  "wings dominate at large Δλ".
- A tiny annotation pointing at the core reads "core stays Gaussian".

{_STYLE_BLOCK}
""",
    ),
    FigureSpec(
        id="atmosphere-layers",
        page="docs/physics/radiative-transfer.md",
        title="1-D plane-parallel photosphere with optical depth axis",
        aspect="3:2 landscape",
        prompt=f"""
A side-on cartoon of a 1-D plane-parallel stellar photosphere, showing
that radiative transfer integrates emission and absorption through a
stack of layers.

Composition:
- A vertical stack of about 8–10 thin horizontal layers (rectangles)
  drawn with hand-wobbly outlines, becoming progressively darker /
  denser from top to bottom. Each layer carries a tiny temperature
  number such as 4500 K, 5200 K, 6000 K, 7500 K, 9000 K (top to bottom).
- A vertical axis on the right labelled "optical depth τ" with arrow
  pointing downward and tick marks for τ = 0 (top) and τ = 1, τ = 10
  (bottom).
- A wavy outgoing ray near the top labelled "emergent photon" with a
  small arrow pointing up and slightly right.
- A short downward arrow labelled "incoming photon (none — vacuum)"
  at the very top, with a tiny strikethrough or "0" marker.
- Inside one of the middle layers, a small annotation "S_λ ≈ B_λ(T)"
  pointing at a sketched emission squiggle.
- Above the whole stack, a hand-written title "stellar photosphere
  (1-D)".

{_STYLE_BLOCK}
""",
    ),
    FigureSpec(
        id="opacity-stack",
        page="docs/physics/opacity.md",
        title="Continuum opacity contributions on a wavelength axis",
        aspect="3:2 landscape",
        prompt=f"""
A schematic plot of continuum opacity vs. wavelength for a Sun-like
photosphere, showing the contributions of different sources stacked on
a log-y axis.

Composition:
- A square plot region with hand-drawn axes. X-axis: "wavelength λ
  (nm)" from about 200 to 2000 (log scale, with major ticks at 200,
  500, 1000, 2000). Y-axis: "opacity κ_λ (log)".
- Several smooth labelled curves inside:
  * A broad bump centred near 500 nm rising on the red side, labelled
    "H⁻ bound-free" (highest in the optical).
  * A sharp step labelled "H I Balmer edge" at λ ≈ 365 nm.
  * A line that rises steeply blueward labelled "Rayleigh ~λ⁻⁴".
  * A flat horizontal line labelled "Thomson scattering".
  * A wide low bump in the IR labelled "H⁻ free-free".
- A short marginal note "(solar-type)" in the corner.
- Make sure curves do not overlap labels. Use the muted palette only.

{_STYLE_BLOCK}
""",
    ),
    FigureSpec(
        id="molecular-equilibrium",
        page="docs/physics/molecular-equilibrium.md",
        title="Molecules form deeper / cooler in the photosphere",
        aspect="3:2 landscape",
        prompt=f"""
A side-on cartoon of a cool dwarf photosphere illustrating that
molecules form preferentially in the deeper, cooler layers and
dissociate higher up.

Composition:
- A vertical stack of layers as in the radiative-transfer figure, but
  now annotated with cartoon "molecules" (a few O–H, C–O, Ti–O
  doublet circles) more densely populated in the lower layers and
  sparsely in the upper.
- Top layer label: "T ≈ 7000 K — atoms only".
- Middle layer label: "T ≈ 4000 K — diatomics start: CO, OH".
- Bottom layer label: "T ≈ 2800 K — TiO, H₂O, FeH dominate".
- A hand-drawn arrow on the left side labelled "decreasing T" pointing
  from top to bottom.
- A small inset on the right side: a tiny diagram with two atoms A and
  B and a dimer AB, with the equation "P_A · P_B / P_AB = K(T)"
  written next to it.
- A tiny pencil note at the bottom: "everything is coupled — solve as
  a system".

{_STYLE_BLOCK}
""",
    ),
    FigureSpec(
        id="hydrogen-stark",
        page="docs/physics/hydrogen-helium.md",
        title="Holtsmark microfield distribution → Stark-broadened H wings",
        aspect="3:2 landscape",
        prompt=f"""
A two-panel diagram illustrating why hydrogen lines have such broad,
non-Voigt wings: an ensemble of microfields from neighbouring charged
particles produces a Holtsmark distribution which then yields wide
Stark-broadened line wings.

Composition:
- Left panel: a schematic of a single hydrogen absorber (a small dot
  labelled "H atom") surrounded by a few wandering charged particles
  (electrons "e⁻", ions "p⁺") at irregular distances. Tiny arrows
  from each particle point at the H atom. Below the panel, a label
  "microfield F = sum of contributions".
- Middle separator: a small ">" or arrow with the text "ensemble
  average".
- Right panel: a plot with hand-drawn axes. X-axis labelled "Δλ from
  line centre". On the y-axis a Lorentzian curve (labelled "what a
  Voigt would predict") is drawn lightly, and an obviously broader,
  flatter curve (labelled "Stark-broadened wings (HPROF4)") is drawn
  on top in the navy accent. Both curves share the same peak at Δλ
  = 0.
- A small annotation "Holtsmark distribution → broad wings" between
  the two panels.

{_STYLE_BLOCK}
""",
    ),
    FigureSpec(
        id="resolution-comparison",
        page="docs/examples/resolution-comparison.md",
        title="Same line at four resolving powers",
        aspect="3:2 landscape",
        prompt=f"""
A four-panel comparison sketch showing how a single absorption line
appears at increasing resolving power R.

Composition:
- Four small panels arranged in a 2×2 grid (or single row, model's
  choice). Each panel has a hand-drawn pair of axes; the y-axis is
  "F_λ / F_cont" with tick marks at 0 and 1, the x-axis is
  "wavelength" (no tick numbers needed).
- Panel 1: label "R = 2 000". A wide, shallow saucer-shaped dip — the
  line is barely resolved, looks like a single broad smear.
- Panel 2: label "R = 20 000". The same dip but with visible deeper
  core; still smooth.
- Panel 3: label "R = 100 000". A sharper V-shape; the Voigt wings
  start to be visible.
- Panel 4: label "R = 300 000". A sharp narrow core that reaches
  ~10% of the continuum, with extended wings clearly visible — this
  is the line's "true" shape.
- A small horizontal arrow at the bottom labelled
  "increasing resolving power →".
- A small note in the corner: "same line, different R; pykurucz default
  R = 300 000".

{_STYLE_BLOCK}
""",
    ),
    FigureSpec(
        id="cool-star-tio",
        page="docs/examples/cool-star.md",
        title="How individual TiO lines pile up to make a bandhead",
        aspect="3:2 landscape",
        prompt=f"""
An explanatory sketch showing how the dense forest of individual TiO
rotational lines piles up at the blue edge of a vibrational band to
form a sharp "bandhead" feature, characteristic of cool-star spectra.

Composition:
- Top half: a sea of many vertical lines of varying depth standing on
  a horizontal axis labelled "wavelength λ". The lines are densely
  packed but become MORE densely packed as you move toward the LEFT
  (blue) side of the panel — this density gradient is the key thing
  to convey. Label this panel "individual TiO rotational lines".
- A short vertical "→" or downward bracket connecting top half to
  bottom half, labelled "summed opacity".
- Bottom half: the smooth resulting absorption profile — a sharp,
  near-vertical drop on the LEFT (the bandhead) followed by a slow
  recovery toward 1.0 on the RIGHT (the band tail). The y-axis is
  "F_λ / F_cont" with the absorption reaching deep, perhaps 0.3.
  Label this panel "observed bandhead".
- An annotation arrow pointing to the sharp blue edge labelled
  "bandhead — characteristic blue edge".
- A small note: "λ ≈ 705 nm (TiO γ system)".

{_STYLE_BLOCK}
""",
    ),
    FigureSpec(
        id="warm-start-convergence",
        page="docs/user-guide/emulator.md",
        title="Warm-start converges; cold-start can stall",
        aspect="3:2 landscape",
        prompt=f"""
A comparison sketch showing why the kurucz-a1 emulator matters: warm
starting from the emulator's prediction lets ATLAS converge cleanly,
whereas a generic cold start can stall in a local minimum and fail to
reach the convergence threshold within a reasonable iteration budget.

Composition:
- A single plot panel with hand-drawn axes. X-axis: "iteration number"
  with tick marks at 0, 5, 10, 15, 20, 25, 30. Y-axis: "max temperature
  change between iterations (log scale)" with a horizontal dashed line
  near the bottom labelled "convergence threshold (1e-3)".
- Two curves:
  * Curve A (in the warm beige tone): labelled "cold start (grey
    atmosphere)". Starts near the top, decreases for a few iterations,
    then FLATTENS OUT and OSCILLATES well ABOVE the convergence
    threshold for the rest of the run. The curve never crosses the
    dashed line — it bounces around at roughly 1e-2 with visible
    wobble. The shape suggests the iteration is stuck in a local
    minimum.
  * Curve B (in the navy/slate tone): labelled "warm start (kurucz-a1
    emulator)". Starts much LOWER than curve A at iteration 0, drops
    quickly and smoothly, and crosses the threshold around iteration
    10–15. After crossing, the curve stops or trails off below the
    line.
- A small annotation pointing at the flat oscillating beige curve:
  "stalled — does not converge".
- A small annotation pointing at the descending navy curve:
  "smooth descent into the basin of attraction".
- A short pencil note in a corner: "warm start lands inside the
  convergence basin; cold start can miss it entirely".

{_STYLE_BLOCK}
""",
    ),
    FigureSpec(
        id="pipeline-cartoon",
        page="docs/getting-started/first-spectrum.md",
        title="Stellar parameters → atmosphere → opacity → spectrum",
        aspect="3:2 landscape",
        prompt=f"""
A friendly four-step cartoon summarising the pykurucz pipeline for
first-time readers, drawn as a left-to-right sequence with arrows.

Composition:
1. LEFT-MOST: a small floating note pad with the four numbers written
   on it: "Teff = 5770", "log g = 4.44", "[M/H] = 0", "[α/M] = 0".
   Label "stellar parameters".
2. ARROW labelled "atmosphere emulator + atlas_py" pointing right.
3. SECOND PANEL: a tiny stack of horizontal layers (like the
   atmosphere-layers figure) with cooler outer layers and hotter
   inner layers; label "model atmosphere".
4. ARROW labelled "synthe_py: line opacity + RT" pointing right.
5. THIRD PANEL: a small chunk of synthetic spectrum (a few absorption
   lines on a continuum), drawn as if printed on graph paper; label
   "synthetic spectrum".
6. ABOVE the whole sequence, a small handwritten title:
   "what pykurucz does, in one line".
7. NO mention of equations or specific formulas; this is a friendly
   overview.

{_STYLE_BLOCK}
""",
    ),
    FigureSpec(
        id="fortran-parity",
        page="docs/architecture/fortran-parity.md",
        title="Fortran vs Python flux comparison with residual band",
        aspect="3:2 landscape",
        prompt=f"""
A three-panel diagram illustrating the Fortran-parity validation
methodology used by pykurucz: top spectrum (Fortran reference), middle
spectrum (Python output), and bottom residual.

Composition:
- THREE stacked panels, each with shared X-axis "wavelength λ" (no
  tick numbers needed; they share the same range).
- Top panel: a sketched stellar spectrum with several absorption
  lines, labelled "Fortran ATLAS12 + SYNTHE (reference)" in the
  beige/warm tone.
- Middle panel: an essentially-identical sketched spectrum overlaid in
  the navy/slate tone, labelled "pykurucz (Python)". The two
  spectra should look visually indistinguishable.
- Bottom panel: a much shorter residual panel (about 1/3 the height of
  the others) showing a flat near-zero curve hovering very close to
  zero, with horizontal dashed lines at +1e-3 and -1e-3 labelled
  "±0.1% threshold". The residual stays well within the band.
  Y-axis label: "(Python − Fortran) / continuum".
- On the right of the residual panel, a short text note in the navy
  tone: "validated end-to-end on the test grid".

{_STYLE_BLOCK}
""",
    ),
]


# --- API plumbing -----------------------------------------------------------


def _load_api_key() -> str:
    """Resolve the Google API key from ~/.env or the environment."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(os.path.expanduser("~/.env"))
    except Exception:  # pragma: no cover - dotenv is optional
        pass
    key = os.environ.get("GOOGLE") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            "Google API key not found.\n"
            "  Set GOOGLE in ~/.env (preferred), or export GOOGLE_API_KEY."
        )
    return key


def _generate(spec: FigureSpec, *, force: bool = False) -> Path:
    """Generate the image for *spec* and write it to OUT_DIR / <id>.png.

    Uses Gemini 3 Pro image-preview via ``client.models.generate_content``,
    requesting only the IMAGE modality. The aspect-ratio hint is folded
    into the prompt because the Gemini image API does not yet expose an
    explicit size parameter (the model picks a sensible canvas size).
    """
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{spec.id}.png"
    if out_path.exists() and not force:
        print(f"  [skip] {spec.id}: {out_path} already exists (use --force)")
        return out_path

    client = genai.Client(api_key=_load_api_key())
    full_prompt = spec.prompt + (
        f"\n\nFRAMING: render at a {spec.aspect} aspect ratio.\n"
        "RETURN: only the finished image, no surrounding chrome.\n"
    )

    print(f"  [gen ] {spec.id} ({spec.aspect}) — this can take 15–60 s …")
    t0 = time.time()

    # Stream the response and pick out the inline image bytes.
    saved = False
    for chunk in client.models.generate_content_stream(
        model="gemini-3-pro-image-preview",
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=full_prompt)]),
        ],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    ):
        if (
            not chunk.candidates
            or chunk.candidates[0].content is None
            or not chunk.candidates[0].content.parts
        ):
            continue
        part = chunk.candidates[0].content.parts[0]
        if part.inline_data and part.inline_data.data:
            out_path.write_bytes(part.inline_data.data)
            saved = True
            break
    if not saved:
        raise RuntimeError(f"no image data returned for {spec.id}")

    dt = time.time() - t0
    print(f"  [ok  ] {spec.id} -> {out_path}  ({dt:.1f} s)")
    return out_path


def _list() -> None:
    print(f"{'id':<24} {'page':<40} {'aspect':<14}  title")
    print("-" * 100)
    for s in FIGURES:
        print(f"{s.id:<24} {s.page:<40} {s.aspect:<14}  {s.title}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ids",
        nargs="*",
        help="Figure ids to generate (default: all). Use --list to see them.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the available figure specs and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if the output file already exists.",
    )
    args = parser.parse_args()

    if args.list:
        _list()
        return

    by_id = {s.id: s for s in FIGURES}
    if args.ids:
        unknown = [i for i in args.ids if i not in by_id]
        if unknown:
            sys.exit(f"unknown figure ids: {unknown!r}; use --list to see them.")
        targets = [by_id[i] for i in args.ids]
    else:
        targets = FIGURES

    for spec in targets:
        try:
            _generate(spec, force=args.force)
        except Exception as e:  # pragma: no cover
            print(f"  [fail] {spec.id}: {e}")


if __name__ == "__main__":
    main()
