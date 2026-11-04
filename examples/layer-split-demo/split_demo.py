#!/usr/bin/env python3
"""Layer-split demo: original pixels -> parts -> exact recomposite.

Only Pillow + numpy. Run:  python3 split_demo.py
Everything is procedural; no real character art is used.

What it shows
  1. A transparent source with anti-aliased edges and a semi-transparent outline.
  2. Splitting by assigning ORIGINAL pixels to parts with authored masks that
     have their own x/y and crop boxes.
  3. A semi-transparent edge shared by two parts (fringe over face/eye):
     "reverse alpha / foreground colour" unmixing  C = F*a + B*(1-a).
  4. A hidden underlay painted under the fringe, revealed only when it moves.
  5. layers.json in the tutorial's manifest format, recomposite in draw order,
     assert RGBA max diff == 0.
  6. The "faint scatter inflates getbbox()" pitfall and a component-based box.
"""
from collections import deque
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
W = H = 160            # canvas of this demo (your project: the source image size)
SS = 4                 # supersampling factor used to make soft (anti-aliased) masks

SKIN = (246, 214, 190)
OUTLINE = (90, 60, 50)
WHITE = (250, 250, 252)
IRIS = (40, 150, 140)
LASH = (45, 30, 30)
HAIR = (30, 30, 40)


def soft(draw_fn):
    """Rasterise a shape at SS x and box-filter down -> float coverage in [0,1]."""
    im = Image.new("L", (W * SS, H * SS), 0)
    draw_fn(ImageDraw.Draw(im), SS)
    return np.asarray(im.resize((W, H), Image.Resampling.BOX), dtype=np.float64) / 255.0


# ---- authored shapes (the same shapes are used as part masks later) ----------
face_shape = lambda d, s: d.ellipse([30 * s, 20 * s, 130 * s, 145 * s], fill=255)
ring_shape = lambda d, s: d.ellipse([27 * s, 17 * s, 133 * s, 148 * s], fill=255)
white_shape = lambda d, s: d.ellipse([52 * s, 68 * s, 90 * s, 88 * s], fill=255)
iris_shape = lambda d, s: d.ellipse([63 * s, 68 * s, 81 * s, 88 * s], fill=255)
lash_shape = lambda d, s: d.arc([50 * s, 62 * s, 92 * s, 90 * s], 195, 345, fill=255, width=4 * s)
fringe_shape = lambda d, s: d.polygon([(70 * s, 22 * s), (84 * s, 22 * s), (80 * s, 70 * s),
                                       (74 * s, 98 * s), (66 * s, 100 * s), (70 * s, 66 * s)], fill=255)


def over(dst_rgb, dst_a, rgb, a):
    """Straight-alpha 'over' in float. Returns new (rgb, a)."""
    out_a = a + dst_a * (1 - a)
    num = rgb * a[..., None] + dst_rgb * (dst_a * (1 - a))[..., None]
    out_rgb = np.where(out_a[..., None] > 0, num / np.maximum(out_a, 1e-12)[..., None], 0)
    return out_rgb, out_a


def to_u8(rgb, a):
    rgba = np.dstack([np.rint(rgb), np.rint(a * 255)]).clip(0, 255).astype(np.uint8)
    rgba[rgba[..., 3] == 0, :3] = 0          # transparent pixels store black
    return rgba


def draw_source():
    rgb, a = np.zeros((H, W, 3)), np.zeros((H, W))
    layers = [(OUTLINE, soft(ring_shape) * 0.55),   # semi-transparent outline
              (SKIN, soft(face_shape)),
              (WHITE, soft(white_shape)),
              (IRIS, soft(iris_shape) * soft(white_shape)),
              (LASH, soft(lash_shape)),
              (HAIR, soft(fringe_shape))]
    for colour, cov in layers:
        rgb, a = over(rgb, a, np.broadcast_to(np.array(colour, float), (H, W, 3)), cov)
    return to_u8(rgb, a)


def redraw_without_fringe():
    """Stand-in for 'paint the hidden region' (real projects: image generation)."""
    rgb, a = np.zeros((H, W, 3)), np.zeros((H, W))
    for colour, cov in [(SKIN, np.ones((H, W))), (WHITE, soft(white_shape)),
                        (IRIS, soft(iris_shape) * soft(white_shape)), (LASH, soft(lash_shape))]:
        rgb, a = over(rgb, a, np.broadcast_to(np.array(colour, float), (H, W, 3)), cov)
    return np.rint(rgb).clip(0, 255).astype(np.uint8)


def main_component_box(alpha, threshold=16, pad=2):
    """Bounding box of the largest 4-connected component with alpha >= threshold,
    padded so the anti-aliased rim (below threshold) is kept."""
    on = alpha >= threshold
    seen = np.zeros_like(on)
    best = []
    for y0, x0 in zip(*np.nonzero(on)):
        if seen[y0, x0]:
            continue
        comp, q = [], deque([(y0, x0)])
        seen[y0, x0] = True
        while q:
            y, x = q.popleft(); comp.append((y, x))
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < on.shape[0] and 0 <= xx < on.shape[1] and on[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; q.append((yy, xx))
        if len(comp) > len(best):
            best = comp
    ys, xs = zip(*best)
    h, w = alpha.shape
    return (int(max(min(xs) - pad, 0)), int(max(min(ys) - pad, 0)), int(min(max(xs) + 1 + pad, w)), int(min(max(ys) + 1 + pad, h)))


def save_part(name, rgba, order, visible, sdk_name, mask=None):
    box = Image.fromarray(rgba).getbbox()      # parts here have no scatter; see scatter demo
    x0, y0, x1, y1 = box
    path = f"layers/{order:02d}-{name}.png"
    Image.fromarray(rgba[y0:y1, x0:x1]).save(OUT / path)
    entry = {"name": name, "sdk_name": sdk_name, "texture": path, "x": x0, "y": y0,
             "width": x1 - x0, "height": y1 - y0, "order": order, "visible": visible}
    if mask is not None:
        entry["mask"] = mask
    return entry


def save_mask(name, cov_bool, purpose):
    m = (cov_bool * 255).astype(np.uint8)
    x0, y0, x1, y1 = Image.fromarray(m).getbbox()
    path = f"masks/{name}.png"
    Image.fromarray(m[y0:y1, x0:x1]).save(OUT / path)
    return {"texture": path, "x": x0, "y": y0, "purpose": purpose}


def composite(manifest, shift=None, skip=()):
    rgb, a = np.zeros((H, W, 3)), np.zeros((H, W))
    for e in sorted(manifest["layers"], key=lambda e: e["order"]):
        if not e["visible"] or e["name"] in skip:
            continue
        im = np.asarray(Image.open(OUT / e["texture"]).convert("RGBA"), dtype=np.float64)
        part = np.zeros((H, W, 4))
        dx = (shift or {}).get(e["name"], 0)
        x, y = e["x"] + dx, e["y"]
        part[y:y + im.shape[0], x:x + im.shape[1]] = im
        if "mask" in e:                          # masks have their own x/y
            mk = np.asarray(Image.open(OUT / e["mask"]["texture"]), dtype=np.float64) / 255
            full = np.zeros((H, W))
            mx, my = e["mask"]["x"], e["mask"]["y"]
            full[my:my + mk.shape[0], mx:mx + mk.shape[1]] = mk
            part[..., 3] *= full
        rgb, a = over(rgb, a, part[..., :3], part[..., 3] / 255)
    return to_u8(rgb, a)


def main():
    (OUT / "layers").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)

    src = draw_source()
    Image.fromarray(src).save(OUT / "source.png")
    C = src[..., :3].astype(np.float64)
    A = src[..., 3].astype(np.float64)

    # ---- authored ownership (hard labels, topmost wins) + fringe soft mask ----
    white_m = soft(white_shape) >= 0.5
    iris_m = (soft(iris_shape) >= 0.5) & white_m
    lash_m = soft(lash_shape) >= 0.5
    label = np.where(A > 0, 1, 0)                 # 1 face (incl. its outline ring)
    label[white_m] = 2; label[iris_m] = 3; label[lash_m] = 4
    w = soft(fringe_shape)                         # fringe coverage = authored soft mask
    names = {1: ("Face", "Face"), 2: ("EyeWhiteL", "EyeWhiteR"), 3: ("IrisL", "IrisR"), 4: ("UpperLashL", "UpperLashR")}

    base = {k: np.zeros((H, W, 4)) for k in names}
    fringe = np.zeros((H, W, 4))
    full = w >= 1.0
    edge = (w > 0) & ~full
    assert (A[w > 0] == 255).all(), "demo keeps the fringe inside the opaque face"

    # 1) pixels owned by one part: copy original RGBA
    for k in names:
        sel = (label == k) & (w == 0)
        base[k][sel, :3] = C[sel]; base[k][sel, 3] = A[sel]
    # 2) fully covered by fringe: fringe takes the original pixel, base gets nothing
    fringe[full, :3] = C[full]; fringe[full, 3] = 255
    # 3) shared edge: unmix  C = F*a + B*(1-a), F = fringe colour, a = coverage
    F = np.median(C[full], axis=0)
    fallback = 0
    for y, x in zip(*np.nonzero(edge)):
        au = int(round(w[y, x] * 255)); a = au / 255
        B = (C[y, x] - F * a) / (1 - a)
        k = label[y, x]
        if (B < 0).any() or (B > 255).any() or au == 0:
            fallback += 1
            fringe[y, x, :3] = C[y, x]; fringe[y, x, 3] = au
            base[k][y, x, :3] = C[y, x]; base[k][y, x, 3] = 255
        else:
            fringe[y, x, :3] = F; fringe[y, x, 3] = au
            base[k][y, x, :3] = B; base[k][y, x, 3] = 255

    def u8(p):
        q = np.rint(p).clip(0, 255).astype(np.uint8); q[q[..., 3] == 0, :3] = 0; return q

    # 4) hidden underlay: under the area the fringe fully covers (dilated by 1px),
    #    only where the source is opaque, painted by "redraw without fringe".
    need = full.copy()
    need[1:, :] |= full[:-1, :]; need[:-1, :] |= full[1:, :]; need[:, 1:] |= full[:, :-1]; need[:, :-1] |= full[:, 1:]
    need &= A == 255
    paint = redraw_without_fringe()
    under = np.zeros((H, W, 4), np.uint8)
    under[need, :3] = paint[need]; under[need, 3] = 255

    iris_aperture = save_mask("EyeApertureL", soft(white_shape) >= 0.5,
                              "iris clip; own x/y, larger than the iris crop, does not move with the iris")
    layers = [
        save_part("UnderlayFringe", under, 0, True, "UnderlayFringe"),
        save_part("Face", u8(base[1]), 1, True, "Face"),
        save_part("EyeWhiteL", u8(base[2]), 2, True, "EyeWhiteR"),
        save_part("IrisL", u8(base[3]), 3, True, "IrisR", mask=iris_aperture),
        save_part("UpperLashL", u8(base[4]), 4, True, "UpperLashR"),
        save_part("HairFringe", u8(fringe), 5, True, "HairFringe"),
    ]
    manifest = {
        "version": "layer-split-demo",
        "width": W, "height": H,
        "coordinate_origin": "top-left",
        "side_convention": "L/R in part names are SCREEN sides; sdk_name gives the CHARACTER-side name used when building the model",
        "source": {"path": "source.png", "width": W, "height": H},
        "resolution_note": "native canvas, no upscaling",
        "draw_order": "ascending order; apply every layer mask",
        "layers": layers,
    }
    (OUT / "layers.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # ---- recomposite from the files, in draw order --------------------------
    manifest = json.loads((OUT / "layers.json").read_text())
    comp = composite(manifest)
    diff = np.abs(comp.astype(int) - src.astype(int))
    Image.fromarray((diff[..., :3].max(axis=2) * 40).clip(0, 255).astype(np.uint8)).save(OUT / "diff.png")
    Image.fromarray(comp).save(OUT / "composite.png")
    moved = composite(manifest, shift={"HairFringe": 6})
    moved_no_under = composite(manifest, shift={"HairFringe": 6}, skip=("UnderlayFringe",))

    # ---- faint scatter pitfall ----------------------------------------------
    gen = np.zeros((48, 48), np.float64)
    yy, xx = np.mgrid[:48, :48]
    gen = np.clip(9 - np.hypot(yy - 24, xx - 22), 0, 1) * 255          # anti-aliased disk r~9
    for (y, x) in [(1, 2), (3, 45), (44, 6), (46, 40)]:
        gen[y, x] = 3                                                    # near-invisible specks
    gen = gen.astype(np.uint8)
    naive = Image.fromarray(gen).getbbox()
    good = main_component_box(gen)

    # ---- contact sheet ------------------------------------------------------
    def tile(rgba):
        checker = ((np.indices((H, W)).sum(0) // 8) % 2) * 40 + 200
        bg = np.dstack([checker] * 3).astype(np.float64)
        a = rgba[..., 3:4] / 255
        img = (rgba[..., :3] * a + bg * (1 - a)).astype(np.uint8)
        return Image.fromarray(img).resize((W * 2, H * 2), Image.Resampling.NEAREST)
    panels = [("source", tile(src)), ("composite", tile(comp)),
              ("diff x40 (max %d)" % diff.max(), Image.open(OUT / "diff.png").convert("RGB").resize((W * 2, H * 2), Image.Resampling.NEAREST)),
              ("fringe +6px, underlay", tile(moved)), ("fringe +6px, NO underlay", tile(moved_no_under))]
    sheet = Image.new("RGB", (len(panels) * (W * 2 + 8) + 8, H * 2 + 28), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (label_text, im) in enumerate(panels):
        x = 8 + i * (W * 2 + 8)
        sheet.paste(im, (x, 22)); d.text((x, 6), label_text, fill=(0, 0, 0))
    sheet.save(OUT / "contact-sheet.png", optimize=True)

    hole = int(((moved_no_under[..., 3] < 255) & (src[..., 3] == 255)).sum())
    print(f"source.png: {W}x{H} RGBA, alpha 0/255/partial = "
          f"{int((A == 0).sum())}/{int((A == 255).sum())}/{int(((A > 0) & (A < 255)).sum())}")
    print(f"parts: {len(layers)} (incl. 1 hidden underlay); shared edge pixels unmixed: {int(edge.sum())}, fallback: {fallback}")
    print(f"recomposite vs source: RGBA max diff = {int(diff.max())}")
    print(f"fringe moved 6px: holes inside opaque face without underlay = {hole}, with underlay = "
          f"{int(((moved[..., 3] < 255) & (src[..., 3] == 255)).sum())}")
    print(f"scatter pitfall: getbbox() = {naive}  vs  main-component box = {good}")
    assert diff.max() == 0, "recomposite must equal the source exactly"
    print("OK")


if __name__ == "__main__":
    main()
