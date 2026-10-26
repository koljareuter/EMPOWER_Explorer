"""Draws the app icon: the 'M' of the EMPOWER logo as a VLT unit-telescope enclosure (silver
shutters, warm-lit base, slit with the orange laser guide star) on Paranal under a starry sky,
framed by the EMPOWER colour gradient.

    python empower_explorer/assets/make_icon.py      ->  empower_icon.ico (16-256 px) + empower_icon.png

Small sizes (<= 32 px) use a simplified drawing (no ribs, stars or laser) so the M stays crisp.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
S = 1024                    # design units; drawn at 2x and reduced for anti-aliasing
SS = 2

EMPOWER = [(242, 140, 28), (33, 196, 120), (38, 200, 230), (37, 99, 235)]   # orange, green, cyan, blue


def _lerp(c0, c1, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))


def _gradient(size, stops, axis):
    """RGB image with a linear gradient through `stops` along axis 0 (vertical) or 1 (horizontal)."""
    n = size[1] if axis == 0 else size[0]
    t = np.linspace(0, 1, n)
    seg = np.clip(t * (len(stops) - 1), 0, len(stops) - 1 - 1e-9)
    i = seg.astype(int)
    f = (seg - i)[:, None]
    c = np.array(stops, float)
    line = c[i] * (1 - f) + c[i + 1] * f
    arr = np.repeat(line[:, None, :], size[0], axis=1) if axis == 0 else np.repeat(line[None, :, :], size[1], axis=0)
    return Image.fromarray(arr.astype(np.uint8), 'RGB')


def _rounded_mask(size, box, radius):
    m = Image.new('L', size, 0)
    ImageDraw.Draw(m).rounded_rectangle(box, radius=radius, fill=255)
    return m


def _m_polygon(x0, x1, yt, yb, notch_top, notch_bottom):
    """Silhouette of the logo's M: a block with a V cut from the top (the dome slit) and a small
    inverted V from the bottom."""
    w = x1 - x0
    xm = (x0 + x1) / 2
    leg = 0.305 * w
    bw = 0.17 * w
    return [(x0, yt), (x0 + leg, yt), (xm, notch_top), (x1 - leg, yt), (x1, yt), (x1, yb),
            (xm + bw, yb), (xm, notch_bottom), (xm - bw, yb), (x0, yb)]


def draw(px):
    """The icon at px x px (RGBA)."""
    detail = px > 32
    k = SS * px / S if px < 256 else SS
    size = int(round(S * k))

    def sc(v):
        return v * k

    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))

    # squircle with a night-sky gradient and a soft glow above the dome
    pad, rad = (40, 230) if detail else (24, 210)
    box = [sc(pad), sc(pad), sc(S - pad), sc(S - pad)]
    sky = _gradient((size, size), [(8, 14, 38), (16, 36, 88), (26, 58, 124)], axis=0).convert('RGBA')
    glow = Image.new('L', (size, size), 0)
    ImageDraw.Draw(glow).ellipse([sc(212), sc(60), sc(812), sc(620)], fill=70 if detail else 40)
    glow = glow.filter(ImageFilter.GaussianBlur(sc(110)))
    sky = Image.composite(Image.new('RGBA', (size, size), (110, 160, 255, 255)), sky, glow)
    img.paste(sky, (0, 0), _rounded_mask((size, size), box, sc(rad)))
    d = ImageDraw.Draw(img)

    if detail:
        # stars (the big ones as small four-point sparkles)
        for x, y, r in [(175, 200, 7), (300, 135, 4), (840, 215, 6), (885, 345, 4), (150, 430, 4),
                        (620, 120, 3), (420, 210, 3), (905, 470, 3)]:
            d.ellipse([sc(x - r), sc(y - r), sc(x + r), sc(y + r)], fill=(255, 255, 255, 235))
            if r >= 6:
                for dx, dy in ((1, 0), (0, 1)):
                    d.line([sc(x - 4 * r * dx), sc(y - 4 * r * dy), sc(x + 4 * r * dx), sc(y + 4 * r * dy)],
                           fill=(255, 255, 255, 170), width=max(1, int(sc(3))))
        # laser guide star beam out of the slit
        beam = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        bd = ImageDraw.Draw(beam)
        x_s, y_s, x_e, y_e = 512, 470, 735, 110
        for i in range(24):
            t0, t1 = i / 24, (i + 1) / 24
            a = int(230 * (1 - t0) ** 0.8)
            bd.line([sc(x_s + (x_e - x_s) * t0), sc(y_s + (y_e - y_s) * t0),
                     sc(x_s + (x_e - x_s) * t1), sc(y_s + (y_e - y_s) * t1)],
                    fill=(255, 176, 52, a), width=int(sc(16)))
        beam = beam.filter(ImageFilter.GaussianBlur(sc(2.5)))
        img.alpha_composite(beam)
        d = ImageDraw.Draw(img)
        d.ellipse([sc(x_e - 14), sc(y_e - 14), sc(x_e + 14), sc(y_e + 14)], fill=(255, 214, 120, 255))

    # Cerro Paranal: a dark hill the enclosure stands on
    hill = Image.new('L', (size, size), 0)
    ImageDraw.Draw(hill).ellipse([sc(-260), sc(770), sc(S + 260), sc(1500)], fill=255)
    hill = Image.composite(hill, Image.new('L', (size, size), 0), _rounded_mask((size, size), box, sc(rad)))
    img.paste((10, 18, 40, 255), (0, 0), hill)

    # the M: geometry
    if detail:
        x0, x1, yt, yb, nt, nb, split = 212, 812, 300, 830, 515, 700, 585
    else:
        x0, x1, yt, yb, nt, nb, split = 150, 874, 215, 860, 470, 730, 560
    poly = [(sc(x), sc(y)) for x, y in _m_polygon(x0, x1, yt, yb, nt, nb)]
    m_mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(m_mask).polygon(poly, fill=255)

    # soft shadow under the M
    shadow = m_mask.filter(ImageFilter.GaussianBlur(sc(22)))
    sh = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    sh.paste((0, 0, 0, 150), (int(sc(10)), int(sc(22))), shadow)
    img.alpha_composite(sh)

    # silver shutters (top) and warm-lit base (bottom)
    body = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    top = _gradient((size, int(sc(split - yt))), [(236, 239, 244), (176, 184, 196), (134, 143, 158)], axis=0)
    body.paste(top, (0, int(sc(yt))))
    base = _gradient((size, int(sc(yb - split)) + 2), [(255, 214, 142), (238, 160, 72), (196, 112, 46)], axis=0)
    body.paste(base, (0, int(sc(split))))
    bd = ImageDraw.Draw(body)
    if detail:
        for y in range(int(yt + 34), int(split - 10), 30):              # louvres of the enclosure
            bd.line([sc(x0), sc(y), sc(x1), sc(y)], fill=(112, 121, 136, 255), width=max(1, int(sc(6))))
        leg = 0.305 * (x1 - x0)
        for x in (x0 + leg * 0.5, x1 - leg * 0.5):                        # seams of the two shutters
            bd.line([sc(x), sc(yt), sc(x), sc(split)], fill=(150, 158, 172, 255), width=max(1, int(sc(5))))
        for wx in (x0 + 55, x1 - 105):                                     # lit windows in the base
            bd.rounded_rectangle([sc(wx), sc(split + 85), sc(wx + 50), sc(split + 135)], radius=sc(10),
                                 fill=(255, 246, 196, 255))
    # the ring between dome and base
    bd.rectangle([sc(x0), sc(split - 8), sc(x1), sc(split + 16)], fill=(96, 70, 58, 255))
    img.paste(body, (0, 0), m_mask)

    # crisp outline of the M (lighter at the top edge)
    d = ImageDraw.Draw(img)
    d.line(poly + [poly[0]], fill=(20, 28, 52, 255), width=max(1, int(sc(10 if detail else 22))), joint='curve')
    d.line([poly[0], poly[1]], fill=(255, 255, 255, 220), width=max(1, int(sc(6))))
    d.line([poly[3], poly[4]], fill=(255, 255, 255, 220), width=max(1, int(sc(6))))

    # EMPOWER gradient frame
    ring_w = 26 if detail else 60
    grad = _gradient((size, size), EMPOWER, axis=1).convert('RGBA')
    outer = _rounded_mask((size, size), box, sc(rad))
    inner = _rounded_mask((size, size), [b + (sc(ring_w) if i < 2 else -sc(ring_w)) for i, b in enumerate(box)],
                          sc(rad - ring_w))
    ring = Image.fromarray(np.clip(np.array(outer, int) - np.array(inner, int), 0, 255).astype(np.uint8), 'L')
    img.paste(grad, (0, 0), ring)

    return img.resize((px, px), Image.LANCZOS)


def main():
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    imgs = {s: draw(s) for s in sizes}
    imgs[256].save(os.path.join(HERE, 'empower_icon.png'))
    draw(512).save(os.path.join(HERE, 'empower_icon_512.png'))
    imgs[256].save(os.path.join(HERE, 'empower_icon.ico'), sizes=[(s, s) for s in sizes],
                   append_images=[imgs[s] for s in sizes if s != 256])
    print('written:', os.listdir(HERE))


if __name__ == '__main__':
    main()
