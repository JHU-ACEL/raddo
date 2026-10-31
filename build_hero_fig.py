import matplotlib

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams['mathtext.fontset'] = 'cm'

# ---------- palette ----------
INK      = '#1B2A41'
PANEL    = '#EEF3F9'
ADAPT    = "#4AB646" # '#C1443C'
ADAPT_D  = "#31922E" #'#96332B'
STEEL    = '#2E5C8A'
SKY      = '#0E1928'
PLANET   = '#24405E'
PLANET_HI= '#4E7BA8'
FIELD    = '#2E5578'
FIELD_HI = '#4A7AA8'
SILVER   = '#E9EEF5'
GRAY_LINE= '#7C8BA0'
MUTED    = '#93A7BF'

SANS = 'DejaVu Sans'
DOT  = (0, (1.5, 2.1))
DASH = (0, (5, 3))

# ---------- layout ----------
# 3 main columns now: [Offline+Learned-Field-Model+Dynamics stack, MPC, Spacecraft].
# The offline-pretraining box and the learned-field-model box are nested
# side by side *within* the same horizontal span as the Dynamics Model box
# above them, instead of offline getting its own column further left --
# that's what keeps the whole figure narrower.
col_w = [6.45, 3.55, 4.90]
gap, margin = 0.45, 0.40
col_x = [margin]
for w in col_w[:-1]:
    col_x.append(col_x[-1] + w + gap)

sub_gap    = 0.40                              # gap between offline box and field-model box
offline_w  = 2.05
fieldmdl_w = col_w[0] - offline_w - sub_gap    # remaining width for the field-model box

bottom_y0, row_h, gap_rows = 1.95, 3.05, 1.55
top_y0 = bottom_y0 + row_h + gap_rows
FIGW = col_x[-1] + col_w[-1] + margin
FIGH = top_y0 + row_h + 0.55

fig, ax = plt.subplots(figsize=(FIGW, FIGH))
ax.set_xlim(0, FIGW); ax.set_ylim(0, FIGH)
ax.set_aspect('equal'); ax.axis('off')
fig.patch.set_facecolor('white')

def rbox(x, y, w, h, fc, ec, lw=2.6, rounding=0.16, zorder=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
                        joinstyle='round')
    ax.add_patch(p); return p

def arrow(x0,y0,x1,y1, color=INK, lw=3.1, ms=25, zorder=4, ls='solid'):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1), arrowstyle='-|>', mutation_scale=ms,
                 linewidth=lw, color=color, zorder=zorder, linestyle=ls,
                 shrinkA=0, shrinkB=0, capstyle='round', joinstyle='round'))

def title_above(x, y_top, text, size=27, color=INK):
    ax.text(x, y_top + 0.30, text, ha='center', va='bottom', fontsize=size,
            family=SANS, fontweight='bold', color=color, zorder=5)

def colored_math(xc, yc, parts, fontsize, zorder=5, gap_pad=0.0):
    """Lay out one math expression from pieces so individual terms can be colored."""
    r = fig.canvas.get_renderer()
    objs = [ax.text(0, yc, s, ha='left', va='center', fontsize=fontsize,
                    color=c, zorder=zorder) for s, c in parts]
    inv = ax.transData.inverted()
    widths = []
    for t in objs:
        bb = t.get_window_extent(renderer=r)
        p0 = inv.transform((bb.x0, bb.y0)); p1 = inv.transform((bb.x1, bb.y1))
        widths.append(p1[0]-p0[0])
    total = sum(widths) + gap_pad*(len(objs)-1)
    x = xc - total/2
    for t, w in zip(objs, widths):
        t.set_x(x); x += w + gap_pad
    return objs

# ---------- panels ----------
P1 = (col_x[0], bottom_y0, offline_w, row_h)                                    # Offline pretraining
P2 = (col_x[0]+offline_w+sub_gap, bottom_y0, fieldmdl_w, row_h)                 # Learned field model
P3 = (col_x[0], top_y0,    col_w[0], row_h)                                     # Dynamics (spans both above)
P4 = (col_x[1], top_y0,    col_w[1], row_h)                                     # QP MPC
P5 = (col_x[2], bottom_y0, col_w[2], (top_y0+row_h)-bottom_y0)                  # Spacecraft

rbox(*P1, SKY, GRAY_LINE)     # offline stage: muted border
for p in (P2, P3, P4):
    rbox(*p, PANEL, INK)
p5_patch = rbox(*P5, SKY, INK)

title_above(P1[0]+P1[2]/2, P1[1]+P1[3], 'Offline Pretraining', size=23, color=GRAY_LINE)
title_above(P3[0]+P3[2]/2, P3[1]+P3[3], 'Dynamics Model')
title_above(P4[0]+P4[2]/2, P4[1]+P4[3], 'MPC')
title_above(P5[0]+P5[2]/2, P5[1]+P5[3], 'Spacecraft')

midy_bottom = bottom_y0 + row_h*0.5
midy_top    = top_y0 + row_h*0.5

# ---------- offline block: low-fidelity dipole model ----------
def dipole_icon(cx, cy, R=0.32, Ls=(0.46,0.69,0.86), color=FIELD_HI, zbase=3):
    th = np.linspace(0.14, np.pi-0.14, 140)
    for L in Ls:
        r = L*np.sin(th)**2
        x, z = r*np.sin(th), 2*r*np.cos(th)
        ax.plot(cx+x, cy+z, color=color, lw=1.5, alpha=0.8, zorder=zbase, solid_capstyle='round')
        ax.plot(cx-x, cy+z, color=color, lw=1.5, alpha=0.8, zorder=zbase, solid_capstyle='round')
    ax.add_patch(mpatches.Circle((cx,cy), R, facecolor=PLANET, edgecolor=PLANET_HI, lw=1.3, zorder=zbase+1.1))

dipole_icon(P1[0]+P1[2]/2, P1[1]+P1[3]/2)

# offline -> network (dashed + gray: this happens before flight)
arrow(P1[0]+P1[2], midy_bottom, P2[0], midy_bottom, color=GRAY_LINE, lw=2.2, ls=DASH)
# ax.text((P1[0]+P1[2]+P2[0])/2, midy_bottom+0.16, r'$\theta_{\mathrm{init}}$',
#         ha='center', va='bottom', fontsize=16, color=GRAY_LINE, zorder=5)

# ---------- learned field model ----------
nn_xfrac  = [0.22, 0.42, 0.62, 0.84]
nn_layers = [3, 3, 3, 1]
node_r = 0.135
highlight_idx = 2

ax.text(P2[0]+P2[2]/2, P2[1]+P2[3]-0.24, 'Learned Field Model',
        ha='center', va='top', fontsize=17, family=SANS, fontweight='bold', color=INK, zorder=5)
# ax.text(P2[0]+0.24, P2[1]+P2[3]-0.62, r'$\theta_{1:\ell-1}$ frozen',
#         ha='left', va='top', fontsize=12.5, family=SANS, style='italic', color=GRAY_LINE, zorder=5)

def layer_ys(n):
    if n == 1:
        return [P2[1]+P2[3]*0.51]
    return [P2[1]+P2[3]*f for f in np.linspace(0.28, 0.74, n)]

nn_coords = [[(P2[0]+P2[2]*xf, y) for y in layer_ys(n)] for xf, n in zip(nn_xfrac, nn_layers)]

hl_x  = P2[0] + P2[2]*nn_xfrac[highlight_idx]
hl_ys = [p[1] for p in nn_coords[highlight_idx]]
pad = 0.12
hb_x, hb_y = hl_x-node_r-pad*0.55, min(hl_ys)-node_r-pad
hb_w, hb_h = (node_r+pad*0.55)*2, (max(hl_ys)-min(hl_ys))+2*(node_r+pad)
for lwid, fc, al in ((2.4, ADAPT, 0.16), (2.4, 'none', 1.0)):
    ax.add_patch(FancyBboxPatch((hb_x,hb_y), hb_w, hb_h,
                 boxstyle="round,pad=0,rounding_size=0.12", linewidth=lwid,
                 edgecolor=ADAPT, facecolor=fc, alpha=al, zorder=3.3))
# ax.text(hb_x+hb_w+0.10, hb_y+hb_h-0.04, r'$\theta_\ell$', ha='left', va='top',
#         fontsize=17, color=ADAPT_D, zorder=5)

# ax.text(P2[0]+P2[2]/2, P2[1]+0.44, 'Online Last-Layer Adaptation', ha='center', va='center',
#         fontsize=12.5, family=SANS, fontweight='bold', color=ADAPT_D, zorder=5)
 
for li in range(len(nn_coords)-1):
    for (x0,y0) in nn_coords[li]:
        for (x1,y1) in nn_coords[li+1]:
            ax.plot([x0,x1],[y0,y1], color=GRAY_LINE, lw=1.0, alpha=0.55, zorder=3.5)
for li, layer in enumerate(nn_coords):
    fc = ADAPT if li == highlight_idx else INK
    for (x,y) in layer:
        ax.add_patch(mpatches.Circle((x,y), node_r, facecolor=fc, edgecolor='white',
                                      linewidth=1.4, zorder=3.7))

ax.text(P2[0]+P2[2]*nn_xfrac[2]-.07, P2[1]+0.26, r'$\theta_\ell \leftarrow \theta_\ell - \alpha\nabla\theta$',
        ha='center', va='center', fontsize=19, color=ADAPT_D, zorder=5)


# ---------- forward pipeline ----------
bx = P2[0] + P2[2]/2
arrow(bx, P2[1]+P2[3], bx, P3[1])
ax.text(bx+0.16, (P2[1]+P2[3]+P3[1])/2, r'$B_\theta$',
        ha='left', va='center', fontsize=24, color=ADAPT_D, zorder=5)

arrow(P3[0]+P3[2], midy_top, P4[0], midy_top)
arrow(P4[0]+P4[2], midy_top, P5[0], midy_top)
# ax.text((P4[0]+P4[2]+P5[0])/2, midy_top+0.16, r'$u$',
#         ha='center', va='bottom', fontsize=20, color=INK, zorder=5)

# ---------- equations ----------
colored_math(P3[0]+P3[2]/2, P3[1]+P3[3]/2,
             [(r'$\dot{\omega} = J^{-1}(\vec{\mu} \times$', INK),
              (r'$B_\theta(\vec{r})$', ADAPT_D),
              (r'$- \omega \times J\omega)$', INK)],
             fontsize=26, gap_pad=0.09)

lx = P4[0] + P4[2]*0.21
lx2 = P4[0] + P4[2]*0.41
rx = P4[0] + P4[2]*0.45
rx2 = P4[0] + P4[2]*0.04
ys = [P4[1]+P4[3]*0.7, P4[1]+P4[3]*0.5, P4[1]+P4[3]*0.28]
ax.text(lx, ys[0], r'$\min_{x,u} \; C(x,u)$', ha='left', va='center', fontsize=26, color=INK, zorder=5)
ax.text(lx2, ys[1], r'$\mathrm{s.t.}$', ha='left', va='center', fontsize=26, color=INK, zorder=5)
ax.text(rx2, ys[2], r'$x_{k+1}=f(x_k, u_k)$', ha='left', va='center', fontsize=26, color=INK, zorder=5)

# ================= adaptation feedback loop =================
y_fb  = 0.98
x_src = P5[0] + P5[2]*0.30
x_dst = hl_x

for seg in ([[x_src,x_src],[P5[1],y_fb]], [[x_dst,x_src],[y_fb,y_fb]], [[x_dst,x_dst],[y_fb,P2[1]-0.34]]):
    ax.plot(seg[0], seg[1], color=ADAPT, lw=2.6, ls=DOT, zorder=4,
            solid_capstyle='round', dash_capstyle='round')
arrow(x_dst, P2[1]-0.34, x_dst, P2[1]-0.02, color=ADAPT, lw=2.6, ms=20)
ax.add_patch(mpatches.Circle((x_src, P5[1]), 0.085, facecolor=ADAPT, edgecolor='white', lw=1.2, zorder=5))

ax.text((x_src+x_dst)/2, y_fb+0.15, r'$\vec{r},\ \vec{B}_{\mathrm{meas}}$',
        ha='center', va='bottom', fontsize=26, color=ADAPT_D, zorder=5)

# ================= spacecraft panel =================
px, py, pw, ph = P5

# ---- traced non-dipolar planetary field (superposed tilted dipoles) ----
def unit(a):
    a = np.deg2rad(a); return np.array([np.cos(a), np.sin(a)])

def field_at(p, dips):
    B = np.zeros(2)
    for pos, m, s in dips:
        d = p - pos; r = np.hypot(*d)
        if r < 1e-6: continue
        rh = d/r
        B += s*(3*rh*np.dot(m, rh) - m)/r**3
    return B

def trace(seed, dips, direction, pcen, R, box, nmax=1400, h=0.05):
    pts = [seed.copy()]; p = seed.copy()
    for _ in range(nmax):
        B = field_at(p, dips); n = np.hypot(*B)
        if n < 1e-12: break
        k1 = B/n
        B2 = field_at(p + direction*h*0.5*k1, dips); n2 = np.hypot(*B2)
        if n2 < 1e-12: break
        p = p + direction*h*(B2/n2)
        if np.hypot(*(p-pcen)) < R*0.995: break
        if not (box[0] < p[0] < box[1] and box[2] < p[1] < box[3]): break
        pts.append(p.copy())
    return np.array(pts)

R_pl = 1.50
pcen = np.array([px + pw*0.50, py + 0.50])
dips = [(pcen,                               unit(55),  1.00),
        (pcen + R_pl*np.array([ 0.40, 0.20]), unit(-35), 0.40),
        (pcen + R_pl*np.array([-0.35, 0.10]), unit(150), 0.35)]
box = (px-0.1, px+pw+0.1, py-0.1, py+ph+0.1)

for rad, angs, col, lw, al in [(1.02, np.linspace(6,174,22),  FIELD,    1.0, 1.00),
                               (1.30, np.linspace(10,170,14), FIELD,    1.0, 0.90),
                               (1.70, np.linspace(18,162,10), FIELD_HI, 0.95, 0.70),
                               (2.20, np.linspace(28,152,8),  FIELD_HI, 0.85, 0.50)]:
    for a in angs:
        seed = pcen + rad*R_pl*unit(a)
        for d in (1, -1):
            L = trace(seed, dips, d, pcen, R_pl, box)
            if len(L) > 12:
                ln, = ax.plot(L[:,0], L[:,1], color=col, lw=lw, alpha=al,
                              zorder=2.5, solid_capstyle='round')
                ln.set_clip_path(p5_patch)

# planet body over the field-line footpoints
planet = mpatches.Circle(pcen, R_pl, facecolor=PLANET, edgecolor='none', zorder=2.9)
ax.add_patch(planet); planet.set_clip_path(p5_patch)
rim = mpatches.Circle(pcen, R_pl, facecolor='none', edgecolor=PLANET_HI, lw=1.8, alpha=0.9, zorder=3.0)
ax.add_patch(rim); rim.set_clip_path(p5_patch)

# ax.text(pcen[0], py+0.34, 'true planetary field', ha='center', va='center',
#         fontsize=12.5, family=SANS, style='italic', color='#B7C7DA', zorder=5)
# ax.text(px+0.32, py+2.95, r'$\vec{B}(\vec{r})$', ha='left', va='center',
#         fontsize=15, color='#9CBBD8', zorder=5)

# ---- satellite ----
body_cx, body_cy = px + pw*0.34, midy_top
bw = bh = 0.60
body = FancyBboxPatch((body_cx-bw/2, body_cy-bh/2), bw, bh,
                       boxstyle="round,pad=0,rounding_size=0.08", linewidth=1.6,
                       edgecolor=INK, facecolor=SILVER, zorder=4)
ax.add_patch(body); body.set_clip_path(p5_patch)

# tsp = np.linspace(0, 3.2*np.pi, 120)
# ln, = ax.plot(body_cx+0.015*tsp*np.cos(tsp), body_cy+0.015*tsp*np.sin(tsp),
#               color=STEEL, lw=1.0, alpha=0.8, zorder=4.1)
ln.set_clip_path(p5_patch)

def solar_panel(cx0, cy0, sign):
    pwid, phgt = 0.58, 0.42
    x0 = cx0 + sign*0.05
    rect = mpatches.Rectangle((x0, cy0-phgt/2), sign*pwid, phgt, facecolor='#1B3A63',
                               edgecolor=ADAPT_D, linewidth=1.3, zorder=3.8)
    ax.add_patch(rect); rect.set_clip_path(p5_patch)
    for k in range(1,4):
        xg = x0 + sign*pwid*k/4
        l, = ax.plot([xg,xg],[cy0-phgt/2, cy0+phgt/2], color='#3E6FA0', lw=0.9, zorder=3.9)
        l.set_clip_path(p5_patch)

solar_panel(body_cx-bw/2, body_cy, -1)
solar_panel(body_cx+bw/2, body_cy, 1)

# def coil_rod(cx, cy, length, thick, angle_deg, nwind=6):
#     t = mtransforms.Affine2D().rotate_deg_around(cx, cy, angle_deg) + ax.transData
#     rod = FancyBboxPatch((cx-length/2, cy-thick/2), length, thick,
#                           boxstyle=f"round,pad=0,rounding_size={thick/2}", linewidth=1.3,
#                           edgecolor=INK, facecolor='#C9812E', zorder=4.4, transform=t)
#     ax.add_patch(rod); rod.set_clip_path(p5_patch)
#     for xw in np.linspace(cx-length/2+thick*0.7, cx+length/2-thick*0.7, nwind):
#         l, = ax.plot([xw,xw],[cy-thick*0.62, cy+thick*0.62], color=INK, lw=1.0,
#                       transform=t, zorder=4.5)
#         l.set_clip_path(p5_patch)
#     for ex in (cx-length/2, cx+length/2):
#         cap = mpatches.Circle((ex, cy), thick*0.52, facecolor=INK, edgecolor=INK,
#                                zorder=4.45, transform=t)
#         ax.add_patch(cap); cap.set_clip_path(p5_patch)

# rod_cx, rod_cy = body_cx, body_cy + bh/2 + 0.28
# coil_rod(rod_cx, rod_cy, 0.62, 0.11, 0)
# coil_rod(rod_cx, rod_cy, 0.46, 0.095, 90)

# l, = ax.plot([rod_cx+0.12, rod_cx+0.42],[rod_cy+0.14, rod_cy+0.40], color='#8FA3BC', lw=1.0, zorder=4.4)
# l.set_clip_path(p5_patch)
# ax.text(rod_cx+0.47, rod_cy+0.42, 'magnetorquers', ha='left', va='bottom', fontsize=12,
#         style='italic', family=SANS, color='#C3D2E3', zorder=5)

# # ---- goal attitude marker ----
gx, gy = px + pw*0.80, py + 5.30
# for rr, aa in ((0.20, 0.9), (0.11, 1.0)):
#     c = mpatches.Circle((gx, gy), rr, facecolor='none', edgecolor=SILVER, lw=1.6, alpha=aa, zorder=4.6)
#     ax.add_patch(c); c.set_clip_path(p5_patch)
# for dx, dy in ((0.30,0),(0,0.30)):
#     l, = ax.plot([gx-dx, gx+dx],[gy-dy, gy+dy], color=SILVER, lw=1.2, alpha=0.75, zorder=4.6)
#     l.set_clip_path(p5_patch)
# ax.text(gx+0.34, gy+0.06, r'$q_g$', ha='left', va='center', fontsize=16,
#         color=SILVER, zorder=5)

# ---- two conceptual attitude trajectories ----
def bezier(p0, p1, p2, n=160):
    t = np.linspace(0,1,n)[:,None]
    return (1-t)**2*np.array(p0) + 2*(1-t)*t*np.array(p1) + t**2*np.array(p2)

def path_with_head(pts, color, lw, ls='solid', ms=19, z=4.7):
    l, = ax.plot(pts[:,0], pts[:,1], color=color, lw=lw, ls=ls, zorder=z,
                 solid_capstyle='round', dash_capstyle='round')
    l.set_clip_path(p5_patch)
    a = FancyArrowPatch(tuple(pts[-6]), tuple(pts[-1]), arrowstyle='-|>',
                        mutation_scale=ms, linewidth=lw, color=color, zorder=z+0.1,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a); a.set_clip_path(p5_patch)

# adapted model: converges onto the goal attitude
adapt_path = bezier((body_cx, body_cy-0.30), (body_cx+1.30, body_cy-1.35), (gx-0.24, gy+0.10))
path_with_head(adapt_path, ADAPT, 2.8)

# dipole-only model: loses control authority, oscillates and drifts off
s_na = np.array([body_cx, body_cy-0.30])
e_na = np.array([px+0.98, py+3.55])
u = e_na - s_na
L_na = np.hypot(*u); u = u/L_na
n_hat = np.array([-u[1], u[0]])
s = np.linspace(0, 1, 200)
wob = 0.30*s*np.sin(2*np.pi*2.0*s)
noad_path = (s_na[None,:] + np.outer(s*L_na, u) + np.outer(wob, n_hat))
path_with_head(noad_path, '#C2CEDC', 2.4, ls=(0,(4.5,2.6)), ms=16)

ax.text(body_cx+1.62, body_cy-1.4, 'with\nadaptation', ha='center', va='center',
        fontsize=18.5, family=SANS, fontweight='bold', color=ADAPT, zorder=5, linespacing=1.2)
ax.text(px+1.62, py+3.55, 'no\nadaptation', ha='left', va='center',
        fontsize=18.5, family=SANS, fontweight='bold', color='#C2CEDC', zorder=5, linespacing=1.2)

plt.savefig('hero.png', dpi=500, bbox_inches='tight', pad_inches=0.15, facecolor='white')
# plt.savefig('/mnt/user-data/outputs/hero_figure.pdf', bbox_inches='tight', pad_inches=0.15, facecolor='white')
# plt.savefig('/mnt/user-data/outputs/hero_figure.svg', bbox_inches='tight', pad_inches=0.15, facecolor='white')
print('saved', FIGW, FIGH)