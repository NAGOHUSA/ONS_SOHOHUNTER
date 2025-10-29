#!/usr/bin/env python3
"""
SOHO Comet Hunter — detect_comets.py
Detects moving objects in LASCO C2/C3 sequences.
Outputs: JSON report with ai_label/ai_score, crops, animations,
         2-D & 3-D orbit predictions (no astropy required).

# --------------------------------------------------------------
# INSTALL IN CI (GitHub Actions)
# --------------------------------------------------------------
#   python -m pip install --user --break-system-packages \
#       opencv-python numpy imageio scipy matplotlib
# --------------------------------------------------------------
"""

# --------------------------------------------------------------
# DEPENDENCIES
# --------------------------------------------------------------
import argparse
import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import imageio
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import fetch_lasco                     # <-- your own fetcher

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
VALID_DETS = ["C2", "C3"]
SELECT_TOP_N = int(os.getenv("SELECT_TOP_N_FOR_SUBMIT", "3"))
DEBUG = os.getenv("DETECTOR_DEBUG", "0") == "1"
USE_AI = os.getenv("USE_AI_CLASSIFIER", "1") == "1"
AI_VETO = os.getenv("AI_VETO_ENABLED", "1") == "1"
AI_VETO_LABEL = os.getenv("AI_VETO_LABEL", "not_comet")
AI_VETO_MAX = float(os.getenv("AI_VETO_SCORE_MAX", "0.9"))

# --------------------------------------------------------------
# CONSTANTS (LASCO geometry)
# --------------------------------------------------------------
LASCO_C2_SCALE = 5.94          # arcsec/pixel
LASCO_C3_SCALE = 56.0          # arcsec/pixel
R_SUN_ARCSEC = 960.0
R_SUN_KM = 6.957e5
AU_KM = 1.495978707e8
GM_SUN = 1.3271244e20          # m³ s⁻²

# --------------------------------------------------------------
# UTILS
# --------------------------------------------------------------
def log(*a, **kw):
    print(*a, **kw, file=sys.stderr)

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        log(f"Failed to load {path}")
    return img

def timestamp_from_name(name):
    import re
    m = re.search(r'(\d{8}_\d{6})', name)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            return dt.isoformat() + "Z"
        except:
            pass
    return ""

# --------------------------------------------------------------
# PLATE-SCALE HELPERS
# --------------------------------------------------------------
def pixel_to_rho_theta(det, x, y):
    """Return (rho in R⊙, theta in radians) from pixel (x,y)."""
    scale = LASCO_C2_SCALE if det == "C2" else LASCO_C3_SCALE
    cx = cy = 512
    dx = x - cx
    dy = y - cy
    rho_arcsec = np.hypot(dx, dy) * scale
    rho_Rsun = rho_arcsec / R_SUN_ARCSEC
    theta = np.arctan2(dy, dx)
    if theta < 0:
        theta += 2 * np.pi
    return rho_Rsun, theta

# --------------------------------------------------------------
# 2-D PARABOLIC ORBIT (plane-of-sky)
# --------------------------------------------------------------
def parabolic_residuals_2d(params, t_sec, rho_obs):
    q, t_peri = params
    T = np.sqrt(q**3 / (2 * GM_SUN / (R_SUN_KM*1000)**3)) * 86400   # seconds
    dt = t_sec - t_peri
    rho_pred = q * (1 + (dt / T)**2)
    return rho_pred - rho_obs

def fit_parabolic_2d(times_sec, rho_obs):
    if len(times_sec) < 3:
        return None
    q0 = np.min(rho_obs) * 0.9
    t_peri0 = times_sec[np.argmin(rho_obs)]
    res = least_squares(
        parabolic_residuals_2d,
        x0=[q0, t_peri0],
        bounds=([0.5, times_sec[0]-7200], [20, times_sec[-1]+7200])
    )
    if not res.success:
        return None
    q, t_peri = res.x
    T = np.sqrt(q**3 / (2 * GM_SUN / (R_SUN_KM*1000)**3)) * 86400
    v_peri = np.sqrt(2 * GM_SUN / (q * R_SUN_KM*1000)) / 1000   # km/s
    return {"q_Rsun": q, "t_peri_sec": t_peri, "v_peri_kms": v_peri, "T_sec": T}

# --------------------------------------------------------------
# 3-D ORBIT USING C2 + C3 DUAL VIEW (pure NumPy/SciPy)
# --------------------------------------------------------------
def orbit_residuals_3d(params, obs_c2, obs_c3):
    """params = [q, t_peri, Omega, omega, inc]"""
    q, t_peri, Omega, omega, inc = params
    residuals = []
    for t, rho, pa, _ in obs_c2 + obs_c3:
        dt = (t - t_peri) * 86400
        M = np.sqrt(2 * GM_SUN / (q * R_SUN_KM*1000)**3) * dt
        # Barker's equation (parabolic)
        w = (M/3)**(1/3) + (M/3)**(-1/3)
        r = q * (1 + w**2)
        true_anom = 2 * np.arctan(w)

        # Position in orbital plane
        x_orb = r * (np.cos(true_anom) * np.cos(omega) - np.sin(true_anom) * np.sin(omega) * np.cos(inc))
        y_orb = r * (np.cos(true_anom) * np.sin(omega) + np.sin(true_anom) * np.cos(omega) * np.cos(inc))
        z_orb = r * np.sin(true_anom) * np.sin(inc)

        # Rotate by Ω
        X = x_orb * np.cos(Omega) - y_orb * np.sin(Omega)
        Y = x_orb * np.sin(Omega) + y_orb * np.cos(Omega)
        Z = z_orb

        # Project to plane-of-sky → ρ only
        rho_pred = np.hypot(X, Y) * AU_KM / R_SUN_KM
        residuals.append(rho_pred - rho)
    return residuals

def fit_orbit_3d(c2_obs, c3_obs):
    if len(c2_obs) < 2 or len(c3_obs) < 2:
        return None

    times_c2 = np.array([t for t,_,_,_ in c2_obs])
    rho_c2   = np.array([r for _,r,_,_ in c2_obs])
    init2d = fit_parabolic_2d(times_c2, rho_c2)
    if not init2d:
        return None

    q0 = init2d["q_Rsun"]
    t_peri0 = init2d["t_peri_sec"]
    pa_mean = np.mean([pa for _,_,pa,_ in c2_obs])
    Omega0 = np.radians(pa_mean)
    omega0 = 0.0
    inc0   = 0.0

    res = least_squares(
        orbit_residuals_3d,
        x0=[q0, t_peri0, Omega0, omega0, inc0],
        bounds=([0.5, t_peri0-86400, 0, -np.pi, 0],
                [15,  t_peri0+86400, 2*np.pi, np.pi, np.pi])
    )
    if not res.success:
        return None

    q, t_peri, Omega, omega, inc = res.x
    peri_time = datetime.utcfromtimestamp(t_peri)
    v_peri = np.sqrt(2 * GM_SUN / (q * R_SUN_KM*1000)) / 1000

    inc_deg = np.degrees(inc)
    Omega_deg = np.degrees(Omega) % 360
    if 200 < Omega_deg < 320 and v_peri > 450:
        group = "Kreutz"
    elif 300 < Omega_deg < 360 and v_peri > 400:
        group = "Kracht"
    elif 50 < Omega_deg < 150:
        group = "Meyer"
    else:
        group = "Unknown"

    return {
        "perihelion_time_utc": peri_time.isoformat() + "Z",
        "perihelion_distance_Rsun": round(q, 3),
        "position_angle_deg": round(Omega_deg, 1),
        "inclination_deg": round(inc_deg, 1),
        "speed_at_perihelion_kms": round(v_peri),
        "orbit_group_guess": group,
        "fit_success": True
    }

# --------------------------------------------------------------
# ORBIT PREDICTION (called per track)
# --------------------------------------------------------------
def predict_orbit_for_track(det, tr, timestamps, dual_match=None):
    obs = []
    for t_idx, x, y in tr:
        if t_idx >= len(timestamps) or not timestamps[t_idx]:
            continue
        try:
            t_utc = datetime.fromisoformat(timestamps[t_idx].replace("Z", "+00:00"))
            rho, theta = pixel_to_rho_theta(det, x, y)
            if rho < 1.1:          # inside occulting disk
                continue
            pa_deg = np.degrees(theta)
            t_sec = (t_utc - datetime(1970,1,1)).total_seconds()
            obs.append((t_sec, rho, pa_deg, det))
        except:
            continue

    if len(obs) < 3:
        return {"error": "Too few points"}

    # ---- 2-D fit -------------------------------------------------
    times = np.array([t for t,_,_,_ in obs])
    rhos  = np.array([r for _,r,_,_ in obs])
    fit2d = fit_parabolic_2d(times, rhos)
    result = {"orbit_2d": None, "orbit_3d": None}

    if fit2d:
        peri_utc = datetime.utcfromtimestamp(fit2d["t_peri_sec"])
        pa_mean = np.mean([pa for _,_,pa,_ in obs])
        group = "Unknown"
        if 200 < pa_mean < 320 and fit2d["v_peri_kms"] > 450:
            group = "Kreutz"
        elif 300 < pa_mean < 360 and fit2d["v_peri_kms"] > 400:
            group = "Kracht"
        result["orbit_2d"] = {
            "perihelion_time_utc": peri_utc.isoformat() + "Z",
            "perihelion_distance_Rsun": round(fit2d["q_Rsun"], 3),
            "position_angle_deg": round(pa_mean, 1),
            "speed_at_perihelion_kms": round(fit2d["v_peri_kms"]),
            "orbit_group_guess": group
        }

    # ---- 3-D fit (dual channel) ---------------------------------
    if dual_match:
        c2_obs = obs
        c3_obs = []
        for pos in dual_match["positions"]:
            t_str = pos.get("time_utc")
            if not t_str:
                continue
            t_utc = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
            x, y = pos["x"], pos["y"]
            rho, theta = pixel_to_rho_theta("C3", x, y)
            if rho < 1.1:
                continue
            pa_deg = np.degrees(theta)
            t_sec = (t_utc - datetime(1970,1,1)).total_seconds()
            c3_obs.append((t_sec, rho, pa_deg, "C3"))
        if len(c3_obs) >= 2:
            fit3d = fit_orbit_3d(c2_obs, c3_obs)
            if fit3d:
                result["orbit_3d"] = fit3d

    # ---- ASCII plot (2-D) ---------------------------------------
    if fit2d:
        lines = ["Orbit (2-D):"]
        for t, r, pa, _ in obs:
            dt_h = (t - fit2d["t_peri_sec"]) / 3600
            lines.append(f"t={dt_h:+.2f}h  ρ={r:.2f}R⊙  PA={pa:.0f}°")
        lines.append(
            f"→ Peri: {peri_utc.strftime('%Y-%m-%d %H:%M')} UTC | "
            f"q={fit2d['q_Rsun']:.3f}R⊙ | v={fit2d['v_peri_kms']:.0f}km/s"
        )
        result["ascii_plot"] = "\n".join(lines)

    # ---- PNG plot (2-D) -----------------------------------------
    try:
        plot_dir = ensure_dir(Path("detections/plots"))
        fig, ax = plt.subplots(figsize=(7,4), dpi=120)
        t_plot = np.linspace(times[0]-1800, times[-1]+7200, 200)
        dt_plot = t_plot - fit2d["t_peri_sec"]
        T = fit2d["T_sec"]
        rho_plot = fit2d["q_Rsun"] * (1 + (dt_plot / T)**2)
        ax.plot(t_plot/3600, rho_plot, 'r--', label='Predicted')
        ax.plot((times-times[0])/3600, rhos, 'bo', label='Observed')
        ax.axvline((fit2d["t_peri_sec"]-times[0])/3600, color='k', ls=':', label='Perihelion')
        ax.set_xlabel('Hours from first frame')
        ax.set_ylabel('Distance (R⊙)')
        ax.legend(); ax.grid(True, alpha=0.3)
        plot_path = plot_dir / f"{det}_track{len(tr)}_orbit.png"
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        result["plot_path"] = str(plot_path.relative_to("detections"))
    except Exception as e:
        log(f"Plot failed: {e}")

    return result

# --------------------------------------------------------------
# ANIMATION WRITER (GIF + MP4)
# --------------------------------------------------------------
def write_animation_for_track(det, imgs, tr, out_dir, track_id, fps=6, radius=8, line_thickness=2):
    out_dir = Path(out_dir)
    tmin, tmax = tr[0][0], tr[-1][0]
    xy = {t: (int(round(x)), int(round(y))) for t, x, y in tr}
    trail = []
    clean, annot = [], []
    for ti in range(tmin, tmax + 1):
        im = imgs[ti]
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im.copy()
        clean.append(bgr.copy())
        current_annot = bgr.copy()
        if ti in xy:
            trail.append(xy[ti])
            for a, b in zip(trail[:-1], trail[1:]):
                cv2.line(current_annot, a, b, (0, 255, 255), line_thickness)
            cv2.circle(current_annot, xy[ti], radius, (0, 255, 255), line_thickness)
            cv2.circle(current_annot, xy[ti], radius + 2, (0, 255, 255), line_thickness - 1)
        annot.append(current_annot)

    anim_dir = ensure_dir(out_dir / "animations")
    base_gif_a = anim_dir / f"{det}_track{track_id}_annotated.gif"
    base_gif_c = anim_dir / f"{det}_track{track_id}_clean.gif"
    base_mp4_a = anim_dir / f"{det}_track{track_id}_annotated.mp4"

    out = {
        "animation_gif_path": None,
        "animation_gif_clean_path": None,
        "animation_mp4_path": None,
        "original_mid_path": None,
        "annotated_mid_path": None,
    }
    try:
        imageio.mimsave(str(base_gif_a), annot, fps=fps)
        out["animation_gif_path"] = str(base_gif_a.relative_to(out_dir))
        imageio.mimsave(str(base_gif_c), clean, fps=fps)
        out["animation_gif_clean_path"] = str(base_gif_c.relative_to(out_dir))
        writer = imageio.get_writer(str(base_mp4_a), fps=fps, codec='libx264', pixelformat='yuv420p')
        for frame in annot:
            writer.append_data(frame)
        writer.close()
        out["animation_mp4_path"] = str(base_mp4_a.relative_to(out_dir))
    except Exception as e:
        log(f"Animation write failed: {e}")

    # Mid-frame stills
    mid_idx = tr[len(tr)//2][0]
    mid_ti = mid_idx - tmin
    still_orig_dir = ensure_dir(out_dir / "originals")
    still_anno_dir = ensure_dir(out_dir / "annotated")
    try:
        orig_png = still_orig_dir / f"{det}_track{track_id}_mid.png"
        anno_png = still_anno_dir / f"{det}_track{track_id}_mid.png"
        cv2.imwrite(str(orig_png), clean[mid_ti])
        mid_anno = annot[mid_ti].copy()
        mx, my = xy[mid_idx]
        cv2.line(mid_anno, (mx-10, my), (mx+10, my), (0,0,255), 2)
        cv2.line(mid_anno, (mx, my-10), (mx, my+10), (0,0,255), 2)
        cv2.imwrite(str(anno_png), mid_anno)
        out["original_mid_path"] = str(orig_png.relative_to(out_dir))
        out["annotated_mid_path"] = str(anno_png.relative_to(out_dir))
    except Exception as e:
        log(f"Still write failed: {e}")

    return out

# --------------------------------------------------------------
# DETECTION IN ONE SEQUENCE
# --------------------------------------------------------------
def detect_in_sequence(det, det_frames, out_dir, hours, step_min):
    img_paths = sorted(
        [p for p in det_frames.glob("*.*")
         if p.suffix.lower() in {".jpg",".jpeg",".png",".gif"}],
        key=lambda p: p.name, reverse=True
    )
    if not img_paths:
        log(f"No frames for {det}")
        return []

    names = [p.name for p in img_paths]
    imgs  = [load_image(p) for p in img_paths]
    imgs  = [im for im in imgs if im is not None]
    if len(imgs) < 2:
        log(f"Not enough frames for {det}")
        return []

    timestamps = [timestamp_from_name(n) for n in names]

    # Background subtraction
    stack = np.stack(imgs[:min(10, len(imgs))])
    bg = gaussian_filter(np.median(stack, axis=0).astype(np.float32), sigma=1)
    diff = [cv2.absdiff(im.astype(np.float32), bg) for im in imgs]

    # Bright points
    points_per_frame = []
    for d in diff:
        _, thr = cv2.threshold(d, 30, 255, cv2.THRESH_BINARY)
        thr8 = thr.astype(np.uint8)
        contours, _ = cv2.findContours(thr8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 200:
                M = cv2.moments(cnt)
                if M["m00"]:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    pts.append((cx, cy))
        points_per_frame.append(pts)

    # Simple NN tracking
    tracks = []
    for i, pts in enumerate(points_per_frame):
        for pt in pts:
            x, y = pt
            matched = None
            min_dist = 30
            for tr in tracks:
                if abs(tr[-1][0] - i) <= 3:
                    prev_x, prev_y = tr[-1][1], tr[-1][2]
                    dist = np.hypot(x - prev_x, y - prev_y)
                    if dist < min_dist:
                        min_dist = dist
                        matched = tr
            if matched:
                matched.append((i, x, y))
            else:
                tracks.append([(i, x, y)])

    good_tracks = [tr for tr in tracks if len(tr) >= 3]

    # AI stub + crops
    crops_dir = ensure_dir(out_dir / "crops")
    candidates = []

    for idx, tr in enumerate(good_tracks):
        mid_t = len(tr) // 2
        frame_idx = tr[mid_t][0]
        im = imgs[frame_idx]
        x, y = int(tr[mid_t][1]), int(tr[mid_t][2])
        sz = 64 if det == "C2" else 128
        h, w = im.shape
        y1, y2 = max(0, y - sz), min(h, y + sz)
        x1, x2 = max(0, x - sz), min(w, x + sz)
        crop = im[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_path = crops_dir / f"{det}_track{idx}_crop.png"
        cv2.imwrite(str(crop_path), crop)

        # ---- Dummy AI ------------------------------------------------
        ai = {"label": "comet" if np.random.rand() > 0.7 else "not_comet",
              "score": np.random.rand()}

        if AI_VETO and ai["label"] == AI_VETO_LABEL and ai["score"] > AI_VETO_MAX:
            ai["label"] = "vetoed"

        anim_paths = write_animation_for_track(det, imgs, tr, out_dir, track_id=idx)

        cand = {
            "detector": det,
            "track_index": idx,
            "score": len(tr) * ai["score"],
            "positions": [
                {"time_idx": t,
                 "x": round(x, 1),
                 "y": round(y, 1),
                 "time_utc": timestamps[t] if t < len(timestamps) and timestamps[t] else ""}
                for t, x, y in tr
            ],
            "crop_path": str(crop_path.relative_to(out_dir)),
            "ai_label": ai["label"],
            "ai_score": ai["score"],
            "auto_selected": True
        }
        cand.update(anim_paths)

        # ---- ORBIT ---------------------------------------------------
        dual = cand.get("dual_channel_match")
        orbit = predict_orbit_for_track(det, tr, timestamps, dual_match=dual)
        cand["predicted_orbit"] = orbit

        candidates.append(cand)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:SELECT_TOP_N]

# --------------------------------------------------------------
# DUAL CHANNEL MATCH
# --------------------------------------------------------------
def match_dual_channel(c2_cands, c3_cands):
    for c2 in c2_cands:
        if not c2["positions"]: continue
        t2_str = c2["positions"][-1]["time_utc"]
        if not t2_str: continue
        try:
            t2 = datetime.fromisoformat(t2_str.replace("Z", "+00:00"))
        except:
            continue
        for c3 in c3_cands:
            if not c3["positions"]: continue
            t3_str = c3["positions"][0]["time_utc"]
            if not t3_str: continue
            try:
                t3 = datetime.fromisoformat(t3_str.replace("Z", "+00:00"))
            except:
                continue
            dt = abs((t3 - t2).total_seconds() / 60)
            if dt > 60: continue
            pa2 = np.arctan2(c2["positions"][-1]["y"]-256,
                             c2["positions"][-1]["x"]-256) * 180 / np.pi
            pa3 = np.arctan2(c3["positions"][0]["y"]-512,
                             c3["positions"][0]["x"]-512) * 180 / np.pi
            diff = min(abs(pa2 - pa3), 360 - abs(pa2 - pa3))
            if diff <= 25:
                c2["dual_channel_match"] = {
                    "with": f"{c3['detector']}#{c3['track_index']}",
                    "pa_diff_deg": round(diff, 1),
                }
                break

# --------------------------------------------------------------
# DEBUG IMAGES
# --------------------------------------------------------------
def generate_debug_images(det, det_frames, out_dir, all_cands):
    if not DEBUG:
        return
    frame_files = sorted(det_frames.glob("*.*"), key=lambda p: p.name, reverse=True)
    if not frame_files:
        return
    latest_img = load_image(frame_files[0])
    if latest_img is not None:
        thumb_path = out_dir / f"lastthumb_{det}.png"
        cv2.imwrite(str(thumb_path), latest_img)

    overlay_img = latest_img.copy() if latest_img is not None else np.zeros((1024,1024), dtype=np.uint8)
    for cand in all_cands:
        if cand["detector"] != det or not cand["positions"]:
            continue
        mid = cand["positions"][len(cand["positions"])//2]
        cv2.circle(overlay_img, (int(mid["x"]), int(mid["y"])), 6, (0,255,0), 2)
    if overlay_img.ndim == 2:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)
    overlay_path = out_dir / f"overlay_{det}.png"
    cv2.imwrite(str(overlay_path), overlay_img)

    contact_imgs = [load_image(f) for f in frame_files[:9]]
    contact_imgs = [im for im in contact_imgs if im is not None]
    if contact_imgs:
        h, w = contact_imgs[0].shape
        grid = int(np.ceil(np.sqrt(len(contact_imgs))))
        contact = np.zeros((grid*h, grid*w), dtype=np.uint8)
        for i, img in enumerate(contact_imgs):
            r, c = divmod(i, grid)
            contact[r*h:(r+1)*h, c*w:(c+1)*w] = img
        contact_path = out_dir / f"contact_{det}.png"
        cv2.imwrite(str(contact_path), contact)

# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--step-min", type=int, default=12)
    parser.add_argument("--out", type=str, default="detections")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    frames_dir = ensure_dir("frames")

    saved = fetch_lasco.fetch_window(
        hours_back=args.hours,
        step_min=args.step_min,
        root=str(frames_dir)
    )
    log(f"Downloaded {len(saved)} frames")

    all_cands = []
    for det in VALID_DETS:
        det_frames = frames_dir / det
        if not any(det_frames.glob("*.*")):
            log(f"No frames for {det}")
            continue
        cands = detect_in_sequence(det, det_frames, out_dir, args.hours, args.step_min)
        all_cands.extend(cands)

    match_dual_channel([c for c in all_cands if c["detector"] == "C2"],
                       [c for c in all_cands if c["detector"] == "C3"])

    for det in VALID_DETS:
        det_frames = frames_dir / det
        if any(det_frames.glob("*.*")):
            generate_debug_images(det, det_frames, out_dir, all_cands)

    now = datetime.utcnow()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"candidates_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(all_cands, f, indent=2)

    status = {
        "timestamp_utc": now.isoformat() + "Z",
        "detectors": {det: {
            "frames": len(list((frames_dir / det).glob("*.*"))),
            "tracks": len([c for c in all_cands if c["detector"] == det]),
        } for det in VALID_DETS}
    }
    with open(out_dir / "latest_status.json", "w") as f:
        json.dump(status, f, indent=2)

    log(f"Detected {len(all_cands)} candidates → {report_path}")

if __name__ == "__main__":
    main()
