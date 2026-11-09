"""Classical-CV perception on the night dashcam clip (no deep nets available offline).

Per sampled frame (15 fps, 320x180) we extract:
  * ego forward-speed proxy from ground-plane optical flow (relative units, then scaled with stated assumptions)
  * yaw proxy (median horizontal flow, upper region)
  * tail-light (red) / head-light (white) blob counts in the road ROI
  * tail-light pair -> lead-vehicle range via pinhole model (assumed 1.5 m lamp spacing)
  * independent-motion fraction in the ego corridor (flow residual after ego-motion model)
  * HOG pedestrian detections
Outputs data/perception.csv
"""
import sys, cv2, numpy as np, pandas as pd

VID = sys.argv[1]
OUT = sys.argv[2]
START, END = int(sys.argv[3]), int(sys.argv[4])   # frame indices (30 fps source)
STRIDE = 2
W, H = 320, 180
HOOD_Y = int(0.66 * H)     # rows below this are bonnet reflections
HORIZON = int(0.40 * H)    # approx horizon row (camera roughly level)
CORR = (int(0.30 * W), int(0.70 * W))   # ego corridor columns

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(VID)
cap.set(cv2.CAP_PROP_POS_FRAMES, START)
prev = None
rows = []
idx = START
while idx < END:
    ok, fr = cap.read()
    if not ok:
        break
    if (idx - START) % STRIDE:
        idx += 1
        continue
    small = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(2.0, (4, 4)).apply(gray)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    rec = dict(frame=idx, t=idx / 30.0)

    # ---- optical flow --------------------------------------------------
    if prev is not None:
        flow = cv2.calcOpticalFlowFarneback(prev, gray_eq, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # ground-plane speed proxy: dy ~ v (y-yh)^2/(fH)  -> v_rel = dy/(y-yh)^2
        ys, xs = np.mgrid[0:H, 0:W]
        dy = flow[..., 1]
        yy = (ys - HORIZON).astype(np.float32)
        gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1)
        tex = np.sqrt(gx ** 2 + gy ** 2)
        mask = (ys > HORIZON + 14) & (ys < HOOD_Y) & (tex > 40) & (xs > 0.15 * W) & (xs < 0.85 * W)
        if mask.sum() > 150:
            # robust fit dy = c + k*(y-yh)^2 ; c absorbs camera pitch (bumps), k ~ forward speed
            A = np.stack([np.ones(mask.sum()), yy[mask] ** 2], 1).astype(np.float64)
            b = dy[mask].astype(np.float64); w = np.ones_like(b)
            for _ in range(6):
                sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
                r = b - A @ sol; sc = 1.4826 * np.median(np.abs(r)) + 1e-3
                w = 1.0 / np.maximum(1.0, np.abs(r) / (1.5 * sc))
            rec['v_med'] = float(np.median(dy[mask] / (yy[mask] ** 2)))
            rec['v_rel'] = float(sol[1]); rec['pitch_c'] = float(sol[0]); rec['v_valid'] = int(mask.sum())
        else:
            rec['v_rel'] = np.nan; rec['pitch_c'] = np.nan; rec['v_valid'] = int(mask.sum())
        # yaw proxy: horizontal flow in the far/upper band
        band = (ys > HORIZON - 25) & (ys < HORIZON + 10) & (tex > 30)
        rec['yaw'] = float(np.median(flow[..., 0][band])) if band.sum() > 80 else np.nan
        # independent motion in corridor: residual vs radial ego-motion field
        foe = np.array([W / 2, HORIZON], np.float32)
        rx = xs - foe[0]; ry = ys - foe[1]
        r2 = rx ** 2 + ry ** 2 + 1e-3
        # radial expansion model flow = k * r  (k fit by robust median over valid pixels)
        valid = (ys < HOOD_Y) & (tex > 25)
        k = np.median(((flow[..., 0] * rx + flow[..., 1] * ry) / r2)[valid]) if valid.sum() > 200 else 0
        resx = flow[..., 0] - k * rx - (np.nanmedian(flow[..., 0][valid]) if valid.sum() > 200 else 0) * 0
        resy = flow[..., 1] - k * ry
        res = np.sqrt(resx ** 2 + resy ** 2)
        res[~valid] = 0
        thr = 2.2
        indep = (res > thr)
        indep = cv2.morphologyEx(indep.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cor = indep[HORIZON - 20:HOOD_Y, CORR[0]:CORR[1]]
        rec['k_exp'] = float(k)
        rec['indep_corr'] = float(cor.mean())
        rec['indep_all'] = float(indep[:HOOD_Y].mean())
        # lateral drift of independent motion in corridor (mean signed horizontal residual)
        m = indep.astype(bool)
        rec['indep_vx'] = float(resx[m].mean()) if m.sum() > 60 else 0.0
    else:
        rec.update(v_rel=np.nan, v_med=np.nan, k_exp=np.nan, pitch_c=np.nan, v_valid=0, yaw=np.nan, indep_corr=np.nan, indep_all=np.nan, indep_vx=np.nan)
    prev = gray_eq

    # ---- light blobs ---------------------------------------------------
    roi = slice(HORIZON - 30, HOOD_Y)
    v = hsv[..., 2]; s = hsv[..., 1]; h = hsv[..., 0]
    bright = (v > 225).astype(np.uint8)
    bright[:HORIZON - 30] = 0; bright[HOOD_Y:] = 0
    red = (((h < 10) | (h > 165)) & (s > 90) & (v > 150)).astype(np.uint8)
    red[:HORIZON - 30] = 0; red[HOOD_Y:] = 0
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(red)
    reds = [(cen[i], st[i]) for i in range(1, n) if 3 <= st[i, cv2.CC_STAT_AREA] <= 400]
    n2, _, st2, cen2 = cv2.connectedComponentsWithStats(bright)
    whites = [(cen2[i], st2[i]) for i in range(1, n2) if 4 <= st2[i, cv2.CC_STAT_AREA] <= 900]
    rec['n_red'] = len(reds); rec['n_white'] = len(whites)
    # pair tail lights: similar row, separation 7..70 px
    best = np.nan
    for i in range(len(reds)):
        for j in range(i + 1, len(reds)):
            (c1, s1), (c2, s2) = reds[i], reds[j]
            if abs(c1[1] - c2[1]) < 4 and 7 <= abs(c1[0] - c2[0]) <= 70 and \
               0.5 < s1[cv2.CC_STAT_AREA] / max(1, s2[cv2.CC_STAT_AREA]) < 2.0:
                sep = abs(c1[0] - c2[0])
                cx = (c1[0] + c2[0]) / 2
                if CORR[0] - 30 < cx < CORR[1] + 30:
                    best = sep if np.isnan(best) else max(best, sep)
    rec['lamp_sep_px'] = best
    # ---- pedestrians (HOG on full-res-ish crop) ------------------------
    rects, wts = hog.detectMultiScale(cv2.resize(gray_eq, (W * 2, H * 2))[: HOOD_Y * 2], winStride=(8, 8),
                                       padding=(8, 8), scale=1.07, hitThreshold=0.2)
    rec['n_ped'] = int(sum(1 for w in (wts.ravel() if len(wts) else []) if w > 0.6))
    # global brightness (scene lighting)
    rec['luma'] = float(gray[:HOOD_Y].mean())
    rows.append(rec)
    idx += 1
pd.DataFrame(rows).to_csv(OUT, index=False)
print('done', OUT, len(rows))
