"""Per-frame ODD-relevant attributes from the clip (classical CV proxies, 15 fps, 320x180)."""
import sys, cv2, numpy as np, pandas as pd
VID, OUT, START, END = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
W, H = 320, 180; HOOD = int(.66 * H); HOR = int(.40 * H)
cap = cv2.VideoCapture(VID); cap.set(cv2.CAP_PROP_POS_FRAMES, START); rows = []; i = START
clahe = cv2.createCLAHE(2.0, (4, 4))
while i < END:
    ok, fr = cap.read()
    if not ok: break
    if (i - START) % 2: i += 1; continue
    g = cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    roi = g[HOR - 10:HOOD]
    ge = clahe.apply(g)[HOR + 5:HOOD]
    ed = cv2.Canny(cv2.GaussianBlur(ge, (3, 3), 0), 40, 110)
    ln = cv2.HoughLinesP(ed, 1, np.pi / 180, 18, minLineLength=22, maxLineGap=4)
    sc = 0.0
    if ln is not None:
        for x1, y1, x2, y2 in ln.reshape(-1, 4):
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if 20 < ang < 80: sc += np.hypot(x2 - x1, y2 - y1)
    rows.append(dict(t=i / 30.0, luma_roi=float(roi.mean()), dark_frac=float((roi < 30).mean()), glare_frac=float((roi >= 250).mean()),
                     contrast=float(roi.std()), sharp=float(cv2.Laplacian(roi, cv2.CV_32F).var()), lane_score=sc / W))
    i += 1
pd.DataFrame(rows).to_csv(OUT, index=False)
