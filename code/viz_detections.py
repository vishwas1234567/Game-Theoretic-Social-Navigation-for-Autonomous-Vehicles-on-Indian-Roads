import cv2, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
V = sys.argv[1]; W, H = 320, 180; HOOD = int(.66 * H); HOR = int(.40 * H)
hog = cv2.HOGDescriptor(); hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture(V); out = []
for tsec in [20, 62, 95, 128, 172, 205]:
    f = int(tsec * 30); cap.set(cv2.CAP_PROP_POS_FRAMES, f - 2); ok, a = cap.read(); cap.read(); ok, b = cap.read()
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA); b = cv2.resize(b, (W, H), interpolation=cv2.INTER_AREA)
    c = cv2.createCLAHE(2.0, (4, 4)); ga = c.apply(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)); gb = c.apply(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, .5, 3, 15, 3, 5, 1.2, 0)
    ys, xs = np.mgrid[0:H, 0:W]; rx = xs - W / 2; ry = ys - HOR; r2 = rx ** 2 + ry ** 2 + 1e-3
    gx = cv2.Sobel(gb, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gb, cv2.CV_32F, 0, 1); tex = np.hypot(gx, gy)
    valid = (ys < HOOD) & (tex > 25)
    k = np.median(((flow[..., 0] * rx + flow[..., 1] * ry) / r2)[valid]) if valid.sum() > 200 else 0
    res = np.hypot(flow[..., 0] - k * rx, flow[..., 1] - k * ry); res[~valid] = 0
    m = cv2.morphologyEx((res > 2.2).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    big = cv2.resize(b, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC); mm = cv2.resize(m, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
    big[mm > 0] = (0.5 * big[mm > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 10) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 150)).astype(np.uint8); red[:HOR - 30] = 0; red[HOOD:] = 0
    n, _, st, cen = cv2.connectedComponentsWithStats(red)
    for i in range(1, n):
        if 3 <= st[i, 4] <= 400: cv2.circle(big, (int(cen[i][0] * 2), int(cen[i][1] * 2)), 7, (0, 0, 255), 1)
    wh = (hsv[..., 2] > 225).astype(np.uint8); wh[:HOR - 30] = 0; wh[HOOD:] = 0
    n, _, st, cen = cv2.connectedComponentsWithStats(wh)
    for i in range(1, n):
        if 4 <= st[i, 4] <= 900: cv2.circle(big, (int(cen[i][0] * 2), int(cen[i][1] * 2)), 7, (255, 255, 0), 1)
    rects, w = hog.detectMultiScale(cv2.resize(gb, (W * 2, H * 2))[:HOOD * 2], winStride=(8, 8), padding=(8, 8), scale=1.07, hitThreshold=.2)
    for (x, y, ww, hh), s in zip(rects, w.ravel() if len(w) else []):
        if s > .6: cv2.rectangle(big, (x, y), (x + ww, y + hh), (255, 0, 255), 1)
    cv2.line(big, (0, HOOD * 2), (W * 2, HOOD * 2), (200, 200, 200), 1)
    out.append((tsec, cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))
fig, ax = plt.subplots(2, 3, figsize=(7.4, 4.3))
for a_, (ts, im) in zip(ax.ravel(), out): a_.imshow(im); a_.set_title(f't = {ts} s', fontsize=8); a_.axis('off')
plt.tight_layout(); plt.savefig('figures/detections.png', dpi=170)
