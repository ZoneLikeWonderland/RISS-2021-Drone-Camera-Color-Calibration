# import cv2.ximgproc
from functools import cmp_to_key
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# x_base, y_base = np.meshgrid(range(320), range(240))
x_base, y_base = None, None


AREA_THRESHOLD = 0.1
STD_THRESHOLD = 0.1
LINE_MIN_LENGTH = 4

plt.ion()


def pickout(raw, rect, c, show_list):

    global x_base, y_base
    if x_base is None:
        x_base, y_base = np.meshgrid(range(c.shape[1]), range(c.shape[0]))

    if rect is not None:
        rect = cv2.dilate(rect, np.ones((10, 10)))
        rect_b = (rect > 0.05).astype(np.uint8)
        #cv2.imshow("rect_b", rect_b * 255)
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(rect_b)
        bgid = labels[rect_b == 0].min()
        stats[bgid, 4] = -1
        tgtid = np.argmax(stats[:, -1])
        rect[labels != tgtid] = 0
        c *= rect

    c_mask = (c > 0.1).astype(np.uint8) * 255

    #cv2.imshow("c_mask", c_mask)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(c_mask)

    ps = []

    c_show = c.copy()

    for i in range(1, retval):
        mask = (labels == i).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((10, 10)))
        pc = mask * c
        x = (pc * x_base).sum() / pc.sum()
        y = (pc * y_base).sum() / pc.sum()
        ps.append((-pc.sum(), (x, y)))
        cv2.circle(c_show, (int(x), int(y)), 10, 0.2)

    ps = [i[1] for i in sorted(ps)[:4]]
    ps = np.array(ps)
    center = ps.mean(axis=0)
    # print("center", center)
    ps = sorted(ps, key=lambda p: np.arctan2((p - center)[1], (p - center)[0]))
    ps = np.array(ps)

    for i, p in enumerate(ps):
        cv2.circle(c_show, (int(p[0]), int(p[1])), 10, 255)
        cv2.putText(c_show, str(i), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    #cv2.imshow("c_show", c_show)
    show_list["c_show"] = c_show

    if len(ps) < 4:
        return

    ps[:, 0] = ps[:, 0] / c.shape[1] * raw.shape[1]
    ps[:, 1] = ps[:, 1] / c.shape[0] * raw.shape[0]

    tw, th = 200, 200
    cp = np.array([(0, 0), (tw, 0), (tw, th), (0, th)])

    H, _ = cv2.findHomography(
        ps,
        cp
    )

    warp = cv2.warpPerspective(raw, H, (tw, th))
    #cv2.imshow("warp", warp)
    show_list["warp"] = warp

    warp = warp.astype(np.float32)

    canny = cv2.Canny((warp * 255).astype(np.uint8), 30, 200)
    canny = cv2.dilate(canny, None)
    canny = cv2.erode(canny, None)
    canny = 255 - canny

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(canny, connectivity=4)
    area = stats[:, -1:]
    widths = stats[:, 2]
    heights = stats[:, 3]

    expected_area = widths * heights
    exp_ratio = expected_area / area[:, 0]
    # print(exp_ratio)
    reject = (exp_ratio > 1 + AREA_THRESHOLD) | (exp_ratio < 1 - AREA_THRESHOLD)
    # print(reject)
    stats_id = np.concatenate((
        area,
        np.arange(0, retval)[:, None],
    ), axis=1)

    stats_ratio = np.array(sorted(
        stats_id.tolist()
    ), dtype=np.float32)
    if len(stats) <= 2:
        return
    stats_ratio[1:, 0] /= stats_ratio[:-1, 0]
    stats_ratio[0, 0] = 0

    # print(stats_ratio)

    start = None
    end = None

    best = (0, None, None)

    notbad = []

    for i in range(1, retval):
        if start is None and stats_ratio[i, 0] < 1.1:
            start = i - 1
        if end is None and start is not None and stats_ratio[i, 0] > 1.1:
            end = i
            best = max(best, (end - start, start, end))
            if end - start > 4:
                notbad.append((end - start, start, end))
            start = None
            end = None

    new_label = np.zeros_like(labels, np.float32)
    select = []

    length, start, end = best
    if start is None or end is None:
        return

    for length, start, end in notbad:
        for i in range(start, end):
            c = int(stats_ratio[i, 1])
            if reject[c]:
                continue
            std = np.std(warp[labels == c], axis=0)
            if (std > STD_THRESHOLD).any():
                continue
            if (labels == c).sum() < tw * th / 40:
                continue

            select.append(c)

            new_label[labels == c] = c * 0.02
            cv2.putText(new_label, str(c), tuple([int(i) for i in centroids[c]]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #cv2.imshow("canny_labels", new_label)
    show_list["canny_labels"] = new_label

    #cv2.imshow("canny", canny)
    show_list["canny"] = canny
    if len(select) <= 2:
        return

    def get_half(ar):
        xs = np.array(sorted(ar))
        # print(xs)
        xs = np.array(sorted(xs[1:] - xs[:-1]))
        half = xs[xs > xs.mean()].mean() / 2
        # print(xs, half)
        jl = np.zeros(ar.shape, np.int)
        # jl2 = np.zeros(ar.shape, np.int)
        for i in range(ar.shape[0]):
            jl[(ar < ar[i] + half) & (ar > ar[i] - half)] = i
        # print(ar)

        ms = []
        for i, j in enumerate(sorted(np.unique(jl))):
            ms.append(ar[jl == j].mean())
        ms = sorted(ms)

        return ms, half

    xs, x_half = get_half(centroids[select][:, 0])
    ys, y_half = get_half(centroids[select][:, 1])

    for x in xs:
        cv2.line(new_label, (int(x), 0), (int(x), 999), 0.6)
    for y in ys:
        cv2.line(new_label, (0, int(y)), (999, int(y)), 0.6)

    width = np.median(widths[select])
    height = np.median(heights[select])
    # print(widths, heights)
    scale = 0.4

    matrix = np.zeros((len(ys), len(xs), 3))
    matrix_use = np.zeros((len(ys), len(xs)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            select_area = np.zeros_like(new_label)
            cv2.circle(select_area, (int(x), int(y)), int(min(width, height) * scale), 255, -1)
            data = warp[select_area > 0]
            std = np.std(data, axis=0)
            mean = np.mean(data, axis=0)
            # print(std)
            if (std > STD_THRESHOLD).any():
                continue

            matrix[j, i] = mean
            matrix_use[j, i] = 1

            cv2.circle(new_label, (int(x), int(y)), int(min(width, height) * scale), 0.6, 1)

    #cv2.imshow("canny_labels", new_label)
    show_list["canny_labels"] = new_label
    matrix_show = cv2.resize(matrix, warp.shape[:2], interpolation=cv2.INTER_NEAREST)
    #cv2.imshow("matrix", matrix_show)
    show_list["matrix"] = matrix_show

    best = (-1, None, None, None)

    for i, x in enumerate(xs):
        line = matrix[:, i][matrix_use[:, i] > 0]
        if line.shape[0] < LINE_MIN_LENGTH:
            continue
        diff = line[1:] - line[:-1]
        diff *= 1 if diff.mean() > 0 else - 1
        if diff.min() < 0:
            continue
        crstd = np.std(line / line.mean(axis=1)[..., None], axis=0).sum()
        # plt.plot(line/line.mean(axis=1)[...,None])
        # plt.show()
        best = max(best, (-crstd, (slice(None, None, None), i), line, diff))

    for j, y in enumerate(ys):
        line = matrix[j][matrix_use[j] > 0]
        if line.shape[0] < LINE_MIN_LENGTH:
            continue
        diff = line[1:] - line[:-1]
        diff *= 1 if diff.mean() > 0 else - 1
        if diff.min() < 0:
            continue
        crstd = np.std(line / line.mean(axis=1)[..., None], axis=0).sum()
        # plt.plot(line/line.mean(axis=1)[...,None])
        # plt.show()
        best = max(best, (-crstd, (j, slice(None, None, None)), line, diff))

    show_key = np.zeros((len(ys), len(xs)))

    # print(best)

    _, idx, line, diff = best
    if idx is None:
        return
    show_key[idx] = 1

    good = np.ones(line.shape[0], np.int)
    good[0] = 0
    good[-1] = 0

    for i in range(len(diff)):
        if (diff[i] <= 0).any():
            good[i] = 0
            good[i + 1] = 0

    show_key[idx][matrix_use[idx] > 0] = good * 2

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if matrix_use[j, i] > 0:
                if show_key[j, i] == 1:
                    cv2.circle(new_label, (int(x), int(y)), int(min(width, height) * scale), 0.8, 2)
                if show_key[j, i] == 2:
                    cv2.circle(new_label, (int(x), int(y)), int(min(width, height) * scale), 1, 4)

    # print(show_key)

    #cv2.imshow("canny_labels", new_label)
    show_list["canny_labels"] = new_label

    # print(good)
    to_be_calibrate = line[good > 0]
    # print(to_be_calibrate)
    coeff = 1 / to_be_calibrate
    coeff /= coeff.min(axis=1)[..., None]
    coeff = coeff.mean(axis=0)
    # print(coeff)

    warp_calibrated = coeff * warp
    #cv2.imshow("warp_calibrated", warp_calibrated)
    show_list["warp_calibrated"] = warp_calibrated

    cv2.waitKey(1)

    bright = warp.mean()

    x = ps[:, 0].mean()
    y = ps[:, 1].mean()

    dist = ((x - raw.shape[1] / 2)**2 + (y - raw.shape[0] / 2)**2)**0.5 / max(raw.shape)

    plt.scatter(dist, bright)
    plt.draw()
    plt.pause(0.01)

    return coeff, [dist.item(), bright.item()]
