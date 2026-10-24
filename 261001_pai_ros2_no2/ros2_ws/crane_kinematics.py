"""CRANE+ V2 (4関節) の順運動学と数値逆運動学。ROS に依存しない純Python実装。

座標系は MoveIt/URDF の base_link（ロボット根元の床面、x:前方, z:上）。
ドリル先端の位置(x, y, z)と、ドリル軸の傾き(pitch)から joint1〜4 を求める。
"""
import math

import numpy as np

_HOME = math.radians(150.0)
JOINT_LIMITS = np.array([
    [math.radians(0.0) - _HOME, math.radians(300.0) - _HOME],
    [math.radians(45.45) - _HOME, math.radians(252.20) - _HOME],
    [math.radians(3.52) - _HOME, math.radians(290.62) - _HOME],
    [math.radians(44.57) - _HOME, math.radians(251.32) - _HOME],
])

# (関節原点の親リンクからの平行移動, 回転軸) crane_plus.xacro の値
_CHAIN = [
    (np.array([0.0, 0.0, 0.0454]), np.array([0.0, 0.0, 1.0])),
    (np.array([0.0, 0.0, 0.026]), np.array([0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, 0.083]), np.array([0.0, -1.0, 0.0])),
    (np.array([0.0, 0.0, 0.0935]), np.array([0.0, -1.0, 0.0])),
]

# crane_plus_link4 の座標系で見たドリル先端位置（hand関節 + drill_mount + 円柱の半分の長さ）。
# URDFでドリルの取り付け位置を変えたらここも合わせる。move_to_xyz.py はTFから自動で求める
DEFAULT_TIP_IN_LINK4 = np.array([0.0, -0.0145 + 0.06, 0.045 + 0.07 + 0.045])


def _rot(axis, angle):
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = 1.0 - c
    return np.array([
        [c + x * x * k, x * y * k - z * s, x * z * k + y * s],
        [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
        [z * x * k - y * s, z * y * k + x * s, c + z * z * k],
    ])


def forward(q, tip_in_link4=DEFAULT_TIP_IN_LINK4):
    """関節角 q[4] から (ドリル先端位置, link4のz軸方向ベクトル) を返す。"""
    pos = np.zeros(3)
    rot = np.eye(3)
    for (offset, axis), angle in zip(_CHAIN, q):
        pos = pos + rot @ offset
        rot = rot @ _rot(axis, angle)
    return pos + rot @ np.asarray(tip_in_link4), rot @ np.array([0.0, 0.0, 1.0])


def _residual(q, target, pitch, tip_in_link4):
    pos, _ = forward(q, tip_in_link4)
    # link4 のピッチ角: 軸 y,-y,-y の回転なので q2 - q3 - q4
    err_pitch = (q[1] - q[2] - q[3] - pitch + math.pi) % (2 * math.pi) - math.pi
    return np.append(pos - target, 0.05 * err_pitch)


def solve_ik(target_xyz, pitch=math.pi, tip_in_link4=DEFAULT_TIP_IN_LINK4,
             seed=None, tol=1e-4, n_seeds=20):
    """ドリル先端を target_xyz[m] に置く関節角 [q1..q4] を返す。解けなければ None。

    pitch: link4 の z 軸（ドリル軸）の傾き[rad]。0 で真上、pi で真下を向く。
    seed: 初期値。連続した点を解くときは前の解を渡すと関節が飛びにくい。
    """
    target = np.asarray(target_xyz, dtype=float)
    low, high = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    rng = np.random.default_rng(0)
    seeds = [] if seed is None else [np.asarray(seed, dtype=float)]
    seeds += [rng.uniform(low, high)     for _ in range(n_seeds)]

    best = None
    for q in seeds:
        q = np.clip(q, low, high)
        for _ in range(60):
            r = _residual(q, target, pitch, tip_in_link4)
            jac = np.empty((4, 4))
            for i in range(4):
                dq = q.copy()
                dq[i] += 1e-6
                jac[:, i] = (_residual(dq, target, pitch, tip_in_link4) - r) / 1e-6
            step = np.linalg.lstsq(jac.T @ jac + 1e-6 * np.eye(4), -jac.T @ r, rcond=None)[0]
            q = np.clip(q + step, low, high)
            if np.linalg.norm(step) < 1e-9:
                break
        pos, _ = forward(q, tip_in_link4)
        err = np.linalg.norm(pos - target)
        pitch_err = abs((q[1] - q[2] - q[3] - pitch + math.pi) % (2 * math.pi) - math.pi)
        if err < tol and pitch_err < 1e-2:
            if seed is None:
                return q
            # seed がある場合は seed に近い解を優先する
            if best is None or np.linalg.norm(q - seeds[0]) < np.linalg.norm(best - seeds[0]):
                best = q
                if np.linalg.norm(best - seeds[0]) < 0.3:
                    return best
    return best
