"""Gコード(NCプログラム)の最小パーサ。ROS非依存。

対応: G00 G01 G81 G80 G90 G91 G21 / X Y Z R F / M03 M05 M30 / ( ) と ; のコメント / N行番号
未対応のGコードは行番号付きでエラーにする（黙って無視して誤動作しないため）。
"""
import re
from dataclasses import dataclass

_WORD = re.compile(r'([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))')


class GCodeError(ValueError):
    pass


@dataclass
class Move:
    """work座標[mm]での1回の移動。rapid=True は G00。"""
    x: float
    y: float
    z: float
    rapid: bool
    feed: float  # mm/min（rapidのときは使わない）
    line: int


@dataclass
class Event:
    """主軸などの補助指令。実機のドリル回転は未モデル化のためログ用。"""
    text: str
    line: int


def _clean(raw):
    text = re.sub(r'\([^)]*\)', ' ', raw)
    text = text.split(';')[0]
    return text.upper().strip()


def parse(source):
    """Gコード文字列を [Move | Event] のリストに展開する。"""
    pos = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
    absolute = True
    feed = 0.0
    motion = None          # 'G0' / 'G1' / 'G81' / None
    retract_r = None       # G81 の R
    items = []

    for lineno, raw in enumerate(source.splitlines(), start=1):
        text = _clean(raw)
        if not text or text.startswith('%'):
            continue
        # 単語として解釈できない文字が残っていないか確認
        if _WORD.sub('', text).strip():
            raise GCodeError(f'{lineno}行目: 解釈できない記述です: {raw.strip()}')

        words = [(m.group(1), float(m.group(2))) for m in _WORD.finditer(text)]
        target = {}
        for letter, value in words:
            if letter == 'N':
                continue
            if letter == 'G':
                code = int(value)
                if code in (0, 1, 81):
                    motion = f'G{code}'
                elif code == 80:
                    motion = None
                elif code == 90:
                    absolute = True
                elif code == 91:
                    absolute = False
                elif code == 21:
                    pass
                else:
                    raise GCodeError(f'{lineno}行目: 未対応の G{code} です')
            elif letter == 'M':
                code = int(value)
                if code in (3, 4, 5, 30, 2):
                    items.append(Event(f'M{code:02d}', lineno))
                else:
                    raise GCodeError(f'{lineno}行目: 未対応の M{code} です')
            elif letter == 'F':
                feed = value
            elif letter in 'XYZ':
                target[letter] = value
            elif letter == 'R':
                retract_r = value
            elif letter in 'S':
                items.append(Event(f'S{value:g}', lineno))
            else:
                raise GCodeError(f'{lineno}行目: 未対応のアドレス {letter} です')

        if not target:
            continue
        if motion is None:
            raise GCodeError(f'{lineno}行目: 移動指令(G00/G01/G81)が指定されていません')

        new = {}
        for axis in 'XYZ':
            if axis in target:
                new[axis] = target[axis] if absolute else pos[axis] + target[axis]
            else:
                new[axis] = pos[axis]

        if motion == 'G81':
            if retract_r is None or 'Z' not in target:
                raise GCodeError(f'{lineno}行目: G81 には R と Z が必要です')
            if feed <= 0:
                raise GCodeError(f'{lineno}行目: G81 の前に F（送り速度）が必要です')
            r = retract_r if absolute else pos['Z'] + retract_r
            items.append(Move(new['X'], new['Y'], pos['Z'], True, 0.0, lineno))
            items.append(Move(new['X'], new['Y'], r, True, 0.0, lineno))
            items.append(Move(new['X'], new['Y'], new['Z'], False, feed, lineno))
            items.append(Move(new['X'], new['Y'], r, True, 0.0, lineno))
            pos = {'X': new['X'], 'Y': new['Y'], 'Z': r}
            continue

        rapid = motion == 'G0'
        if not rapid and feed <= 0:
            raise GCodeError(f'{lineno}行目: G01 の前に F（送り速度）が必要です')
        items.append(Move(new['X'], new['Y'], new['Z'], rapid, feed, lineno))
        pos = new

    return items
