"""Read public leaderboard endpoints exposed by Codabench's own web client."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import urllib.request
import urllib.error

OUT = Path(__file__).resolve().parent / 'review'
PHASES = {1: 30683, 2: 30684, 3: 30685, 4: 30686, 5: 30687, 6: 30688, 7: 30701}


def get(item):
    phase, pk = item
    url = f'https://www.codabench.org/api/phases/{pk}/get_leaderboard/'
    req = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://www.codabench.org/competitions/18186/'})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            raw = r.read()
        (OUT / f'codabench_leaderboard_p{phase}.json').write_bytes(raw)
        data = json.loads(raw)
        text = json.dumps(data)
        pos = text.find('a_240131')
        return {'phase': phase, 'status': 200, 'bytes': len(raw), 'keys': list(data) if isinstance(data, dict) else 'list',
                'user_found': pos >= 0, 'user_excerpt': text[max(0, pos-200):pos+800] if pos >= 0 else None}
    except Exception as e:
        return {'phase': phase, 'url': url, 'error': str(e)}


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(get, PHASES.items()):
            print(json.dumps(r))
