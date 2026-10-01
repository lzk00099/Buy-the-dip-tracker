"""Run exactly the UI's scoring pipeline without an open browser.

SENTINEL_DATA_DIR points at the checked-out market-data branch. Credentials
come only from Actions secrets; observation exports never include credentials.
"""
import os
from pathlib import Path
import sys

os.environ['SENTINEL_COLLECTOR'] = '1'
from streamlit.testing.v1 import AppTest


def main():
    app = AppTest.from_file(str(Path(__file__).with_name('main2.py')),default_timeout=240).run()
    if app.exception:
        # UI errors are reported, but no secrets/config environment is dumped.
        print('Collector failed:', [e.message for e in app.exception])
        return 1
    try:
        record = app.session_state['latest_observation']
    except KeyError:
        print('Collector did not produce an observation.')
        return 1
    print('Observed:', record['observed_at'], 'data date:',record['trade_date'],
          'complete:',record['quality_ok'])
    return 0 if record['quality_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
