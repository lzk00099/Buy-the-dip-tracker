import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

import market_runtime as mr


@pytest.mark.parametrize('time,state',[
    ('2026-07-03 10:00','closed'), # Independence Day observed
    ('2026-09-30 09:29','premarket'),
    ('2026-09-30 09:30','regular'),
    ('2026-11-27 12:59','regular'), # half day
    ('2026-11-27 13:00','postmarket'),
    ('2026-09-26 12:00','closed'),
])
def test_calendar(time,state):
    assert mr.market_session(pd.Timestamp(time,tz=mr.ET))['session'] == state


def frame(rows):
    return pd.DataFrame(rows,columns=['symbol','price','timestamp','source']).set_index('symbol')


def test_each_cached_quote_checked():
    now=pd.Timestamp('2026-09-30T15:00:00Z')
    data=frame([('QQQ',600,now-pd.Timedelta(minutes=5),'a'),
                ('SPY',650,now-pd.Timedelta(days=1),'a'),
                ('^VIX',20,now+pd.Timedelta(minutes=10),'a'),
                ('IWM',float('inf'),now,'a'),('RSP',-5,now,'a')])
    assert mr.validate_quotes(data,now).index.tolist()==['QQQ']


def test_newest_source_wins():
    now=pd.Timestamp('2026-09-30T15:00:00Z')
    data=frame([('QQQ',600,now,'fresh'),('QQQ',590,now-pd.Timedelta(minutes=20),'old')])
    assert mr.validate_quotes(data,now).loc['QQQ','source']=='fresh'


def test_overlay_never_fills_missing_symbol(monkeypatch):
    now=pd.Timestamp.now(tz='UTC')
    prior=pd.Timestamp(now.tz_convert(mr.ET).date())-pd.Timedelta(days=1)
    daily=pd.DataFrame({'QQQ':[600],'SPY':[650]},index=[prior])
    q=frame([('QQQ',605,now,'test')])
    out=mr.overlay(daily,q,['QQQ'])
    assert pd.isna(out.SPY.iloc[-1])
    rejected=mr.overlay(daily,q,['QQQ','SPY'],True)
    pd.testing.assert_frame_equal(rejected,daily)
    assert rejected.attrs['quote_mode']=='daily_fallback'


def test_unsynchronised_ratio_rejected():
    now=pd.Timestamp.now(tz='UTC')
    q=frame([('^VIX',20,now,'a'),('^VIX3M',25,now-pd.Timedelta(minutes=25),'b')])
    daily=pd.DataFrame({'^VIX':[18],'^VIX3M':[24]},index=[pd.Timestamp('2026-09-29')])
    assert mr.overlay(daily,q,['^VIX','^VIX3M'],True).attrs['quote_mode']=='daily_fallback'


def test_dix_percent_consistent():
    df=mr.normalize_dix(pd.DataFrame({'date':['2026-09-29','2026-09-28'],
                                     'dix':[.4692,44.5],'gex':[5e9,-1e9]}))
    assert df.dix.tolist()==[44.5,46.92]
    assert (df.dix >= 45).tolist()==[False,True]


def test_dix_html_not_data():
    with pytest.raises(ValueError):
        mr.normalize_dix(pd.DataFrame({'html':['error']}))


def test_technical_levels_exclude_unfinished_day():
    idx=pd.bdate_range(end='2026-09-30',periods=150)
    values=pd.Series(range(100,250),index=idx,dtype=float)
    expected=values.iloc[:-1].tail(20).mean()
    values.iloc[-1]=999999
    levels=mr.technical_levels(values,pd.Timestamp('2026-09-30 12:00',tz=mr.ET))
    assert levels['asof']=='2026-09-29'
    assert levels['trend_triggers']['21']==expected
    assert levels['support_high']<1000


def test_reports_expire_and_reject_futures(tmp_path):
    path=tmp_path/'levels.json'
    report=dict(instrument='NDX',basis='spot',verified=True,publisher='Test',
                source_url='https://example.com/report',asof='2026-09-25',expires='2026-10-02',
                levels=[{'label':'test','value':30000}])
    path.write_text(json.dumps(report))
    assert mr.verified_levels(path,'2026-09-30')
    assert mr.verified_levels(path,'2026-10-03') is None
    report['basis']='futures'
    path.write_text(json.dumps(report))
    assert mr.verified_levels(path,'2026-09-30') is None


def test_spx_report_cannot_leak_into_ndx_chart():
    path=Path(__file__).parents[1]/'spx_levels.json'
    report=mr.verified_levels(path,'2026-09-30',instrument='SPX')
    assert [x['value'] for x in report['levels']]==[7665,7444,6972]
    assert report['verification']=='public_secondary_report'
    assert mr.verified_levels(path,'2026-09-30',instrument='NDX') is None
    assert mr.verified_levels(path,'2026-10-07',instrument='SPX') is None


def test_persistence_survives_reopen_concurrent_and_deduplicates(tmp_path,monkeypatch):
    monkeypatch.setenv('SENTINEL_DATA_DIR',str(tmp_path))
    day=mr.market_session()['last_completed']
    scores=dict(weighted_risk=35.,weighted_opportunity=10.,macro_adjustment=5.,neutral_baseline=7.,net_risk=23.)
    def record(_):
        return mr.save_observation(scores,[],{'date':day},True,day,now=pd.Timestamp('2026-09-30T16:00:00Z'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record,range(8)))
    latest,rows=record(0)
    assert len(rows)==1
    assert len(mr.daily_records(rows))==1
    assert mr.history_frame(rows).net_risk.iloc[0]==23.


def test_incomplete_later_snapshot_does_not_replace_valid():
    base=dict(version=mr.VERSION,quality_ok=True,trade_date='2026-09-29',observed_at='2026-09-29T20:00:00Z')
    invalid=dict(base,quality_ok=False,observed_at='2026-09-29T21:00:00Z')
    assert mr.daily_records([base,invalid])==[base]
    assert mr.daily_records([dict(base,version='old')])==[]


def test_error_never_discloses_query_key():
    assert 'test-secret' not in mr.safe_error(ValueError('https://x/?apiKey=test-secret&x=1'))


def test_delayed_massive_indices_accepted_and_symbol_checked(monkeypatch):
    monkeypatch.setenv('MASSIVE_API_KEY','test-key')
    monkeypatch.setenv('ENABLE_PAID_VOLATILITY_SOURCE','true')
    monkeypatch.setattr(mr,'json_get',lambda *a,**kw: {'results':[
        {'ticker':'I:VIX','value':20,'last_updated':1790770216000000000,'timeframe':'DELAYED'},
        {'ticker':'I:FAKE','value':20,'last_updated':1790770216000000000,'timeframe':'REAL-TIME'}]})
    rows=mr.massive_quotes()
    assert len(rows)==1 and rows[0]['symbol']=='^VIX'
    assert rows[0]['source']=='Massive DELAYED'
