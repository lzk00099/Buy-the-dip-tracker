"""End-to-end UI and collector tests; deterministic fixtures, never live fallback data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest
import yfinance

import market_runtime as mr


def fixtures(monkeypatch,tmp_path,missing_dix=False):
    st.cache_data.clear()
    monkeypatch.setenv('SENTINEL_DATA_DIR',str(tmp_path))
    # A known regular trading day makes these tests independent of clock/weekend.
    real_session=mr.market_session
    monkeypatch.setattr(mr,'market_session',lambda now_et=None:real_session(pd.Timestamp('2026-09-30 12:00',tz=mr.ET)))
    save=mr.save_observation
    monkeypatch.setattr(mr,'save_observation',lambda *a,**kw:save(*a,**kw,now=pd.Timestamp('2026-09-30T16:00:00Z')))
    monkeypatch.setattr(mr,'remote_json',lambda name:None)
    monkeypatch.setattr(mr,'remote_frame',lambda name:pd.DataFrame())
    dates=pd.bdate_range(end='2026-09-29',periods=300)
    symbols=mr.ETF+['^NDX','^GSPC','^MOVE']
    market=pd.DataFrame({('Close',s):np.arange(300)*.2+100+i*15 for i,s in enumerate(symbols)},index=dates)
    monkeypatch.setattr(yfinance,'download',lambda *args,**kw:market)
    monkeypatch.setattr(mr,'fetch_quotes',lambda *args:pd.DataFrame(columns=['price','timestamp','source']))
    class Response:
        status_code=200
        def __init__(self,text='',payload=None):self.text,self.payload=text,payload
        def raise_for_status(self):pass
        def json(self):return self.payload
    def get(url,params=None,**kwargs):
        if 'squeezemetrics' in url:
            if missing_dix: raise requests.ConnectionError('fixture failure')
            return Response(pd.DataFrame({'date':dates,'dix':.46,'gex':5e9}).to_csv(index=False))
        if 'cboe.com' in url:
            symbol=url.split('/')[-1].split('_')[0]
            offset={'VIX':18,'VIX3M':22,'VXN':24,'VVIX':85,'DSPX':40,'COR1M':25}[symbol]
            return Response(pd.DataFrame({'DATE':dates,'CLOSE':offset+np.sin(np.arange(300)/13)}).to_csv(index=False))
        days=pd.date_range(end='2026-09-29',periods=60,tz='UTC')[::-1]
        millis=[int(x.timestamp()*1000) for x in days]
        if 'history-candles' in url:
            rows=[[str(t),'1','1','1',str(80000+i*10),'1','1','1','1'] for i,t in enumerate(millis)]
        elif 'open-interest-volume' in url:
            rows=[[str(t),str(3e9+i*1e6),'1'] for i,t in enumerate(millis)]
        elif 'funding-rate-history' in url:
            rows=[{'fundingTime':str(t),'fundingRate':'.0001'} for t in millis]
        else:
            raise AssertionError('Unexpected fixture URL: '+url)
        return Response(payload={'code':'0','data':rows})
    monkeypatch.setattr(requests,'get',get)


def test_full_page_and_saved_score_use_same_math(monkeypatch,tmp_path):
    fixtures(monkeypatch,tmp_path)
    app=AppTest.from_file(str(Path(__file__).parents[1]/'main2.py')).run(timeout=30)
    assert not app.exception, [x.message for x in app.exception]
    record=app.session_state['latest_observation']
    assert record['quality_ok']
    components=record['components']
    weight=sum(x['effective_weight'] for x in components)
    risk=sum(x['risk_score']*x['effective_weight'] for x in components)/weight
    opportunity=sum(x['opportunity_score']*x['effective_weight'] for x in components)/weight
    scores=record['scores']
    assert scores['net_risk']==pytest.approx(risk-opportunity+scores['macro_adjustment']-scores['neutral_baseline'])
    assert record['inputs']['dix']['dix']==46
    assert record['inputs']['dix']['df']['rows'][-1]['dix']==46
    assert app.get('plotly_chart')


def test_collector_preserves_failures_without_fake_score(monkeypatch,tmp_path):
    fixtures(monkeypatch,tmp_path,missing_dix=True)
    monkeypatch.setenv('SENTINEL_COLLECTOR','1')
    app=AppTest.from_file(str(Path(__file__).parents[1]/'main2.py')).run(timeout=30)
    assert not app.exception,[x.message for x in app.exception]
    record=app.session_state['latest_observation']
    assert not record['quality_ok']
    assert record['inputs']['dix']['error']
    assert not record['inputs']['dix']['is_mock']
    assert json.loads((tmp_path/'daily_history.json').read_text())['records']==[]
    assert list((tmp_path/'observations').rglob('*.json'))
