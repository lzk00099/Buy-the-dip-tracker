# 行情、实录与指数点位修复说明

本次修复对象是线上应用入口 `main2.py`。原 `main.py` 保留。

## 报价与刷新

- 页面每5分钟使用 Streamlit 原生 fragment 触发完整重跑，跨休市到开盘也会重新检查；手动刷新清除行情缓存。
- ETF：已有 Alpaca 配置优先；逐标的补用 Yahoo chart 与 Nasdaq 官方网站公开报价。独立的 GitHub 采集节点还可提供短时缓存。
- 指数：Yahoo chart；显式启用 Massive 后接受 REAL-TIME **或 DELAYED**，并保留标签。`^NDX`和`^GSPC`纳入指数快照。
- 每条数据按**原始行情时间**检查：超过35分钟、超过当前时刻1分钟、非当日、无效价格均不能称为盘中报价。缓存读出后也重新检查。
- 同一比率的两边及 CTA 的 QQQ/SPY/IWM，时间跨度超过15分钟时整组退回日线。RSP不再阻止CTA三个核心ETF更新。
- 每个标的显示价格、源和行情时间；页面刷新时间不作为行情时间。美国交易日、节假日和提前收市由交易所日历处理。
- 公开网站接口不提供可用性保证。若要持续稳定地读取所有指数，需要供应商提供对应指数权限；代码不能凭空获得授权。SqueezeMetrics DIX/GEX、CBOE COR1M/DSPX 与当前 OKX 日线模型不会因刷新按钮变成逐笔数据。

## 净风险准确性与留档

- 净风险 = 加权风险 − 加权机会 + 宏观修正 − 中性基线。保存每个开关的原始分数和实际权重，可重算核对。
- DIX的**整列**统一转为0–100百分比，卡片、图表、阈值共用同一单位；删除随机DIX/GEX兜底。
- 删除缺少历史OI/资金费率时用中性常数填补的回放曲线。旧回放还漏了MOVE宏观修正，与实时分类器并不完全相同，不能当作每天实录。
- COR1M/DSPX不再补造常数；其均线只使用各自发布的真实交易日。VIX相关比率不再前向填充不同日期的指数。
- 失效或落后输入会暂停有效综合信号，保存诊断，不把缺失当作中性。
- 图表只画该版本**真正采集过的有效观察**。每天取最后一次有效观察；这是混合频率模型的日内/盘后观察，不宣称是全部指标同一秒或官方最终收盘分数。采集日期不倒填为输入日期，避免将翌日输入写回昨日。
- SQLite用于当前实例并发安全保存；GitHub `market-data` 分支是持久记录。Streamlit休眠、重启或重新部署后会重新加载数据分支。
- 后台约每30分钟采集，并在夜间/次晨补采，不依赖网页是否打开。GitHub计划任务可能延迟；超过36小时没有更新时页面警示。GitHub可能停用长期无活动公共仓库的计划任务，应留意Actions通知。
- `observations/YYYY-MM-DD/<hash>.json`：逐次输入、指标末两行、源时间、开关分数、权重、公式组成、质量和模型版本；`daily_history.json`：每天最后有效观察；`cache/`：已取得的原始历史序列；`health.json`：最近采集状态。
- 失败观察不覆盖以前的有效日线。没有真实记录的历史日期保留空缺，不能从已经丢失的临时文件恢复出当时真实模型状态。
- 分数是启发式规则，不是下跌概率；本次核验计算与数据一致性，未证明买卖策略收益或预测有效性。

## SPX报道枢轴与NDX技术参考

默认图已按用户要求切换为SPX。2026-09-29 Tickmill / Patrick Munnelly的公开报道，归属Cullen Morgan / Goldman Sachs，列出短/中/长期CTA枢轴 **7,665 / 7,444 / 6,972**。已核对[报道全文](https://www.tickmill.com/blog/institutional-insights-goldman-sachs-sp500-positioning-key-levles-29926)，未独立取得高盛原始报告。这些是报道中的趋势枢轴，不是保证执行的抛售线。

`spx_levels.json`保存明确的二手来源属性、报道日期和来源链接。最多7天展示期是看板的陈旧保护规则，不是发布者承诺的有效期；2026-10-06之后隐藏，等待更新。公开报告不能保证每日免费发布，不能用技术估算冒充最新机构数据。

SPX/NDX可切换，分别使用各自现货行情；技术低位区也按所选指数每日重算，不能宣称有真实承接资金。移除无日期/无来源的28,500、26,500及27,200–28,000区域。新图每天以已完成NDX现货日线计算：

1. 21/63/126日趋势失守参考：分别取最近20/62/125日收盘均值。这是下一收盘价与对应移动均线相交的价格。
2. 20/63日收盘低位区：最近20日和63日最低收盘价之间的范围。

这些是透明的价格模型，**不是机构CTA仓位、真实抛售阈值或已观测承接资金**。

检索至2026-09-30：可找到9月25日BofA报告的公开摘要，但摘要没有可独立核验的NDX精确触发点；也不能将SPX触发位、NQ期货位或一个月前的评论当作NDX现货最新位。

`ndx_levels.json`只在有原始机构报告时填写：`instrument=NDX`、`basis=spot`、发布日期、到期日（最多7天）、机构、来源URL、levels和`verified=true`缺一不可。过期即隐藏。当前保持未验证，防止旧数字再次长期挂在图上。

## 已有账号配置

密钥只保存在Streamlit Secrets和GitHub Actions Secrets，不提交进仓库。

Streamlit可继续使用：

```toml
ALPACA_API_KEY = "填写你自己的key"
ALPACA_API_SECRET = "填写你自己的secret"
ALPACA_FEED = "iex"
# IEX是单交易所；sip/delayed_sip需以自己的账户权限为准。
ENABLE_PAID_VOLATILITY_SOURCE = "false"
# 有对应指数权限并希望启用时，将上行改为true，再配置MASSIVE_API_KEY。
```

后台也使用Alpaca/Massive时，在仓库Actions Secrets保存同名密钥；ALPACA_FEED、ENABLE_PAID_VOLATILITY_SOURCE及可选MASSIVE_VOL_TICKERS放Actions Variables。没有密钥仍尝试公开报价，缺源会明确标注。

第一次运行会在确认不存在时自动创建`market-data`分支。采集工作流只写该数据分支，不因每次采样触发Streamlit代码重新部署。仓库须允许GitHub Actions运行及工作流申请的contents写权限；无需给网页进程配置GitHub写入密钥。

## 验证与参考

本地执行 `python -m pytest -q`；`python collect.py`执行与页面相同的分数计算，并把结果写入SENTINEL_DATA_DIR指定目录。测试使用隔离的确定性样本；生产抓取失败不会调用测试样本。

- [Streamlit原生周期刷新](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment)
- [Alpaca市场数据及订阅限制](https://docs.alpaca.markets/us/docs/market-data-faq)
- [Massive指数快照的REAL-TIME/DELAYED和时间字段](https://massive.com/docs/rest/indices/snapshots/indices-snapshot)
- [GitHub工作流必须位于.github/workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows)
- [9月25日BofA报告公开摘要（非原始报告、非实时行情）](https://finvaulta.com/research/bank-of-america/systematic-flows-monitor-ctas-ride-the-equity-rally-2026-09-25)
