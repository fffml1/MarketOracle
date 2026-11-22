import { useCallback, useLayoutEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { MarkdownAnswer } from './MarkdownAnswer'
import './App.css'

type Evidence = { evidence_id: string; title: string; summary: string; source: string; published_at?: string; payload?: Record<string, unknown> }
type Speech = { round: number; speaker: string; text: string; state: string }
type ChartPoint = { date: string; close: number }
type Request = { action: 'ask' | 'report'; text: string }
type Turn = Request & { id: number; answer: string; failed?: boolean; fromComposer: boolean }
type Report = { headline_verdict: string; executive_summary: string; rating: string; confidence_score: number; confidence_rationale: string; debate_synthesis: { bull_strongest_argument: string; bear_strongest_argument: string; unresolved_conflicts: string[] }; action_plan: { entry_zone: string; stop_loss_rule: string }; key_risk_warnings: string[] }
type StreamEvent =
  | { type: 'debate'; kind: string; round_num: number; speaker: string; text: string }
  | { type: 'answer_delta'; text: string }
  | { type: 'answer_reset' }
  | { type: 'phase'; stage: 'collecting' }
  | { type: 'result'; answer: string; incomplete: boolean; evidence: Evidence[]; report: Report | null; entry: string | null }
  | { type: 'error'; message: string }

const sampleEvidence: Evidence[] = [
  { evidence_id: 'EV_TECH_001', title: '量价与均线结构', summary: '收盘 245.60 元，近 20 日上涨 6.32%，量比 1.23。该记录用于演示证据回溯。', source: '模拟行情', published_at: '2026-11-20' },
  { evidence_id: 'EV_NEWS_001', title: '储能业务进展', summary: '样例资讯提及储能订单，首次披露日期和订单金额仍待核实。', source: '模拟资讯', published_at: '2026-11-18' },
  { evidence_id: 'EV_RISK_001', title: '短线回撤风险', summary: '样例风险材料提示，放量上行后仍需观察承接，成交量本身无法确认资金方向。', source: '模拟风险记录', published_at: '2026-11-20' },
]
const sampleSpeeches: Speech[] = [
  { round: 1, speaker: 'Bull', state: 'end', text: '近 20 日量价表现偏强，收盘站上短期均线。若后续成交量维持，趋势有进一步确认的机会。[EV_TECH_001]' },
  { round: 1, speaker: 'Bear', state: 'end', text: '放量本身不能确认资金方向。储能订单的首次披露时间待核实，当前证据还不足以支持把它视作新增催化。[EV_NEWS_001]' },
]
const sampleReport: Report = {
  headline_verdict: '趋势改善，等待催化确认',
  executive_summary: '近期量价有所走强，持久性仍取决于后续承接。订单信息的首次披露日期是下一项值得核实的证据。',
  rating: '中性观望', confidence_score: 62, confidence_rationale: '量价依据完整，订单催化仍待核实。',
  debate_synthesis: { bull_strongest_argument: sampleSpeeches[0].text, bear_strongest_argument: sampleSpeeches[1].text, unresolved_conflicts: ['订单是否属于新增催化'] },
  action_plan: { entry_zone: '观察成交量能否持续', stop_loss_rule: '若量价结构转弱，重新评估入场假设' },
  key_risk_warnings: ['订单披露时点仍待核实', '成交量无法单独确认资金方向'],
}
const history = [224, 226, 223, 225, 227, 226, 231, 229, 227, 229, 232, 233, 231.5, 234, 238, 235, 239, 237, 236, 240, 238.8, 241, 239, 244, 242, 245.6]
  .map((close, index) => {
    const date = new Date(Date.UTC(2026, 9, 16))
    let days = index
    while (days > 0) {
      date.setUTCDate(date.getUTCDate() + 1)
      if (date.getUTCDay() !== 0 && date.getUTCDay() !== 6) days--
    }
    return { date: `${date.getUTCMonth() + 1}/${date.getUTCDate()}`, close }
  })
const presets = [
  { label: '分析近期走势', text: '最近走势和成交量怎么样？' },
  { label: '发起短线辩论', text: '我想做短线，先给我听听多空理由。' },
  { label: '补充风险上限', text: '我最多接受从拟入场价下跌 3%，其他条件不变。' },
]
const sampleSession = { symbol: '300750.SZ', asOf: '2026-11-20', mock: true }

function App() {
  const [view, setView] = useState<'workspace' | 'report'>('workspace')
  const [tab, setTab] = useState<'debate' | 'evidence'>('debate')
  const [mode, setMode] = useState('小白')
  const [settings, setSettings] = useState(false)
  const [session, setSession] = useState(sampleSession)
  const [draft, setDraft] = useState(sampleSession)
  const [example, setExample] = useState(true)
  const [period, setPeriod] = useState('近一个月')
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState<Turn[]>([])
  const [activeTurn, setActiveTurn] = useState<number | null>(null)
  const [progress, setProgress] = useState('')
  const [retryTurn, setRetryTurn] = useState<Turn | null>(null)
  const [speech, setSpeech] = useState(sampleSpeeches)
  const [evidence, setEvidence] = useState(sampleEvidence)
  const [report, setReport] = useState<Report | null>(sampleReport)
  const [entry, setEntry] = useState<string | null>(null)
  const [selected, setSelected] = useState<Evidence | null>(null)
  const [status, setStatus] = useState('已完成 1 轮 · 样例实录')
  const busy = activeTurn !== null
  const [error, setError] = useState('')
  const identity = useRef('')
  const input = useRef<HTMLTextAreaElement>(null)
  const feed = useRef<HTMLElement>(null)
  const transcript = useRef<HTMLDivElement>(null)
  const followTranscript = useRef(true)
  const inFlight = useRef(false)
  const turnSequence = useRef(0)
  const evidenceIds = useMemo(() => new Set(evidence.map(item => item.evidence_id)), [evidence])
  const openEvidence = useCallback((id: string) => setSelected(evidence.find(item => item.evidence_id === id) ?? null), [evidence])

  useLayoutEffect(() => {
    const box = transcript.current
    if (box && followTranscript.current) box.scrollTop = box.scrollHeight
  }, [messages, progress, error, view])

  useLayoutEffect(() => {
    const field = input.current
    if (!field) return
    field.style.height = 'auto'
    field.style.height = `${Math.min(field.scrollHeight, 144)}px`
  }, [question, view])

  function reset(next = session) {
    setSession(next)
    setEvidence([])
    setExample(false)
    setSpeech([])
    setReport(null)
    setEntry(null)
    setMessages([])
    setStatus('等待发起辩论')
    setError('')
    setRetryTurn(null)
    identity.current = ''
  }

  function applySession(event: FormEvent) {
    event.preventDefault()
    const symbol = draft.symbol.trim().toUpperCase()
    if (!/^\d{6}(\.(SH|SZ|BJ))?$/.test(symbol) || !draft.asOf) return
    reset({ ...draft, symbol })
    setQuestion('')
    setTab('debate')
    setSettings(false)
    setView('workspace')
  }

  async function submit(action: Request['action'], retry?: Turn, query?: string) {
    const text = retry?.text ?? query ?? (action === 'ask' ? question.trim() : '')
    if (inFlight.current || (action === 'ask' && !text)) return
    inFlight.current = true
    const fromComposer = retry?.fromComposer ?? (action === 'ask' && query === undefined)
    const turn = { id: retry?.id ?? ++turnSequence.current, action, text, answer: '', fromComposer }
    if (example) reset()
    followTranscript.current = true
    setActiveTurn(turn.id)
    setProgress('正在连接研究服务…')
    setError('')
    setRetryTurn(null)
    setMessages(previous => retry ? previous.map(item => item.id === turn.id ? turn : item) : [...previous, turn])
    if (fromComposer) setQuestion(current => current.trim() === text ? '' : current)
    if (action === 'ask') input.current?.focus()
    const updateAnswer = (answer: (previous: string) => string, failed = false) => {
      setMessages(previous => previous.map(item => item.id === turn.id ? { ...item, answer: answer(item.answer), failed } : item))
    }
    const fail = (message: string) => {
      setError(message)
      setRetryTurn(turn)
      if (fromComposer) setQuestion(current => current || text)
    }
    let sawDebate = false
    let gotResult = false
    const process = (event: StreamEvent) => {
      if (event.type === 'error') throw new Error(event.message)
      if (event.type === 'answer_delta') {
        setProgress('正在生成回答…')
        updateAnswer(previous => previous + event.text)
        return
      }
      if (event.type === 'answer_reset' || event.type === 'phase') {
        if (event.type === 'answer_reset') updateAnswer(() => '')
        setProgress('正在收集研究资料…')
        return
      }
      if (event.type === 'result') {
        gotResult = true
        setEvidence(event.evidence)
        setReport(event.report)
        setEntry(event.entry)
        updateAnswer(() => event.answer, event.incomplete)
        if (action === 'report' && event.report && !event.incomplete) setView('report')
        if (event.incomplete) fail('本次研究未完成，可重新发起。')
        return
      }
      if (!sawDebate) {
        sawDebate = true
        setSpeech([])
      }
      const round = event.round_num
      if (event.kind === 'start') {
        setSpeech(previous => [...previous, { round, speaker: event.speaker, state: 'start', text: '' }])
        setStatus(`第 ${round} 轮 · ${event.speaker === 'Bull' ? '多头' : '空头'}准备发言`)
        setProgress(`第 ${round} 轮辩论 · ${event.speaker === 'Bull' ? '多头' : '空头'}`)
      } else if (event.kind === 'continue' || event.kind === 'stop') {
        setStatus(event.kind === 'continue' ? `续辩 · ${event.text}` : `已完成 ${round} 轮 · ${event.text}`)
        if (event.kind === 'stop') setProgress(action === 'report' ? '正在整理研报…' : '正在汇总辩论结果…')
      } else {
        setSpeech(previous => previous.map((turn, index) => {
          const target = round === 0 ? index === previous.length - 1 && turn.state !== 'end' : turn.round === round && turn.speaker === event.speaker
          return target ? { ...turn, state: event.kind, text: event.kind === 'text' ? turn.text + event.text : turn.text } : turn
        }))
      }
    }
    try {
      const health = await fetch('/api/health')
      if (health.status !== 204) throw new Error('本地后端未连接。启动 backend.web 后重试。')
      identity.current ||= crypto.randomUUID()
      setProgress('正在收集研究资料…')
      const response = await fetch('/api/research', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...session, session: identity.current, mode, action, question: text }),
      })
      if (!response.ok || !response.body) throw new Error(`无法开始研究（HTTP ${response.status}）。请检查本地后端和模型配置。`)
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
      let buffer = ''
      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += value
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''
          lines.filter(Boolean).forEach(line => process(JSON.parse(line) as StreamEvent))
        }
        if (buffer.trim()) process(JSON.parse(buffer) as StreamEvent)
      } finally {
        await reader.cancel()
        reader.releaseLock()
      }
      if (!gotResult) throw new Error('响应已中断，已收到的内容保留在当前页面。')
    } catch (reason) {
      fail(reason instanceof Error ? reason.message : '本地请求失败，请检查后端连接。')
      updateAnswer(previous => previous, true)
      setSpeech(previous => previous.map(turn => turn.state === 'start' || turn.state === 'text' ? { ...turn, state: 'interrupted' } : turn))
    } finally {
      inFlight.current = false
      setActiveTurn(null)
      setProgress('')
    }
  }

  function showEvidence() {
    setTab('evidence')
    feed.current?.focus({ preventScroll: true })
    feed.current?.scrollIntoView({ block: window.matchMedia('(max-width: 680px)').matches ? 'start' : 'nearest' })
    if (feed.current) feed.current.scrollTop = 0
  }

  function withCitations(text: string) {
    return text.split(/(\[EV_[A-Z]+_\d+\])/g).map((part, index) => {
      const item = evidence.find(value => `[${value.evidence_id}]` === part)
      return item ? <button className="citation" key={index} onClick={() => setSelected(item)}>{item.evidence_id}</button> : part
    })
  }

  const market = [...evidence].reverse().find(item => item.evidence_id.startsWith('EV_TECH_'))?.payload
  const rawSeries = market?.close_series
  const series = example ? history : (Array.isArray(rawSeries) ? rawSeries as ChartPoint[] : [])
  const chartData = series.slice(period === '近两周' ? -10 : -20)
  const number = (key: string, suffix = '') => typeof market?.[key] === 'number' ? `${Number(market[key]).toFixed(2)}${suffix}` : '—'
  const trackedName = session.symbol === '300750.SZ' || session.symbol === '300750' ? '宁德时代' : session.symbol
  const openSettings = () => { setDraft(session); setSettings(true) }
  const renderError = () => error && <div className="error"><span role="alert">{error}</span>{retryTurn && <button type="button" className="text-button" onClick={() => void submit(retryTurn.action, retryTurn)}>重试</button>}</div>
  const sourceUrl = selected?.payload?.source_url
  const pages = selected?.payload?.pages
  const pageStart = selected?.payload?.page_start
  const pageEnd = selected?.payload?.page_end
  const pdfPages = Array.isArray(pages) ? pages.filter(page => Number.isInteger(page)).join('、')
    : typeof pageStart === 'number' ? typeof pageEnd === 'number' && pageEnd !== pageStart ? `${pageStart}–${pageEnd}` : String(pageStart) : ''
  const metrics = [
    ['近 20 日涨跌', example ? '+6.32%' : number('ret_20d_pct', '%'), '区间涨跌'],
    ['20 日量比', example ? '1.23×' : number('vol_ratio_20d', '×'), '当日成交量 / 20 日均量'],
    ['20 日均线', example ? '¥233.18' : number('ma20', ' 元'), 'MA20'],
    ['60 日最大回撤', example ? '−9.42%' : number('max_drawdown_60d_pct', '%'), '区间最大回撤'],
  ]

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={event => { event.preventDefault(); setView('workspace') }}><span className="brand-mark">M</span><span>MarketOracle<small>单股研究</small></span></a>
        <nav aria-label="工作区导航">
          <button className={view === 'workspace' ? 'nav-item active' : 'nav-item'} onClick={() => setView('workspace')}><span>◫</span>研究工作台</button>
          <button className={view === 'report' ? 'nav-item active' : 'nav-item'} onClick={() => setView('report')}><span>▤</span>研究报告</button>
          <button className="nav-item" onClick={showEvidence}><span>▧</span>证据库</button>
        </nav>
        <div className="sidebar-bottom">
          <span className="sidebar-label">当前标的</span>
          <div className="sidebar-stock"><span className="stock-icon">{trackedName.slice(0, 1)}</span><div><strong>{trackedName}</strong><small>{session.symbol}</small></div></div>
        </div>
      </aside>
      <div className="workspace">
        <main className="content">
          <div className="page-title"><h1>{view === 'workspace' ? '单股研究工作台' : '研究报告'}</h1><button className="control report-action" onClick={() => void submit('report')} disabled={busy}>生成完整研报</button></div>
          <div className="dashboard">
            <div className="main-column">
              {view === 'workspace' ? <>
                <section className="panel market">
                  <div className="market-header"><div className="stock-title"><div className="stock-icon">{trackedName.slice(0, 1)}</div><div><div className="stock-name"><h2>{trackedName}</h2><button type="button" className="swap-stock" aria-label="切换股票" title="切换股票" onClick={openSettings} disabled={busy}><svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M4 8h15m-4-4 4 4-4 4M20 16H5m4-4-4 4 4 4" /></svg></button></div><p>{session.symbol} <span>·</span> 基准日 {session.asOf}</p></div></div><span className="badge"><i />{example ? '示例数据' : session.mock ? '模拟行情' : '真实行情'}</span></div>
                  <div className="price-row"><div><small>最近收盘</small><div className="price">{example ? '¥245.60' : number('close', ' 元')}</div></div>{example && <span className="change">+2.46% <small>日涨跌</small></span>}</div>
                  {chartData.length > 0 ? <><div className="chart-heading"><span>收盘价走势</span><div className="segmented">{['近两周', '近一个月'].map(value => <button key={value} className={period === value ? 'chosen' : ''} onClick={() => setPeriod(value)}>{value}</button>)}</div></div>
                  <div className="chart" aria-label="收盘价走势图"><ResponsiveContainer width="100%" height={185} initialDimension={{ width: 720, height: 185 }}><AreaChart data={chartData} margin={{ top: 10, right: 1, bottom: 0, left: -14 }}>
                    <defs><linearGradient id="market-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#087f75" stopOpacity=".23" /><stop offset="100%" stopColor="#087f75" stopOpacity=".025" /></linearGradient></defs>
                    <CartesianGrid vertical={false} stroke="#e6eeed" strokeDasharray="3 4" />
                    <XAxis dataKey="date" tickFormatter={value => String(value).length === 10 ? String(value).slice(5).replace('-', '/') : String(value)} axisLine={false} tickLine={false} tickMargin={8} minTickGap={27} tick={{ fill: '#61727c', fontSize: 12 }} />
                    <YAxis domain={['dataMin - 3', 'dataMax + 2']} tickCount={5} axisLine={false} tickLine={false} tick={{ fill: '#61727c', fontSize: 12 }} />
                    <Tooltip formatter={value => [`¥${value}`, '收盘价']} />
                    <Area type="linear" dataKey="close" stroke="#087f75" fill="url(#market-fill)" strokeWidth={2.5} isAnimationActive={false} />
                  </AreaChart></ResponsiveContainer></div></> : <div className="market-note">{market ? `数据截至 ${String(market.latest_trade_date ?? session.asOf)}` : <><span>当前会话还没有行情。</span><button className="control report-action" disabled={busy} onClick={() => void submit('ask', undefined, presets[0].text)}>查询行情</button></>}</div>}
                  <div className="metrics">{metrics.map(([label, value, detail]) => <div key={label}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>)}</div>
                </section>
                <section className="panel chat">
                  <div className="section-heading"><span className="section-icon">✦</span><h2>研究问答</h2>{tab !== 'evidence' && evidence.length > 0 && <button className="text-button chat-evidence" onClick={showEvidence}>查看证据</button>}</div>
                  {!!messages.length && <div className="messages" ref={transcript} aria-label="对话记录" onScroll={event => { const box = event.currentTarget; followTranscript.current = box.scrollHeight - box.clientHeight - box.scrollTop < 36 }}>
                    {messages.map(turn => <article className="turn" key={turn.id}>
                      <div className="user-message">{turn.action === 'report' ? '生成完整研报' : turn.text}</div>
                      {(turn.answer || turn.failed || turn.id === activeTurn) && <div className="assistant-message">
                        <span className="assistant-mark" aria-hidden="true">✦</span>
                        <div className="assistant-content"><strong className="assistant-name">MarketOracle</strong>
                          {turn.answer && <MarkdownAnswer text={turn.answer} evidenceIds={evidenceIds} onEvidence={openEvidence} />}
                          {turn.failed && <small className="muted">{turn.answer ? '回答未完成' : '未收到回答'}</small>}
                          {turn.id === activeTurn && <p className="chat-progress" role="status"><span />{progress}</p>}
                        </div>
                      </div>}
                    </article>)}
                  </div>}
                  {entry && <details className="entry"><summary>最近一次短线评估</summary><MarkdownAnswer text={entry} evidenceIds={evidenceIds} onEvidence={openEvidence} /></details>}
                  {renderError()}
                  {!messages.length && <div className="presets">{presets.map(preset => <button key={preset.label} onClick={() => { setQuestion(preset.text); input.current?.focus() }} disabled={busy || !!question.trim()}>{preset.label}</button>)}</div>}
                  <form className="compose" onSubmit={event => { event.preventDefault(); void submit('ask') }}>
                    <textarea ref={input} rows={1} aria-label="输入研究问题" value={question} maxLength={2000} onChange={event => setQuestion(event.target.value)} placeholder="问问这只股票的走势、风险或研报依据…" onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229 && window.matchMedia('(pointer: fine)').matches) { event.preventDefault(); void submit('ask') } }} />
                    <div className="compose-actions"><label className="mode"><select aria-label="回答模式" value={mode} onChange={event => setMode(event.target.value)} disabled={busy}><option>小白</option><option>专业</option></select></label><button type="submit" className="primary send-button" aria-label="发送问题" title="发送问题" disabled={busy || !question.trim()}><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 19V5m-6 6 6-6 6 6" /></svg></button></div>
                  </form>
                </section>
              </> : <section className="panel report">
                <div className="section-heading"><span className="section-icon">▤</span><div><h2>{trackedName} · 完整研报</h2><p>{session.asOf}{example ? ' · 示例研报' : ''}</p></div></div>
                {busy && <p className="chat-progress" role="status"><span />{progress}</p>}
                {renderError()}
                {report ? <><div className="report-lead"><div><span className="badge">{report.rating}</span><h3>{report.headline_verdict}</h3></div><span className="score"><strong>{report.confidence_score}</strong> / 100</span></div>
                  <p>{withCitations(report.executive_summary)}</p><p className="muted">{report.confidence_rationale}</p>
                  <h3>多空分歧</h3><p>{withCitations(report.debate_synthesis.bull_strongest_argument)}</p><p>{withCitations(report.debate_synthesis.bear_strongest_argument)}</p>
                  <h3>风险与重评条件</h3><p>{withCitations(report.action_plan.entry_zone)}</p><p>{withCitations(report.action_plan.stop_loss_rule)}</p>
                  <ul>{[...report.key_risk_warnings, ...report.debate_synthesis.unresolved_conflicts].map((risk, index) => <li key={index}>{withCitations(risk)}</li>)}</ul>
                </> : <p className="empty">当前会话还没有完整研报。</p>}
              </section>}
            </div>
            <aside className="panel research-feed" ref={feed} tabIndex={-1} aria-label="研究现场">
              <div className="section-heading feed-heading"><span className="section-icon">◈</span><div><h2>研究现场</h2><p>查看交锋原话和引用证据</p></div></div>
              <div className="feed-tabs" aria-label="研究现场视图"><button className={tab === 'debate' ? 'selected' : ''} onClick={() => setTab('debate')}>多空辩论</button><button className={tab === 'evidence' ? 'selected' : ''} onClick={() => setTab('evidence')}>证据库 <span>{evidence.length}</span></button></div>
              {tab === 'debate' ? <div className="feed-body"><div className="feed-status" role="status"><i className={busy ? 'running' : ''} />{status}</div>
                {speech.length ? speech.map((turn, index) => <div key={`${turn.round}-${turn.speaker}-${index}`}>
                  {(index === 0 || speech[index - 1].round !== turn.round) && <div className="round-label">第 {turn.round} 轮</div>}
                  <article className={turn.speaker === 'Bull' ? 'speech bull' : 'speech bear'}><div className="speaker"><span>{turn.speaker === 'Bull' ? '多' : '空'}</span><strong>{turn.speaker === 'Bull' ? '多头主理人' : '空头质询官'}</strong>{turn.state === 'interrupted' && <small>未完成</small>}</div>
                    <p>{turn.text ? withCitations(turn.text) : '正在准备发言…'}</p></article>
                </div>) : <p className="empty">{busy ? '正在收集研究材料…' : '发起短线评估或完整研报后，辩手的公开发言会在这里逐段出现。'}</p>}
                {example && <button className="text-button" onClick={() => setView('report')}>查看示例研报</button>}
              </div> : <div className="feed-body">{evidence.length ? evidence.map(item => <button className="evidence-row" key={item.evidence_id} onClick={() => setSelected(item)}><span>{item.evidence_id}</span><strong>{item.title}</strong><small>{item.source}{item.published_at && ` · ${item.published_at}`}</small></button>) : <p className="empty">当前会话尚未取得证据。</p>}</div>}
            </aside>
          </div>
        </main>
      </div>
      {settings && <div className="overlay" onClick={() => setSettings(false)}><form className="panel dialog" role="dialog" aria-modal="true" aria-label="切换研究标的" onClick={event => event.stopPropagation()} onSubmit={applySession}><h2>新建研究会话</h2><label>股票代码<input autoFocus required pattern="[0-9]{6}(\.[sS][hHzZ]|\.[bB][jJ])?" value={draft.symbol} onChange={event => setDraft({ ...draft, symbol: event.target.value.trim().toUpperCase() })} placeholder="300750.SZ" /></label><label>研究基准日<input type="date" required value={draft.asOf} onChange={event => setDraft({ ...draft, asOf: event.target.value })} /></label><label className="checkbox"><input type="checkbox" checked={draft.mock} onChange={event => setDraft({ ...draft, mock: event.target.checked })} />使用模拟行情和资讯</label><div className="dialog-actions"><button type="button" className="control" onClick={() => setSettings(false)}>取消</button><button className="primary">开始会话</button></div></form></div>}
      {selected && <div className="overlay" onClick={() => setSelected(null)}><section className="panel dialog" role="dialog" aria-modal="true" aria-label="证据详情" onClick={event => event.stopPropagation()}><span className="badge">{selected.evidence_id}</span><h2>{selected.title}</h2><p>{selected.summary}</p><small>{selected.source}{selected.published_at && ` · ${selected.published_at}`}{pdfPages && ` · PDF 物理页 ${pdfPages}`}</small><div className="dialog-actions">{typeof sourceUrl === 'string' && /^https?:\/\//i.test(sourceUrl) && <a className="control source-link" href={sourceUrl} target="_blank" rel="noopener noreferrer">打开原文 PDF ↗</a>}<button autoFocus className="control" onClick={() => setSelected(null)}>关闭</button></div></section></div>}
    </div>
  )
}

export default App
