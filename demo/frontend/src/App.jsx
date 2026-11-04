import { useCallback, useEffect, useRef, useState } from 'react'

const STEP_ORDER = ['original', 'preprocessed', 'leaf mask', 'lesion candidates', 'cleaned lesions', 'k-means clusters', 'overlay']
const SEVERITY_CLASS = { Healthy: 'sev-healthy', Mild: 'sev-mild', Moderate: 'sev-moderate', Severe: 'sev-severe' }

const pretty = (label) => {
  if (!label) return '—'
  const [plant, disease] = label.split('___')
  return `${plant.replaceAll('_', ' ')} · ${(disease || '').replaceAll('_', ' ')}`
}
const pct = (x) => (x == null ? '—' : `${(x * 100).toFixed(1)}%`)
const num = (x, d = 3) => (x == null ? 'n/a' : x.toFixed(d))

function SevPill({ label }) {
  return <span className={`sev small ${SEVERITY_CLASS[label] || ''}`}>{label}</span>
}

function Diagnosis({ c }) {
  return (
    <div className="card">
      <h2>Diagnosis (CNN)</h2>
      {!c.available && <p className="warn">No classifier loaded — generic profile used.</p>}
      {c.available && (
        <>
          <div className="big">{c.uncertain ? 'Uncertain disease' : pretty(c.label)}</div>
          <div className="muted">confidence {pct(c.confidence)} · plant: {c.plant} ({pct(c.plant_confidence)})</div>
          {c.uncertain && <p className="warn">Confidence below {pct(c.min_confidence)}: treat the disease name as a guess (top-3 below).</p>}
          <ol className="top3">
            {c.top3.map((t) => (
              <li key={t.label}>
                <span>{pretty(t.label)}</span>
                <span className="bar"><span style={{ width: pct(t.confidence) }} /></span>
                <span className="num">{pct(t.confidence)}</span>
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  )
}

function Severity({ s }) {
  return (
    <div className="card">
      <h2>Severity (image processing)</h2>
      <div className={`sev ${SEVERITY_CLASS[s.severity]}`}>{s.severity}</div>
      <div className="stats">
        <div><span className="num">{s.infected_pct.toFixed(1)}%</span><span>measured lesion area</span></div>
        <div><span className="num">{s.n_spots}</span><span>spots</span></div>
        <div><span className="num">{Math.round(s.mean_spot_area_px)}</span><span>mean spot px</span></div>
      </div>
      {s.estimated_affected_pct != null && (
        <p className="muted">
          Learned estimate of expert-annotated affected area: <b>{s.estimated_affected_pct.toFixed(1)}%</b> ({s.estimated_severity}),
          from image-processing features (README §10.6)
        </p>
      )}
      <div className="muted">symptom group <b>{s.group}</b> · leaf: {s.leaf_method} · lesions: {s.lesion_method}</div>
      {s.non_leaf_warning && <p className="warn">This disease mainly affects fruit / heads / stems: leaf-area severity is approximate.</p>}
    </div>
  )
}

/* Side-by-side: standard colour thresholding vs our approach (+ optional expert ground truth) */
function Comparison({ result }) {
  const std = result.standard
  const ours = result.severity
  const exp = result.expert
  const cols = [
    {
      key: 'std', title: 'Standard colour thresholding',
      sub: 'ExG + Otsu leaf · one fixed HSV "not green" rule',
      img: std.images.overlay, pctv: std.infected_pct, sev: std.severity, spots: std.n_spots, scores: std.scores,
    },
    {
      key: 'ours', title: 'Our approach',
      sub: `ExG+ leaf · ${ours.lesion_method} (chosen by CNN: ${ours.group})`,
      img: result.images.overlay, pctv: ours.infected_pct, sev: ours.severity, spots: ours.n_spots, scores: ours.scores,
      extra: ours.estimated_affected_pct,
    },
  ]
  if (exp) cols.push({ key: 'gt', title: 'Expert ground truth', sub: exp.healthy ? 'healthy leaf: no lesions' : 'PlantSeg annotation', img: exp.overlay, pctv: exp.infected_pct, sev: exp.severity })

  return (
    <section className="compare">
      <h2 className="section">{exp ? 'Standard vs ours vs expert' : 'Standard vs ours'}</h2>
      <div className={`cmpgrid n${cols.length}`}>
        {cols.map((c) => (
          <figure key={c.key} className={c.key === 'ours' ? 'ours' : ''}>
            <figcaption className="cmphead"><b>{c.title}</b><span className="muted">{c.sub}</span></figcaption>
            <img src={c.img} alt={c.title} />
            <div className="cmpstats">
              <div><span className="num">{c.pctv.toFixed(1)}%</span><span>infected</span></div>
              <div><SevPill label={c.sev} /><span>grade</span></div>
              {c.spots != null && <div><span className="num">{c.spots}</span><span>spots</span></div>}
              {exp && c.scores && <div><span className="num">{num(c.scores.dice)}</span><span>Dice vs expert</span></div>}
            </div>
            {c.extra != null && <div className="muted pad">learned estimate: {c.extra.toFixed(1)}%</div>}
          </figure>
        ))}
      </div>
      <figure className="diff">
        <img src={result.comparison.diff} alt="pixel difference" />
        <figcaption>
          Pixel difference: <span className="sw blue" /> only ours · <span className="sw red" /> only standard ·{' '}
          <span className="sw yellow" /> both{exp ? <> · white outline = expert lesions</> : null}
        </figcaption>
      </figure>
      {!exp && (
        <p className="muted">
          Uploaded photos have no ground truth, so this shows how the methods differ, not which is right. On 1,554 unseen
          test images the standard leaf mask kept 26 % of expert lesion pixels (ours 94 %), and a single generic detector gave
          20.8 % false-positive area on healthy leaves (ours 4.3 %). Use the example gallery for a comparison against experts.
        </p>
      )}
    </section>
  )
}

function Steps({ images }) {
  const [show, setShow] = useState(false)
  return (
    <section className="steps">
      <button className="btn secondary" onClick={() => setShow((v) => !v)}>{show ? 'Hide' : 'Show'} intermediate image-processing steps</button>
      {show && (
        <div className="strip">
          {STEP_ORDER.filter((k) => images[k]).map((k, i) => (
            <figure key={k}><img src={images[k]} alt={k} /><figcaption>{i + 1}. {k}</figcaption></figure>
          ))}
        </div>
      )}
    </section>
  )
}

function Result({ result, preview }) {
  return (
    <>
      <section className="cards">
        <Diagnosis c={result.classifier} />
        <Severity s={result.severity} />
      </section>
      <section className="pair">
        {preview && (<figure><img src={preview} alt="uploaded leaf" /><figcaption>Original</figcaption></figure>)}
        <figure><img src={result.images.overlay} alt="lesion overlay" /><figcaption>Our lesions (red fill / yellow edge), leaf boundary in green</figcaption></figure>
        {result.images.gradcam && (<figure><img src={result.images.gradcam} alt="Grad-CAM" /><figcaption>Grad-CAM: where the CNN looked</figcaption></figure>)}
      </section>
      <Comparison result={result} />
      <Steps images={result.images} />
      <footer className="muted">Processed in {result.timing_s}s on CPU</footer>
    </>
  )
}

function Gallery({ onPick, selected, onAnalyse, loading }) {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)
  useEffect(() => {
    fetch('/api/examples').then((r) => (r.ok ? r.json() : Promise.reject(r.status))).then(setData).catch(() => setErr('Examples not available (run python -m src.data.make_demo_examples).'))
  }, [])
  if (err) return <p className="warn">{err}</p>
  if (!data) return <p className="muted">Loading examples…</p>
  const sel = data.examples.find((e) => e.id === selected)
  return (
    <>
      <div className="galleryhead">
        <p className="muted">{data.note} Select an image, then press <b>Analyse</b> to compare the standard pipeline, ours and the expert annotation.</p>
        {sel && (
          <div className="selbox">
            <span className="muted">Selected: <b>{pretty(sel.label)}</b></span>
            <button className="btn" disabled={loading} onClick={onAnalyse}>{loading ? 'Analysing…' : 'Analyse'}</button>
          </div>
        )}
      </div>
      <div className="gallery">
        {data.examples.map((e) => (
          <button key={e.id} className={`thumb ${selected === e.id ? 'sel' : ''}`} onClick={() => onPick(e.id)}>
            <img src={e.thumb} alt={e.label} />
            <span>{pretty(e.label)}</span>
          </button>
        ))}
      </div>
    </>
  )
}

export default function App() {
  const [tab, setTab] = useState('upload')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [exId, setExId] = useState(null)
  const resultsRef = useRef(null)

  // when a result (or the loading state) appears, scroll the page down to it
  useEffect(() => {
    if ((result || loading) && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [result, loading])

  const pick = useCallback((f) => {
    if (!f) return
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError(null)
  }, [])

  const run = async (req) => {
    setLoading(true); setError(null); setResult(null)
    try {
      const r = await req()
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`)
      setResult(await r.json())
    } catch (e) {
      setError(`Analysis failed: ${e.message}. Is the backend running on port 8000?`)
    } finally {
      setLoading(false)
    }
  }

  const analyze = () => {
    const body = new FormData(); body.append('file', file)
    run(() => fetch('/api/analyze', { method: 'POST', body }))
  }
  const selectExample = (id) => { setExId(id); setResult(null); setError(null) }      // select only, no run
  const compareExample = () => { if (exId != null) run(() => fetch(`/api/examples/${exId}/compare`)) }
  const switchTab = (t) => { setTab(t); setResult(null); setError(null) }

  return (
    <div className="page">
      <header>
        <h1>Leaf Disease Spot Detection &amp; Severity</h1>
        <p className="sub">CNN tells <b>which</b> disease · classical image processing measures <b>how bad</b></p>
        <nav className="tabs">
          <button className={tab === 'upload' ? 'on' : ''} onClick={() => switchTab('upload')}>Analyse a photo</button>
          <button className={tab === 'examples' ? 'on' : ''} onClick={() => switchTab('examples')}>Example gallery (vs expert)</button>
        </nav>
      </header>

      {tab === 'upload' && (
        <section
          className={`drop ${dragging ? 'dragging' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0]) }}
        >
          <input id="file" type="file" accept="image/*" onChange={(e) => pick(e.target.files[0])} hidden />
          <label htmlFor="file" className="btn secondary">Choose a leaf photo</label>
          <span className="hint">{file ? file.name : 'or drag & drop an image here'}</span>
          <button className="btn" disabled={!file || loading} onClick={analyze}>{loading ? 'Analysing…' : 'Analyse'}</button>
        </section>
      )}
      {tab === 'examples' && <Gallery onPick={selectExample} selected={exId} onAnalyse={compareExample} loading={loading} />}

      <div ref={resultsRef} className="results-anchor">
        {loading && <div className="loading">Running our system and the standard pipeline… (~3 s on CPU)</div>}
        {error && <div className="error">{error}</div>}
        {result && <Result result={result} preview={tab === 'upload' ? preview : null} />}
      </div>
    </div>
  )
}
