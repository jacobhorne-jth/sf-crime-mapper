// frontend/src/App.jsx
import { useEffect, useRef, useState, useMemo } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { getNeighborhoods, getPredictions, getSpike } from "./api";

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;

export default function App() {
  const mapRef = useRef(null);
  const mapEl = useRef(null);
  const idMapRef = useRef({});            // name -> feature index (for feature-state)
  const featureCountRef = useRef(0);      // number of features
  const predsByNameRef = useRef(new Map());
  const requestIdRef = useRef(0);         // ignore stale responses
  const modeRef = useRef("forecast");     // keep popup in sync with current mode

  // ---- UI state
  const [mode, setMode] = useState("forecast"); // "forecast" | "spike"
  const [date, setDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [crime, setCrime] = useState("all");
  const [tod, setTod] = useState("all");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [servedWeek, setServedWeek] = useState(null); // spike mode may clamp to next supported week

  // keep a live ref of mode for event handlers
  useEffect(() => { modeRef.current = mode; }, [mode]);

  // StrictMode/HMR-safe cleanup
  if (import.meta.hot) {
    import.meta.hot.dispose(() => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    });
  }

  // 1) Initialize base map
  useEffect(() => {
    if (mapRef.current) return;
    if (!mapboxgl.accessToken) {
      setErr("Mapbox token missing. Put VITE_MAPBOX_TOKEN in frontend/.env and restart Vite.");
      return;
    }
    if (!mapboxgl.supported()) {
      setErr("WebGL unsupported. Enable hardware acceleration in your browser.");
      return;
    }
    const m = new mapboxgl.Map({
      container: mapEl.current,
      style: "mapbox://styles/mapbox/light-v11",
      center: [-122.431297, 37.773972],
      zoom: 11,
    });
    m.addControl(new mapboxgl.NavigationControl(), "top-right");
    mapRef.current = m;

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null; // critical for StrictMode double-mount
      }
    };
  }, []);

  // 2) After map load, add neighborhoods, layers, and hover popup
  useEffect(() => {
    const m = mapRef.current;
    if (!m) return;

    async function onLoad() {
      try {
        const raw = await getNeighborhoods();

        // index by neighborhood/name property (must match backend predictions)
        const features = raw.features.map((f, idx) => {
          const name = f.properties?.neighborhood || f.properties?.name;
          const key = String(name ?? f.id ?? idx);
          idMapRef.current[key] = idx;
          return { ...f, id: idx };
        });
        featureCountRef.current = features.length;
        const gj = { ...raw, features };

        if (!m.getSource("neigh")) {
          m.addSource("neigh", { type: "geojson", data: gj });
          m.addLayer({
            id: "neigh-fill",
            type: "fill",
            source: "neigh",
            paint: {
              // If feature-state 'risk' is missing => gray; otherwise color by risk 0..100
              "fill-color": [
                "case",
                ["==", ["feature-state", "risk"], null],
                "#e5e7eb",
                [
                  "interpolate",
                  ["linear"],
                  ["feature-state", "risk"],
                  0,   "#e0f2fe", // low
                  50,  "#fb923c", // mid
                  100, "#7f1d1d"  // high
                ]
              ],
              "fill-opacity": 0.72,
            },
          });
          m.addLayer({
            id: "neigh-line",
            type: "line",
            source: "neigh",
            paint: { "line-color": "#111827", "line-width": 0.6, "line-opacity": 0.5 },
          });
        }

        // hover popup
        const popup = new mapboxgl.Popup({ closeButton: false, closeOnClick: false });
        m.on("mousemove", "neigh-fill", (e) => {
          const f = e.features?.[0];
          if (!f) return;
          const name = f.properties?.neighborhood || f.properties?.name;
          const state = m.getFeatureState({ source: "neigh", id: f.id }) || {};
          const pred = predsByNameRef.current.get(String(name));
          const risk = state.risk ?? 0;

          // show different details depending on mode
          const currentMode = modeRef.current;
          let html = `<div style="font-weight:600">${name ?? "?"}</div>
                      <div>Risk: ${Number(risk).toFixed(1)}</div>`;
          if (currentMode === "forecast") {
            const mean = pred ? pred.mean_incidents : 0;
            html += `<div>Mean incidents: ${Number(mean).toFixed(2)}</div>`;
          } else {
            const probPct = pred ? (Number(pred.prob) * 100).toFixed(1) : Number(risk).toFixed(1);
            html += `<div>Spike probability: ${probPct}%</div>`;
          }

          popup.setLngLat(e.lngLat).setHTML(html).addTo(m);
        });
        m.on("mouseleave", "neigh-fill", () => popup.remove());

        // initial paint
        paintChoropleth(mode, date, crime, tod);
      } catch (e) {
        console.error(e);
        setErr(e?.message || "Failed to load neighborhoods.");
      }
    }

    if (m.loaded()) onLoad();
    else m.once("load", onLoad);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 3) Fetch + paint when controls change
  useEffect(() => {
    paintChoropleth(mode, date, crime, tod);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, date, crime, tod]);

  async function paintChoropleth(currentMode, d, ct, time) {
    const m = mapRef.current;
    if (!m || !m.getSource("neigh")) return;
    setLoading(true);
    setErr("");
    setServedWeek(null);
    const rid = ++requestIdRef.current;

    try {
      if (currentMode === "spike") {
        const resp = await getSpike(d, ct, time);
        if (rid !== requestIdRef.current) return;

        // resp.predictions: [{ neighborhood_id, prob, risk }]
        const byName = new Map(resp.predictions.map((p) => [String(p.neighborhood_id), p]));
        predsByNameRef.current = byName;
        setServedWeek(resp.served_week || null);

        // reset risk
        for (let i = 0; i < featureCountRef.current; i++) {
          m.setFeatureState({ source: "neigh", id: i }, { risk: 0 });
        }
        // set spike risk (0..100)
        for (const [name, idx] of Object.entries(idMapRef.current)) {
          const p = byName.get(name);
          const risk = p ? Number(p.risk) : 0;
          m.setFeatureState({ source: "neigh", id: Number(idx) }, { risk: Number.isFinite(risk) ? risk : 0 });
        }
      } else {
        // forecast mode (Prophet)
        const data = await getPredictions(d, ct, time);
        if (rid !== requestIdRef.current) return;

        const byName = new Map(data.predictions.map((p) => [String(p.neighborhood_id), p]));
        predsByNameRef.current = byName;

        for (let i = 0; i < featureCountRef.current; i++) {
          m.setFeatureState({ source: "neigh", id: i }, { risk: 0 });
        }
        for (const [name, idx] of Object.entries(idMapRef.current)) {
          const p = byName.get(name);
          const risk = p ? Number(p.risk) : 0;
          m.setFeatureState({ source: "neigh", id: Number(idx) }, { risk: Number.isFinite(risk) ? risk : 0 });
        }
      }
    } catch (e) {
      console.error(e);
      setErr(e?.message || "Failed to load data.");
    } finally {
      if (rid === requestIdRef.current) setLoading(false);
    }
  }

  // Legend (works for both modes because both use risk 0..100)
  const Legend = useMemo(
    () => (
      <div
        style={{
          position: "absolute",
          bottom: 16,
          left: 16,
          background: "rgba(255,255,255,0.9)",
          padding: "8px 10px",
          borderRadius: 8,
          boxShadow: "0 1px 4px rgba(0,0,0,0.15)",
          fontSize: 12,
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 6 }}>Risk</div>
        <div
          style={{
            width: 180,
            height: 10,
            background: "linear-gradient(to right, #e0f2fe, #fb923c, #7f1d1d)",
            borderRadius: 4,
          }}
        />
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
          <span>0</span>
          <span>50</span>
          <span>100</span>
        </div>
      </div>
    ),
    []
  );

  return (
    <div style={{ height: "100vh", display: "grid", gridTemplateRows: "48px 1fr" }}>
      <div
        style={{
          padding: 8,
          borderBottom: "1px solid #eee",
          display: "flex",
          gap: 16,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <strong>SF Crime Mapper</strong>

        <label> Mode:{" "}
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="forecast">Forecast</option>
            <option value="spike">Spike Risk</option>
          </select>
        </label>

        <label> Date:{" "}
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
        </label>

        <label> Crime Type:{" "}
          <select value={crime} onChange={(e) => setCrime(e.target.value)}>
            <option value="all">All</option>
            <option value="violent">Violent</option>
            <option value="property">Property</option>
          </select>
        </label>

        <label> Time of Day:{" "}
          <select value={tod} onChange={(e) => setTod(e.target.value)}>
            <option value="all">All</option>
            <option value="day">Day</option>
            <option value="night">Night</option>
          </select>
        </label>

        {loading && <span style={{ color: "#6b7280" }}>loading…</span>}
        {err && <span style={{ color: "#b91c1c" }}>{err}</span>}
        {mode === "spike" && servedWeek && (
          <span style={{ color: "#6b7280" }}>
            (served week: {servedWeek})
          </span>
        )}
      </div>

      <div style={{ position: "relative" }}>
        <div ref={mapEl} style={{ width: "100%", height: "100%" }} />
        {Legend}
      </div>
    </div>
  );
}
