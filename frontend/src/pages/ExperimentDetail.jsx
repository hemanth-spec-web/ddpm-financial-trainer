import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { getExperiment } from "../api/experiments";

export default function ExperimentDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [exp, setExp] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getExperiment(id).then((res) => {
      setExp(res.data);
      setLoading(false);
    });
  }, [id]);

  if (loading) return <div style={styles.page}><p style={styles.muted}>Loading...</p></div>;
  if (!exp) return <div style={styles.page}><p style={styles.muted}>Not found</p></div>;

  const curves = exp.metrics?.schedule_summary?.curves;
  const checkpointTable = exp.metrics?.schedule_summary?.checkpoint_table || [];
  const tests = exp.metrics?.unit_tests?.tests || [];
  const allPassed = exp.metrics?.unit_tests?.all_passed;

  // Build chart data for schedule curves
  const scheduleData = curves
    ? curves.timesteps.map((t, i) => ({
        t,
        beta: curves.betas[i],
        alpha_bar: curves.alpha_bars[i],
        sqrt_alpha_bar: curves.sqrt_alpha_bars[i],
        sqrt_one_minus_alpha_bar: curves.sqrt_one_minus_alpha_bars[i],
      }))
    : [];

  // Build chart data for forward process snapshots
  const snapshots = exp.generated_samples?.snapshots || [];
  const cleanSignal = exp.generated_samples?.clean_signal || [];

  return (
    <div style={styles.page}>
      <button style={styles.backBtn} onClick={() => navigate("/dashboard")}>
        ← Back to Dashboard
      </button>

      <div style={styles.header}>
        <h1 style={styles.title}>{exp.name}</h1>
        <p style={styles.subtitle}>{exp.description}</p>
      </div>

      <div style={styles.metaRow}>
        <div style={styles.metaItem}><span style={styles.metaLabel}>T</span><span style={styles.metaValue}>{exp.T}</span></div>
        <div style={styles.metaItem}><span style={styles.metaLabel}>β start</span><span style={styles.metaValue}>{exp.beta_start}</span></div>
        <div style={styles.metaItem}><span style={styles.metaLabel}>β end</span><span style={styles.metaValue}>{exp.beta_end}</span></div>
        <div style={styles.metaItem}><span style={styles.metaLabel}>Seq Length</span><span style={styles.metaValue}>{exp.sequence_length}</span></div>
        <div style={styles.metaItem}>
          <span style={styles.metaLabel}>Unit Tests</span>
          <span style={{ ...styles.metaValue, color: allPassed ? "#00e5a0" : "#ef4444" }}>
            {allPassed ? "✓ All Passed" : "✗ Failed"}
          </span>
        </div>
      </div>

      {/* Unit test results */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>Unit Test Results</h2>
        <div style={styles.testGrid}>
          {tests.map((t, i) => (
            <div key={i} style={styles.testCard}>
              <div style={styles.testHeader}>
                <span style={{ color: t.passed ? "#00e5a0" : "#ef4444" }}>
                  {t.passed ? "✓" : "✗"}
                </span>
                <span style={styles.testName}>{t.name}</span>
              </div>
              <p style={styles.testDetail}>{t.detail}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Noise schedule chart */}
      {scheduleData.length > 0 && (
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Signal vs Noise Coefficients</h2>
          <p style={styles.sectionDesc}>
            √ᾱ_t (signal, blue) decays while √(1-ᾱ_t) (noise, red) grows — this is what the neural network learns to reverse.
          </p>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={scheduleData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e222b" />
              <XAxis dataKey="t" stroke="#5a6478" tick={{ fontSize: 11 }} />
              <YAxis stroke="#5a6478" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0c0f16", border: "1px solid #1f2535" }} />
              <Legend />
              <Line type="monotone" dataKey="sqrt_alpha_bar" stroke="#5b9cf6" name="√ᾱ_t (signal)" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="sqrt_one_minus_alpha_bar" stroke="#ef4444" name="√(1-ᾱ_t) (noise)" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Beta schedule chart */}
      {scheduleData.length > 0 && (
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Beta Schedule (β_t)</h2>
          <p style={styles.sectionDesc}>Noise added at each individual timestep — linearly increasing.</p>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={scheduleData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e222b" />
              <XAxis dataKey="t" stroke="#5a6478" tick={{ fontSize: 11 }} />
              <YAxis stroke="#5a6478" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0c0f16", border: "1px solid #1f2535" }} />
              <Line type="monotone" dataKey="beta" stroke="#e8b44b" name="β_t" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Forward process snapshots */}
      {snapshots.length > 0 && (
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Forward Process — Signal Dissolving Into Noise</h2>
          <p style={styles.sectionDesc}>
            The same clean signal at 5 different noise levels — this is x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε applied at increasing t.
          </p>
          {snapshots.map((snap, i) => {
            const chartData = snap.values.map((v, idx) => ({ idx, value: v }));
            return (
              <div key={i} style={{ marginBottom: "16px" }}>
                <p style={styles.snapLabel}>
                  t = {snap.t} &nbsp;|&nbsp; SNR = {snap.snr.toFixed(4)}
                </p>
                <ResponsiveContainer width="100%" height={100}>
                  <LineChart data={chartData}>
                    <XAxis dataKey="idx" hide />
                    <YAxis domain={[-4, 4]} hide />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke={snap.t === 0 ? "#00e5a0" : "#5b9cf6"}
                      dot={false}
                      strokeWidth={1.2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#06080d",
    padding: "32px",
    fontFamily: "'Inter', sans-serif",
    maxWidth: "1000px",
    margin: "0 auto",
  },
  backBtn: {
    background: "transparent",
    border: "none",
    color: "#5b9cf6",
    fontSize: "13px",
    cursor: "pointer",
    marginBottom: "20px",
    padding: 0,
  },
  header: { marginBottom: "24px" },
  title: { fontSize: "22px", fontWeight: 700, color: "#dde3ef" },
  subtitle: { fontSize: "13px", color: "#8893a8", marginTop: "4px" },
  metaRow: {
    display: "flex",
    gap: "24px",
    padding: "16px 0",
    borderTop: "1px solid #171c27",
    borderBottom: "1px solid #171c27",
    marginBottom: "28px",
    flexWrap: "wrap",
  },
  metaItem: { display: "flex", flexDirection: "column", gap: "4px" },
  metaLabel: { fontSize: "10px", color: "#5a6478", textTransform: "uppercase", letterSpacing: "0.05em" },
  metaValue: { fontSize: "14px", color: "#dde3ef", fontFamily: "monospace" },
  section: { marginBottom: "36px" },
  sectionTitle: { fontSize: "15px", fontWeight: 600, color: "#dde3ef", marginBottom: "6px" },
  sectionDesc: { fontSize: "12.5px", color: "#8893a8", marginBottom: "16px" },
  testGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
    gap: "10px",
  },
  testCard: {
    background: "#0c0f16",
    border: "1px solid #171c27",
    borderRadius: "6px",
    padding: "12px 14px",
  },
  testHeader: { display: "flex", gap: "8px", alignItems: "center", marginBottom: "6px" },
  testName: { fontSize: "12.5px", color: "#dde3ef", fontWeight: 500 },
  testDetail: { fontSize: "11px", color: "#8893a8", fontFamily: "monospace" },
  snapLabel: { fontSize: "11px", color: "#8893a8", fontFamily: "monospace", marginBottom: "4px" },
  muted: { color: "#5a6478", fontSize: "13px" },
};