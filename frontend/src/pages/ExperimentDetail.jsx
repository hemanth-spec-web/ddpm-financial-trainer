import { useEffect, useState, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { getExperiment, startTraining, generateSamples } from "../api/experiments";

function FactCard({ label, real, generated, note }) {
  const hasGenerated = generated !== undefined && generated !== null;

  return (
    <div style={styles.factCard}>
      <p style={styles.factLabel}>{label}</p>
      <div style={styles.factValues}>
        <div style={styles.factValueBlock}>
          <span style={styles.factValueLabel}>Real</span>
          <span style={styles.factValueReal}>{real?.toFixed(4)}</span>
        </div>
        {hasGenerated && (
          <div style={styles.factValueBlock}>
            <span style={styles.factValueLabel}>Generated</span>
            <span style={styles.factValueGenerated}>{generated.toFixed(4)}</span>
          </div>
        )}
      </div>
      <p style={styles.factNote}>{note}</p>
    </div>
  );
}

export default function ExperimentDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [exp, setExp] = useState(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const pollRef = useRef(null);

  const fetchExperiment = async () => {
    const res = await getExperiment(id);
    setExp(res.data);
    return res.data;
  };

  useEffect(() => {
    fetchExperiment().finally(() => setLoading(false));

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [id]);

  // Poll every 2 seconds while training is running
  useEffect(() => {
    if (exp?.status === "running") {
      pollRef.current = setInterval(async () => {
        const updated = await fetchExperiment();
        if (updated.status !== "running") {
          clearInterval(pollRef.current);
        }
      }, 2000);
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [exp?.status]);

  const handleStartTraining = async () => {
    setStarting(true);
    setError("");
    try {
      await startTraining(id);
      await fetchExperiment();
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to start training");
    } finally {
      setStarting(false);
    }
  };

  const handleGenerate = async () => {
    setGenerating(true);
    setError("");
    try {
      await generateSamples(id, 4);
      const pollGen = setInterval(async () => {
        const updated = await fetchExperiment();
        if (updated.generated_samples?.model_generated) {
          clearInterval(pollGen);
          setGenerating(false);
        }
      }, 3000);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to generate samples");
      setGenerating(false);
    }
  };

  if (loading) return <div style={styles.page}><p style={styles.muted}>Loading...</p></div>;
  if (!exp) return <div style={styles.page}><p style={styles.muted}>Not found</p></div>;

  const curves = exp.metrics?.schedule_summary?.curves;
  const tests = exp.metrics?.unit_tests?.tests || [];
  const allPassed = exp.metrics?.unit_tests?.all_passed;

  const scheduleData = curves
    ? curves.timesteps.map((t, i) => ({
        t,
        beta: curves.betas[i],
        sqrt_alpha_bar: curves.sqrt_alpha_bars[i],
        sqrt_one_minus_alpha_bar: curves.sqrt_one_minus_alpha_bars[i],
      }))
    : [];

  const snapshots = exp.generated_samples?.snapshots || [];
  const modelGenerated = exp.generated_samples?.model_generated?.sequences || [];
  const stylizedReal = exp.metrics?.stylized_facts_real_data;
  const stylizedGenerated = exp.metrics?.stylized_facts_generated;

  const lossData = (exp.train_losses || []).map((loss, i) => ({
    epoch: i + 1,
    loss,
  }));

  const hasTrainingModel = exp.model_weights_path;

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
        <div style={styles.metaItem}><span style={styles.metaLabel}>Epochs</span><span style={styles.metaValue}>{exp.epochs}</span></div>
        <div style={styles.metaItem}>
          <span style={styles.metaLabel}>Data Source</span>
          <span style={styles.metaValue}>
            {exp.data_source === "financial" ? `📈 ${exp.ticker}` : "Synthetic"}
          </span>
        </div>
        <div style={styles.metaItem}>
          <span style={styles.metaLabel}>Unit Tests</span>
          <span style={{ ...styles.metaValue, color: allPassed ? "#00e5a0" : "#ef4444" }}>
            {allPassed ? "✓ Passed" : "✗ Failed"}
          </span>
        </div>
      </div>

      {/* Training Control Panel */}
      <div style={styles.trainingPanel}>
        <div style={styles.trainingHeader}>
          <h2 style={styles.sectionTitle}>Model Training</h2>
          <span style={{ ...styles.badge, ...statusStyle(exp.status) }}>{exp.status}</span>
        </div>

        {error && <div style={styles.error}>{error}</div>}

        {exp.status === "running" ? (
          <div>
            <p style={styles.trainingStatus}>
              Training in progress — epoch {exp.current_epoch} / {exp.epochs}
            </p>
            <div style={styles.progressBarBg}>
              <div
                style={{
                  ...styles.progressBarFill,
                  width: `${(exp.current_epoch / exp.epochs) * 100}%`,
                }}
              />
            </div>
          </div>
        ) : hasTrainingModel ? (
          <p style={styles.trainingStatus}>
            ✓ Training complete — final loss: <strong style={{ color: "#00e5a0" }}>{exp.final_loss}</strong>
          </p>
        ) : (
          <button style={styles.trainBtn} onClick={handleStartTraining} disabled={starting}>
            {starting ? "Starting..." : "▶ Start Training"}
          </button>
        )}
      </div>

      {/* Live loss curve */}
      {lossData.length > 0 && (
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Training Loss</h2>
          <p style={styles.sectionDesc}>
            MSE between predicted and actual noise — should trend downward as the U-Net learns.
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e222b" />
              <XAxis dataKey="epoch" stroke="#5a6478" tick={{ fontSize: 11 }} label={{ value: "Epoch", position: "insideBottom", offset: -5, fill: "#5a6478", fontSize: 11 }} />
              <YAxis stroke="#5a6478" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0c0f16", border: "1px solid #1f2535" }} />
              <Line type="monotone" dataKey="loss" stroke="#00e5a0" name="Training Loss (MSE)" dot={{ r: 3 }} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Model-generated samples */}
      {hasTrainingModel && (
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Generated Samples (Reverse Diffusion)</h2>
          <p style={styles.sectionDesc}>
            New sequences generated entirely from pure noise, using the trained U-Net
            to reverse the diffusion process 1000 steps back to a clean signal.
          </p>
          {modelGenerated.length === 0 ? (
            <button style={styles.trainBtn} onClick={handleGenerate} disabled={generating}>
              {generating ? "Generating... (~30s)" : "✨ Generate New Samples"}
            </button>
          ) : (
            <>
              {modelGenerated.map((seq, i) => {
                const chartData = seq.map((v, idx) => ({ idx, value: v }));
                return (
                  <div key={i} style={{ marginBottom: "14px" }}>
                    <p style={styles.snapLabel}>Generated Sample #{i + 1}</p>
                    <ResponsiveContainer width="100%" height={100}>
                      <LineChart data={chartData}>
                        <XAxis dataKey="idx" hide />
                        <YAxis hide />
                        <Line type="monotone" dataKey="value" stroke="#c084fc" dot={false} strokeWidth={1.4} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                );
              })}
              <button style={styles.regenBtn} onClick={handleGenerate} disabled={generating}>
                {generating ? "Generating..." : "↻ Generate 4 More"}
              </button>
            </>
          )}
        </div>
      )}

      {/* Stylized Facts Comparison — the quant evaluation */}
      {stylizedReal && (
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Stylized Facts — Real vs. Generated</h2>
          <p style={styles.sectionDesc}>
            The statistical properties every credible market simulator must reproduce
            (Cont, 2001). Comparing real {exp.ticker} returns against the model's generated samples.
          </p>

          <div style={styles.factsGrid}>
            <FactCard
              label="Kurtosis (fat tails)"
              real={stylizedReal.kurtosis}
              generated={stylizedGenerated?.kurtosis}
              note="Normal distribution = 0. Real markets are always > 0."
            />
            <FactCard
              label="Skewness"
              real={stylizedReal.skewness}
              generated={stylizedGenerated?.skewness}
              note="Real markets typically show slight negative skew (crashes > rallies)."
            />
            <FactCard
              label="Raw Return Autocorrelation (lag 1)"
              real={stylizedReal.raw_return_autocorr_lag1}
              generated={stylizedGenerated?.raw_return_autocorr_lag1}
              note="Should be near zero — markets are efficient, returns aren't predictable from their own past."
            />
            <FactCard
              label="Avg. Volatility Clustering"
              real={stylizedReal.avg_volatility_clustering}
              generated={stylizedGenerated?.avg_volatility_clustering}
              note="Positive value = volatility clusters in time (calm/turbulent periods), as in real markets."
            />
          </div>

          {!stylizedGenerated && (
            <p style={styles.mutedNote}>
              Generate samples above to see how the model's synthetic data compares.
            </p>
          )}
        </div>
      )}

      {/* Unit test results */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>Unit Test Results</h2>
        <div style={styles.testGrid}>
          {tests.map((t, i) => (
            <div key={i} style={styles.testCard}>
              <div style={styles.testHeader}>
                <span style={{ color: t.passed ? "#00e5a0" : "#ef4444" }}>{t.passed ? "✓" : "✗"}</span>
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
          <ResponsiveContainer width="100%" height={280}>
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

      {/* Forward process snapshots */}
      {snapshots.length > 0 && (
        <div style={styles.section}>
          <h2 style={styles.sectionTitle}>Forward Process — Signal Dissolving Into Noise</h2>
          {snapshots.map((snap, i) => {
            const chartData = snap.values.map((v, idx) => ({ idx, value: v }));
            return (
              <div key={i} style={{ marginBottom: "16px" }}>
                <p style={styles.snapLabel}>t = {snap.t} &nbsp;|&nbsp; SNR = {snap.snr.toFixed(4)}</p>
                <ResponsiveContainer width="100%" height={100}>
                  <LineChart data={chartData}>
                    <XAxis dataKey="idx" hide />
                    <YAxis domain={[-4, 4]} hide />
                    <Line type="monotone" dataKey="value" stroke={snap.t === 0 ? "#00e5a0" : "#5b9cf6"} dot={false} strokeWidth={1.2} />
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

function statusStyle(status) {
  switch (status) {
    case "completed":
      return { background: "rgba(0,229,160,0.1)", color: "#00e5a0", borderColor: "rgba(0,229,160,0.3)" };
    case "running":
      return { background: "rgba(245,166,35,0.1)", color: "#f5a623", borderColor: "rgba(245,166,35,0.3)" };
    case "failed":
      return { background: "rgba(239,68,68,0.1)", color: "#ef4444", borderColor: "rgba(239,68,68,0.3)" };
    default:
      return { background: "rgba(100,116,139,0.1)", color: "#64748b", borderColor: "rgba(100,116,139,0.3)" };
  }
}

const styles = {
  page: { minHeight: "100vh", background: "#06080d", padding: "32px", fontFamily: "'Inter', sans-serif", maxWidth: "1000px", margin: "0 auto" },
  backBtn: { background: "transparent", border: "none", color: "#5b9cf6", fontSize: "13px", cursor: "pointer", marginBottom: "20px", padding: 0 },
  header: { marginBottom: "24px" },
  title: { fontSize: "22px", fontWeight: 700, color: "#dde3ef" },
  subtitle: { fontSize: "13px", color: "#8893a8", marginTop: "4px" },
  metaRow: { display: "flex", gap: "24px", padding: "16px 0", borderTop: "1px solid #171c27", borderBottom: "1px solid #171c27", marginBottom: "24px", flexWrap: "wrap" },
  metaItem: { display: "flex", flexDirection: "column", gap: "4px" },
  metaLabel: { fontSize: "10px", color: "#5a6478", textTransform: "uppercase", letterSpacing: "0.05em" },
  metaValue: { fontSize: "14px", color: "#dde3ef", fontFamily: "monospace" },
  trainingPanel: { background: "#0c0f16", border: "1px solid #171c27", borderRadius: "8px", padding: "20px", marginBottom: "28px" },
  trainingHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" },
  badge: { fontSize: "10px", fontFamily: "monospace", padding: "3px 10px", borderRadius: "3px", border: "1px solid", textTransform: "uppercase" },
  trainBtn: { background: "#00e5a0", color: "#06080d", border: "none", borderRadius: "4px", padding: "12px 24px", fontSize: "13px", fontWeight: 600, cursor: "pointer" },
  trainingStatus: { fontSize: "13px", color: "#dde3ef", marginBottom: "10px" },
  progressBarBg: { background: "#171c27", borderRadius: "4px", height: "8px", overflow: "hidden" },
  progressBarFill: { background: "#f5a623", height: "100%", transition: "width 0.3s ease" },
  section: { marginBottom: "36px" },
  sectionTitle: { fontSize: "15px", fontWeight: 600, color: "#dde3ef", marginBottom: "6px" },
  sectionDesc: { fontSize: "12.5px", color: "#8893a8", marginBottom: "16px" },
  testGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))", gap: "10px" },
  testCard: { background: "#0c0f16", border: "1px solid #171c27", borderRadius: "6px", padding: "12px 14px" },
  testHeader: { display: "flex", gap: "8px", alignItems: "center", marginBottom: "6px" },
  testName: { fontSize: "12.5px", color: "#dde3ef", fontWeight: 500 },
  testDetail: { fontSize: "11px", color: "#8893a8", fontFamily: "monospace" },
  snapLabel: { fontSize: "11px", color: "#8893a8", fontFamily: "monospace", marginBottom: "4px" },
  muted: { color: "#5a6478", fontSize: "13px" },
  error: { background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", color: "#ef4444", borderRadius: "4px", padding: "10px 12px", fontSize: "13px", marginBottom: "14px" },
  regenBtn: { background: "transparent", border: "1px solid rgba(192,132,252,0.3)", color: "#c084fc", borderRadius: "4px", padding: "10px 18px", fontSize: "13px", cursor: "pointer", marginTop: "8px" },
  factsGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: "14px" },
  factCard: { background: "#0c0f16", border: "1px solid #171c27", borderRadius: "6px", padding: "16px" },
  factLabel: { fontSize: "12.5px", fontWeight: 600, color: "#dde3ef", marginBottom: "10px" },
  factValues: { display: "flex", gap: "20px", marginBottom: "10px" },
  factValueBlock: { display: "flex", flexDirection: "column", gap: "2px" },
  factValueLabel: { fontSize: "10px", color: "#5a6478", textTransform: "uppercase" },
  factValueReal: { fontSize: "16px", fontFamily: "monospace", color: "#5b9cf6", fontWeight: 600 },
  factValueGenerated: { fontSize: "16px", fontFamily: "monospace", color: "#c084fc", fontWeight: 600 },
  factNote: { fontSize: "11px", color: "#5a6478", lineHeight: "1.4" },
  mutedNote: { fontSize: "12px", color: "#5a6478", fontStyle: "italic" },
};