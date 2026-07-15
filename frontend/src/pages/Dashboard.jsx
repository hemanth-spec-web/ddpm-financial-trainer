import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listExperiments, createExperiment, deleteExperiment } from "../api/experiments";
import { useAuthStore } from "../store/authStore";

export default function Dashboard() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState("");

  const navigate = useNavigate();
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);

  const [form, setForm] = useState({
    name: "",
    description: "",
    T: 1000,
    beta_start: 0.0001,
    beta_end: 0.02,
    sequence_length: 128,
    data_source: "synthetic",
    ticker: "^GSPC",
  });

  const fetchExperiments = async () => {
    setLoading(true);
    try {
      const res = await listExperiments();
      setExperiments(res.data);
    } catch (err) {
      setError("Failed to load experiments");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExperiments();
  }, []);

  const handleCreate = async (e) => {
    e.preventDefault();
    setCreating(true);
    setError("");
    try {
      await createExperiment({
        ...form,
        epochs: 15,
        batch_size: 16,
        learning_rate: 0.0002,
        d_model: 16,
      });
      setShowForm(false);
      setForm({ ...form, name: "", description: "" });
      fetchExperiments();
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to create experiment");
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id) => {
    if (!confirm("Delete this experiment?")) return;
    await deleteExperiment(id);
    fetchExperiments();
  };

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>DDPM Trainer</h1>
          <p style={styles.subtitle}>{user?.username}</p>
        </div>
        <button style={styles.logoutBtn} onClick={handleLogout}>Log Out</button>
      </div>

      <div style={styles.toolbar}>
        <h2 style={styles.sectionTitle}>Your Experiments</h2>
        <button style={styles.newBtn} onClick={() => setShowForm(!showForm)}>
          {showForm ? "Cancel" : "+ New Experiment"}
        </button>
      </div>

      {error && <div style={styles.error}>{error}</div>}

      {showForm && (
        <form onSubmit={handleCreate} style={styles.form}>
          <div style={styles.formRow}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Name</label>
              <input
                style={styles.input}
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                required
              />
            </div>
            <div style={styles.formGroup}>
              <label style={styles.label}>Description</label>
              <input
                style={styles.input}
                value={form.description}
                onChange={(e) => setForm({ ...form, description: e.target.value })}
              />
            </div>
          </div>
          <div style={styles.formRow}>
            <div style={styles.formGroup}>
              <label style={styles.label}>T (timesteps)</label>
              <input
                type="number"
                style={styles.input}
                value={form.T}
                onChange={(e) => setForm({ ...form, T: parseInt(e.target.value) })}
              />
            </div>
            <div style={styles.formGroup}>
              <label style={styles.label}>Beta Start</label>
              <input
                type="number"
                step="0.00001"
                style={styles.input}
                value={form.beta_start}
                onChange={(e) => setForm({ ...form, beta_start: parseFloat(e.target.value) })}
              />
            </div>
            <div style={styles.formGroup}>
              <label style={styles.label}>Beta End</label>
              <input
                type="number"
                step="0.001"
                style={styles.input}
                value={form.beta_end}
                onChange={(e) => setForm({ ...form, beta_end: parseFloat(e.target.value) })}
              />
            </div>
            <div style={styles.formGroup}>
              <label style={styles.label}>Sequence Length</label>
              <input
                type="number"
                style={styles.input}
                value={form.sequence_length}
                onChange={(e) => setForm({ ...form, sequence_length: parseInt(e.target.value) })}
              />
            </div>
          </div>
          <div style={styles.formRow}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Data Source</label>
              <select
                style={styles.input}
                value={form.data_source}
                onChange={(e) => setForm({ ...form, data_source: e.target.value })}
              >
                <option value="synthetic">Synthetic (sine waves)</option>
                <option value="financial">Real Financial Data</option>
              </select>
            </div>
            {form.data_source === "financial" && (
              <div style={styles.formGroup}>
                <label style={styles.label}>Ticker Symbol</label>
                <input
                  style={styles.input}
                  value={form.ticker}
                  onChange={(e) => setForm({ ...form, ticker: e.target.value })}
                  placeholder="^GSPC"
                />
              </div>
            )}
          </div>
          <button type="submit" style={styles.submitBtn} disabled={creating}>
            {creating ? "Running Phase 1..." : "Create & Run"}
          </button>
        </form>
      )}

      {loading ? (
        <p style={styles.muted}>Loading...</p>
      ) : experiments.length === 0 ? (
        <p style={styles.muted}>No experiments yet. Create one above.</p>
      ) : (
        <div style={styles.grid}>
          {experiments.map((exp) => (
            <div key={exp.id} style={styles.card}>
              <div style={styles.cardHeader}>
                <h3 style={styles.cardTitle}>{exp.name}</h3>
                <span style={{ ...styles.badge, ...statusStyle(exp.status) }}>
                  {exp.status}
                </span>
              </div>
              <p style={styles.cardDesc}>{exp.description || "No description"}</p>
              <div style={styles.cardMeta}>
                <span>T={exp.T}</span>
                <span>β: {exp.beta_start}→{exp.beta_end}</span>
                <span>L={exp.sequence_length}</span>
                {exp.data_source === "financial" && (
                  <span style={{ color: "#e8b44b" }}>📈 {exp.ticker}</span>
                )}
              </div>
              <div style={styles.cardActions}>
                <button
                  style={styles.viewBtn}
                  onClick={() => navigate(`/experiments/${exp.id}`)}
                >
                  View Results
                </button>
                <button
                  style={styles.deleteBtn}
                  onClick={() => handleDelete(exp.id)}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
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
  page: {
    minHeight: "100vh",
    background: "#06080d",
    padding: "32px",
    fontFamily: "'Inter', sans-serif",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "32px",
    paddingBottom: "20px",
    borderBottom: "1px solid #171c27",
  },
  title: { fontSize: "20px", fontWeight: 700, color: "#dde3ef" },
  subtitle: { fontSize: "12px", color: "#5a6478", marginTop: "2px" },
  logoutBtn: {
    background: "transparent",
    border: "1px solid #1f2535",
    color: "#8893a8",
    borderRadius: "4px",
    padding: "8px 16px",
    fontSize: "13px",
    cursor: "pointer",
  },
  toolbar: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "20px",
  },
  sectionTitle: { fontSize: "16px", fontWeight: 600, color: "#dde3ef" },
  newBtn: {
    background: "#00e5a0",
    color: "#06080d",
    border: "none",
    borderRadius: "4px",
    padding: "10px 18px",
    fontSize: "13px",
    fontWeight: 600,
    cursor: "pointer",
  },
  form: {
    background: "#0c0f16",
    border: "1px solid #171c27",
    borderRadius: "8px",
    padding: "24px",
    marginBottom: "28px",
  },
  formRow: {
    display: "flex",
    gap: "16px",
    marginBottom: "16px",
    flexWrap: "wrap",
  },
  formGroup: { flex: 1, minWidth: "140px", display: "flex", flexDirection: "column", gap: "6px" },
  label: { fontSize: "11px", color: "#8893a8", textTransform: "uppercase", letterSpacing: "0.05em" },
  input: {
    background: "#06080d",
    border: "1px solid #1f2535",
    borderRadius: "4px",
    padding: "9px 12px",
    color: "#dde3ef",
    fontSize: "13px",
    outline: "none",
  },
  submitBtn: {
    background: "#e8b44b",
    color: "#06080d",
    border: "none",
    borderRadius: "4px",
    padding: "11px 20px",
    fontSize: "13px",
    fontWeight: 600,
    cursor: "pointer",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
    gap: "16px",
  },
  card: {
    background: "#0c0f16",
    border: "1px solid #171c27",
    borderRadius: "8px",
    padding: "20px",
  },
  cardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: "8px",
  },
  cardTitle: { fontSize: "15px", fontWeight: 600, color: "#dde3ef" },
  badge: {
    fontSize: "10px",
    fontFamily: "monospace",
    padding: "3px 8px",
    borderRadius: "3px",
    border: "1px solid",
    textTransform: "uppercase",
  },
  cardDesc: { fontSize: "12.5px", color: "#8893a8", marginBottom: "14px" },
  cardMeta: {
    display: "flex",
    gap: "12px",
    fontSize: "11px",
    fontFamily: "monospace",
    color: "#5a6478",
    marginBottom: "16px",
    flexWrap: "wrap",
  },
  cardActions: { display: "flex", gap: "8px" },
  viewBtn: {
    flex: 1,
    background: "rgba(91,156,246,0.1)",
    border: "1px solid rgba(91,156,246,0.3)",
    color: "#5b9cf6",
    borderRadius: "4px",
    padding: "8px",
    fontSize: "12px",
    cursor: "pointer",
  },
  deleteBtn: {
    background: "transparent",
    border: "1px solid #1f2535",
    color: "#5a6478",
    borderRadius: "4px",
    padding: "8px 12px",
    fontSize: "12px",
    cursor: "pointer",
  },
  muted: { color: "#5a6478", fontSize: "13px" },
  error: {
    background: "rgba(239,68,68,0.1)",
    border: "1px solid rgba(239,68,68,0.3)",
    color: "#ef4444",
    borderRadius: "4px",
    padding: "10px 12px",
    fontSize: "13px",
    marginBottom: "16px",
  },
};