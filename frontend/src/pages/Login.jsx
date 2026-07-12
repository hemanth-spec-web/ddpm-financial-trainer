import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { loginUser, getCurrentUser } from "../api/auth";
import { useAuthStore } from "../store/authStore";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();
  const setAuth = useAuthStore((s) => s.setAuth);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await loginUser(email, password);
      const { access_token, refresh_token, user } = res.data;
      setAuth(user, access_token, refresh_token);
      navigate("/dashboard");
    } catch (err) {
      setError(err.response?.data?.detail || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.title}>DDPM Trainer</h1>
        <p style={styles.subtitle}>Log in to your account</p>

        {error && <div style={styles.error}>{error}</div>}

        <form onSubmit={handleSubmit} style={styles.form}>
          <label style={styles.label}>Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={styles.input}
            required
          />

          <label style={styles.label}>Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={styles.input}
            required
          />

          <button type="submit" style={styles.button} disabled={loading}>
            {loading ? "Logging in..." : "Log In"}
          </button>
        </form>

        <p style={styles.footer}>
          Don't have an account? <Link to="/register" style={styles.link}>Register</Link>
        </p>
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#0a0c10",
    fontFamily: "'Inter', sans-serif",
  },
  card: {
    background: "#111318",
    border: "1px solid #1e222b",
    borderRadius: "8px",
    padding: "40px",
    width: "360px",
  },
  title: {
    color: "#e2e8f0",
    fontSize: "22px",
    fontWeight: 700,
    marginBottom: "4px",
  },
  subtitle: {
    color: "#64748b",
    fontSize: "13px",
    marginBottom: "24px",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
  },
  label: {
    color: "#94a3b8",
    fontSize: "12px",
    marginTop: "12px",
  },
  input: {
    background: "#0a0c10",
    border: "1px solid #1e222b",
    borderRadius: "4px",
    padding: "10px 12px",
    color: "#e2e8f0",
    fontSize: "14px",
    outline: "none",
  },
  button: {
    marginTop: "24px",
    background: "#00e5a0",
    color: "#06080d",
    border: "none",
    borderRadius: "4px",
    padding: "12px",
    fontSize: "14px",
    fontWeight: 600,
    cursor: "pointer",
  },
  error: {
    background: "rgba(239,68,68,0.1)",
    border: "1px solid rgba(239,68,68,0.3)",
    color: "#ef4444",
    borderRadius: "4px",
    padding: "10px 12px",
    fontSize: "13px",
    marginBottom: "16px",
  },
  footer: {
    marginTop: "20px",
    color: "#64748b",
    fontSize: "13px",
    textAlign: "center",
  },
  link: {
    color: "#00e5a0",
    textDecoration: "none",
  },
};