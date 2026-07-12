import api from "./client";

export const registerUser = (email, username, password) =>
  api.post("/api/auth/register", { email, username, password });

export const loginUser = (email, password) =>
  api.post("/api/auth/login", { email, password });

export const getCurrentUser = () => api.get("/api/auth/me");

export const logoutUser = (refreshToken) =>
  api.post("/api/auth/logout", { refresh_token: refreshToken });