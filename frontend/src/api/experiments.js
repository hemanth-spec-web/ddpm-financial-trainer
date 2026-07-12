import api from "./client";

export const createExperiment = (data) =>
  api.post("/api/experiments/", data);

export const listExperiments = () =>
  api.get("/api/experiments/");

export const getExperiment = (id) =>
  api.get(`/api/experiments/${id}`);

export const deleteExperiment = (id) =>
  api.delete(`/api/experiments/${id}`);

export const startTraining = (id) =>
  api.post(`/api/experiments/${id}/train`);

export const generateSamples = (id, nSamples = 4) =>
  api.post(`/api/experiments/${id}/generate?n_samples=${nSamples}`);