import api from "./client";

export const createExperiment = (data) =>
  api.post("/api/experiments/", data);

export const listExperiments = () =>
  api.get("/api/experiments/");

export const getExperiment = (id) =>
  api.get(`/api/experiments/${id}`);

export const deleteExperiment = (id) =>
  api.delete(`/api/experiments/${id}`);