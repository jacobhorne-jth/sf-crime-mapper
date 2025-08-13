import axios from "axios";
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || "http://localhost:8000",
  timeout: 15000,
});
export const getNeighborhoods = () => api.get("/api/neighborhoods").then(r => r.data);
export const getPredictions   = (date, crime="all", tod="all") =>
  api.get("/api/predict", { params: { date, crime_type: crime, time_of_day: tod }}).then(r => r.data);
export const getSpike = (date, crime="all", tod="all") =>
  api.get("/api/spike", { params: { date, crime_type: crime, time_of_day: tod }}).then(r => r.data);
