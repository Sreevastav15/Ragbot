// src/api/axiosinstance.js
import axios from "axios";

const axiosInstant = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 120000,
});

export default axiosInstant;