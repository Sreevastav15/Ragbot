// src/api/uploadApi.js
import axiosInstance from "./axiosinstance.js";

export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await axiosInstance.post("/upload/", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

export const uploadMultiple = async (files) => {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  const res = await axiosInstance.post("/upload/multi", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data; // { uploaded: [...] }
};