// src/api/delete.js
import axiosInstant from "./axiosinstance";

export const deleteChat = async (doc_id) => {
  try {
    await axiosInstant.delete("/delete/", { params: { doc_id } });
  } catch (error) {
    console.error("Failed to delete chat", error);
    throw error;
  }
};