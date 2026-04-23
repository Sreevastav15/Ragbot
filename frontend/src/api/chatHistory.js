// src/api/chatHistory.js
import axiosInstant from "./axiosinstance";

export const fileHistory = async () => {
  try {
    const res = await axiosInstant.get("/chathistory/all");
    return res.data;
  } catch (error) {
    console.error("Error fetching chat history:", error);
    throw error;
  }
};

export const fullChatHistory = async (doc_id) => {
  try {
    const res = await axiosInstant.get(`/chathistory/full?doc_id=${doc_id}`);
    return res.data;
  } catch (error) {
    console.error("Error fetching Chat history", error);
    throw error;
  }
};

export const loadChatSession = async (doc_id) => {
  try {
    const res = await axiosInstant.get("/chathistory/session", {
      params: { doc_id },
    });
    return res.data;
  } catch (error) {
    console.error("Error loading chat session:", error);
    throw error;
  }
};