// src/api/qaApi.js
import axiosInstance from "./axiosinstance";

/**
 * POST /answer/  — returns the full answer in a single response (non-streaming).
 *
 * Response shape:
 *  {
 *    answer:           string,
 *    sources:          [{ filename, page }],
 *    rewritten_query:  string,
 *    k_used:           number,
 *    response_time_ms: number,
 *  }
 */
export const askQuestion = async (question, documentIds) => {
  const ids = Array.isArray(documentIds) ? documentIds : [documentIds];
  const res = await axiosInstance.post("/answer/", {
    question,
    document_ids: ids,
    stream: false,
  });
  return res.data;
};