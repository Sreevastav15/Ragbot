// src/api/qaApi.js
import axiosInstance from "./axiosinstance";

export const askQuestion = async (question, documentIds) => {
  const ids = Array.isArray(documentIds) ? documentIds : [documentIds];
  const res = await axiosInstance.post("/answer/", {
    question,
    document_ids: ids,
    stream: false,
  });
  return res.data;
};

/**
 * Streaming version – calls /answer/stream and returns a ReadableStream.
 * The caller receives chunks of JSON lines to parse.
 */
export const askQuestionStream = async (question, documentIds, onToken, onMeta, onDone) => {
  const ids = Array.isArray(documentIds) ? documentIds : [documentIds];

  const response = await fetch("http://localhost:8000/answer/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_ids: ids, stream: true }),
  });

  if (!response.ok) throw new Error(`HTTP ${response.status}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop(); // keep incomplete line

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;
        try {
          const data = JSON.parse(jsonStr);
          if (data.type === "meta") onMeta?.(data);
          else if (data.type === "token") onToken?.(data.text);
          else if (data.type === "done") onDone?.(data);
        } catch {}
      }
    }
  }
};