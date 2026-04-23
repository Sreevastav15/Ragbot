// src/App.jsx
import { Toaster } from "react-hot-toast";
import Home from "./pages/Home";
import "./index.css";

function App() {
  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "#181c25",
            color: "#e8eaf0",
            border: "1px solid #252b3b",
            borderRadius: "10px",
            fontSize: "0.85rem",
          },
          success: { iconTheme: { primary: "#34d39a", secondary: "#0c0e12" } },
          error:   { iconTheme: { primary: "#f87171", secondary: "#0c0e12" } },
        }}
      />
      <Home />
    </>
  );
}

export default App;