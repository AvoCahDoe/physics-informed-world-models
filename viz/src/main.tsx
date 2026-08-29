import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import "./index.css";
import Layout from "./Layout";
import Docs from "./pages/Docs";

// three.js and recharts are ~1.2 MB between them and are only used by the
// rollout player, so /results loads them on demand rather than making the
// landing page wait on them.
const Results = lazy(() => import("./pages/Results"));
const Try = lazy(() => import("./pages/Try"));

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Suspense fallback={<div className="route-loading">loading…</div>}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Navigate to="/docs" replace />} />
            <Route path="docs" element={<Docs />} />
            <Route path="results" element={<Results />} />
            <Route path="try" element={<Try />} />
            <Route path="*" element={<Navigate to="/docs" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  </StrictMode>,
);
