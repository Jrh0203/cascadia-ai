import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import ClusterDashboard from "./ClusterDashboard";
import "./styles.css";

const currentPath = window.location.pathname.replace(/\/+$/, "") || "/";
const RootView = currentPath === "/cluster" ? ClusterDashboard : App;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RootView />
  </StrictMode>,
);
