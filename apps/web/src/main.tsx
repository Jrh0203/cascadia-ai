import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import ClusterDashboard from "./ClusterDashboard";
import WildlifeAtlas from "./WildlifeAtlas";
import "./styles.css";

const currentPath = window.location.pathname.replace(/\/+$/, "") || "/";
const RootView =
  currentPath === "/cluster"
    ? ClusterDashboard
    : currentPath === "/wildlife-catalog"
      ? WildlifeAtlas
      : App;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RootView />
  </StrictMode>,
);
