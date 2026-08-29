import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useEffect } from "react";

const TABS = [
  { to: "/docs", label: "Docs" },
  { to: "/results", label: "Results" },
  { to: "/try", label: "Try" },
];

export default function Layout() {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <NavLink to="/docs" className="brand-link">
            <span className="brand-mark" aria-hidden="true" />
            <span>
              <strong>Physics-Informed World Models</strong>
              <small>HNN vs. Neural ODE vs. MLP on a bouncing ball</small>
            </span>
          </NavLink>
        </div>

        <nav className="tabs">
          {TABS.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              className={({ isActive }) => (isActive ? "tab active" : "tab")}
            >
              {t.label}
            </NavLink>
          ))}
        </nav>

        <a
          className="ghost"
          href="https://github.com/AvoCahDoe/physics-informed-world-models"
          target="_blank"
          rel="noreferrer"
        >
          GitHub
        </a>
      </header>

      <Outlet />

      <footer className="sitefoot">
        Ground truth uses velocity Verlet with event-resolved restitution. Models
        are free-running rollouts from the same initial state, trained in PyTorch.
      </footer>
    </div>
  );
}
