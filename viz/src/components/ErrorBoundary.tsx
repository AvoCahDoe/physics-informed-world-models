import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/** Keeps a render fault in one route from blanking the whole site.
 *
 *  The live simulator and the WebGL scene both run every frame, so a single bad
 *  frame would otherwise unmount everything and leave a white page with no
 *  indication of what happened. */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error("Unhandled render error:", error);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="page crash">
        <h1>Something broke while rendering this page.</h1>
        <p className="lede">
          That is a bug, not something you did. The written analysis is on
          GitHub if you need the results in the meantime.
        </p>
        <pre className="code">{error.message}</pre>
        <div className="hero-actions">
          <button className="btn primary" onClick={() => this.setState({ error: null })}>
            Try again
          </button>
          <a
            className="btn"
            href="https://github.com/AvoCahDoe/physics-informed-world-models"
            target="_blank"
            rel="noreferrer"
          >
            Open the repo
          </a>
        </div>
      </div>
    );
  }
}
