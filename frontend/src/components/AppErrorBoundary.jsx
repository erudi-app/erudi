import React from "react";
import PropTypes from "prop-types";
import { AlertTriangle, RotateCcw } from "lucide-react";
import { withTranslation } from "react-i18next";

import ReportProblem from "./ReportProblem";
import { recordSessionError } from "../utils/errorCapture";

/**
 * Root error boundary.
 *
 * `window.onerror` does not see an exception thrown during a React render:
 * React catches it, unmounts the tree and rethrows into its own reporting.
 * Without a boundary that is a white window and no trace in either log file.
 * This one records the error through the same path the global handlers use,
 * so it lands in `erudi-backend.log` and in the session buffer the Diagnostics
 * page reads, and it offers the user the shared report block plus a reload.
 *
 * A class component because that is the only way to declare a boundary.
 * `withTranslation` supplies `t`, since hooks are unavailable here.
 */
class AppErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null, componentStack: "" };
    this.handleReload = this.handleReload.bind(this);
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    const componentStack = info?.componentStack ?? "";
    this.setState({ componentStack });
    recordSessionError({
      origin: "react.errorBoundary",
      message: error?.message ?? String(error),
      stack: `${error?.stack ?? ""}${componentStack}`,
    });
  }

  handleReload() {
    // A full renderer reload: the React tree is gone, and everything that
    // matters lives in the backend's database, not in this window's memory.
    window.location.reload();
  }

  render() {
    const { children, t } = this.props;
    const { error, componentStack } = this.state;
    if (!error) return children;

    const diagnostics = [
      `Erudi renderer error: ${error?.message ?? String(error)}`,
      error?.stack ?? "",
      componentStack,
    ]
      .filter(Boolean)
      .join("\n");

    return (
      <div className="min-h-screen w-full bg-[var(--canvas)] flex items-center justify-center p-8">
        <div className="w-full max-w-2xl space-y-6 rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-8">
          <div className="flex items-start gap-3.5">
            <div className="mt-0.5 rounded-xl border border-[var(--line)] bg-[var(--surface-2)] p-2.5">
              <AlertTriangle className="w-5 h-5 text-[var(--fit-poor,#f87171)]" />
            </div>
            <div>
              <h1 className="text-[17px] font-semibold text-[var(--ink)] tracking-tight">
                {t("diagnostics:boundary.title")}
              </h1>
              <p className="text-[13px] text-[var(--ink-dim)] mt-1.5 leading-relaxed">
                {t("diagnostics:boundary.body")}
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={this.handleReload}
            className="inline-flex items-center gap-1.5 text-[13px] rounded-lg border border-[var(--line)] bg-[var(--canvas)] text-[var(--ink)] px-3 py-1.5 hover:border-[var(--fit-good)] focus:outline-none focus:border-[var(--fit-good)] transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            {t("diagnostics:boundary.reload")}
          </button>

          <ReportProblem diagnostics={diagnostics} />
        </div>
      </div>
    );
  }
}

AppErrorBoundary.propTypes = {
  children: PropTypes.node,
  t: PropTypes.func.isRequired,
};

export default withTranslation()(AppErrorBoundary);
