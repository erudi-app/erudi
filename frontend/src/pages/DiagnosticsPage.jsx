import React, { useEffect } from "react";
import { Bug } from "lucide-react";
import { useTranslation } from "react-i18next";
import Sidebar from "../components/Sidebar";
import DiagnosticsPanel from "../components/DiagnosticsPanel";
import { markDiagnosticsVisited } from "../utils/bugCounter";

/**
 * Diagnostics page (bug icon at the bottom of the sidebar rail).
 *
 * Everything a bug report needs about this machine, read locally and handed
 * to the user to copy: the environment summary, the recent warnings and
 * errors, the log folder and the report block. It sends nothing anywhere.
 * The page is the same chrome as Settings around the diagnostics content,
 * which owns its own data loading and its own loading and error states.
 *
 * Opening this page is also what clears the bug icon's counter badge (#485):
 * mounting records "now" as the last visit, so anything recorded before this
 * moment stops counting, immediately (the sidebar's badge subscribes to the
 * same marker) and on every future visit until the next new error.
 */
export default function DiagnosticsPage() {
  const { t } = useTranslation();

  useEffect(() => {
    markDiagnosticsVisited();
  }, []);

  return (
    <div className="flex h-screen bg-[#071b18]">
      <Sidebar />
      <main className="flex-1 bg-[var(--canvas)] relative custom-scroll overflow-auto">
        <div className="mx-auto max-w-3xl px-8 py-10 space-y-9">
          <header className="rise flex items-start gap-3.5">
            <div className="mt-1 rounded-xl border border-[var(--line)] bg-[var(--surface-2)] p-2.5">
              <Bug className="w-5 h-5 text-[var(--fit-good)]" />
            </div>
            <div>
              <span className="eyebrow">{t("diagnostics:page.eyebrow")}</span>
              <h1 className="text-2xl font-semibold text-[var(--ink)] tracking-tight mt-1.5">
                {t("diagnostics:page.title")}
              </h1>
              <p className="text-[13px] text-[var(--ink-dim)] mt-1.5 max-w-xl leading-relaxed">
                {t("diagnostics:page.description")}
              </p>
              <p className="text-[12px] text-[var(--ink-faint)] mt-2 leading-relaxed">
                {t("diagnostics:page.note")}
              </p>
            </div>
          </header>

          <DiagnosticsPanel />
        </div>
      </main>
    </div>
  );
}
