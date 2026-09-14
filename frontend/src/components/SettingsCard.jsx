import React from "react";
import PropTypes from "prop-types";

/**
 * One settings card: icon, title, description, optional fine-print note and
 * the control on the right. Shared by every section of the Settings page so
 * they look identical, and by the cards that live in their own file because
 * they carry state of their own (see `UpdatesCard`).
 */
export default function SettingsCard({ icon, title, description, note, control }) {
  return (
    <section className="relative overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--surface)] rise">
      <div
        className="pointer-events-none absolute -right-24 -top-24 w-72 h-72 rounded-full blur-3xl"
        style={{
          background: "radial-gradient(circle, rgba(52,214,165,0.10), transparent 70%)",
        }}
      />
      <div className="relative p-6">
        <div className="flex items-start justify-between gap-6">
          <div className="flex items-start gap-3.5">
            <div className="mt-0.5 rounded-xl border border-[var(--line)] bg-[var(--surface-2)] p-2.5">
              {icon}
            </div>
            <div>
              <h2 className="text-[15px] font-semibold text-[var(--ink)] tracking-tight">
                {title}
              </h2>
              <p className="text-[13px] text-[var(--ink-dim)] mt-1.5 max-w-md leading-relaxed">
                {description}
              </p>
              {note && (
                <p className="text-[12px] text-[var(--ink-faint)] mt-2.5 leading-relaxed">{note}</p>
              )}
            </div>
          </div>
          <div className="pt-1">{control}</div>
        </div>
      </div>
    </section>
  );
}

SettingsCard.propTypes = {
  icon: PropTypes.node.isRequired,
  title: PropTypes.string.isRequired,
  description: PropTypes.string.isRequired,
  note: PropTypes.string,
  control: PropTypes.node.isRequired,
};
