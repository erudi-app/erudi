// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

const { getMock, putMock } = vi.hoisted(() => ({
  getMock: vi.fn(),
  putMock: vi.fn(),
}));

vi.mock("../services/api/client", () => ({
  default: { get: getMock, put: putMock },
  apiClient: { get: getMock, put: putMock },
}));

vi.mock("../components/Sidebar", () => ({
  default: () => <div data-testid="sidebar" />,
}));

import SettingsPage from "./SettingsPage";
import i18n from "../i18n";

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/erudi/settings"]}>
      <SettingsPage />
    </MemoryRouter>
  );
}

beforeEach(() => {
  getMock.mockResolvedValue({
    web_search_enabled: false,
    language: "en",
    auto_update_enabled: true,
    inference_backend: "auto",
  });
  putMock.mockResolvedValue({
    web_search_enabled: true,
    language: "en",
    auto_update_enabled: true,
    inference_backend: "auto",
  });
});

afterEach(async () => {
  cleanup();
  vi.clearAllMocks();
  delete window.languageAPI;
  delete window.updaterAPI;
  await i18n.changeLanguage("en");
});

describe("SettingsPage — application language", () => {
  it("renders the language section with the four languages named natively", async () => {
    renderPage();
    const select = await screen.findByRole("combobox", { name: "Application language" });
    const labels = Array.from(select.querySelectorAll("option")).map((o) => o.textContent);
    expect(labels).toEqual(["English", "Français", "Español", "中文"]);
    const values = Array.from(select.querySelectorAll("option")).map((o) => o.value);
    expect(values).toEqual(["en", "fr", "es", "zh"]);
  });

  it("reflects the active language on the select (applied at boot by the App sync)", async () => {
    await i18n.changeLanguage("es");
    renderPage();
    const select = await screen.findByRole("combobox", { name: "Idioma de la aplicación" });
    expect(select.value).toBe("es");
    expect(screen.getByText("Ajustes")).toBeTruthy();
  });

  it("PUTs the new language, switches the UI immediately and notifies main", async () => {
    const set = vi.fn();
    window.languageAPI = { set };
    putMock.mockResolvedValue({ web_search_enabled: false, language: "fr" });
    renderPage();
    const select = await screen.findByRole("combobox", { name: "Application language" });
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    fireEvent.change(select, { target: { value: "fr" } });

    await waitFor(() =>
      expect(putMock).toHaveBeenCalledWith("/user_settings/", { language: "fr" })
    );
    expect(await screen.findByText("Paramètres")).toBeTruthy();
    expect(screen.getByText("Recherche web")).toBeTruthy();
    expect(i18n.language).toBe("fr");
    expect(set).toHaveBeenCalledWith("fr");
  });

  it("keeps the new language in the UI even if persisting fails", async () => {
    putMock.mockRejectedValue(new Error("offline"));
    renderPage();
    const select = await screen.findByRole("combobox", { name: "Application language" });
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    fireEvent.change(select, { target: { value: "zh" } });

    expect(await screen.findByText("设置")).toBeTruthy();
    expect(i18n.language).toBe("zh");
  });
});

describe("SettingsPage", () => {
  it("renders the settings heading and the web search section", async () => {
    renderPage();
    expect(screen.getByText("Settings")).toBeTruthy();
    expect(await screen.findByText("Web Search")).toBeTruthy();
  });

  it("fetches the user settings on mount", async () => {
    renderPage();
    await waitFor(() => expect(getMock).toHaveBeenCalledWith("/user_settings/"));
  });

  it("reflects the fetched value on the toggle (off by default)", async () => {
    renderPage();
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("false"));
  });

  it("reflects an enabled global setting", async () => {
    getMock.mockResolvedValue({ web_search_enabled: true });
    renderPage();
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("true"));
  });

  it("PUTs the new value when toggled and updates the UI", async () => {
    renderPage();
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    await waitFor(() => expect(getMock).toHaveBeenCalled());
    fireEvent.click(toggle);
    await waitFor(() =>
      expect(putMock).toHaveBeenCalledWith("/user_settings/", {
        web_search_enabled: true,
      })
    );
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("true"));
  });

  it("explains the privacy trade-off and the inheritance rule", async () => {
    renderPage();
    await screen.findByText("Web Search");
    const page = document.body.textContent;
    expect(page).toMatch(/search engines/i);
    expect(page).toMatch(/new conversations/i);
  });

  it("renders the sidebar", () => {
    renderPage();
    expect(screen.getByTestId("sidebar")).toBeTruthy();
  });
});

describe("SettingsPage — automatic updates", () => {
  it("shows automatic updates on, the behaviour of an install nobody configured", async () => {
    renderPage();
    const toggle = await screen.findByRole("switch", { name: "Automatic updates" });
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("true"));
  });

  it("reflects a refusal that was persisted earlier", async () => {
    getMock.mockResolvedValue({
      web_search_enabled: false,
      language: "en",
      auto_update_enabled: false,
    });
    renderPage();
    const toggle = await screen.findByRole("switch", { name: "Automatic updates" });
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("false"));
  });

  it("persists a refusal and tells the main process to stop checking", async () => {
    const setAutoUpdateEnabled = vi.fn();
    window.updaterAPI = { setAutoUpdateEnabled };
    putMock.mockResolvedValue({
      web_search_enabled: false,
      language: "en",
      auto_update_enabled: false,
    });
    renderPage();
    const toggle = await screen.findByRole("switch", { name: "Automatic updates" });
    await waitFor(() => expect(getMock).toHaveBeenCalled());

    fireEvent.click(toggle);

    await waitFor(() =>
      expect(putMock).toHaveBeenCalledWith("/user_settings/", { auto_update_enabled: false })
    );
    await waitFor(() => expect(setAutoUpdateEnabled).toHaveBeenCalledWith(false));
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("false"));
  });

  it("says that updates are downloaded and installed without asking", async () => {
    renderPage();
    await screen.findByText("Automatic updates");
    expect(document.body.textContent).toMatch(/install/i);
  });
});

describe("SettingsPage inference engine", () => {
  it("shows Automatic by default and says the engine restarts", async () => {
    renderPage();
    expect(await screen.findByText("Inference engine")).toBeTruthy();
    const select = screen.getByLabelText("Inference engine");
    await waitFor(() => expect(select.value).toBe("auto"));
    expect(document.body.textContent).toMatch(/restarts its engine/i);
  });

  it("pins the processor and restarts the backend so the choice applies", async () => {
    window.backendAPI = { restartBackend: vi.fn().mockResolvedValue(undefined) };
    renderPage();
    const select = await screen.findByLabelText("Inference engine");

    fireEvent.change(select, { target: { value: "cpu" } });

    await waitFor(() =>
      expect(putMock).toHaveBeenCalledWith("/user_settings/", { inference_backend: "cpu" })
    );
    await waitFor(() => expect(window.backendAPI.restartBackend).toHaveBeenCalled());
    delete window.backendAPI;
  });

  it("reflects a persisted processor preference and switches back", async () => {
    getMock.mockResolvedValue({
      web_search_enabled: false,
      language: "en",
      auto_update_enabled: true,
      inference_backend: "cpu",
    });
    window.backendAPI = { restartBackend: vi.fn().mockResolvedValue(undefined) };
    renderPage();
    const select = await screen.findByLabelText("Inference engine");
    await waitFor(() => expect(select.value).toBe("cpu"));

    fireEvent.change(select, { target: { value: "auto" } });

    await waitFor(() =>
      expect(putMock).toHaveBeenCalledWith("/user_settings/", { inference_backend: "auto" })
    );
    delete window.backendAPI;
  });
});

describe("SettingsPage", () => {
  it("renders the sidebar", () => {
    renderPage();
    expect(screen.getByTestId("sidebar")).toBeTruthy();
  });

  it("links to the 'what leaves your machine' page in the system browser", async () => {
    renderPage();
    expect(await screen.findByText("What leaves your machine")).toBeTruthy();
    const link = screen.getByRole("link", { name: "Read the page" });
    expect(link.getAttribute("href")).toBe("https://erudi-app.github.io/erudi/privacy/");
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toBe("noreferrer");
  });
});
