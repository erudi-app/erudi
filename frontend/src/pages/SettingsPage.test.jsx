// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";
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

import SettingsPage, { showsInferenceEngineSetting } from "./SettingsPage";
import i18n from "../i18n";

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/erudi/settings"]}>
      <SettingsPage />
    </MemoryRouter>
  );
}

/**
 * Both `/user_settings/` and `/hardware/app_startup` go through the same
 * mocked `apiClient.get`, so tests that care about `backend_type` route the
 * mock by URL instead of relying on call order.
 */
function mockGetByUrl({ userSettings, appStartup }) {
  getMock.mockImplementation((url) => {
    if (url === "/hardware/app_startup") {
      return Promise.resolve(appStartup);
    }
    return Promise.resolve(userSettings);
  });
}

beforeEach(() => {
  mockGetByUrl({
    userSettings: {
      web_search_enabled: false,
      language: "en",
      auto_update_enabled: true,
      inference_backend: "auto",
    },
    appStartup: { backend_type: "cuda" },
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
    mockGetByUrl({
      userSettings: { web_search_enabled: true },
      appStartup: { backend_type: "cuda" },
    });
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

describe("SettingsPage — Updates (#563)", () => {
  const SETTINGS = {
    web_search_enabled: false,
    language: "en",
    auto_update_enabled: true,
    auto_download_updates: false,
    auto_install_updates: false,
    inference_backend: "auto",
  };
  const CHECK = "Check for updates";
  const DOWNLOAD = "Download updates automatically";
  const INSTALL = "Install downloaded updates when I quit";

  const switchNamed = (name) => screen.findByRole("switch", { name });
  const checked = (toggle) => toggle.getAttribute("aria-checked");

  it("shows Check on and the two opt-ins off on a fresh install", async () => {
    mockGetByUrl({ userSettings: SETTINGS, appStartup: { backend_type: "cuda" } });
    renderPage();
    const check = await switchNamed(CHECK);
    await waitFor(() => expect(checked(check)).toBe("true"));
    expect(checked(await switchNamed(DOWNLOAD))).toBe("false");
    expect(checked(await switchNamed(INSTALL))).toBe("false");
  });

  it("says what each switch sends and does", async () => {
    mockGetByUrl({ userSettings: SETTINGS, appStartup: { backend_type: "cuda" } });
    renderPage();
    await switchNamed(CHECK);
    const text = document.body.textContent;
    expect(text).toMatch(/GitHub releases/);
    expect(text).toMatch(/nothing but your version and platform/);
    expect(text).toMatch(/fetches the installer/);
    expect(text).toMatch(/replaces the app.*when you quit/);
  });

  it("disables the two opt-ins while Check is off, and says why", async () => {
    mockGetByUrl({
      userSettings: { ...SETTINGS, auto_update_enabled: false, auto_install_updates: true },
      appStartup: { backend_type: "cuda" },
    });
    renderPage();
    const check = await switchNamed(CHECK);
    await waitFor(() => expect(checked(check)).toBe("false"));
    expect((await switchNamed(DOWNLOAD)).disabled).toBe(true);
    expect((await switchNamed(INSTALL)).disabled).toBe(true);
    expect(check.disabled).toBe(false);
    expect(document.body.textContent).toMatch(/Turn on Check for updates/);
  });

  it.each([
    [CHECK, "auto_update_enabled", false],
    [DOWNLOAD, "auto_download_updates", true],
    [INSTALL, "auto_install_updates", true],
  ])("persists %s and hands main the full snapshot", async (name, field, next) => {
    const setPreferences = vi.fn();
    window.updaterAPI = { setPreferences };
    mockGetByUrl({ userSettings: SETTINGS, appStartup: { backend_type: "cuda" } });
    const saved = { ...SETTINGS, [field]: next };
    putMock.mockResolvedValue(saved);
    renderPage();
    const toggle = await switchNamed(name);
    await waitFor(() => expect(toggle.disabled).toBe(false));

    fireEvent.click(toggle);

    await waitFor(() => expect(putMock).toHaveBeenCalledWith("/user_settings/", { [field]: next }));
    await waitFor(() =>
      expect(setPreferences).toHaveBeenCalledWith({
        check: saved.auto_update_enabled,
        download: saved.auto_download_updates,
        install: saved.auto_install_updates,
      })
    );
    await waitFor(() => expect(checked(toggle)).toBe(String(next)));
  });

  it("keeps the manual Update Erudi card", async () => {
    renderPage();
    expect(await screen.findByText("Update Erudi")).toBeTruthy();
  });

  it("warns on macOS that an update already handed to the system may still install", async () => {
    window.updaterAPI = {
      setPreferences: vi.fn(),
      getState: vi
        .fn()
        .mockResolvedValue({ available: true, phase: "downloaded", handedOff: true }),
      onUpdaterEvent: vi.fn(() => () => {}),
    };
    mockGetByUrl({ userSettings: SETTINGS, appStartup: { backend_type: "mlx" } });
    renderPage();

    expect(await screen.findByText(/may still install when you quit/)).toBeTruthy();
  });

  it("does not warn while install stays on after the hand-off", async () => {
    window.updaterAPI = {
      setPreferences: vi.fn(),
      getState: vi
        .fn()
        .mockResolvedValue({ available: true, phase: "downloaded", handedOff: true }),
      onUpdaterEvent: vi.fn(() => () => {}),
    };
    mockGetByUrl({
      userSettings: { ...SETTINGS, auto_install_updates: true },
      appStartup: { backend_type: "mlx" },
    });
    renderPage();
    const install = await switchNamed(INSTALL);
    await waitFor(() => expect(checked(install)).toBe("true"));

    expect(screen.queryByText(/may still install when you quit/)).toBeNull();
  });

  it("learns about a hand-off from the updater event", async () => {
    // Both update cards listen; main broadcasts to every listener.
    const listeners = [];
    const emit = (payload) => listeners.forEach((listener) => listener(payload));
    window.updaterAPI = {
      setPreferences: vi.fn(),
      getState: vi.fn().mockResolvedValue({ available: true, phase: "idle", handedOff: false }),
      onUpdaterEvent: vi.fn((callback) => {
        listeners.push(callback);
        return () => {};
      }),
    };
    mockGetByUrl({ userSettings: SETTINGS, appStartup: { backend_type: "mlx" } });
    renderPage();
    await switchNamed(CHECK);
    expect(screen.queryByText(/may still install when you quit/)).toBeNull();

    act(() => emit({ event: "install-handed-off" }));

    expect(await screen.findByText(/may still install when you quit/)).toBeTruthy();
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
    // #511: on a CUDA machine pinned back to the processor, backend_type
    // reports the running engine ("cpu"), not the raw hardware -- the card
    // must stay visible so the user has a way back to Automatic.
    mockGetByUrl({
      userSettings: {
        web_search_enabled: false,
        language: "en",
        auto_update_enabled: true,
        inference_backend: "cpu",
      },
      appStartup: { backend_type: "cpu" },
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

/**
 * #511: the Inference engine card saves a preference the backend only ever
 * reads on the CUDA leg (`apply_inference_backend_preference` returns
 * immediately unless the running engine is `CUDA_Engine`). On any other
 * machine the card was a no-op that still triggered a full backend restart.
 * It must render only where the choice is real.
 */
describe("SettingsPage — Inference engine card visibility (#511)", () => {
  it("hides the card on macOS (backend_type mlx) and never restarts the backend for it", async () => {
    window.backendAPI = { restartBackend: vi.fn().mockResolvedValue(undefined) };
    mockGetByUrl({
      userSettings: {
        web_search_enabled: false,
        language: "en",
        auto_update_enabled: true,
        inference_backend: "auto",
      },
      appStartup: { backend_type: "mlx" },
    });
    renderPage();

    await screen.findByText("Web Search");
    expect(screen.queryByText("Inference engine")).toBeNull();
    expect(screen.queryByLabelText("Inference engine", { selector: "select" })).toBeNull();
    expect(window.backendAPI.restartBackend).not.toHaveBeenCalled();
    delete window.backendAPI;
  });

  it("shows the card on a machine running the CUDA engine", async () => {
    mockGetByUrl({
      userSettings: {
        web_search_enabled: false,
        language: "en",
        auto_update_enabled: true,
        inference_backend: "auto",
      },
      appStartup: { backend_type: "cuda" },
    });
    renderPage();

    expect(await screen.findByLabelText("Inference engine")).toBeTruthy();
  });

  it("shows the card when pinned to the processor on a machine that runs CUDA (the way back to Automatic)", async () => {
    mockGetByUrl({
      userSettings: {
        web_search_enabled: false,
        language: "en",
        auto_update_enabled: true,
        inference_backend: "cpu",
      },
      appStartup: { backend_type: "cpu" },
    });
    renderPage();

    expect(await screen.findByLabelText("Inference engine")).toBeTruthy();
  });

  it("hides the card when backend_type is cpu without a processor pin (never had CUDA)", async () => {
    mockGetByUrl({
      userSettings: {
        web_search_enabled: false,
        language: "en",
        auto_update_enabled: true,
        inference_backend: "auto",
      },
      appStartup: { backend_type: "cpu" },
    });
    renderPage();

    await screen.findByText("Web Search");
    expect(screen.queryByText("Inference engine")).toBeNull();
  });
});

describe("showsInferenceEngineSetting", () => {
  it("shows on a CUDA-running machine regardless of the stored preference", () => {
    expect(showsInferenceEngineSetting("cuda", "auto")).toBe(true);
    expect(showsInferenceEngineSetting("cuda", "cpu")).toBe(true);
  });

  it("shows on a CUDA machine pinned back to the processor", () => {
    expect(showsInferenceEngineSetting("cpu", "cpu")).toBe(true);
  });

  it("hides on a machine reporting cpu that was never pinned from CUDA", () => {
    expect(showsInferenceEngineSetting("cpu", "auto")).toBe(false);
  });

  it("hides on macOS (mlx) whatever the stored preference", () => {
    expect(showsInferenceEngineSetting("mlx", "auto")).toBe(false);
    expect(showsInferenceEngineSetting("mlx", "cpu")).toBe(false);
  });

  it("hides while backend_type is unknown (loading or failed to load)", () => {
    expect(showsInferenceEngineSetting(null, "auto")).toBe(false);
    expect(showsInferenceEngineSetting(undefined, "auto")).toBe(false);
    expect(showsInferenceEngineSetting(null, "cpu")).toBe(false);
    expect(showsInferenceEngineSetting(undefined, "cpu")).toBe(false);
  });
});

/**
 * One failure, one record. The settings hook logs the failed PUT where the
 * request is made; the page adding its own turned every failed write into two
 * entries on the Diagnostics page, reading as two defects. What the page does
 * own is the restart, whose failure means something else entirely.
 */
describe("SettingsPage — how a failure is reported", () => {
  let errors;

  beforeEach(() => {
    errors = vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    errors.mockRestore();
  });

  const messages = () => errors.mock.calls.map((call) => String(call[0]));

  it("leaves a failed setting to its single owner, the hook", async () => {
    putMock.mockRejectedValue(new Error("backend is down"));
    renderPage();
    const select = await screen.findByLabelText("Inference engine");

    fireEvent.change(select, { target: { value: "cpu" } });

    await waitFor(() => expect(putMock).toHaveBeenCalled());
    await waitFor(() => expect(messages().length).toBe(1));
    expect(messages()[0]).toMatch(/user settings/i);
  });

  it("does not restart the backend for a preference that was not saved", async () => {
    putMock.mockRejectedValue(new Error("backend is down"));
    window.backendAPI = { restartBackend: vi.fn().mockResolvedValue(undefined) };
    renderPage();
    const select = await screen.findByLabelText("Inference engine");

    fireEvent.change(select, { target: { value: "cpu" } });

    await waitFor(() => expect(putMock).toHaveBeenCalled());
    expect(window.backendAPI.restartBackend).not.toHaveBeenCalled();
    delete window.backendAPI;
  });

  it("reports a failed restart as a failed restart, not as a failed setting", async () => {
    window.backendAPI = {
      restartBackend: vi.fn().mockRejectedValue(new Error("spawn refused")),
    };
    renderPage();
    const select = await screen.findByLabelText("Inference engine");

    fireEvent.change(select, { target: { value: "cpu" } });

    await waitFor(() => expect(window.backendAPI.restartBackend).toHaveBeenCalled());
    await waitFor(() => expect(messages().length).toBe(1));
    expect(messages()[0]).toMatch(/restart/i);
    // The choice IS saved: saying "failed setting" sends people to the wrong
    // place entirely.
    expect(messages()[0]).toMatch(/saved/i);
    delete window.backendAPI;
  });

  it("only tells main about an update choice that persisted", async () => {
    putMock.mockRejectedValue(new Error("backend is down"));
    window.updaterAPI = { setPreferences: vi.fn() };
    renderPage();
    const toggle = await screen.findByLabelText("Check for updates");
    await waitFor(() => expect(toggle.disabled).toBe(false));

    fireEvent.click(toggle);

    await waitFor(() => expect(putMock).toHaveBeenCalled());
    expect(window.updaterAPI.setPreferences).not.toHaveBeenCalled();
    expect(messages().length).toBe(1);
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
    expect(link.getAttribute("href")).toBe(
      "https://www.erudi.app/open-source/desktop/docs/privacy"
    );
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toBe("noreferrer");
  });
});
