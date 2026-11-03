// @vitest-environment jsdom
/**
 * Switching the inference engine restarts the backend, and the card's
 * visibility depends on the engine that RUNS (`backend_type`). The card must
 * stay on screen through the restart, re-read the engine once the backend is
 * ready, and give the select back on every way the switch can end.
 */
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

import SettingsPage from "./SettingsPage";
import { DownloadModalContext } from "../contexts/DownloadModalContext";
import i18n from "../i18n";

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/erudi/settings"]}>
      <SettingsPage />
    </MemoryRouter>
  );
}

function pageWithDownload(download) {
  return (
    <DownloadModalContext.Provider value={download}>
      <MemoryRouter initialEntries={["/erudi/settings"]}>
        <SettingsPage />
      </MemoryRouter>
    </DownloadModalContext.Provider>
  );
}

let appStartup;
let listeners;
let errors;

function settingsWith(inferenceBackend) {
  return {
    web_search_enabled: false,
    language: "en",
    auto_update_enabled: true,
    inference_backend: inferenceBackend,
  };
}

function routeGets(userSettings) {
  getMock.mockImplementation((url) => {
    if (url === "/hardware/app_startup") {
      return typeof appStartup === "function" ? appStartup() : Promise.resolve(appStartup);
    }
    return Promise.resolve(userSettings);
  });
}

function installBridge(overrides = {}) {
  window.backendAPI = {
    restartBackend: vi.fn().mockResolvedValue({ ok: true }),
    onBackendEvent: vi.fn((cb) => {
      listeners.push(cb);
      return () => {
        listeners = listeners.filter((l) => l !== cb);
      };
    }),
    ...overrides,
  };
}

const emitReady = () =>
  act(() => listeners.forEach((cb) => cb({ event: "backend_ready", port: 27182 })));

const startupGets = () => getMock.mock.calls.filter(([url]) => url === "/hardware/app_startup");

beforeEach(() => {
  listeners = [];
  errors = vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(async () => {
  cleanup();
  vi.clearAllMocks();
  errors.mockRestore();
  delete window.backendAPI;
  await i18n.changeLanguage("en");
});

describe("SettingsPage — switching the inference engine", () => {
  it("keeps the card through a switch back to Automatic and re-reads the engine once the backend is ready", async () => {
    // A CUDA machine pinned to the processor: the running engine is CPU.
    appStartup = { backend_type: "cpu" };
    routeGets(settingsWith("cpu"));
    putMock.mockResolvedValue(settingsWith("auto"));
    installBridge();
    renderPage();
    const select = await screen.findByLabelText("Inference engine");
    await waitFor(() => expect(select.value).toBe("cpu"));

    fireEvent.change(select, { target: { value: "auto" } });

    await waitFor(() => expect(window.backendAPI.restartBackend).toHaveBeenCalled());
    // Still there while the backend restarts, and locked: the stale engine
    // (cpu) with the new preference (auto) would otherwise hide it.
    expect(screen.getByLabelText("Inference engine")).toBe(select);
    expect(select.value).toBe("auto");
    expect(select.disabled).toBe(true);

    appStartup = { backend_type: "cuda" };
    await emitReady();

    await waitFor(() => expect(startupGets()).toHaveLength(2));
    await waitFor(() => expect(screen.getByLabelText("Inference engine").disabled).toBe(false));
    expect(screen.getByLabelText("Inference engine").value).toBe("auto");
  });

  it("keeps the select disabled until the restarted backend is ready", async () => {
    appStartup = { backend_type: "cuda" };
    routeGets(settingsWith("auto"));
    putMock.mockResolvedValue(settingsWith("cpu"));
    installBridge();
    renderPage();
    const select = await screen.findByLabelText("Inference engine");
    await waitFor(() => expect(select.disabled).toBe(false));

    fireEvent.change(select, { target: { value: "cpu" } });

    await waitFor(() => expect(window.backendAPI.restartBackend).toHaveBeenCalled());
    // The restart call resolves before the backend is up (main.js does not
    // await it): nothing has been re-read yet.
    await act(async () => {});
    expect(select.disabled).toBe(true);
    expect(startupGets()).toHaveLength(1);

    appStartup = { backend_type: "cpu" };
    await emitReady();

    await waitFor(() => expect(select.disabled).toBe(false));
    expect(screen.getByLabelText("Inference engine").value).toBe("cpu");
  });

  it("gives the select back when the preference could not be saved", async () => {
    appStartup = { backend_type: "cuda" };
    routeGets(settingsWith("auto"));
    putMock.mockRejectedValue(new Error("backend is down"));
    installBridge();
    renderPage();
    const select = await screen.findByLabelText("Inference engine");
    await waitFor(() => expect(select.disabled).toBe(false));

    fireEvent.change(select, { target: { value: "cpu" } });

    await waitFor(() => expect(putMock).toHaveBeenCalled());
    await waitFor(() => expect(select.disabled).toBe(false));
    expect(select.value).toBe("auto");
    expect(window.backendAPI.restartBackend).not.toHaveBeenCalled();
  });

  it("gives the select back when the restart is refused", async () => {
    appStartup = { backend_type: "cuda" };
    routeGets(settingsWith("auto"));
    putMock.mockResolvedValue(settingsWith("cpu"));
    installBridge({ restartBackend: vi.fn().mockRejectedValue(new Error("spawn refused")) });
    renderPage();
    const select = await screen.findByLabelText("Inference engine");
    await waitFor(() => expect(select.disabled).toBe(false));

    fireEvent.change(select, { target: { value: "cpu" } });

    await waitFor(() => expect(window.backendAPI.restartBackend).toHaveBeenCalled());
    await waitFor(() => expect(select.disabled).toBe(false));
  });

  it("gives the select back when there is no Electron bridge to restart through", async () => {
    appStartup = { backend_type: "cuda" };
    routeGets(settingsWith("auto"));
    putMock.mockResolvedValue(settingsWith("cpu"));
    renderPage();
    const select = await screen.findByLabelText("Inference engine");
    await waitFor(() => expect(select.disabled).toBe(false));

    fireEvent.change(select, { target: { value: "cpu" } });

    await waitFor(() => expect(putMock).toHaveBeenCalled());
    await waitFor(() => expect(select.disabled).toBe(false));
  });

  it("gives the select back when the re-read after the restart fails", async () => {
    appStartup = { backend_type: "cuda" };
    routeGets(settingsWith("auto"));
    putMock.mockResolvedValue(settingsWith("cpu"));
    installBridge();
    renderPage();
    const select = await screen.findByLabelText("Inference engine");
    await waitFor(() => expect(select.disabled).toBe(false));

    fireEvent.change(select, { target: { value: "cpu" } });
    await waitFor(() => expect(window.backendAPI.restartBackend).toHaveBeenCalled());

    appStartup = () => Promise.reject(new Error("not up yet"));
    await emitReady();

    await waitFor(() => expect(startupGets()).toHaveLength(2));
    await waitFor(() => expect(select.disabled).toBe(false));
  });
});

/**
 * The restart kills a download in progress, which then comes back "interrupted"
 * and has to start over. While one runs, switching asks first, names that
 * consequence, and on confirmation cancels the download cleanly before the
 * switch.
 */
describe("SettingsPage — switching the engine during a download", () => {
  let order;
  let download;

  beforeEach(() => {
    order = [];
    appStartup = { backend_type: "cuda" };
    routeGets(settingsWith("auto"));
    putMock.mockImplementation(async () => {
      order.push("put");
      return settingsWith("cpu");
    });
    installBridge({
      restartBackend: vi.fn(async () => {
        order.push("restart");
        return { ok: true };
      }),
    });
    download = {
      open: vi.fn(),
      isDownloading: true,
      completionCount: 0,
      lastCompletedAt: null,
      cancelDownload: vi.fn(async () => {
        order.push("cancel");
      }),
    };
  });

  const askForCpu = async () => {
    const select = await screen.findByLabelText("Inference engine");
    await waitFor(() => expect(select.disabled).toBe(false));
    fireEvent.change(select, { target: { value: "cpu" } });
    return select;
  };

  it("asks first, naming the cancelled download, and changes nothing yet", async () => {
    render(pageWithDownload(download));
    const select = await askForCpu();

    const dialog = await screen.findByRole("dialog", { name: "A download is in progress" });
    expect(dialog.textContent).toMatch(/cancels the model download in progress/);
    expect(dialog.textContent).toMatch(/start it again from the beginning/);
    expect(putMock).not.toHaveBeenCalled();
    expect(window.backendAPI.restartBackend).not.toHaveBeenCalled();
    expect(download.cancelDownload).not.toHaveBeenCalled();
    expect(select.value).toBe("auto");
  });

  it("on confirmation cancels the download, then saves and restarts", async () => {
    render(pageWithDownload(download));
    await askForCpu();

    fireEvent.click(await screen.findByRole("button", { name: "Cancel the download and switch" }));

    await waitFor(() => expect(order).toEqual(["cancel", "put", "restart"]));
    expect(putMock).toHaveBeenCalledWith("/user_settings/", { inference_backend: "cpu" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("keeps everything as it was when the user keeps downloading", async () => {
    render(pageWithDownload(download));
    const select = await askForCpu();

    fireEvent.click(await screen.findByRole("button", { name: "Keep downloading" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(order).toEqual([]);
    expect(select.value).toBe("auto");
    expect(select.disabled).toBe(false);
  });

  it("does not cancel a download that finished while the question was open", async () => {
    const { rerender } = render(pageWithDownload(download));
    await askForCpu();
    await screen.findByRole("dialog", { name: "A download is in progress" });

    rerender(pageWithDownload({ ...download, isDownloading: false }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel the download and switch" }));

    await waitFor(() => expect(order).toEqual(["put", "restart"]));
    expect(download.cancelDownload).not.toHaveBeenCalled();
  });

  it("switches straight away when no download is running", async () => {
    render(pageWithDownload({ ...download, isDownloading: false }));
    await askForCpu();

    await waitFor(() => expect(order).toEqual(["put", "restart"]));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
